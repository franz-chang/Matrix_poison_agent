from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class CamelRuntimeConfig:
    enabled: bool = True
    model: str = "openai:gpt-4o-mini"
    use_stub_when_no_api_key: bool = True
    openai_api_key_env: str = "OPENAI_API_KEY"


class CamelMatrixPoisonCoordinator:
    """LLM role coordinator built on CAMEL ChatAgent.

    This class is intentionally fail-safe:
    - If CAMEL is unavailable, it becomes inactive.
    - If model/API key is unavailable, it falls back to STUB mode.
    - If the model output is not valid JSON, deterministic fallbacks are used.
    """

    def __init__(self, cfg: CamelRuntimeConfig):
        self.cfg = cfg
        self.active = False
        self.using_stub = False
        self._analysis_agent = None
        self._planning_agent = None
        self._judge_agent = None
        self._evolve_agent = None
        self.error_reason: Optional[str] = None

        if not cfg.enabled:
            self.error_reason = "camel disabled by config"
            return

        try:
            from camel.agents import ChatAgent
            from camel.types import ModelPlatformType, ModelType
        except Exception as exc:
            self.error_reason = f"camel import failed: {exc}"
            return

        model_spec: Any = cfg.model
        api_key = os.getenv(cfg.openai_api_key_env, "").strip()
        if not api_key:
            if cfg.use_stub_when_no_api_key:
                model_spec = (ModelPlatformType.STUB, ModelType.STUB)
                self.using_stub = True
            else:
                self.error_reason = f"missing {cfg.openai_api_key_env}"
                return

        try:
            self._analysis_agent = ChatAgent(
                system_message=(
                    "You are the Analysis Agent for a recommender-system poisoning framework. "
                    "Given vulnerability scores of attack methods, output strict JSON."
                ),
                model=model_spec,
            )
            self._planning_agent = ChatAgent(
                system_message=(
                    "You are the Planning Agent. Optimize attack-mixture weights to maximize objective "
                    "while maintaining stealth. Output strict JSON only."
                ),
                model=model_spec,
            )
            self._judge_agent = ChatAgent(
                system_message=(
                    "You are the Judgement Agent. Assess candidate attack quality and safety constraints. "
                    "Output strict JSON only."
                ),
                model=model_spec,
            )
            self._evolve_agent = ChatAgent(
                system_message=(
                    "You are the Self-Evolving Optimizer Agent. Update epsilon for next iteration "
                    "according to attack outcome. Output strict JSON only."
                ),
                model=model_spec,
            )
            self.active = True
        except Exception as exc:
            self.error_reason = f"camel init failed: {exc}"
            self.active = False

    @staticmethod
    def _normalize_weights(weights: Dict[str, float], methods: List[str]) -> Dict[str, float]:
        merged = {m: float(max(0.0, weights.get(m, 0.0))) for m in methods}
        s = sum(merged.values())
        if s <= 0:
            return {m: (1.0 if i == 0 else 0.0) for i, m in enumerate(methods)}
        return {m: float(v / s) for m, v in merged.items()}

    @staticmethod
    def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
        text = (text or "").strip()
        if not text:
            return None
        # Fast path: direct JSON
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        # Robust path: extract first {...} block
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _agent_json(self, agent: Any, prompt: str) -> Optional[Dict[str, Any]]:
        if not self.active or agent is None:
            return None
        try:
            response = agent.step(prompt)
            msgs = getattr(response, "msgs", None) or []
            if not msgs:
                return None
            content = str(getattr(msgs[0], "content", "") or "")
            return self._extract_json_obj(content)
        except Exception:
            return None

    def analysis_rank(
        self,
        method_sensitivity: Dict[str, float],
        intent: str,
        target_category: str,
    ) -> Optional[List[str]]:
        methods = list(method_sensitivity.keys())
        default_rank = [k for k, _ in sorted(method_sensitivity.items(), key=lambda kv: kv[1], reverse=True)]
        if not self.active:
            return default_rank

        prompt = (
            "Rank attack methods by expected category-level vulnerability.\n"
            f"intent={intent}, target_category={target_category}\n"
            f"method_sensitivity={json.dumps(method_sensitivity, ensure_ascii=False)}\n"
            "Return JSON ONLY: "
            '{"ranking": ["m1", "m2", "m3", "m4"], "reason": "short"}'
        )
        obj = self._agent_json(self._analysis_agent, prompt)
        if not obj:
            return default_rank
        ranking = obj.get("ranking")
        if not isinstance(ranking, list):
            return default_rank
        filtered = [str(x) for x in ranking if str(x) in methods]
        missing = [m for m in methods if m not in filtered]
        out = filtered + missing
        if len(out) != len(methods):
            return default_rank
        return out

    def planning_weights(
        self,
        base_plan: Dict[str, float],
        current_weights: Dict[str, float],
        iteration: int,
        stagnation_steps: int,
        intent: str,
    ) -> Optional[Dict[str, float]]:
        methods = list(base_plan.keys())
        default_plan = self._normalize_weights(base_plan, methods)
        if not self.active:
            return default_plan

        prompt = (
            "Optimize mixture weights for attack methods.\n"
            f"intent={intent}, iteration={iteration}, stagnation_steps={stagnation_steps}\n"
            f"base_plan={json.dumps(base_plan, ensure_ascii=False)}\n"
            f"current_weights={json.dumps(current_weights, ensure_ascii=False)}\n"
            "Return JSON ONLY: "
            '{"weights": {"random": 0.25, "average": 0.25, "bandwagon": 0.25, "segmented": 0.25}, '
            '"reason": "short"}'
        )
        obj = self._agent_json(self._planning_agent, prompt)
        if not obj:
            return default_plan
        weights = obj.get("weights")
        if not isinstance(weights, dict):
            return default_plan
        return self._normalize_weights({k: float(v) for k, v in weights.items()}, methods)

    def judge_adjustment(
        self,
        metrics: Dict[str, float],
        is_push: bool,
        clean_drop_tolerance: float,
        push_min_shift: float,
    ) -> Optional[Dict[str, Any]]:
        default_pass = metrics.get("objective", 0.0) > 0
        if is_push:
            default_pass = (
                default_pass
                and metrics.get("clean_drop", 0.0) <= clean_drop_tolerance
                and metrics.get("prediction_shift", 0.0) > push_min_shift
            )
        if not self.active:
            return {"pass_judge": bool(default_pass), "reward_adjust": 0.0, "reason": "rule_fallback"}

        prompt = (
            "Assess whether the candidate attack should pass judging.\n"
            f"is_push={is_push}, clean_drop_tolerance={clean_drop_tolerance}, push_min_shift={push_min_shift}\n"
            f"metrics={json.dumps(metrics, ensure_ascii=False)}\n"
            "Return JSON ONLY: "
            '{"pass_judge": true, "reward_adjust": 0.0, "reason": "short"}'
        )
        obj = self._agent_json(self._judge_agent, prompt)
        if not obj:
            return {"pass_judge": bool(default_pass), "reward_adjust": 0.0, "reason": "rule_fallback"}

        try:
            return {
                "pass_judge": bool(obj.get("pass_judge", default_pass)),
                "reward_adjust": float(obj.get("reward_adjust", 0.0)),
                "reason": str(obj.get("reason", "")),
            }
        except Exception:
            return {"pass_judge": bool(default_pass), "reward_adjust": 0.0, "reason": "rule_fallback"}

    def evolve_epsilon(
        self,
        current_epsilon: float,
        objective: float,
        stealth_penalty: float,
        prediction_shift_value: float,
        epsilon_min: float,
        epsilon_max: float,
        is_push: bool,
        push_target_shift: float,
    ) -> Optional[float]:
        if not self.active:
            return None
        prompt = (
            "Update epsilon for next iteration.\n"
            f"is_push={is_push}, current_epsilon={current_epsilon}, objective={objective}, "
            f"stealth_penalty={stealth_penalty}, prediction_shift={prediction_shift_value}, "
            f"epsilon_min={epsilon_min}, epsilon_max={epsilon_max}, push_target_shift={push_target_shift}\n"
            "Return JSON ONLY: "
            '{"next_epsilon": 0.01, "reason": "short"}'
        )
        obj = self._agent_json(self._evolve_agent, prompt)
        if not obj:
            return None
        try:
            next_eps = float(obj.get("next_epsilon", current_epsilon))
        except Exception:
            return None
        if not np.isfinite(next_eps):
            return None
        return float(np.clip(next_eps, epsilon_min, epsilon_max))
