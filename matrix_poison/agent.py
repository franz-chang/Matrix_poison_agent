from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .attacks import ATTACK_METHODS, AttackRequest, mixed_attack
from .camel_orchestrator import CamelMatrixPoisonCoordinator, CamelRuntimeConfig
from .data import PreparedData, append_malicious_rows, build_user_item_sets, build_user_sequences
from .metrics import hit_ndcg_at_k, objective_from_intent, prediction_shift, stealth_metrics
from .models import TrainConfig, train_model


@dataclass
class MatrixPoisonConfig:
    intent: str = "nuke"  # push | nuke
    model_name: str = "bpr"
    epsilon: float = 0.005
    epsilon_min: float = 0.001
    epsilon_max: float = 0.2
    max_iters: int = 5
    analysis_probe_ratio: float = 0.4
    clean_epochs: int = 20
    probe_epochs: int = 4
    judge_epochs: int = 8
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-6
    embedding_dim: int = 64
    lightgcn_layers: int = 2
    sasrec_max_len: int = 50
    sasrec_heads: int = 2
    sasrec_layers: int = 2
    k: int = 10
    stealth_lambda: float = 0.25
    shift_weight: float = 0.2
    rank_weight: float = 1.0
    target_eval_size: int = 1
    clean_drop_tolerance: float = 0.06
    profile_max_len: int = 120
    push_profile_max_len: int = 24
    positive_rating_threshold: float = 4.0
    push_use_explicit_neg: bool = False
    nuke_use_explicit_neg: bool = True
    push_neg_sample_bias: float = 0.0
    nuke_neg_sample_bias: float = 0.8
    push_min_shift: float = 1e-4
    push_negative_penalty: float = 3.0
    push_shift_on_unseen_only: bool = True
    push_target_shift: float = 0.8
    push_epsilon_max: float = 0.5
    enable_analysis_phase: bool = True
    enable_planning_phase: bool = True
    enable_judgment_phase: bool = True
    enable_self_evolving: bool = True
    fixed_attack_weights: Optional[Dict[str, float]] = None
    camel_enabled: bool = True
    camel_model: str = "openai:gpt-4o-mini"
    camel_use_stub_when_no_api_key: bool = True
    seed: int = 42
    device: str = "auto"


@dataclass
class IterationResult:
    iteration: int
    epsilon: float
    weights: Dict[str, float]
    objective: float
    prediction_shift: float
    hit_at_k: float
    ndcg_at_k: float
    stealth_penalty: float
    reward: float
    clean_drop: float
    pass_judge: bool


@dataclass
class AgentResult:
    target_item: int
    target_category: str
    clean_metrics: Dict[str, float]
    best_iteration: int
    best_reward: float
    best_weights: Dict[str, float]
    best_epsilon: float
    best_metrics: Dict[str, float]
    history: List[Dict[str, object]] = field(default_factory=list)
    malicious_rows: Optional[pd.DataFrame] = None


class MatrixPoisonAgent:
    def __init__(
        self,
        data: PreparedData,
        target_category: str,
        target_item: int,
        cfg: MatrixPoisonConfig,
    ):
        if target_category not in data.category_to_items:
            raise ValueError(f"Unknown target category: {target_category}")
        self.data = data
        self.target_category = target_category
        self.target_item = int(target_item)
        if self.target_item not in set(data.category_to_items[target_category]):
            raise ValueError("target_item must belong to target_category")
        self.cfg = cfg
        self.device = self._resolve_device(cfg.device)
        cat_items = [int(i) for i in self.data.category_to_items[self.target_category]]
        item_pop = self.data.train["item"].value_counts().to_dict()
        cat_items_sorted = sorted(cat_items, key=lambda i: item_pop.get(int(i), 0), reverse=True)
        if self._is_push():
            # Push attacks optimize one designated target item (standard shilling setup).
            self.target_items = {self.target_item}
        elif self.cfg.target_eval_size > 0:
            self.target_items = set(cat_items_sorted[: self.cfg.target_eval_size])
            self.target_items.add(self.target_item)
        else:
            self.target_items = set(cat_items_sorted)

        self.clean_train_user_items = build_user_item_sets(data.train)
        self.eval_users = sorted(set(data.test["user"].astype(int).tolist()) or set(data.train["user"].astype(int).tolist()))
        self.current_epsilon = float(cfg.epsilon)
        if self.cfg.intent.lower() == "push":
            self.current_weights = self._push_weight_prior()
        else:
            self.current_weights = {"random": 0.5, "average": 0.5, "bandwagon": 0.0, "segmented": 0.0}
        self.best_effective_weights: Optional[Dict[str, float]] = None
        self.best_stealth_weights: Optional[Dict[str, float]] = None
        self.stagnation_steps = 0
        self.analysis_ranking: List[str] = []
        self.iter_history: List[IterationResult] = []
        self.camel = CamelMatrixPoisonCoordinator(
            CamelRuntimeConfig(
                enabled=self.cfg.camel_enabled,
                model=self.cfg.camel_model,
                use_stub_when_no_api_key=self.cfg.camel_use_stub_when_no_api_key,
            )
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _is_push(self) -> bool:
        return self.cfg.intent.lower() == "push"

    @staticmethod
    def _normalize_attack_weights(weights: Dict[str, float]) -> Dict[str, float]:
        merged = {k: float(max(0.0, weights.get(k, 0.0))) for k in ATTACK_METHODS}
        s = sum(merged.values())
        if s <= 0:
            return {"random": 1.0, "average": 0.0, "bandwagon": 0.0, "segmented": 0.0}
        return {k: float(v / s) for k, v in merged.items()}

    def _attack_profile_max_len(self) -> int:
        if self._is_push():
            base = int(self.cfg.push_profile_max_len)
            model = self.cfg.model_name.lower()
            if model == "sasrec":
                return int(max(8, min(base, 12)))
            if model == "bpr":
                return int(max(10, min(base, 16)))
            return base
        return int(self.cfg.profile_max_len)

    def _push_weight_prior(self) -> Dict[str, float]:
        model = self.cfg.model_name.lower()
        if model == "lightgcn":
            return {"random": 0.1, "average": 0.55, "bandwagon": 0.1, "segmented": 0.25}
        if model == "neumf":
            return {"random": 0.55, "average": 0.2, "bandwagon": 0.05, "segmented": 0.2}
        if model == "sasrec":
            return {"random": 0.2, "average": 0.1, "bandwagon": 0.5, "segmented": 0.2}
        # bpr
        return {"random": 0.1, "average": 0.45, "bandwagon": 0.05, "segmented": 0.4}

    def _push_epsilon_cap(self) -> float:
        cap = float(self.cfg.push_epsilon_max)
        model = self.cfg.model_name.lower()
        if model == "lightgcn":
            return min(cap, 0.25)
        if model == "sasrec":
            return min(cap, 0.12)
        return min(cap, 0.5)

    def _push_target_shift(self) -> float:
        model = self.cfg.model_name.lower()
        if model == "lightgcn":
            return max(self.cfg.push_target_shift, 1.2)
        if model == "neumf":
            return max(self.cfg.push_target_shift, 1.0)
        if model == "sasrec":
            return max(0.6, self.cfg.push_target_shift * 0.8)
        return max(0.7, self.cfg.push_target_shift * 0.9)

    def _push_objective_weights(self) -> Tuple[float, float]:
        model = self.cfg.model_name.lower()
        if model == "lightgcn":
            return 0.85, 0.35
        if model == "neumf":
            return 0.9, 0.35
        if model == "sasrec":
            return 0.75, 0.45
        return 0.9, 0.4

    def _build_train_cfg(self, epochs: int, seed_offset: int = 0) -> TrainConfig:
        use_explicit_neg = self.cfg.push_use_explicit_neg if self._is_push() else self.cfg.nuke_use_explicit_neg
        neg_bias = self.cfg.push_neg_sample_bias if self._is_push() else self.cfg.nuke_neg_sample_bias
        return TrainConfig(
            model_name=self.cfg.model_name,
            embedding_dim=self.cfg.embedding_dim,
            lightgcn_layers=self.cfg.lightgcn_layers,
            sasrec_max_len=self.cfg.sasrec_max_len,
            sasrec_heads=self.cfg.sasrec_heads,
            sasrec_layers=self.cfg.sasrec_layers,
            epochs=epochs,
            batch_size=self.cfg.batch_size,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            seed=self.cfg.seed + seed_offset,
            device=self.device,
            positive_threshold=self.cfg.positive_rating_threshold,
            use_explicit_neg=use_explicit_neg,
            explicit_neg_bias=neg_bias,
        )

    def _train_model(self, train_df: pd.DataFrame, epochs: int, seed_offset: int = 0):
        n_users = int(train_df["user"].max()) + 1 if not train_df.empty else self.data.num_users
        return train_model(
            train_df=train_df,
            num_users=n_users,
            num_items=self.data.num_items,
            cfg=self._build_train_cfg(epochs=epochs, seed_offset=seed_offset),
        )

    def _category_sensitivity(
        self,
        clean_model,
        poisoned_model,
        clean_sequences: Dict[int, List[int]],
        poison_sequences: Dict[int, List[int]],
    ) -> float:
        cat_items = self.data.category_to_items[self.target_category]
        sample_items = cat_items if len(cat_items) <= 30 else cat_items[:30]
        shifts = []
        users = self.eval_users[: min(len(self.eval_users), 500)]
        for item in sample_items:
            shifts.append(
                abs(
                    prediction_shift(
                        clean_model=clean_model,
                        poisoned_model=poisoned_model,
                        users=users,
                        target_item=int(item),
                        clean_sequences=clean_sequences,
                        poisoned_sequences=poison_sequences,
                    )
                )
            )
        if not shifts:
            return 0.0
        return float(np.mean(shifts))

    def _evaluate_candidate(
        self,
        clean_model,
        clean_hit: float,
        clean_ndcg: float,
        train_poisoned: pd.DataFrame,
        seed_offset: int,
        epochs: int,
    ) -> Dict[str, float]:
        poison_model = self._train_model(train_poisoned, epochs=epochs, seed_offset=seed_offset)

        clean_seq = build_user_sequences(self.data.train, self.cfg.sasrec_max_len)
        poison_seq = build_user_sequences(train_poisoned, self.cfg.sasrec_max_len)
        shift_users = self.eval_users[: min(len(self.eval_users), 1000)]
        if self._is_push() and self.cfg.push_shift_on_unseen_only:
            unseen = [
                int(u) for u in shift_users
                if self.target_item not in self.clean_train_user_items.get(int(u), set())
            ]
            if unseen:
                shift_users = unseen
        pred_shift = prediction_shift(
            clean_model=clean_model,
            poisoned_model=poison_model,
            users=shift_users,
            target_item=self.target_item,
            clean_sequences=clean_seq,
            poisoned_sequences=poison_seq,
        )
        eval_user_items = build_user_item_sets(train_poisoned)
        quality = hit_ndcg_at_k(
            model=poison_model,
            users=self.eval_users[: min(len(self.eval_users), 1000)],
            train_user_items=eval_user_items,
            target_items=self.target_items,
            num_items=self.data.num_items,
            k=self.cfg.k,
            user_sequences=poison_seq,
        )
        hit_delta = float(quality["hit"] - clean_hit)
        ndcg_delta = float(quality["ndcg"] - clean_ndcg)
        stealth = stealth_metrics(self.data.train, train_poisoned)
        clean_drop = float(
            max(
                0.0,
                clean_hit - float(quality["hit"]),
                clean_ndcg - float(quality["ndcg"]),
            )
        )
        stealth_penalty = stealth["rating_kl"] + stealth["profile_shift"]
        shift_term = objective_from_intent(self.cfg.intent, pred_shift)
        if self._is_push():
            rank_term = hit_delta + ndcg_delta
            shift_w, rank_w = self._push_objective_weights()
            obj = shift_w * shift_term + rank_w * rank_term
            if pred_shift <= self.cfg.push_min_shift:
                # Hard guard for push: reject candidates with non-positive target shift.
                obj -= self.cfg.push_negative_penalty * (self.cfg.push_min_shift - pred_shift + 1.0)
        else:
            rank_term = (-hit_delta) + (-ndcg_delta)
            # For nuke, jointly optimize target-score suppression and rank collapse.
            obj = self.cfg.shift_weight * shift_term + self.cfg.rank_weight * rank_term
            if pred_shift >= 0.0:
                # Guard against reverse direction: nuke should lower target score.
                obj -= (pred_shift + 1.0)
        reward = obj - self.cfg.stealth_lambda * stealth_penalty
        sensitivity = self._category_sensitivity(clean_model, poison_model, clean_seq, poison_seq)

        return {
            "prediction_shift": float(pred_shift),
            "objective": float(obj),
            "reward": float(reward),
            "hit": float(quality["hit"]),
            "ndcg": float(quality["ndcg"]),
            "hit_delta": float(hit_delta),
            "ndcg_delta": float(ndcg_delta),
            "clean_drop": float(clean_drop),
            "stealth_penalty": float(stealth_penalty),
            "rating_kl": float(stealth["rating_kl"]),
            "profile_shift": float(stealth["profile_shift"]),
            "interaction_ratio": float(stealth["interaction_ratio"]),
            "category_sensitivity": float(sensitivity),
        }

    def analysis_phase(self, clean_model, clean_hit: float, clean_ndcg: float) -> List[str]:
        if not self.cfg.enable_analysis_phase:
            if self._is_push():
                prior = self._push_weight_prior()
                rank = [k for k, _ in sorted(prior.items(), key=lambda kv: kv[1], reverse=True)]
            else:
                rank = list(ATTACK_METHODS.keys())
            self.analysis_ranking = rank
            return rank

        probe_eps = max(self.cfg.epsilon_min, self.current_epsilon * self.cfg.analysis_probe_ratio)
        scores: Dict[str, float] = {}
        for idx, method in enumerate(ATTACK_METHODS.keys()):
            req = AttackRequest(
                epsilon=probe_eps,
                intent=self.cfg.intent,
                target_item=self.target_item,
                target_items=sorted(self.target_items),
                target_category_items=self.data.category_to_items[self.target_category],
                max_profile_len=self._attack_profile_max_len(),
                seed=self.cfg.seed + 100 + idx,
            )
            mal = mixed_attack(
                train_df=self.data.train,
                num_users=self.data.num_users,
                num_items=self.data.num_items,
                req=req,
                weights={method: 1.0},
            )
            poisoned = append_malicious_rows(self.data.train, mal)
            metrics = self._evaluate_candidate(
                clean_model=clean_model,
                clean_hit=clean_hit,
                clean_ndcg=clean_ndcg,
                train_poisoned=poisoned,
                seed_offset=300 + idx,
                epochs=self.cfg.probe_epochs,
            )
            # Analysis phase follows Eq. (6): category sensitivity from probe attack.
            scores[method] = float(metrics["category_sensitivity"])
        rank = self.camel.analysis_rank(
            method_sensitivity=scores,
            intent=self.cfg.intent,
            target_category=self.target_category,
        )
        if not rank:
            rank = [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
        self.analysis_ranking = rank
        return rank

    def planning_phase(self, iteration: int) -> Dict[str, float]:
        if self.cfg.fixed_attack_weights:
            return self._normalize_attack_weights(self.cfg.fixed_attack_weights)

        if not self.cfg.enable_planning_phase:
            return {"random": 1.0, "average": 0.0, "bandwagon": 0.0, "segmented": 0.0}

        if not self.analysis_ranking:
            self.analysis_ranking = list(ATTACK_METHODS.keys())
        rank = self.analysis_ranking
        rank_weights = np.array([1.0 / (i + 1) for i in range(len(rank))], dtype=np.float64)
        rank_weights /= rank_weights.sum()
        plan = {m: 0.0 for m in ATTACK_METHODS}
        for m, w in zip(rank, rank_weights):
            plan[m] = float(w)

        # Distribution hybridization: blend effective and stealth policies.
        if self.best_effective_weights:
            for k in plan:
                plan[k] = 0.7 * plan[k] + 0.3 * self.best_effective_weights.get(k, 0.0)
        if self.best_stealth_weights:
            beta = 0.15 if iteration > 0 else 0.10
            for k in plan:
                plan[k] = (1.0 - beta) * plan[k] + beta * self.best_stealth_weights.get(k, 0.0)

        if self._is_push():
            prior = self._push_weight_prior()
            prior_alpha = 0.7
            for k in plan:
                plan[k] = prior_alpha * prior.get(k, 0.0) + (1.0 - prior_alpha) * plan[k]

        # Mode switching under stagnation.
        if self.stagnation_steps >= 2 and len(rank) > 1:
            top = rank[0]
            alt = rank[min(1, len(rank) - 1)]
            move = min(0.2, plan[top] * 0.5)
            plan[top] -= move
            plan[alt] += move

        s = sum(max(v, 0.0) for v in plan.values())
        if s <= 0:
            if self._is_push():
                return self._push_weight_prior()
            return {"random": 1.0, "average": 0.0, "bandwagon": 0.0, "segmented": 0.0}
        plan = {k: float(max(v, 0.0) / s) for k, v in plan.items()}

        camel_plan = self.camel.planning_weights(
            base_plan=plan,
            current_weights=self.current_weights,
            iteration=iteration,
            stagnation_steps=self.stagnation_steps,
            intent=self.cfg.intent,
        )
        if camel_plan:
            return self._normalize_attack_weights(camel_plan)
        return plan

    def self_evolve(self, objective: float, stealth_penalty: float, prediction_shift_value: float) -> None:
        if not self.cfg.enable_self_evolving:
            return

        epsilon_max = self._push_epsilon_cap() if self._is_push() else self.cfg.epsilon_max
        camel_next = self.camel.evolve_epsilon(
            current_epsilon=self.current_epsilon,
            objective=objective,
            stealth_penalty=stealth_penalty,
            prediction_shift_value=prediction_shift_value,
            epsilon_min=self.cfg.epsilon_min,
            epsilon_max=epsilon_max,
            is_push=self._is_push(),
            push_target_shift=self._push_target_shift(),
        )
        if camel_next is not None:
            self.current_epsilon = camel_next
            return

        # Semantic-gradient inspired update on poisoning ratio epsilon.
        if self._is_push():
            if stealth_penalty > 0.2:
                self.current_epsilon *= 0.85
            elif prediction_shift_value <= self.cfg.push_min_shift:
                self.current_epsilon *= 2.0
            elif prediction_shift_value < self._push_target_shift():
                self.current_epsilon *= 1.6
            elif objective <= 0:
                self.current_epsilon *= 1.3
            else:
                self.current_epsilon *= 1.08
            self.current_epsilon = float(np.clip(self.current_epsilon, self.cfg.epsilon_min, self._push_epsilon_cap()))
            return

        if prediction_shift_value > -0.1:
            self.current_epsilon *= 1.55
        elif prediction_shift_value > -0.3:
            self.current_epsilon *= 1.35
        elif objective <= 0:
            self.current_epsilon *= 1.20
        elif stealth_penalty > 0.35:
            self.current_epsilon *= 0.92
        else:
            self.current_epsilon *= 1.05
        self.current_epsilon = float(np.clip(self.current_epsilon, self.cfg.epsilon_min, self.cfg.epsilon_max))

    def run(self) -> AgentResult:
        clean_model = self._train_model(self.data.train, epochs=self.cfg.clean_epochs, seed_offset=0)
        clean_sequences = build_user_sequences(self.data.train, self.cfg.sasrec_max_len)
        clean_quality = hit_ndcg_at_k(
            model=clean_model,
            users=self.eval_users[: min(len(self.eval_users), 1000)],
            train_user_items=self.clean_train_user_items,
            target_items=self.target_items,
            num_items=self.data.num_items,
            k=self.cfg.k,
            user_sequences=clean_sequences,
        )
        clean_metrics = {"hit": float(clean_quality["hit"]), "ndcg": float(clean_quality["ndcg"])}

        self.analysis_phase(clean_model=clean_model, clean_hit=clean_metrics["hit"], clean_ndcg=clean_metrics["ndcg"])

        best_reward = -1e18
        best_iter = -1
        best_weights = self.current_weights
        best_eps = self.current_epsilon
        best_metrics: Dict[str, float] = {}
        best_malicious = pd.DataFrame(columns=["user", "item", "rating", "timestamp", "is_malicious"])
        has_valid_best = False

        stealth_sorted: List[Tuple[float, Dict[str, float]]] = []
        effective_sorted: List[Tuple[float, Dict[str, float]]] = []

        for it in range(1, self.cfg.max_iters + 1):
            weights = self.planning_phase(iteration=it)
            self.current_weights = weights
            req = AttackRequest(
                epsilon=self.current_epsilon,
                intent=self.cfg.intent,
                target_item=self.target_item,
                target_items=sorted(self.target_items),
                target_category_items=self.data.category_to_items[self.target_category],
                max_profile_len=self._attack_profile_max_len(),
                seed=self.cfg.seed + it * 13,
            )
            malicious = mixed_attack(
                train_df=self.data.train,
                num_users=self.data.num_users,
                num_items=self.data.num_items,
                req=req,
                weights=weights,
            )
            poisoned = append_malicious_rows(self.data.train, malicious)
            metrics = self._evaluate_candidate(
                clean_model=clean_model,
                clean_hit=clean_metrics["hit"],
                clean_ndcg=clean_metrics["ndcg"],
                train_poisoned=poisoned,
                seed_offset=500 + it,
                epochs=self.cfg.judge_epochs,
            )

            pass_judge = metrics["objective"] > 0
            if self._is_push():
                # Push should preserve clean utility and increase the target score.
                pass_judge = (
                    pass_judge
                    and metrics["clean_drop"] <= self.cfg.clean_drop_tolerance
                    and metrics["prediction_shift"] > self.cfg.push_min_shift
                )
            iter_res = IterationResult(
                iteration=it,
                epsilon=self.current_epsilon,
                weights=weights,
                objective=metrics["objective"],
                prediction_shift=metrics["prediction_shift"],
                hit_at_k=metrics["hit"],
                ndcg_at_k=metrics["ndcg"],
                stealth_penalty=metrics["stealth_penalty"],
                reward=metrics["reward"],
                clean_drop=metrics["clean_drop"],
                pass_judge=pass_judge,
            )
            self.iter_history.append(iter_res)

            if self.cfg.enable_judgment_phase:
                judge = self.camel.judge_adjustment(
                    metrics=metrics,
                    is_push=self._is_push(),
                    clean_drop_tolerance=self.cfg.clean_drop_tolerance,
                    push_min_shift=self.cfg.push_min_shift,
                )
                if judge:
                    pass_judge = bool(judge.get("pass_judge", pass_judge))
                    metrics["reward"] = float(metrics["reward"] + float(judge.get("reward_adjust", 0.0)))
                    iter_res.pass_judge = pass_judge
                    iter_res.reward = float(metrics["reward"])

                stealth_sorted.append((metrics["stealth_penalty"], weights))
                stealth_sorted = sorted(stealth_sorted, key=lambda x: x[0])[:3]
                self.best_stealth_weights = stealth_sorted[0][1]
                effective_sorted.append((metrics["objective"], weights))
                effective_sorted = sorted(effective_sorted, key=lambda x: x[0], reverse=True)[:3]
                self.best_effective_weights = effective_sorted[0][1]
            else:
                pass_judge = True
                iter_res.pass_judge = True

            if self._is_push():
                candidate_valid = metrics["prediction_shift"] > self.cfg.push_min_shift
            else:
                # Nuke candidates should preferably produce negative target shift.
                candidate_valid = metrics["prediction_shift"] < 0.0
            if candidate_valid:
                if not has_valid_best:
                    should_update_best = True
                elif self._is_push():
                    should_update_best = metrics["reward"] > best_reward
                else:
                    prev_shift = float(best_metrics.get("prediction_shift", 1e18)) if best_metrics else 1e18
                    cur_shift = float(metrics.get("prediction_shift", 1e18))
                    should_update_best = (cur_shift < prev_shift - 1e-8) or (abs(cur_shift - prev_shift) <= 1e-8 and metrics["reward"] > best_reward)
            else:
                should_update_best = (not has_valid_best) and (metrics["reward"] > best_reward)
            if should_update_best:
                best_reward = float(metrics["reward"])
                best_iter = it
                best_weights = dict(weights)
                best_eps = float(self.current_epsilon)
                best_metrics = dict(metrics)
                best_malicious = malicious.copy()
                has_valid_best = candidate_valid or has_valid_best
                self.stagnation_steps = 0
            else:
                self.stagnation_steps += 1

            self.self_evolve(
                objective=metrics["objective"],
                stealth_penalty=metrics["stealth_penalty"],
                prediction_shift_value=metrics["prediction_shift"],
            )

        history_dicts = [asdict(x) for x in self.iter_history]
        return AgentResult(
            target_item=self.target_item,
            target_category=self.target_category,
            clean_metrics=clean_metrics,
            best_iteration=best_iter,
            best_reward=best_reward,
            best_weights=best_weights,
            best_epsilon=best_eps,
            best_metrics=best_metrics,
            history=history_dicts,
            malicious_rows=best_malicious,
        )
