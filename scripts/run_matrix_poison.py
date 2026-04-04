from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matrix_poison.agent import MatrixPoisonAgent, MatrixPoisonConfig
from matrix_poison.attacks import ATTACK_METHODS
from matrix_poison.data import PreparedData, prepare_data


def _pick_target_item(data: PreparedData, target_category: str) -> int:
    items = data.category_to_items[target_category]
    pop = data.train["item"].value_counts().to_dict()
    items_sorted = sorted(items, key=lambda i: pop.get(int(i), 0), reverse=True)
    return int(items_sorted[0])


def _resolve_target_item(data: PreparedData, raw_target_item: str | None, target_category: str) -> int:
    if not raw_target_item:
        return _pick_target_item(data, target_category)
    if raw_target_item in data.item_id_to_index:
        idx = data.item_id_to_index[raw_target_item]
    else:
        try:
            idx = int(raw_target_item)
        except ValueError as exc:
            raise ValueError(f"target item `{raw_target_item}` not found in item_id mapping.") from exc
    if idx not in data.category_to_items[target_category]:
        raise ValueError(f"target item {idx} is not in target category `{target_category}`")
    return int(idx)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Matrix-Poison reproduction attack.")
    p.add_argument("--interactions", required=True, help="Path to interactions.csv")
    p.add_argument("--items", required=True, help="Path to items.csv")
    p.add_argument("--target-category", required=True, help="Target category label in items.csv")
    p.add_argument("--target-item", default=None, help="Target item raw id (preferred) or encoded item index")
    p.add_argument("--intent", choices=["push", "nuke"], default="nuke")
    p.add_argument("--model", choices=["bpr", "neumf", "lightgcn", "sasrec"], default="bpr")
    p.add_argument("--min-positive-rating", type=float, default=1.0)
    p.add_argument("--epsilon", type=float, default=0.005)
    p.add_argument("--max-iters", type=int, default=5)
    p.add_argument("--clean-epochs", type=int, default=20)
    p.add_argument("--probe-epochs", type=int, default=4)
    p.add_argument("--judge-epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--lightgcn-layers", type=int, default=2)
    p.add_argument("--sasrec-max-len", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--target-eval-size", type=int, default=5, help="Top-N popular items in target category used for Hit/NDCG")
    p.add_argument("--profile-max-len", type=int, default=120, help="Max fake profile length for nuke attack generation")
    p.add_argument("--push-profile-max-len", type=int, default=24, help="Max fake profile length for push attack generation")
    p.add_argument("--positive-rating-threshold", type=float, default=4.0, help="Rating threshold used as positive feedback in training")
    p.add_argument("--push-use-explicit-neg", action="store_true", help="Enable explicit-negative sampling in push mode")
    p.add_argument("--push-neg-sample-bias", type=float, default=0.0, help="Probability to sample explicit negatives in push mode")
    p.add_argument("--nuke-neg-sample-bias", type=float, default=0.8, help="Probability to sample explicit negatives in nuke mode")
    p.add_argument("--push-min-shift", type=float, default=1e-4, help="Minimal positive prediction shift required for push judge")
    p.add_argument("--push-negative-penalty", type=float, default=3.0, help="Penalty weight when push shift is not positive")
    p.add_argument("--push-target-shift", type=float, default=0.8, help="Target prediction shift used for push epsilon evolution")
    p.add_argument("--push-epsilon-max", type=float, default=0.5, help="Max epsilon cap for push mode")
    p.add_argument("--disable-push-shift-unseen-only", action="store_true", help="If set, push shift uses all eval users instead of unseen users")
    p.add_argument("--disable-analysis-phase", action="store_true", help="Disable analysis phase (ablation: w/o analysis)")
    p.add_argument("--disable-planning-phase", action="store_true", help="Disable planning phase (ablation: w/o planning)")
    p.add_argument("--disable-judgment-phase", action="store_true", help="Disable judgment phase (ablation: w/o judgment)")
    p.add_argument("--disable-self-evolving", action="store_true", help="Disable self-evolving phase (ablation: w/o self-evolving)")
    p.add_argument(
        "--fixed-attack-method",
        choices=sorted(ATTACK_METHODS.keys()),
        default=None,
        help="Force a single attack method throughout all iterations (for baseline comparison).",
    )
    p.add_argument("--disable-camel", action="store_true", help="Disable CAMEL orchestrator.")
    p.add_argument("--camel-model", default="openai:gpt-4o-mini", help="CAMEL model spec, e.g. openai:gpt-4o-mini")
    p.add_argument("--camel-no-stub-fallback", action="store_true", help="Disable CAMEL STUB fallback when API key is absent")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto/cpu/cuda")
    p.add_argument("--output-dir", default="outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_data(
        interactions_path=args.interactions,
        items_path=args.items,
        min_positive_rating=args.min_positive_rating,
        seed=args.seed,
    )
    categories = sorted(data.category_to_items.keys())
    if args.target_category not in data.category_to_items:
        raise ValueError(f"Unknown target category `{args.target_category}`. Available: {categories[:30]}")
    target_item = _resolve_target_item(data, args.target_item, args.target_category)

    cfg = MatrixPoisonConfig(
        intent=args.intent,
        model_name=args.model,
        epsilon=args.epsilon,
        max_iters=args.max_iters,
        clean_epochs=args.clean_epochs,
        probe_epochs=args.probe_epochs,
        judge_epochs=args.judge_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        embedding_dim=args.embedding_dim,
        lightgcn_layers=args.lightgcn_layers,
        sasrec_max_len=args.sasrec_max_len,
        k=args.k,
        target_eval_size=args.target_eval_size,
        profile_max_len=args.profile_max_len,
        push_profile_max_len=args.push_profile_max_len,
        positive_rating_threshold=args.positive_rating_threshold,
        push_use_explicit_neg=args.push_use_explicit_neg,
        push_neg_sample_bias=args.push_neg_sample_bias,
        nuke_neg_sample_bias=args.nuke_neg_sample_bias,
        push_min_shift=args.push_min_shift,
        push_negative_penalty=args.push_negative_penalty,
        push_shift_on_unseen_only=not args.disable_push_shift_unseen_only,
        push_target_shift=args.push_target_shift,
        push_epsilon_max=args.push_epsilon_max,
        enable_analysis_phase=not args.disable_analysis_phase,
        enable_planning_phase=not args.disable_planning_phase,
        enable_judgment_phase=not args.disable_judgment_phase,
        enable_self_evolving=not args.disable_self_evolving,
        fixed_attack_weights={args.fixed_attack_method: 1.0} if args.fixed_attack_method else None,
        camel_enabled=not args.disable_camel,
        camel_model=args.camel_model,
        camel_use_stub_when_no_api_key=not args.camel_no_stub_fallback,
        seed=args.seed,
        device=args.device,
    )

    agent = MatrixPoisonAgent(
        data=data,
        target_category=args.target_category,
        target_item=target_item,
        cfg=cfg,
    )
    result = agent.run()

    malicious = result.malicious_rows.copy() if result.malicious_rows is not None else pd.DataFrame()
    if not malicious.empty:
        malicious["item_raw_id"] = malicious["item"].map(data.index_to_item_id).fillna("UNKNOWN")
        malicious.to_csv(out_dir / "best_malicious_profiles.csv", index=False)

    summary = {
        "target_category": result.target_category,
        "target_item_index": result.target_item,
        "target_item_raw_id": data.index_to_item_id.get(result.target_item, "UNKNOWN"),
        "clean_metrics": result.clean_metrics,
        "best_iteration": result.best_iteration,
        "best_reward": result.best_reward,
        "best_weights": result.best_weights,
        "best_epsilon": result.best_epsilon,
        "best_metrics": result.best_metrics,
        "history": result.history,
    }
    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== Matrix-Poison Reproduction Finished ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Artifacts written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
