#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matrix_poison.agent import MatrixPoisonAgent, MatrixPoisonConfig
from matrix_poison.data import build_user_sequences, prepare_data
from matrix_poison.metrics import leave_one_out_hit_ndcg_at_k

MODEL_ORDER = ["bpr", "neumf", "lightgcn", "sasrec"]
MODEL_LABEL = {
    "bpr": "BPR",
    "neumf": "NeuMF",
    "lightgcn": "LightGCN",
    "sasrec": "SASRec",
}
MARKETS = ["US", "JP", "UK"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Table-4 cross-market baseline experiments.")
    p.add_argument("--xmarket-root", default="", help="Optional root with US/JP/UK subdirs containing interactions.csv and items.csv")
    p.add_argument("--ml1m-interactions", default="data/ml1m/interactions.csv")
    p.add_argument("--ml1m-items", default="data/ml1m/items.csv")
    p.add_argument("--pseudo-market-root", default="outputs/table4_baseline_seed42/generated_cross_market")
    p.add_argument("--output-dir", default="outputs/table4_baseline_seed42")
    p.add_argument("--models", default=",".join(MODEL_ORDER))
    p.add_argument("--target-category", default="Action")
    p.add_argument("--target-item-policy", choices=["head", "median", "tail"], default="head")
    p.add_argument("--target-eval-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--clean-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--sasrec-batch-size", type=int, default=4096)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--lightgcn-layers", type=int, default=2)
    p.add_argument("--sasrec-max-len", type=int, default=50)
    p.add_argument("--sasrec-recent-steps", type=int, default=0, help="SASRec one-step samples from recent transitions (<=0 means all).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--min-positive-rating", type=float, default=1.0)
    p.add_argument("--positive-threshold", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--decimals", type=int, default=4)
    p.add_argument("--eval-user-limit", type=int, default=0, help="Max eval users per market (<=0 means all).")
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_markets(args: argparse.Namespace) -> Tuple[Path, bool]:
    if args.xmarket_root:
        root = Path(args.xmarket_root).expanduser().resolve()
        ok = True
        for m in MARKETS:
            if not ((root / m / "interactions.csv").exists() and (root / m / "items.csv").exists()):
                ok = False
                break
        if ok:
            return root, False

    # Fallback: deterministic pseudo-market split from ML-1M users.
    pseudo_root = (ROOT / args.pseudo_market_root).resolve()
    ensure_dir(pseudo_root)

    inter = pd.read_csv((ROOT / args.ml1m_interactions).resolve())
    items = pd.read_csv((ROOT / args.ml1m_items).resolve())

    users = sorted(inter["user_id"].astype(str).unique().tolist())
    buckets = {"US": set(), "JP": set(), "UK": set()}
    for i, u in enumerate(users):
        if i % 3 == 0:
            buckets["US"].add(u)
        elif i % 3 == 1:
            buckets["JP"].add(u)
        else:
            buckets["UK"].add(u)

    for market, uids in buckets.items():
        mdir = pseudo_root / market
        ensure_dir(mdir)
        sub = inter[inter["user_id"].astype(str).isin(uids)].copy()
        sub.to_csv(mdir / "interactions.csv", index=False)
        items.to_csv(mdir / "items.csv", index=False)

    return pseudo_root, True


def pick_target_item(data, target_category: str, policy: str) -> int:
    items = list(data.category_to_items[target_category])
    pop = data.train["item"].value_counts().to_dict()
    items = sorted(items, key=lambda i: pop.get(int(i), 0), reverse=True)
    if not items:
        raise ValueError(f"No items for target category: {target_category}")
    if policy == "tail":
        return int(items[-1])
    if policy == "median":
        return int(items[len(items) // 2])
    return int(items[0])


def resolve_category(data, preferred: str) -> str:
    if preferred in data.category_to_items and len(data.category_to_items[preferred]) > 0:
        return preferred
    # Fallback to most frequent non-unknown category in train-item map.
    for cat, items in sorted(data.category_to_items.items(), key=lambda kv: len(kv[1]), reverse=True):
        if str(cat).upper() in {"UNKNOWN", "OTHER"}:
            continue
        if len(items) > 0:
            return str(cat)
    return sorted(data.category_to_items.keys())[0]


def run_baseline_for_case(
    interactions: Path,
    items: Path,
    model: str,
    target_category: str,
    target_item_policy: str,
    args: argparse.Namespace,
) -> Dict[str, float | int | str]:
    data = prepare_data(
        interactions_path=str(interactions),
        items_path=str(items),
        min_positive_rating=float(args.min_positive_rating),
        seed=int(args.seed),
    )
    category = resolve_category(data, target_category)
    target_item = pick_target_item(data, category, target_item_policy)

    batch_size = int(args.sasrec_batch_size) if model == "sasrec" else int(args.batch_size)
    cfg = MatrixPoisonConfig(
        intent="nuke",
        model_name=model,
        epsilon=0.02,
        max_iters=1,
        clean_epochs=int(args.clean_epochs),
        probe_epochs=1,
        judge_epochs=1,
        batch_size=batch_size,
        embedding_dim=int(args.embedding_dim),
        lightgcn_layers=int(args.lightgcn_layers),
        sasrec_max_len=int(args.sasrec_max_len),
        sasrec_recent_steps=int(args.sasrec_recent_steps),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        k=int(args.k),
        target_eval_size=int(args.target_eval_size),
        positive_rating_threshold=float(args.positive_threshold),
        seed=int(args.seed),
        device=str(args.device),
        camel_enabled=False,
    )
    agent = MatrixPoisonAgent(
        data=data,
        target_category=category,
        target_item=target_item,
        cfg=cfg,
    )

    clean_model = agent._train_model(agent.data.train, epochs=cfg.clean_epochs, seed_offset=0)
    clean_sequences = build_user_sequences(agent.data.train, cfg.sasrec_max_len)
    quality = leave_one_out_hit_ndcg_at_k(
        model=clean_model,
        test_df=agent.data.test,
        train_user_items=agent.clean_train_user_items,
        num_items=agent.data.num_items,
        k=cfg.k,
        user_sequences=clean_sequences,
        user_limit=int(args.eval_user_limit),
    )

    return {
        "hit": float(quality.get("hit", 0.0)),
        "ndcg": float(quality.get("ndcg", 0.0)),
        "target_category": category,
        "target_item_index": int(target_item),
        "target_item_raw_id": data.index_to_item_id.get(int(target_item), "UNKNOWN"),
        "num_users": int(data.num_users),
        "num_items": int(data.num_items),
    }


def fmt(v: float, decimals: int = 4) -> str:
    return f"{float(v):.{decimals}f}"


def build_latex_table(df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Table 4 Baseline Results in Cross-Market Scenarios (Seed=42).}")
    lines.append("\\label{tab:table4-baseline-seed42}")
    lines.append("\\begin{tabular}{llccccccccc}")
    lines.append("\\toprule")
    lines.append("Methods & Model & \\multicolumn{3}{c}{US} & \\multicolumn{3}{c}{JP} & \\multicolumn{3}{c}{UK} \\\\")
    lines.append("\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11}")
    lines.append(" & & Hit@10 & NDCG@10 & P. Shift & Hit@10 & NDCG@10 & P. Shift & Hit@10 & NDCG@10 & P. Shift \\\\")
    lines.append("\\midrule")

    for _, r in df.iterrows():
        lines.append(
            f"{r['Methods']} & {r['Model']} & "
            f"{r['US Hit@10']} & {r['US NDCG@10']} & {r['US P. Shift']} & "
            f"{r['JP Hit@10']} & {r['JP NDCG@10']} & {r['JP P. Shift']} & "
            f"{r['UK Hit@10']} & {r['UK NDCG@10']} & {r['UK P. Shift']} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    out_dir = (ROOT / args.output_dir).resolve()
    ensure_dir(out_dir)

    market_root, used_pseudo = resolve_markets(args)

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    rows: List[Dict[str, str]] = []
    details: List[Dict[str, object]] = []

    for model in models:
        row: Dict[str, str] = {
            "Methods": "Baseline",
            "Model": MODEL_LABEL.get(model, model),
        }
        for market in MARKETS:
            inter = market_root / market / "interactions.csv"
            items = market_root / market / "items.csv"
            result = run_baseline_for_case(
                interactions=inter,
                items=items,
                model=model,
                target_category=args.target_category,
                target_item_policy=args.target_item_policy,
                args=args,
            )
            row[f"{market} Hit@10"] = fmt(float(result["hit"]), args.decimals)
            row[f"{market} NDCG@10"] = fmt(float(result["ndcg"]), args.decimals)
            row[f"{market} P. Shift"] = "-"

            details.append(
                {
                    "market": market,
                    "model": model,
                    **result,
                }
            )
            print(
                f"[baseline][{market}][{model}] "
                f"hit={float(result['hit']):.4f} ndcg={float(result['ndcg']):.4f} "
                f"target_category={result['target_category']} target_item={result['target_item_raw_id']}",
                flush=True,
            )

        rows.append(row)

    table_df = pd.DataFrame(rows)
    table_csv = out_dir / "table4_baseline_seed42.csv"
    table_df.to_csv(table_csv, index=False)

    latex = build_latex_table(table_df)
    tex_path = out_dir / "table4_baseline_seed42.tex"
    tex_path.write_text(latex, encoding="utf-8")

    md_path = out_dir / "table4_baseline_seed42.md"
    md_lines = ["| " + " | ".join(table_df.columns.tolist()) + " |", "| " + " | ".join(["---"] * len(table_df.columns)) + " |"]
    for _, r in table_df.iterrows():
        md_lines.append("| " + " | ".join(str(r[c]) for c in table_df.columns) + " |")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    detail_csv = out_dir / "table4_baseline_seed42_details.csv"
    pd.DataFrame(details).to_csv(detail_csv, index=False)

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(args.seed),
        "market_root": str(market_root),
        "used_pseudo_markets_from_ml1m": bool(used_pseudo),
        "target_category_preference": args.target_category,
        "target_item_policy": args.target_item_policy,
        "models": models,
        "args": vars(args),
    }
    (out_dir / "table4_baseline_seed42_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] table csv: {table_csv}")
    print(f"[done] latex: {tex_path}")
    print(f"[done] details: {detail_csv}")


if __name__ == "__main__":
    main()
