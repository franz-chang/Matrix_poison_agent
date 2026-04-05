#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

import sys

PROJECT_ROOT = Path('/root/matrix-poison-camel')
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matrix_poison.agent import MatrixPoisonAgent, MatrixPoisonConfig
from matrix_poison.data import build_user_sequences, prepare_data
from matrix_poison.metrics import hit_ndcg_at_k


DATASETS = {
    "ml1m": {
        "label": "ML-1M",
        "interactions": "data/ml1m/interactions.csv",
        "items": "data/ml1m/items.csv",
        "target_prefs": ["Thriller", "Action", "Drama"],
        "exclude_categories": {"UNKNOWN", "OTHER"},
        "target_item_policy": "head",
    },
    "lastfm": {
        "label": "LastFM",
        "interactions": "data/lastfm/interactions.csv",
        "items": "data/lastfm/items.csv",
        "target_prefs": ["rock", "alternative", "pop", "electronic"],
        "exclude_categories": {"UNKNOWN", "OTHER"},
        "target_item_policy": "head",
    },
    "automotive": {
        "label": "Automotive",
        "interactions": "data/automotive/interactions.csv",
        "items": "data/automotive/items.csv",
        "target_prefs": ["Automotive"],
        "exclude_categories": set(),
        "target_item_policy": "head",
    },
}

MODEL_ORDER = ["bpr", "neumf", "lightgcn", "sasrec"]
MODEL_LABEL = {
    "bpr": "BPR",
    "neumf": "NeuMF",
    "lightgcn": "LightGCN",
    "sasrec": "SASRec",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baseline-only experiments for selected datasets.")
    p.add_argument("--datasets", default="ml1m,lastfm")
    p.add_argument("--models", default=",".join(MODEL_ORDER))
    p.add_argument("--output-dir", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--clean-epochs", type=int, default=20)
    p.add_argument("--clean-epochs-lastfm", type=int, default=28)
    p.add_argument("--clean-epochs-automotive", type=int, default=28)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--sasrec-batch-size", type=int, default=3072)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--lightgcn-layers", type=int, default=2)
    p.add_argument("--sasrec-max-len", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--target-eval-size", type=int, default=1)
    p.add_argument("--strict-single-target-eval", action="store_true")
    p.add_argument("--min-positive-rating", type=float, default=1.0)
    p.add_argument("--positive-threshold", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--decimals", type=int, default=4)
    p.add_argument(
        "--target-item-overrides",
        default="",
        help="Comma-separated dataset:item override, item can be raw item_id or encoded index, e.g. lastfm:378",
    )
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_float(v, d=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(d)


def _parse_target_item_overrides(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not raw:
        return out
    for part in str(raw).split(","):
        p = part.strip()
        if not p or ":" not in p:
            continue
        ds, item = p.split(":", 1)
        ds = ds.strip().lower()
        item = item.strip()
        if ds and item:
            out[ds] = item
    return out


def _resolve_override_item(data, raw_item: str) -> int:
    if raw_item in data.item_id_to_index:
        return int(data.item_id_to_index[raw_item])
    return int(raw_item)


def _select_target_category(data, dataset_key: str, items_df: pd.DataFrame) -> str:
    cfg = DATASETS[dataset_key]
    candidates = [c for c in cfg["target_prefs"] if c in data.category_to_items and len(data.category_to_items[c]) > 0]
    if candidates:
        return candidates[0]

    counts = items_df["category"].astype(str).value_counts().to_dict()
    excluded = cfg["exclude_categories"]
    sorted_cats = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    for cat, _ in sorted_cats:
        if cat in excluded:
            continue
        if cat in data.category_to_items and len(data.category_to_items[cat]) > 0:
            return str(cat)

    return sorted(data.category_to_items.keys())[0]


def _select_target_item(data, target_category: str, policy: str = "head") -> int:
    cands = list(data.category_to_items[target_category])
    pop = data.train["item"].value_counts().to_dict()
    cands = sorted(cands, key=lambda i: pop.get(int(i), 0), reverse=True)
    if policy == "tail":
        return int(cands[-1])
    if policy == "median":
        return int(cands[len(cands) // 2])
    return int(cands[0])


def _build_agent(data, dataset_key: str, model: str, target_category: str, target_item: int, args: argparse.Namespace, seed_offset: int = 0) -> MatrixPoisonAgent:
    batch_size = args.sasrec_batch_size if model.lower() == "sasrec" else args.batch_size
    if dataset_key == "lastfm":
        clean_epochs = args.clean_epochs_lastfm
    elif dataset_key == "automotive":
        clean_epochs = args.clean_epochs_automotive
    else:
        clean_epochs = args.clean_epochs
    cfg = MatrixPoisonConfig(
        intent="nuke",
        model_name=model,
        epsilon=0.02,
        max_iters=1,
        clean_epochs=int(clean_epochs),
        probe_epochs=1,
        judge_epochs=2,
        batch_size=batch_size,
        embedding_dim=args.embedding_dim,
        lightgcn_layers=args.lightgcn_layers,
        sasrec_max_len=args.sasrec_max_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        k=args.k,
        target_eval_size=args.target_eval_size,
        positive_rating_threshold=args.positive_threshold,
        seed=args.seed + seed_offset,
        device=args.device,
        camel_enabled=False,
    )
    return MatrixPoisonAgent(data=data, target_category=target_category, target_item=target_item, cfg=cfg)


def _compute_clean(agent: MatrixPoisonAgent) -> Tuple[float, float]:
    clean_model = agent._train_model(agent.data.train, epochs=agent.cfg.clean_epochs, seed_offset=0)
    clean_sequences = build_user_sequences(agent.data.train, agent.cfg.sasrec_max_len)
    quality = hit_ndcg_at_k(
        model=clean_model,
        users=agent.eval_users[: min(len(agent.eval_users), 1000)],
        train_user_items=agent.clean_train_user_items,
        target_items=agent.target_items,
        num_items=agent.data.num_items,
        k=agent.cfg.k,
        user_sequences=clean_sequences,
    )
    return _safe_float(quality.get("hit", 0.0)), _safe_float(quality.get("ndcg", 0.0))


def _save_markdown(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_pdf_table(df: pd.DataFrame, path: Path, title: str) -> None:
    ensure_dir(path.parent)
    with PdfPages(path) as pdf:
        fig_h = max(5.0, min(18.0, 1.2 + 0.36 * (len(df) + 2)))
        fig, ax = plt.subplots(figsize=(14, fig_h))
        ax.axis("off")
        ax.text(0.0, 1.02, title, fontsize=13, fontweight="bold", transform=ax.transAxes)
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc="center",
            loc="upper left",
            bbox=[0, 0, 1, 0.95],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.2)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    overrides = _parse_target_item_overrides(args.target_item_overrides)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / f"baseline_only_{ts}"
    ensure_dir(out_root)

    selected = {}
    rows = []

    for ds in datasets:
        inter_path = PROJECT_ROOT / DATASETS[ds]["interactions"]
        items_path = PROJECT_ROOT / DATASETS[ds]["items"]
        data = prepare_data(
            interactions_path=str(inter_path),
            items_path=str(items_path),
            min_positive_rating=args.min_positive_rating,
            seed=args.seed,
        )
        items_df = pd.read_csv(items_path)
        target_category = _select_target_category(data, ds, items_df)
        target_item = _select_target_item(data, target_category, policy=DATASETS[ds]["target_item_policy"])
        if ds in overrides:
            target_item = _resolve_override_item(data, overrides[ds])

        selected[ds] = {
            "dataset_label": DATASETS[ds]["label"],
            "target_category": target_category,
            "target_item_index": int(target_item),
            "target_item_raw_id": data.index_to_item_id.get(int(target_item), "UNKNOWN"),
        }

        for model_idx, model in enumerate(models):
            agent = _build_agent(
                data=data,
                dataset_key=ds,
                model=model,
                target_category=target_category,
                target_item=target_item,
                args=args,
                seed_offset=model_idx,
            )
            if args.strict_single_target_eval:
                agent.target_items = {int(target_item)}

            hit, ndcg = _compute_clean(agent)
            row = {
                "Methods": "Baseline",
                "Model": MODEL_LABEL.get(model, model),
                f"{DATASETS[ds]['label']} Hit@10": f"{hit:.{args.decimals}f}",
                f"{DATASETS[ds]['label']} NDCG@10": f"{ndcg:.{args.decimals}f}",
                f"{DATASETS[ds]['label']} P. Shift": "-",
            }
            rows.append((ds, model, row))
            print(f"[baseline] {ds} {model} hit={hit:.4f} ndcg={ndcg:.4f}", flush=True)

    merged_rows = []
    for model in models:
        out = {"Methods": "Baseline", "Model": MODEL_LABEL.get(model, model)}
        for ds in datasets:
            for dds, mm, r in rows:
                if dds == ds and mm == model:
                    out.update(r)
                    break
        merged_rows.append(out)

    df = pd.DataFrame(merged_rows)
    csv_path = out_root / "table_baseline_only.csv"
    md_path = out_root / "table_baseline_only.md"
    pdf_path = out_root / "table_baseline_only.pdf"
    target_path = out_root / "selected_targets.json"

    df.to_csv(csv_path, index=False)
    _save_markdown(df, md_path)
    _save_pdf_table(df, pdf_path, "Baseline Only (Selected Datasets)")
    target_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved: {csv_path}")
    print(f"saved: {md_path}")
    print(f"saved: {pdf_path}")
    print(f"saved: {target_path}")


if __name__ == "__main__":
    main()
