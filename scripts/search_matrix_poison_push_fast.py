#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matrix_poison.agent import MatrixPoisonAgent, MatrixPoisonConfig
from matrix_poison.data import prepare_data

DATASETS = {
    "ml1m": {
        "label": "ML-1M",
        "interactions": "data/ml1m/interactions.csv",
        "items": "data/ml1m/items.csv",
        "target_category": "Thriller",
        "target_item_raw": "2762",
        "clean_epochs": 20,
    },
    "lastfm": {
        "label": "LastFM",
        "interactions": "data/lastfm/interactions.csv",
        "items": "data/lastfm/items.csv",
        "target_category": "rock",
        "target_item_raw": "378",
        "clean_epochs": 28,
    },
    "automotive": {
        "label": "Automotive",
        "interactions": "data/automotive/interactions.csv",
        "items": "data/automotive/items.csv",
        "target_category": "Automotive",
        "target_item_raw": "406",
        "clean_epochs": 28,
    },
}

MODEL_ORDER = ["bpr", "neumf", "lightgcn", "sasrec"]
MODEL_LABEL = {
    "bpr": "BPR",
    "neumf": "NeuMF",
    "lightgcn": "LightGCN",
    "sasrec": "SASRec",
}
MODEL_KEY_FROM_LABEL = {v: k for k, v in MODEL_LABEL.items()}

PRESETS = [
    {
        "name": "p1_thr4_balanced",
        "positive_threshold": 4.0,
        "epsilon": 0.03,
        "max_iters": 6,
        "probe_epochs": 2,
        "judge_epochs": 8,
        "push_profile_max_len": 24,
        "push_target_shift": 0.8,
        "push_epsilon_max": 0.5,
        "push_negative_penalty": 3.0,
        "push_use_explicit_neg": False,
        "push_neg_sample_bias": 0.0,
        "push_shift_on_unseen_only": False,
    },
    {
        "name": "p2_thr4_aggressive",
        "positive_threshold": 4.0,
        "epsilon": 0.06,
        "max_iters": 8,
        "probe_epochs": 2,
        "judge_epochs": 8,
        "push_profile_max_len": 32,
        "push_target_shift": 1.0,
        "push_epsilon_max": 0.5,
        "push_negative_penalty": 2.5,
        "push_use_explicit_neg": False,
        "push_neg_sample_bias": 0.0,
        "push_shift_on_unseen_only": False,
    },
    {
        "name": "p3_thr1_fallback",
        "positive_threshold": 1.0,
        "epsilon": 0.04,
        "max_iters": 6,
        "probe_epochs": 2,
        "judge_epochs": 8,
        "push_profile_max_len": 24,
        "push_target_shift": 0.8,
        "push_epsilon_max": 0.5,
        "push_negative_penalty": 3.0,
        "push_use_explicit_neg": False,
        "push_neg_sample_bias": 0.0,
        "push_shift_on_unseen_only": False,
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/sota_seed42_matrix_push_tuned")
    p.add_argument("--baseline-table", default="outputs/baseline_seed42_run1/table_baseline_only.csv")
    p.add_argument("--nuke-table", default="outputs/sota_seed42_matrix_nuke_final_best/table_matrix_poison_nuke_seed42_best.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--sasrec-batch-size", type=int, default=4096)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--lightgcn-layers", type=int, default=2)
    p.add_argument("--sasrec-max-len", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--min-positive-rating", type=float, default=1.0)
    p.add_argument("--decimals", type=int, default=4)
    return p.parse_args()


def sf(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return float(d)


def load_baseline(path: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    df = pd.read_csv(path)
    m = {}
    for _, r in df.iterrows():
        model = MODEL_KEY_FROM_LABEL.get(str(r["Model"]), str(r["Model"]).lower())
        m[(model, "ml1m")] = {"hit": sf(r.get("ML-1M Hit@10", 0.0)), "ndcg": sf(r.get("ML-1M NDCG@10", 0.0))}
        m[(model, "lastfm")] = {"hit": sf(r.get("LastFM Hit@10", 0.0)), "ndcg": sf(r.get("LastFM NDCG@10", 0.0))}
        m[(model, "automotive")] = {"hit": sf(r.get("Automotive Hit@10", 0.0)), "ndcg": sf(r.get("Automotive NDCG@10", 0.0))}
    return m


def score(hit: float, ndcg: float, shift: float, b_hit: float, b_ndcg: float) -> float:
    s = (hit - b_hit) + (ndcg - b_ndcg) + 0.2 * shift
    if shift <= 0:
        s -= 2.0 * (1.0 + abs(shift))
    return float(s)


def fmt(v: float, d: int) -> str:
    return f"{float(v):.{d}f}"


def main() -> None:
    args = parse_args()
    out_dir = (ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_baseline((ROOT / args.baseline_table).resolve())

    cache = {}
    targets = {}
    for ds, cfg in DATASETS.items():
        data = prepare_data(
            interactions_path=str(ROOT / cfg["interactions"]),
            items_path=str(ROOT / cfg["items"]),
            min_positive_rating=float(args.min_positive_rating),
            seed=int(args.seed),
        )
        raw = str(cfg["target_item_raw"])
        idx = int(data.item_id_to_index[raw])
        targets[ds] = {
            "category": str(cfg["target_category"]),
            "target_raw": raw,
            "target_idx": idx,
        }
        cache[ds] = data

    candidates_path = out_dir / "all_push_candidates.csv"
    rows: List[Dict[str, object]] = []

    total = len(DATASETS) * len(MODEL_ORDER) * len(PRESETS)
    done = 0

    for ds, dcfg in DATASETS.items():
        data = cache[ds]
        tcat = targets[ds]["category"]
        tidx = targets[ds]["target_idx"]
        traw = targets[ds]["target_raw"]

        for model in MODEL_ORDER:
            for preset in PRESETS:
                done += 1
                bs = int(args.sasrec_batch_size) if model == "sasrec" else int(args.batch_size)
                cfg = MatrixPoisonConfig(
                    intent="push",
                    model_name=model,
                    epsilon=float(preset["epsilon"]),
                    epsilon_min=0.005,
                    epsilon_max=0.5,
                    max_iters=int(preset["max_iters"]),
                    analysis_probe_ratio=0.6,
                    clean_epochs=int(dcfg["clean_epochs"]),
                    probe_epochs=int(preset["probe_epochs"]),
                    judge_epochs=int(preset["judge_epochs"]),
                    batch_size=bs,
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    embedding_dim=int(args.embedding_dim),
                    lightgcn_layers=int(args.lightgcn_layers),
                    sasrec_max_len=int(args.sasrec_max_len),
                    k=int(args.topk),
                    stealth_lambda=0.10,
                    target_eval_size=1,
                    clean_drop_tolerance=0.10,
                    profile_max_len=120,
                    push_profile_max_len=int(preset["push_profile_max_len"]),
                    positive_rating_threshold=float(preset["positive_threshold"]),
                    push_use_explicit_neg=bool(preset["push_use_explicit_neg"]),
                    push_neg_sample_bias=float(preset["push_neg_sample_bias"]),
                    push_min_shift=1e-4,
                    push_negative_penalty=float(preset["push_negative_penalty"]),
                    push_shift_on_unseen_only=bool(preset["push_shift_on_unseen_only"]),
                    push_target_shift=float(preset["push_target_shift"]),
                    push_epsilon_max=float(preset["push_epsilon_max"]),
                    camel_enabled=False,
                    seed=int(args.seed),
                    device=str(args.device),
                )

                t0 = time.time()
                agent = MatrixPoisonAgent(data=data, target_category=tcat, target_item=tidx, cfg=cfg)
                res = agent.run()
                elapsed = time.time() - t0
                bm = res.best_metrics or {}

                hit = sf(bm.get("hit", 0.0))
                ndcg = sf(bm.get("ndcg", 0.0))
                shift = sf(bm.get("prediction_shift", 0.0))
                obj = sf(bm.get("objective", 0.0))
                reward = sf(res.best_reward, -1e18)
                b_hit = baseline[(model, ds)]["hit"]
                b_ndcg = baseline[(model, ds)]["ndcg"]
                s = score(hit, ndcg, shift, b_hit, b_ndcg)

                row = {
                    "dataset": ds,
                    "dataset_label": dcfg["label"],
                    "model": model,
                    "model_label": MODEL_LABEL[model],
                    "preset": str(preset["name"]),
                    "target_category": tcat,
                    "target_item_raw": traw,
                    "target_item_index": tidx,
                    "baseline_hit": b_hit,
                    "baseline_ndcg": b_ndcg,
                    "clean_hit": sf(res.clean_metrics.get("hit", 0.0)),
                    "clean_ndcg": sf(res.clean_metrics.get("ndcg", 0.0)),
                    "hit": hit,
                    "ndcg": ndcg,
                    "prediction_shift": shift,
                    "objective": obj,
                    "reward": reward,
                    "best_iteration": int(res.best_iteration),
                    "best_epsilon": sf(res.best_epsilon, 0.0),
                    "score_vs_baseline": s,
                    "elapsed_sec": float(elapsed),
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(candidates_path, index=False)

                print(
                    f"[{done:03d}/{total}] ds={ds} model={model} preset={preset['name']} "
                    f"hit={hit:.4f} ndcg={ndcg:.4f} shift={shift:.4f} score={s:.4f} t={elapsed:.1f}s",
                    flush=True,
                )

    cdf = pd.DataFrame(rows)
    cdf.to_csv(candidates_path, index=False)

    best_rows = []
    for ds in DATASETS:
        for model in MODEL_ORDER:
            sub = cdf[(cdf["dataset"] == ds) & (cdf["model"] == model)].copy()
            sub = sub.sort_values(["score_vs_baseline", "prediction_shift", "hit", "ndcg", "objective"], ascending=False)
            best_rows.append(sub.iloc[0].to_dict())
    bdf = pd.DataFrame(best_rows)
    bdf.to_csv(out_dir / "best_push_cases_detail.csv", index=False)

    push_table_rows = []
    for model in MODEL_ORDER:
        row = {"Methods": "Matrix Poison-Push", "Model": MODEL_LABEL[model]}
        for ds, dcfg in DATASETS.items():
            best = bdf[(bdf["dataset"] == ds) & (bdf["model"] == model)].iloc[0]
            lbl = dcfg["label"]
            row[f"{lbl} Hit@10"] = fmt(best["hit"], args.decimals)
            row[f"{lbl} NDCG@10"] = fmt(best["ndcg"], args.decimals)
            row[f"{lbl} P. Shift"] = fmt(best["prediction_shift"], args.decimals)
        push_table_rows.append(row)
    push_df = pd.DataFrame(push_table_rows)
    push_df.to_csv(out_dir / "table_matrix_poison_push_seed42_best.csv", index=False)

    baseline_df = pd.read_csv((ROOT / args.baseline_table).resolve())
    baseline_df = baseline_df[baseline_df["Methods"] == "Baseline"].copy()
    cols = [
        "Methods", "Model",
        "ML-1M Hit@10", "ML-1M NDCG@10", "ML-1M P. Shift",
        "LastFM Hit@10", "LastFM NDCG@10", "LastFM P. Shift",
        "Automotive Hit@10", "Automotive NDCG@10", "Automotive P. Shift",
    ]
    baseline_df = baseline_df[cols]
    nuke_df = pd.read_csv((ROOT / args.nuke_table).resolve())[cols]
    table3_df = pd.concat([baseline_df, nuke_df, push_df], ignore_index=True)
    table3_df.to_csv(out_dir / "table3_with_matrix_poison_push_seed42.csv", index=False)

    md = ["| " + " | ".join(table3_df.columns) + " |", "| " + " | ".join(["---"] * len(table3_df.columns)) + " |"]
    for _, r in table3_df.iterrows():
        md.append("| " + " | ".join(str(r[c]) for c in table3_df.columns) + " |")
    (out_dir / "table3_with_matrix_poison_push_seed42.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    summary = {
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "sasrec_batch_size": int(args.sasrec_batch_size),
        "presets": PRESETS,
        "targets": targets,
    }
    (out_dir / "search_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] candidates: {candidates_path}")
    print(f"[done] best push table: {out_dir / 'table_matrix_poison_push_seed42_best.csv'}")
    print(f"[done] merged table3: {out_dir / 'table3_with_matrix_poison_push_seed42.csv'}")


if __name__ == "__main__":
    main()

