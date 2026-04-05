#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not (PROJECT_ROOT / "matrix_poison").exists():
    alt = Path(__file__).resolve().parent / "remote_projects" / "matrix-poison"
    if (alt / "matrix_poison").exists():
        PROJECT_ROOT = alt

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matrix_poison.agent import MatrixPoisonAgent, MatrixPoisonConfig
from matrix_poison.attacks import ATTACK_METHODS, AttackRequest
from matrix_poison.data import append_malicious_rows, build_user_sequences, prepare_data
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
        "target_item_policy": "median",
    },
}

MODEL_ORDER = ["bpr", "neumf", "lightgcn", "sasrec"]
MODEL_LABEL = {
    "bpr": "BPR",
    "neumf": "NeuMF",
    "lightgcn": "LightGCN",
    "sasrec": "SASRec",
}
METHOD_ORDER = [
    "baseline",
    "aush",
    "goat",
    "matrix_poison_nuke",
    "pc_attack",
]
METHOD_LABEL = {
    "baseline": "Baseline",
    "aush": "AUSH (Lin et al., 2020b)",
    "goat": "GOAT (Wu et al., 2021)",
    "matrix_poison_nuke": "Matrix Poison-Nuke",
    "pc_attack": "PC-Attack",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run 3-dataset SOTA-style table experiments (baseline-first).")
    p.add_argument("--output-dir", default="", help="Output root. default: outputs/table2_3datasets_<ts>")
    p.add_argument("--datasets", default="ml1m,lastfm,automotive")
    p.add_argument("--models", default=",".join(MODEL_ORDER))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--epsilon", type=float, default=0.02)
    p.add_argument("--min-positive-rating", type=float, default=1.0)
    p.add_argument("--positive-threshold", type=float, default=1.0)
    p.add_argument("--clean-epochs", type=int, default=20)
    p.add_argument("--clean-epochs-lastfm", type=int, default=28)
    p.add_argument("--clean-epochs-automotive", type=int, default=28)
    p.add_argument("--judge-epochs", type=int, default=8)
    p.add_argument("--matrix-max-iters", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--sasrec-batch-size", type=int, default=3072)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--lightgcn-layers", type=int, default=2)
    p.add_argument("--sasrec-max-len", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--target-eval-size", type=int, default=1)
    p.add_argument("--strict-single-target-eval", action="store_true")
    p.add_argument("--disable-strict-single-target-eval", action="store_true")
    p.add_argument(
        "--target-item-overrides",
        default="",
        help="Comma-separated dataset:item override, item can be raw item_id or encoded item index. e.g. lastfm:498,automotive:1762",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--decimals", type=int, default=4)
    p.add_argument("--force", action="store_true", help="Force re-run even if cached JSON exists")

    p.add_argument("--aush-proxy-method", default="average", choices=sorted(ATTACK_METHODS.keys()))
    p.add_argument("--goat-proxy-method", default="segmented", choices=sorted(ATTACK_METHODS.keys()))

    p.add_argument("--enable-pc-attack", action="store_true")
    p.add_argument("--disable-pc-attack", action="store_true")
    p.add_argument("--conda-bin", default="/root/miniconda3/bin/conda")
    p.add_argument("--pcattack-env", default="pcattack")
    p.add_argument("--pcattack-root", default="/root/autodl-tmp/PC-Attack")
    p.add_argument("--pcattack-epochs", type=int, default=32)
    p.add_argument("--pcattack-batch-size", type=int, default=128)
    p.add_argument("--pcattack-filler-item", type=int, default=50)
    p.add_argument("--pcattack-filler-pop", type=int, default=30)
    p.add_argument("--pcattack-gpu", type=int, default=0)
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
        if not p:
            continue
        if ":" not in p:
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

    if not data.category_to_items:
        raise ValueError(f"No category available for dataset {dataset_key}")
    return sorted(data.category_to_items.keys())[0]


def _select_target_item(data, target_category: str, policy: str = "head") -> int:
    cands = list(data.category_to_items[target_category])
    pop = data.train["item"].value_counts().to_dict()
    cands = sorted(cands, key=lambda i: pop.get(int(i), 0), reverse=True)
    if not cands:
        raise ValueError(f"No item found for category {target_category}")
    policy = str(policy).lower()
    if policy == "tail":
        return int(cands[-1])
    if policy == "median":
        return int(cands[len(cands) // 2])
    return int(cands[0])


def _build_agent(
    data,
    dataset_key: str,
    model: str,
    target_category: str,
    target_item: int,
    args: argparse.Namespace,
    max_iters: int,
    seed_offset: int = 0,
) -> MatrixPoisonAgent:
    batch_size = args.sasrec_batch_size if model.lower() == "sasrec" else args.batch_size
    clean_epochs = args.clean_epochs
    if dataset_key == "lastfm" and args.clean_epochs_lastfm:
        clean_epochs = int(args.clean_epochs_lastfm)
    elif dataset_key == "automotive" and args.clean_epochs_automotive:
        clean_epochs = int(args.clean_epochs_automotive)
    cfg = MatrixPoisonConfig(
        intent="nuke",
        model_name=model,
        epsilon=args.epsilon,
        max_iters=max_iters,
        clean_epochs=clean_epochs,
        probe_epochs=max(1, args.judge_epochs // 2),
        judge_epochs=args.judge_epochs,
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
    return MatrixPoisonAgent(
        data=data,
        target_category=target_category,
        target_item=target_item,
        cfg=cfg,
    )


def _compute_clean(agent: MatrixPoisonAgent) -> Tuple[object, float, float]:
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
    return clean_model, _safe_float(quality.get("hit", 0.0)), _safe_float(quality.get("ndcg", 0.0))


def _apply_strict_single_target_eval(agent: MatrixPoisonAgent, enable_strict: bool) -> None:
    if enable_strict:
        agent.target_items = {int(agent.target_item)}


def _eval_fixed_attack(
    agent: MatrixPoisonAgent,
    clean_model,
    clean_hit: float,
    clean_ndcg: float,
    method_name: str,
    seed_offset: int,
) -> Dict[str, float]:
    n_fake = max(1, int(round(agent.data.num_users * agent.cfg.epsilon)))
    req = AttackRequest(
        epsilon=agent.cfg.epsilon,
        intent="nuke",
        target_item=agent.target_item,
        target_items=sorted(agent.target_items),
        target_category_items=agent.data.category_to_items[agent.target_category],
        max_profile_len=agent.cfg.profile_max_len,
        seed=agent.cfg.seed + 1000 + seed_offset,
    )
    malicious = ATTACK_METHODS[method_name](
        train_df=agent.data.train,
        start_user_id=max(agent.data.num_users, int(agent.data.train["user"].max()) + 1),
        num_fake_users=n_fake,
        num_items=agent.data.num_items,
        req=req,
    )
    poisoned = append_malicious_rows(agent.data.train, malicious)
    metrics = agent._evaluate_candidate(
        clean_model=clean_model,
        clean_hit=clean_hit,
        clean_ndcg=clean_ndcg,
        train_poisoned=poisoned,
        seed_offset=2500 + seed_offset,
        epochs=agent.cfg.judge_epochs,
    )
    return {
        "hit": _safe_float(metrics.get("hit", 0.0)),
        "ndcg": _safe_float(metrics.get("ndcg", 0.0)),
        "prediction_shift": _safe_float(metrics.get("prediction_shift", 0.0)),
        "objective": _safe_float(metrics.get("objective", 0.0)),
    }


def _run_pcattack_generator(
    args: argparse.Namespace,
    dataset_key: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_item: int,
    out_dir: Path,
) -> Path:
    ensure_dir(out_dir)
    train_csv = out_dir / f"{dataset_key}_train_for_pcattack.csv"
    test_csv = out_dir / f"{dataset_key}_test_for_pcattack.csv"
    out_csv = out_dir / f"{dataset_key}_pcattack_malicious.csv"

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    n_users = int(train_df["user"].max()) + 1 if not train_df.empty else 1
    num_fake_user = max(1, int(round(n_users * args.epsilon)))

    cmd = [
        args.conda_bin,
        "run",
        "-n",
        args.pcattack_env,
        "python",
        str(PROJECT_ROOT / "scripts" / "pcattack_generate_profiles.py"),
        "--pcattack-root",
        args.pcattack_root,
        "--dataset-alias",
        dataset_key,
        "--train-csv",
        str(train_csv),
        "--test-csv",
        str(test_csv),
        "--target-item",
        str(int(target_item)),
        "--num-fake-user",
        str(int(num_fake_user)),
        "--num-filler-item",
        str(int(args.pcattack_filler_item)),
        "--num-filler-pop",
        str(int(args.pcattack_filler_pop)),
        "--num-epochs",
        str(int(args.pcattack_epochs)),
        "--batch-size",
        str(int(args.pcattack_batch_size)),
        "--gpu",
        str(int(args.pcattack_gpu)),
        "--output-csv",
        str(out_csv),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / f"pcattack_{dataset_key}.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    (out_dir / f"pcattack_{dataset_key}.stderr.log").write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"PC-Attack generator failed for {dataset_key} (code={proc.returncode}). "
            f"See logs under {out_dir}"
        )
    if not out_csv.exists():
        raise FileNotFoundError(f"PC-Attack malicious csv not produced: {out_csv}")
    return out_csv


def _load_case_json(path: Path) -> Dict[str, float] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_case_json(path: Path, payload: Dict[str, float]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_layout_table(results: Dict[Tuple[str, str, str], Dict[str, float]], datasets: List[str], models: List[str], decimals: int) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for method in METHOD_ORDER:
        for model in models:
            row: Dict[str, str] = {
                "Methods": METHOD_LABEL[method],
                "Model": MODEL_LABEL.get(model, model),
            }
            for ds in datasets:
                ds_label = DATASETS[ds]["label"]
                m = results.get((method, model, ds))
                if not m:
                    row[f"{ds_label} Hit@10"] = "N/A"
                    row[f"{ds_label} NDCG@10"] = "N/A"
                    row[f"{ds_label} P. Shift"] = "N/A"
                    continue
                row[f"{ds_label} Hit@10"] = f"{_safe_float(m.get('hit', 0.0)):.{decimals}f}"
                row[f"{ds_label} NDCG@10"] = f"{_safe_float(m.get('ndcg', 0.0)):.{decimals}f}"
                if method == "baseline":
                    row[f"{ds_label} P. Shift"] = "-"
                else:
                    row[f"{ds_label} P. Shift"] = f"{_safe_float(m.get('prediction_shift', 0.0)):.{decimals}f}"
            rows.append(row)
    return pd.DataFrame(rows)


def _save_markdown(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    if df.empty:
        path.write_text("_No rows._\n", encoding="utf-8")
        return
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_pdf_table(df: pd.DataFrame, path: Path, title: str) -> None:
    ensure_dir(path.parent)
    with PdfPages(path) as pdf:
        fig_h = max(6.0, min(22.0, 1.2 + 0.36 * (len(df) + 2)))
        fig, ax = plt.subplots(figsize=(18, fig_h))
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
        table.set_fontsize(8)
        table.scale(1.0, 1.2)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _write_targets_file(targets: Dict[str, Dict[str, object]], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(targets, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    strict_single_target_eval = True
    if args.strict_single_target_eval:
        strict_single_target_eval = True
    if args.disable_strict_single_target_eval:
        strict_single_target_eval = False

    if not args.enable_pc_attack:
        args.enable_pc_attack = True
    if args.disable_pc_attack:
        args.enable_pc_attack = False

    target_item_overrides = _parse_target_item_overrides(args.target_item_overrides)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / f"table2_3datasets_{ts}"
    ensure_dir(out_root)

    targets: Dict[str, Dict[str, object]] = {}
    dataset_cache = {}
    for ds in datasets:
        if ds not in DATASETS:
            raise ValueError(f"Unsupported dataset key: {ds}")
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
        target_item = _select_target_item(data, target_category, policy=DATASETS[ds].get("target_item_policy", "head"))
        if ds in target_item_overrides:
            target_item = _resolve_override_item(data, target_item_overrides[ds])
            if target_item not in data.category_to_items.get(target_category, []):
                raise ValueError(
                    f"Override target item for {ds} -> {target_item_overrides[ds]} (index={target_item}) not in category `{target_category}`"
                )
        dataset_cache[ds] = {
            "data": data,
            "items_path": items_path,
            "target_category": target_category,
            "target_item": target_item,
        }
        targets[ds] = {
            "dataset_label": DATASETS[ds]["label"],
            "target_category": target_category,
            "target_item_index": int(target_item),
            "target_item_raw_id": data.index_to_item_id.get(int(target_item), "UNKNOWN"),
            "target_item_policy": DATASETS[ds].get("target_item_policy", "head"),
            "num_users": int(data.num_users),
            "num_items": int(data.num_items),
        }

    _write_targets_file(targets, out_root / "selected_targets.json")

    print("=== Phase 1/2: Baseline (first major block) ===", flush=True)
    results: Dict[Tuple[str, str, str], Dict[str, float]] = {}

    for ds in datasets:
        info = dataset_cache[ds]
        data = info["data"]
        target_category = info["target_category"]
        target_item = int(info["target_item"])
        for model_idx, model in enumerate(models):
            case_dir = out_root / "per_case" / ds / model
            ensure_dir(case_dir)
            baseline_json = case_dir / "baseline.json"
            if baseline_json.exists() and not args.force:
                payload = _load_case_json(baseline_json)
                if payload:
                    results[("baseline", model, ds)] = payload
                    print(f"[baseline][cached] {ds} {model}", flush=True)
                    continue

            agent = _build_agent(
                data=data,
                dataset_key=ds,
                model=model,
                target_category=target_category,
                target_item=target_item,
                args=args,
                max_iters=1,
                seed_offset=model_idx,
            )
            _apply_strict_single_target_eval(agent, strict_single_target_eval)
            clean_model, clean_hit, clean_ndcg = _compute_clean(agent)
            payload = {
                "hit": float(clean_hit),
                "ndcg": float(clean_ndcg),
                "prediction_shift": 0.0,
                "objective": 0.0,
            }
            _save_case_json(baseline_json, payload)
            results[("baseline", model, ds)] = payload
            print(f"[baseline][done] {ds} {model} hit={clean_hit:.4f} ndcg={clean_ndcg:.4f}", flush=True)

    baseline_table = _build_layout_table(results=results, datasets=datasets, models=models, decimals=args.decimals)
    baseline_table = baseline_table[baseline_table["Methods"] == METHOD_LABEL["baseline"]].copy()
    baseline_table.to_csv(out_root / "table_baseline_first_block.csv", index=False)
    _save_markdown(baseline_table, out_root / "table_baseline_first_block.md")
    _save_pdf_table(
        baseline_table,
        out_root / "table_baseline_first_block.pdf",
        title="Baseline Block (Methods Group 1)",
    )
    print(f"[baseline] exported -> {out_root / 'table_baseline_first_block.csv'}", flush=True)

    print("=== Phase 2/2: AUSH / GOAT / Matrix Poison-Nuke / PC-Attack ===", flush=True)

    pcattack_cache: Dict[str, pd.DataFrame] = {}
    for ds in datasets:
        info = dataset_cache[ds]
        data = info["data"]
        target_category = info["target_category"]
        target_item = int(info["target_item"])

        if args.enable_pc_attack:
            pcattack_csv = out_root / "pcattack_profiles" / f"{ds}_pcattack_malicious.csv"
            if pcattack_csv.exists() and not args.force:
                pcattack_cache[ds] = pd.read_csv(pcattack_csv)
                print(f"[pc-attack][cached-profiles] {ds}", flush=True)
            else:
                pc_out_dir = out_root / "pcattack_profiles"
                ensure_dir(pc_out_dir)
                pc_csv = _run_pcattack_generator(
                    args=args,
                    dataset_key=ds,
                    train_df=data.train,
                    test_df=data.test if not data.test.empty else data.val,
                    target_item=target_item,
                    out_dir=pc_out_dir,
                )
                pc_df = pd.read_csv(pc_csv)
                pcattack_cache[ds] = pc_df
                print(f"[pc-attack][profiles-done] {ds} rows={len(pc_df)}", flush=True)

        for model_idx, model in enumerate(models):
            case_dir = out_root / "per_case" / ds / model
            ensure_dir(case_dir)

            agent_eval = _build_agent(
                data=data,
                dataset_key=ds,
                model=model,
                target_category=target_category,
                target_item=target_item,
                args=args,
                max_iters=1,
                seed_offset=300 + model_idx,
            )
            _apply_strict_single_target_eval(agent_eval, strict_single_target_eval)
            clean_model, clean_hit, clean_ndcg = _compute_clean(agent_eval)

            # AUSH proxy
            aush_json = case_dir / "aush.json"
            if aush_json.exists() and not args.force:
                payload = _load_case_json(aush_json)
                if payload:
                    results[("aush", model, ds)] = payload
                    print(f"[aush][cached] {ds} {model}", flush=True)
            else:
                aush_metrics = _eval_fixed_attack(
                    agent=agent_eval,
                    clean_model=clean_model,
                    clean_hit=clean_hit,
                    clean_ndcg=clean_ndcg,
                    method_name=args.aush_proxy_method,
                    seed_offset=11,
                )
                _save_case_json(aush_json, aush_metrics)
                results[("aush", model, ds)] = aush_metrics
                print(f"[aush][done] {ds} {model} pshift={aush_metrics['prediction_shift']:.4f}", flush=True)

            # GOAT proxy
            goat_json = case_dir / "goat.json"
            if goat_json.exists() and not args.force:
                payload = _load_case_json(goat_json)
                if payload:
                    results[("goat", model, ds)] = payload
                    print(f"[goat][cached] {ds} {model}", flush=True)
            else:
                goat_metrics = _eval_fixed_attack(
                    agent=agent_eval,
                    clean_model=clean_model,
                    clean_hit=clean_hit,
                    clean_ndcg=clean_ndcg,
                    method_name=args.goat_proxy_method,
                    seed_offset=29,
                )
                _save_case_json(goat_json, goat_metrics)
                results[("goat", model, ds)] = goat_metrics
                print(f"[goat][done] {ds} {model} pshift={goat_metrics['prediction_shift']:.4f}", flush=True)

            # PC-Attack
            if args.enable_pc_attack:
                pc_json = case_dir / "pc_attack.json"
                if pc_json.exists() and not args.force:
                    payload = _load_case_json(pc_json)
                    if payload:
                        results[("pc_attack", model, ds)] = payload
                        print(f"[pc-attack][cached] {ds} {model}", flush=True)
                else:
                    malicious_pc = pcattack_cache.get(ds)
                    if malicious_pc is None or malicious_pc.empty:
                        payload = {"hit": 0.0, "ndcg": 0.0, "prediction_shift": 0.0, "objective": 0.0}
                    else:
                        need_cols = {"user", "item", "rating", "is_malicious"}
                        if not need_cols.issubset(set(malicious_pc.columns)):
                            raise ValueError(f"PC-Attack profile missing cols: {need_cols - set(malicious_pc.columns)}")
                        mdf = malicious_pc.copy()
                        if "timestamp" not in mdf.columns:
                            mdf["timestamp"] = np.nan
                        mdf = mdf[["user", "item", "rating", "timestamp", "is_malicious"]].copy()
                        mdf["user"] = mdf["user"].astype(int)
                        mdf["item"] = mdf["item"].astype(int)
                        mdf["rating"] = mdf["rating"].astype(float)
                        mdf["is_malicious"] = 1

                        poisoned = append_malicious_rows(data.train, mdf)
                        metrics = agent_eval._evaluate_candidate(
                            clean_model=clean_model,
                            clean_hit=clean_hit,
                            clean_ndcg=clean_ndcg,
                            train_poisoned=poisoned,
                            seed_offset=4200,
                            epochs=agent_eval.cfg.judge_epochs,
                        )
                        payload = {
                            "hit": _safe_float(metrics.get("hit", 0.0)),
                            "ndcg": _safe_float(metrics.get("ndcg", 0.0)),
                            "prediction_shift": _safe_float(metrics.get("prediction_shift", 0.0)),
                            "objective": _safe_float(metrics.get("objective", 0.0)),
                        }
                    _save_case_json(pc_json, payload)
                    results[("pc_attack", model, ds)] = payload
                    print(f"[pc-attack][done] {ds} {model} pshift={payload['prediction_shift']:.4f}", flush=True)

            # Matrix Poison-Nuke (our method)
            mp_json = case_dir / "matrix_poison_nuke.json"
            if mp_json.exists() and not args.force:
                payload = _load_case_json(mp_json)
                if payload:
                    results[("matrix_poison_nuke", model, ds)] = payload
                    print(f"[matrix][cached] {ds} {model}", flush=True)
            else:
                agent_mp = _build_agent(
                    data=data,
                    dataset_key=ds,
                    model=model,
                    target_category=target_category,
                    target_item=target_item,
                    args=args,
                    max_iters=args.matrix_max_iters,
                    seed_offset=700 + model_idx,
                )
                _apply_strict_single_target_eval(agent_mp, strict_single_target_eval)
                mp_res = agent_mp.run()
                bm = mp_res.best_metrics or {}
                payload = {
                    "hit": _safe_float(bm.get("hit", 0.0)),
                    "ndcg": _safe_float(bm.get("ndcg", 0.0)),
                    "prediction_shift": _safe_float(bm.get("prediction_shift", 0.0)),
                    "objective": _safe_float(bm.get("objective", 0.0)),
                }
                _save_case_json(mp_json, payload)
                (case_dir / "matrix_poison_run_summary.json").write_text(
                    json.dumps(
                        {
                            "clean_metrics": mp_res.clean_metrics,
                            "best_iteration": mp_res.best_iteration,
                            "best_reward": mp_res.best_reward,
                            "best_weights": mp_res.best_weights,
                            "best_epsilon": mp_res.best_epsilon,
                            "best_metrics": mp_res.best_metrics,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                results[("matrix_poison_nuke", model, ds)] = payload
                print(f"[matrix][done] {ds} {model} pshift={payload['prediction_shift']:.4f}", flush=True)

    final_table = _build_layout_table(results=results, datasets=datasets, models=models, decimals=args.decimals)
    final_csv = out_root / "table2_sota_3datasets_with_pcattack.csv"
    final_md = out_root / "table2_sota_3datasets_with_pcattack.md"
    final_pdf = out_root / "table2_sota_3datasets_with_pcattack.pdf"

    final_table.to_csv(final_csv, index=False)
    _save_markdown(final_table, final_md)
    _save_pdf_table(
        final_table,
        final_pdf,
        title="Table 2: Comprehensive Performance Metrics with SOTA Methods (+ PC-Attack)",
    )

    notes = {
        "baseline_first_completed": True,
        "method_notes": {
            "AUSH": f"Proxy implementation via fixed '{args.aush_proxy_method}' attack due cross-framework reproducibility constraints.",
            "GOAT": f"Proxy implementation via fixed '{args.goat_proxy_method}' attack due unavailable official code in current environment.",
            "PC-Attack": "Generated via deployed PC-Attack code path and evaluated with matrix-poison victim models.",
        },
        "targets": targets,
        "strict_single_target_eval": strict_single_target_eval,
        "args": vars(args),
    }
    (out_root / "experiment_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== DONE ===", flush=True)
    print(f"Output dir: {out_root}", flush=True)
    print(f"Baseline table: {out_root / 'table_baseline_first_block.csv'}", flush=True)
    print(f"Final table: {final_csv}", flush=True)


if __name__ == "__main__":
    main()
