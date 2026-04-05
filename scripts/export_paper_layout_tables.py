#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reformat experiment outputs into paper-style tables and export PDF.")
    p.add_argument("--run-dir", required=True, help="Experiment output root.")
    p.add_argument("--output", default="", help="Output PDF path. Default: <run-dir>/paper_layout_tables.pdf")
    p.add_argument("--decimals", type=int, default=4)
    return p.parse_args()


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt(v, decimals: int = 4) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, str):
        return v
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return str(v)


def _abs_shift(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return abs(float(v))
    except Exception:
        return None


def _draw_table(pdf: PdfPages, title: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        fig, ax = plt.subplots(figsize=(16.5, 5.0))
        ax.axis("off")
        ax.text(0.02, 0.95, title, fontsize=16, fontweight="bold", va="top")
        ax.text(0.02, 0.8, "No data", fontsize=12, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    rows = len(df)
    fig_h = min(22.0, max(5.5, 1.4 + 0.36 * (rows + 1)))
    fig, ax = plt.subplots(figsize=(16.5, fig_h))
    ax.axis("off")
    ax.text(0.01, 1.03, title, fontsize=14, fontweight="bold", transform=ax.transAxes, va="bottom")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns.tolist(),
        cellLoc="center",
        loc="upper left",
        bbox=[0.0, 0.0, 1.0, 0.95],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _collect_single_market(run_dir: Path) -> Dict[Tuple[str, str], dict]:
    out: Dict[Tuple[str, str], dict] = {}
    base = run_dir / "5_3_1_single_market_targeted"
    for p in sorted(base.glob("*/run_summary.json")):
        name = p.parent.name
        parts = name.split("_")
        if len(parts) < 2:
            continue
        model = parts[0]
        intent = parts[1]
        obj = _load_json(p)
        if obj:
            out[(model, intent)] = obj
    return out


def _collect_baseline_compare(run_dir: Path) -> Dict[Tuple[str, str, str], dict]:
    out: Dict[Tuple[str, str, str], dict] = {}
    base = run_dir / "5_3_2_baseline_comparison"
    for p in sorted(base.glob("*/run_summary.json")):
        name = p.parent.name
        parts = name.split("_")
        if len(parts) < 3:
            continue
        model = parts[0]
        intent = parts[1]
        method = "_".join(parts[2:])
        obj = _load_json(p)
        if obj:
            out[(model, intent, method)] = obj
    return out


def _collect_cross_market(run_dir: Path) -> Dict[Tuple[str, str, str], dict]:
    out: Dict[Tuple[str, str, str], dict] = {}
    base = run_dir / "5_4_cross_market"
    for p in sorted(base.glob("*/run_summary.json")):
        name = p.parent.name
        parts = name.split("_")
        if len(parts) < 3:
            continue
        market = parts[0]
        model = parts[1]
        intent = "_".join(parts[2:])
        obj = _load_json(p)
        if obj:
            out[(market, model, intent)] = obj
    return out


def _collect_ablation(run_dir: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    base = run_dir / "5_6_ablation"
    for p in sorted(base.glob("*/run_summary.json")):
        name = p.parent.name
        if "_" not in name:
            continue
        variant = name.split("_", 1)[1]
        obj = _load_json(p)
        if obj:
            out[variant] = obj
    return out


def build_table3_like(run_dir: Path, decimals: int) -> pd.DataFrame:
    single = _collect_single_market(run_dir)
    models = sorted({m for (m, _) in single.keys()})
    rows: List[dict] = []
    for model in models:
        nuke = single.get((model, "nuke"))
        push = single.get((model, "push"))
        ref = nuke or push
        clean = (ref or {}).get("clean_metrics", {})
        rows.append(
            {
                "Methods": "Baseline",
                "Model": model,
                "ML-1M Hit@10": _fmt(clean.get("hit"), decimals),
                "ML-1M NDCG@10": _fmt(clean.get("ndcg"), decimals),
                "ML-1M P.Shift": "-",
            }
        )
        if nuke:
            bm = nuke.get("best_metrics", {})
            rows.append(
                {
                    "Methods": "Matrix Poison-Nuke",
                    "Model": model,
                    "ML-1M Hit@10": _fmt(bm.get("hit"), decimals),
                    "ML-1M NDCG@10": _fmt(bm.get("ndcg"), decimals),
                    "ML-1M P.Shift": _fmt(_abs_shift(bm.get("prediction_shift")), decimals),
                }
            )
        if push:
            bm = push.get("best_metrics", {})
            rows.append(
                {
                    "Methods": "Matrix Poison-Push",
                    "Model": model,
                    "ML-1M Hit@10": _fmt(bm.get("hit"), decimals),
                    "ML-1M NDCG@10": _fmt(bm.get("ndcg"), decimals),
                    "ML-1M P.Shift": _fmt(_abs_shift(bm.get("prediction_shift")), decimals),
                }
            )
    return pd.DataFrame(rows)


def build_table2_like(run_dir: Path, decimals: int) -> pd.DataFrame:
    cmp_map = _collect_baseline_compare(run_dir)
    if not cmp_map:
        return pd.DataFrame()

    preferred_model = sorted({k[0] for k in cmp_map.keys()})[0]
    preferred_intent = "nuke"
    methods_order = ["random", "average", "bandwagon", "segmented", "matrix_poison"]
    labels = {
        "random": "Random",
        "average": "Average",
        "bandwagon": "Bandwagon",
        "segmented": "Segmented",
        "matrix_poison": "Matrix Poison-Nuke",
    }

    rows: List[dict] = []
    ref = cmp_map.get((preferred_model, preferred_intent, "matrix_poison"))
    clean = (ref or {}).get("clean_metrics", {})
    rows.append(
        {
            "Methods": "Baseline",
            "Model": preferred_model,
            "ML-1M Hit@10": _fmt(clean.get("hit"), decimals),
            "ML-1M NDCG@10": _fmt(clean.get("ndcg"), decimals),
            "ML-1M P.Shift": "-",
        }
    )

    for m in methods_order:
        obj = cmp_map.get((preferred_model, preferred_intent, m))
        if not obj:
            continue
        bm = obj.get("best_metrics", {})
        rows.append(
            {
                "Methods": labels.get(m, m),
                "Model": preferred_model,
                "ML-1M Hit@10": _fmt(bm.get("hit"), decimals),
                "ML-1M NDCG@10": _fmt(bm.get("ndcg"), decimals),
                "ML-1M P.Shift": _fmt(_abs_shift(bm.get("prediction_shift")), decimals),
            }
        )
    return pd.DataFrame(rows)


def build_table4_like(run_dir: Path, decimals: int) -> pd.DataFrame:
    cross = _collect_cross_market(run_dir)
    if not cross:
        return pd.DataFrame()

    markets = ["US", "JP", "UK"]
    models = sorted({k[1] for k in cross.keys()})
    rows: List[dict] = []
    for model in models:
        base_row = {"Methods": "Baseline", "Model": model}
        for mk in markets:
            ref = cross.get((mk, model, "nuke")) or cross.get((mk, model, "push"))
            clean = (ref or {}).get("clean_metrics", {})
            base_row[f"{mk} Hit@10"] = _fmt(clean.get("hit"), decimals)
            base_row[f"{mk} NDCG@10"] = _fmt(clean.get("ndcg"), decimals)
            base_row[f"{mk} P.Shift"] = "-"
        rows.append(base_row)

        for intent, method_name in [("push", "Matrix Poison-Push"), ("nuke", "Matrix Poison-Nuke")]:
            row = {"Methods": method_name, "Model": model}
            has_any = False
            for mk in markets:
                obj = cross.get((mk, model, intent))
                if not obj:
                    row[f"{mk} Hit@10"] = "N/A"
                    row[f"{mk} NDCG@10"] = "N/A"
                    row[f"{mk} P.Shift"] = "N/A"
                    continue
                bm = obj.get("best_metrics", {})
                row[f"{mk} Hit@10"] = _fmt(bm.get("hit"), decimals)
                row[f"{mk} NDCG@10"] = _fmt(bm.get("ndcg"), decimals)
                row[f"{mk} P.Shift"] = _fmt(_abs_shift(bm.get("prediction_shift")), decimals)
                has_any = True
            if has_any:
                rows.append(row)
    return pd.DataFrame(rows)


def build_table5_like(run_dir: Path, decimals: int) -> pd.DataFrame:
    ab = _collect_ablation(run_dir)
    order = ["full", "wo_analysis", "wo_planning", "wo_judgment", "wo_self_evolving"]
    label = {
        "full": "Full Model",
        "wo_analysis": "w/o Analysis Phase",
        "wo_planning": "w/o Planning Phase",
        "wo_judgment": "w/o Judgment Phase",
        "wo_self_evolving": "w/o Self-evolving Phase",
    }
    rows: List[dict] = []
    for k in order:
        obj = ab.get(k)
        if not obj:
            continue
        bm = obj.get("best_metrics", {})
        rows.append(
            {
                "Variant": label.get(k, k),
                "Model": "lightgcn",
                "Hit@10": _fmt(bm.get("hit"), decimals),
                "NDCG@10": _fmt(bm.get("ndcg"), decimals),
                "P.Shift": _fmt(_abs_shift(bm.get("prediction_shift")), decimals),
                "Objective": _fmt(bm.get("objective"), decimals),
                "Reward": _fmt(obj.get("best_reward"), decimals),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run-dir not found: {run_dir}")

    output = Path(args.output).expanduser().resolve() if args.output else run_dir / "paper_layout_tables.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)

    table2 = build_table2_like(run_dir, args.decimals)
    table3 = build_table3_like(run_dir, args.decimals)
    table4 = build_table4_like(run_dir, args.decimals)
    table5 = build_table5_like(run_dir, args.decimals)

    # Save CSV artifacts for transparency
    table2.to_csv(run_dir / "table2_paper_layout.csv", index=False)
    table3.to_csv(run_dir / "table3_paper_layout.csv", index=False)
    table4.to_csv(run_dir / "table4_paper_layout.csv", index=False)
    table5.to_csv(run_dir / "table5_ablation_paper_layout.csv", index=False)

    with PdfPages(output) as pdf:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        ax.text(0.02, 0.92, "Paper-Style Experiment Tables", fontsize=22, fontweight="bold", va="top")
        ax.text(0.02, 0.82, f"Run Directory: {run_dir}", fontsize=11, va="top")
        ax.text(0.02, 0.76, f"Generated: {datetime.now().isoformat(timespec='seconds')}", fontsize=11, va="top")
        ax.text(0.02, 0.66, "Table 2  (SOTA/Baseline-Style Comparison, ML-1M)", fontsize=12, va="top")
        ax.text(0.02, 0.60, "Table 3  (Single-Market Baseline vs Matrix Poison)", fontsize=12, va="top")
        ax.text(0.02, 0.54, "Table 4  (Cross-Market: US/JP/UK)", fontsize=12, va="top")
        ax.text(0.02, 0.48, "Table 5  (Ablation Study)", fontsize=12, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        _draw_table(pdf, "Table 2: Comprehensive Performance Metrics with Baseline Methods (ML-1M)", table2)
        _draw_table(pdf, "Table 3: Comprehensive Performance Metrics with Baseline Methods (Single Market)", table3)
        _draw_table(pdf, "Table 4: Cross-Market Performance Metrics (US/JP/UK)", table4)
        _draw_table(pdf, "Table 5: Ablation Study on ML-1M (Nuke)", table5)

    print(f"PDF written to: {output}")
    print(f"CSV written: {run_dir / 'table2_paper_layout.csv'}")
    print(f"CSV written: {run_dir / 'table3_paper_layout.csv'}")
    print(f"CSV written: {run_dir / 'table4_paper_layout.csv'}")
    print(f"CSV written: {run_dir / 'table5_ablation_paper_layout.csv'}")


if __name__ == "__main__":
    main()
