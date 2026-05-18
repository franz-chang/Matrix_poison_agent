#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AblationVariant:
    key: str
    label: str
    flags: Sequence[str]


VARIANTS: List[AblationVariant] = [
    AblationVariant("full", "Full Matrix-Poison", []),
    AblationVariant("wo_analysis", "w/o Analysis Phase", ["--disable-analysis-phase"]),
    AblationVariant("wo_planning", "w/o Planning Phase", ["--disable-planning-phase"]),
    AblationVariant("wo_judgment", "w/o Judgment Phase", ["--disable-judgment-phase"]),
    # Paper text also refers to this as "w/o Self-Reflection Phase".
    AblationVariant("wo_self_evolving", "w/o Self-Reflection Phase", ["--disable-self-evolving"]),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce Table-4 ablation study on ML-1M (Matrix-Poison, Nuke).")
    p.add_argument("--interactions", default="data/ml1m/interactions.csv", help="Path to interactions.csv")
    p.add_argument("--items", default="data/ml1m/items.csv", help="Path to items.csv")
    p.add_argument("--model", default="bpr", choices=["bpr", "neumf", "lightgcn", "sasrec"], help="Victim model for Table 4")
    p.add_argument("--intent", default="nuke", choices=["nuke", "push"], help="Attack intent")
    p.add_argument("--target-category", default="Action")
    p.add_argument("--target-item-policy", default="auto", choices=["auto", "head", "tail", "median"])
    p.add_argument("--target-eval-size", type=int, default=1)

    p.add_argument("--epsilon", type=float, default=0.005)
    p.add_argument("--clean-epochs", type=int, default=50)
    p.add_argument("--probe-epochs", type=int, default=4)
    p.add_argument("--judge-epochs", type=int, default=12)
    p.add_argument("--max-iters", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4096)

    p.add_argument("--min-positive-rating", type=float, default=1.0)
    p.add_argument("--positive-rating-threshold", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")

    p.add_argument("--python-bin", default=sys.executable, help="Python interpreter used to call run_matrix_poison.py")
    p.add_argument("--output-dir", default="outputs/table4_ablation_seed42")
    p.add_argument("--output-prefix", default="table4_ablation_seed42", help="Output file prefix inside output-dir")
    p.add_argument("--decimals", type=int, default=4)
    p.add_argument(
        "--table-caption",
        default="Performance Metrics for Method Variants (Table 4 Ablation, Seed=42).",
        help="LaTeX table caption",
    )
    p.add_argument("--table-label", default="tab:table4-ablation-seed42", help="LaTeX table label")

    p.add_argument("--enable-camel", action="store_true", help="Enable CAMEL orchestrator (disabled by default for reproducibility)")
    p.add_argument("--camel-model", default="openai:gpt-4o-mini", help="CAMEL model spec")
    p.add_argument("--camel-no-stub-fallback", action="store_true", help="Disable CAMEL STUB fallback")
    return p.parse_args()


def _float_or_nan(v) -> float:
    try:
        if v is None:
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def _fmt(v: float, decimals: int) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{float(v):.{decimals}f}"


def _read_summary(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing run summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def detect_llm_driver(args: argparse.Namespace) -> Dict[str, object]:
    has_key = bool(os.getenv("OPENAI_API_KEY", "").strip())
    if not args.enable_camel:
        effective = "disabled (--disable-camel)"
    elif has_key:
        effective = f"CAMEL active via {args.camel_model}"
    elif args.camel_no_stub_fallback:
        effective = "CAMEL configured but unavailable (missing OPENAI_API_KEY, no STUB fallback)"
    else:
        effective = "CAMEL STUB fallback (missing OPENAI_API_KEY)"
    return {
        "camel_enabled": bool(args.enable_camel),
        "camel_model": args.camel_model,
        "camel_no_stub_fallback": bool(args.camel_no_stub_fallback),
        "openai_api_key_set": has_key,
        "effective_driver": effective,
    }


def run_variant(args: argparse.Namespace, variant: AblationVariant, root_out: Path) -> Dict[str, object]:
    variant_out = root_out / variant.key
    variant_out.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        str(args.python_bin),
        str(ROOT / "scripts" / "run_matrix_poison.py"),
        "--interactions",
        str((ROOT / args.interactions).resolve()),
        "--items",
        str((ROOT / args.items).resolve()),
        "--model",
        args.model,
        "--intent",
        args.intent,
        "--target-category",
        args.target_category,
        "--target-item-policy",
        args.target_item_policy,
        "--target-eval-size",
        str(args.target_eval_size),
        "--epsilon",
        str(args.epsilon),
        "--clean-epochs",
        str(args.clean_epochs),
        "--probe-epochs",
        str(args.probe_epochs),
        "--judge-epochs",
        str(args.judge_epochs),
        "--max-iters",
        str(args.max_iters),
        "--batch-size",
        str(args.batch_size),
        "--min-positive-rating",
        str(args.min_positive_rating),
        "--positive-rating-threshold",
        str(args.positive_rating_threshold),
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
        "--output-dir",
        str(variant_out),
    ]

    if not args.enable_camel:
        cmd.append("--disable-camel")
    else:
        cmd.extend(["--camel-model", args.camel_model])
        if args.camel_no_stub_fallback:
            cmd.append("--camel-no-stub-fallback")

    cmd.extend(list(variant.flags))

    print(f"[table4-ablation] running {variant.key}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)

    summary = _read_summary(variant_out / "run_summary.json")
    best = summary.get("best_metrics", {}) or {}

    return {
        "variant": variant.key,
        "Methods": variant.label,
        "Hit@10": _float_or_nan(best.get("hit")),
        "NDCG@10": _float_or_nan(best.get("ndcg")),
        "P. Shift": _float_or_nan(best.get("prediction_shift")),
        "Objective": _float_or_nan(best.get("objective")),
        "Reward": _float_or_nan(summary.get("best_reward")),
        "Best Iter": summary.get("best_iteration", -1),
        "run_dir": str(variant_out),
    }


def build_latex(rows: List[Dict[str, object]], decimals: int, table_caption: str, table_label: str) -> str:
    lines: List[str] = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{table_caption}}}",
        f"\\label{{{table_label}}}",
        "\\scriptsize",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Methods} & \\textbf{Hit@10$\\downarrow$} & \\textbf{NDCG@10$\\downarrow$} & \\textbf{P. Shift} \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['Methods']} & {_fmt(r['Hit@10'], decimals)} & {_fmt(r['NDCG@10'], decimals)} & {_fmt(r['P. Shift'], decimals)} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    inter_path = (ROOT / args.interactions).resolve()
    items_path = (ROOT / args.items).resolve()
    if not inter_path.exists():
        raise FileNotFoundError(f"interactions.csv not found: {inter_path}")
    if not items_path.exists():
        raise FileNotFoundError(f"items.csv not found: {items_path}")

    llm_info = detect_llm_driver(args)
    print("[table4-ablation] LLM validation:")
    print(json.dumps(llm_info, ensure_ascii=False, indent=2), flush=True)

    out_dir = (ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for variant in VARIANTS:
        row = run_variant(args, variant, out_dir)
        rows.append(row)

    table_df = pd.DataFrame(
        [{
            "Methods": r["Methods"],
            "Hit@10↓": _fmt(r["Hit@10"], args.decimals),
            "NDCG@10↓": _fmt(r["NDCG@10"], args.decimals),
            "P. Shift": _fmt(r["P. Shift"], args.decimals),
        } for r in rows]
    )
    detail_df = pd.DataFrame(rows)

    prefix = str(args.output_prefix).strip() or "table4_ablation_seed42"
    table_csv = out_dir / f"{prefix}.csv"
    details_csv = out_dir / f"{prefix}_details.csv"
    tex_path = out_dir / f"{prefix}.tex"
    meta_path = out_dir / f"{prefix}_meta.json"

    table_df.to_csv(table_csv, index=False)
    detail_df.to_csv(details_csv, index=False)
    tex_path.write_text(
        build_latex(
            rows,
            args.decimals,
            table_caption=str(args.table_caption),
            table_label=str(args.table_label),
        ),
        encoding="utf-8",
    )

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "llm_validation": llm_info,
        "output_dir": str(out_dir),
        "table_csv": str(table_csv),
        "details_csv": str(details_csv),
        "latex": str(tex_path),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[table4-ablation] DONE")
    print(f"[table4-ablation] CSV:      {table_csv}")
    print(f"[table4-ablation] Details:  {details_csv}")
    print(f"[table4-ablation] LaTeX:    {tex_path}")
    print(f"[table4-ablation] Meta:     {meta_path}")
    print(table_df.to_string(index=False))


if __name__ == "__main__":
    main()
