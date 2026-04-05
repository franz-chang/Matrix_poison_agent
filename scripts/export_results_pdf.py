#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export experiment summary CSV tables into a PDF report.")
    p.add_argument("--run-dir", required=True, help="Experiment root directory containing summary CSV files.")
    p.add_argument(
        "--output",
        default="",
        help="Output PDF path. Default: <run-dir>/experiment_results_tables.pdf",
    )
    p.add_argument("--max-rows", type=int, default=50, help="Max rows per table page.")
    return p.parse_args()


def _maybe_read(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _format_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: f"{x:.6f}")
    return out


def _draw_table_page(pdf: PdfPages, title: str, df: Optional[pd.DataFrame], max_rows: int = 50) -> None:
    if df is None or df.empty:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape
        ax.axis("off")
        ax.text(0.02, 0.95, title, fontsize=16, fontweight="bold", va="top")
        ax.text(0.02, 0.85, "No data found.", fontsize=12, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    table_df = _format_df(df.head(max_rows))
    n_rows = len(table_df)
    fig_height = min(20.0, max(6.5, 1.2 + 0.33 * (n_rows + 2)))
    fig, ax = plt.subplots(figsize=(16.5, fig_height))
    ax.axis("off")
    ax.text(0.01, 1.03, title, fontsize=14, fontweight="bold", transform=ax.transAxes, va="bottom")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns.tolist(),
        cellLoc="center",
        loc="upper left",
        bbox=[0.0, 0.0, 1.0, 0.95],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.15)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run-dir not found: {run_dir}")

    out_pdf = Path(args.output).expanduser().resolve() if args.output else run_dir / "experiment_results_tables.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    tables = [
        ("5.3.1 Single-Market Targeted Attack", run_dir / "5_3_1_single_market_targeted/summary_5_3_1.csv"),
        ("5.3.2 Baseline Comparison", run_dir / "5_3_2_baseline_comparison/summary_5_3_2.csv"),
        ("5.4 Cross-Market Effectiveness", run_dir / "5_4_cross_market/summary_5_4.csv"),
        ("5.6 Ablation Study", run_dir / "5_6_ablation/summary_5_6.csv"),
        ("RQ4 Vulnerability Summary", run_dir / "rq4_vulnerability_summary.csv"),
    ]

    with PdfPages(out_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape
        ax.axis("off")
        ax.text(0.02, 0.92, "Matrix-Poison Experiment Results", fontsize=22, fontweight="bold", va="top")
        ax.text(0.02, 0.80, f"Run Directory: {run_dir}", fontsize=12, va="top")
        ax.text(0.02, 0.73, f"Generated At: {datetime.now().isoformat(timespec='seconds')}", fontsize=12, va="top")
        ax.text(0.02, 0.62, "Included Tables:", fontsize=12, va="top", fontweight="bold")
        y = 0.56
        for name, _ in tables:
            ax.text(0.04, y, f"- {name}", fontsize=11, va="top")
            y -= 0.06
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for title, csv_path in tables:
            _draw_table_page(pdf=pdf, title=title, df=_maybe_read(csv_path), max_rows=args.max_rows)

    print(f"PDF written to: {out_pdf}")


if __name__ == "__main__":
    main()
