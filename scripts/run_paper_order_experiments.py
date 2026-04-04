#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _split_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = [cols, ["---"] * len(cols)]
    for _, r in df.iterrows():
        rows.append([str(r[c]) for c in cols])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


@dataclass
class RunResult:
    section: str
    experiment: str
    dataset: str
    market: str
    model: str
    intent: str
    method: str
    output_dir: str
    status: str
    best_iteration: int = -1
    best_reward: float = 0.0
    objective: float = 0.0
    prediction_shift: float = 0.0
    hit: float = 0.0
    ndcg: float = 0.0
    clean_hit: float = 0.0
    clean_ndcg: float = 0.0
    stealth_penalty: float = 0.0
    rating_kl: float = 0.0
    profile_shift: float = 0.0
    interaction_ratio: float = 1.0
    hit_delta: float = 0.0
    ndcg_delta: float = 0.0
    note: str = ""

    def as_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Matrix-Poison experiments in paper section order.")
    p.add_argument("--ml1m-interactions", default="data/ml1m/interactions.csv")
    p.add_argument("--ml1m-items", default="data/ml1m/items.csv")
    p.add_argument("--xmarket-root", default="", help="Optional root containing US/JP/UK subdirs with interactions.csv/items.csv")
    p.add_argument("--models", default="bpr,neumf,lightgcn,sasrec", help="Models for section 5.3.1")
    p.add_argument("--comparison-models", default="lightgcn", help="Models for section 5.3.2")
    p.add_argument("--cross-market-models", default="lightgcn", help="Models for section 5.4")
    p.add_argument("--ablation-model", default="lightgcn", help="Model for section 5.6")
    p.add_argument("--comparison-intents", default="nuke", help="Intents for section 5.3.2, e.g. nuke,push")
    p.add_argument("--target-category", default="Action")
    p.add_argument("--push-target-category", default="Thriller")
    p.add_argument("--quick", action="store_true", help="Use quick settings (recommended for first run).")
    p.add_argument("--clean-epochs", type=int, default=None)
    p.add_argument("--probe-epochs", type=int, default=None)
    p.add_argument("--judge-epochs", type=int, default=None)
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--epsilon-nuke", type=float, default=0.005)
    p.add_argument("--epsilon-push", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--camel-model", default="openai:gpt-4o-mini")
    p.add_argument("--disable-camel", action="store_true")
    p.add_argument("--camel-no-stub-fallback", action="store_true")
    p.add_argument("--output-dir", default="")
    return p.parse_args()


def resolve_epochs(args: argparse.Namespace) -> Dict[str, int]:
    if args.quick:
        defaults = {"clean_epochs": 3, "probe_epochs": 1, "judge_epochs": 2, "max_iters": 3}
    else:
        defaults = {"clean_epochs": 50, "probe_epochs": 4, "judge_epochs": 12, "max_iters": 5}
    for k in list(defaults.keys()):
        v = getattr(args, k)
        if v is not None:
            defaults[k] = int(v)
    return defaults


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def prepare_pseudo_markets(ml1m_interactions: Path, ml1m_items: Path, out_dir: Path) -> Dict[str, Dict[str, Path]]:
    out = out_dir / "generated_cross_market"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ml1m_interactions)
    users = sorted(df["user_id"].astype(str).unique().tolist())
    buckets = {"US": set(), "JP": set(), "UK": set()}
    for i, u in enumerate(users):
        if i % 3 == 0:
            buckets["US"].add(u)
        elif i % 3 == 1:
            buckets["JP"].add(u)
        else:
            buckets["UK"].add(u)
    markets: Dict[str, Dict[str, Path]] = {}
    for market, uids in buckets.items():
        mdir = out / market
        mdir.mkdir(parents=True, exist_ok=True)
        sub = df[df["user_id"].astype(str).isin(uids)].copy()
        sub.to_csv(mdir / "interactions.csv", index=False)
        pd.read_csv(ml1m_items).to_csv(mdir / "items.csv", index=False)
        markets[market] = {"interactions": mdir / "interactions.csv", "items": mdir / "items.csv"}
    return markets


def load_cross_markets(args: argparse.Namespace, out_dir: Path) -> Dict[str, Dict[str, Path]]:
    root = Path(args.xmarket_root).expanduser().resolve() if args.xmarket_root else None
    if root and root.exists():
        markets: Dict[str, Dict[str, Path]] = {}
        for market in ["US", "JP", "UK"]:
            mdir = root / market
            i_path = mdir / "interactions.csv"
            item_path = mdir / "items.csv"
            if i_path.exists() and item_path.exists():
                markets[market] = {"interactions": i_path, "items": item_path}
        if len(markets) == 3:
            return markets
    return prepare_pseudo_markets(
        ml1m_interactions=Path(args.ml1m_interactions),
        ml1m_items=Path(args.ml1m_items),
        out_dir=out_dir,
    )


def run_case(
    section: str,
    experiment: str,
    dataset: str,
    market: str,
    model: str,
    intent: str,
    method: str,
    interactions: Path,
    items: Path,
    out_dir: Path,
    base_args: Sequence[str],
) -> RunResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    summary_path = out_dir / "run_summary.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts/run_matrix_poison.py"),
        "--interactions",
        str(interactions),
        "--items",
        str(items),
        "--model",
        model,
        "--intent",
        intent,
        "--output-dir",
        str(out_dir),
        *list(base_args),
    ]
    status = "ok"
    note = ""
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
        ensure_parent(log_path)
        log_path.write_text(proc.stdout + "\n\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")
    except subprocess.CalledProcessError as exc:
        status = "failed"
        note = f"exit={exc.returncode}"
        ensure_parent(log_path)
        log_path.write_text((exc.stdout or "") + "\n\n--- STDERR ---\n" + (exc.stderr or ""), encoding="utf-8")

    row = RunResult(
        section=section,
        experiment=experiment,
        dataset=dataset,
        market=market,
        model=model,
        intent=intent,
        method=method,
        output_dir=str(out_dir),
        status=status,
        note=note,
    )
    if not summary_path.exists():
        return row

    try:
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
        bm = obj.get("best_metrics", {}) or {}
        clean = obj.get("clean_metrics", {}) or {}
        row.best_iteration = int(obj.get("best_iteration", -1))
        row.best_reward = _safe_float(obj.get("best_reward", 0.0))
        row.objective = _safe_float(bm.get("objective", 0.0))
        row.prediction_shift = _safe_float(bm.get("prediction_shift", 0.0))
        row.hit = _safe_float(bm.get("hit", 0.0))
        row.ndcg = _safe_float(bm.get("ndcg", 0.0))
        row.clean_hit = _safe_float(clean.get("hit", 0.0))
        row.clean_ndcg = _safe_float(clean.get("ndcg", 0.0))
        row.stealth_penalty = _safe_float(bm.get("stealth_penalty", 0.0))
        row.rating_kl = _safe_float(bm.get("rating_kl", 0.0))
        row.profile_shift = _safe_float(bm.get("profile_shift", 0.0))
        row.interaction_ratio = _safe_float(bm.get("interaction_ratio", 1.0), default=1.0)
        row.hit_delta = _safe_float(bm.get("hit_delta", row.hit - row.clean_hit))
        row.ndcg_delta = _safe_float(bm.get("ndcg_delta", row.ndcg - row.clean_ndcg))
    except Exception as exc:
        row.status = "failed_parse"
        row.note = f"summary parse error: {exc}"
    return row


def main() -> None:
    args = parse_args()
    epoch_cfg = resolve_epochs(args)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) if args.output_dir else ROOT / "outputs" / f"paper_order_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    models = _split_csv_list(args.models)
    comparison_models = _split_csv_list(args.comparison_models)
    cross_models = _split_csv_list(args.cross_market_models)
    comparison_intents = _split_csv_list(args.comparison_intents)

    common = [
        "--max-iters",
        str(epoch_cfg["max_iters"]),
        "--clean-epochs",
        str(epoch_cfg["clean_epochs"]),
        "--probe-epochs",
        str(epoch_cfg["probe_epochs"]),
        "--judge-epochs",
        str(epoch_cfg["judge_epochs"]),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
        "--camel-model",
        str(args.camel_model),
    ]
    if args.disable_camel:
        common.append("--disable-camel")
    if args.camel_no_stub_fallback:
        common.append("--camel-no-stub-fallback")

    all_rows: List[RunResult] = []

    # 5.3.1 Targeted attack effectiveness (RQ1)
    ml1m_i = Path(args.ml1m_interactions)
    ml1m_items = Path(args.ml1m_items)
    for model in models:
        for intent in ["nuke", "push"]:
            eps = args.epsilon_nuke if intent == "nuke" else args.epsilon_push
            category = args.target_category if intent == "nuke" else args.push_target_category
            row = run_case(
                section="5.3.1",
                experiment="single_market_targeted",
                dataset="ML-1M",
                market="single",
                model=model,
                intent=intent,
                method="matrix_poison",
                interactions=ml1m_i,
                items=ml1m_items,
                out_dir=out_root / "5_3_1_single_market_targeted" / f"{model}_{intent}",
                base_args=[
                    *common,
                    "--target-category",
                    category,
                    "--epsilon",
                    str(eps),
                ],
            )
            all_rows.append(row)

    # 5.3.2 Comparison with baseline poisoning methods (RQ1 extension)
    baseline_methods = ["random", "average", "bandwagon", "segmented", "matrix_poison"]
    for model in comparison_models:
        for intent in comparison_intents:
            eps = args.epsilon_nuke if intent == "nuke" else args.epsilon_push
            category = args.target_category if intent == "nuke" else args.push_target_category
            for method in baseline_methods:
                extra = [*common, "--target-category", category, "--epsilon", str(eps)]
                if method != "matrix_poison":
                    # Baseline method: one fixed heuristic, without adaptive phases.
                    extra.extend(
                        [
                            "--max-iters",
                            "1",
                            "--disable-analysis-phase",
                            "--disable-planning-phase",
                            "--disable-judgment-phase",
                            "--disable-self-evolving",
                            "--fixed-attack-method",
                            method,
                        ]
                    )
                row = run_case(
                    section="5.3.2",
                    experiment="baseline_comparison",
                    dataset="ML-1M",
                    market="single",
                    model=model,
                    intent=intent,
                    method=method,
                    interactions=ml1m_i,
                    items=ml1m_items,
                    out_dir=out_root / "5_3_2_baseline_comparison" / f"{model}_{intent}_{method}",
                    base_args=extra,
                )
                all_rows.append(row)

    # 5.4 Cross-market attack effectiveness (RQ2)
    markets = load_cross_markets(args, out_root)
    for market, paths in markets.items():
        for model in cross_models:
            for intent in ["nuke", "push"]:
                eps = args.epsilon_nuke if intent == "nuke" else args.epsilon_push
                category = args.target_category if intent == "nuke" else args.push_target_category
                row = run_case(
                    section="5.4",
                    experiment="cross_market_effectiveness",
                    dataset="XMarket_or_Pseudo",
                    market=market,
                    model=model,
                    intent=intent,
                    method="matrix_poison",
                    interactions=paths["interactions"],
                    items=paths["items"],
                    out_dir=out_root / "5_4_cross_market" / f"{market}_{model}_{intent}",
                    base_args=[
                        *common,
                        "--target-category",
                        category,
                        "--epsilon",
                        str(eps),
                    ],
                )
                all_rows.append(row)

    # 5.6 Ablation study (RQ4 support)
    ablation_variants = {
        "full": [],
        "wo_analysis": ["--disable-analysis-phase"],
        "wo_planning": ["--disable-planning-phase"],
        "wo_judgment": ["--disable-judgment-phase"],
        "wo_self_evolving": ["--disable-self-evolving"],
    }
    for variant, flags in ablation_variants.items():
        row = run_case(
            section="5.6",
            experiment="ablation",
            dataset="ML-1M",
            market="single",
            model=args.ablation_model,
            intent="nuke",
            method=variant,
            interactions=ml1m_i,
            items=ml1m_items,
            out_dir=out_root / "5_6_ablation" / f"{args.ablation_model}_{variant}",
            base_args=[
                *common,
                "--target-category",
                args.target_category,
                "--epsilon",
                str(args.epsilon_nuke),
                *flags,
            ],
        )
        all_rows.append(row)

    df = pd.DataFrame([r.as_dict() for r in all_rows])
    df.to_csv(out_root / "paper_order_all_results.csv", index=False)

    rq1_df = df[df["section"].isin(["5.3.1", "5.3.2"])].copy()
    rq2_df = df[df["section"] == "5.4"].copy()
    ablation_df = df[df["section"] == "5.6"].copy()

    # 5.5 stealthiness report from all matrix-poison runs
    stealth_df = df[df["method"] == "matrix_poison"].copy()
    stealth_df["utility_drop"] = (
        (stealth_df["clean_hit"] - stealth_df["hit"]).clip(lower=0.0)
        + (stealth_df["clean_ndcg"] - stealth_df["ndcg"]).clip(lower=0.0)
    ) / 2.0
    stealth_df = stealth_df[
        [
            "section",
            "dataset",
            "market",
            "model",
            "intent",
            "utility_drop",
            "rating_kl",
            "profile_shift",
            "interaction_ratio",
            "stealth_penalty",
            "status",
            "output_dir",
        ]
    ]
    stealth_df.to_csv(out_root / "5_5_stealthiness.csv", index=False)

    # RQ4 vulnerability summary
    vuln_df = df[df["method"] == "matrix_poison"].copy()
    vuln_df["vulnerability_score"] = vuln_df["objective"].abs()
    vuln_df = (
        vuln_df.groupby(["dataset", "market", "model", "intent"], as_index=False)["vulnerability_score"]
        .mean()
        .sort_values("vulnerability_score", ascending=False)
    )
    vuln_df.to_csv(out_root / "rq4_vulnerability_summary.csv", index=False)

    report = [
        "# Matrix-Poison Paper-Order Experiment Report",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Output root: `{out_root}`",
        f"- Quick mode: `{args.quick}`",
        f"- CAMEL enabled: `{not args.disable_camel}`",
        f"- CAMEL model: `{args.camel_model}`",
        "",
        "## 5.3.1 Single-Market Targeted Attack (RQ1)",
        _md_table(
            df[df["section"] == "5.3.1"][
                ["dataset", "model", "intent", "best_reward", "prediction_shift", "hit", "ndcg", "status"]
            ]
        ),
        "",
        "## 5.3.2 Baseline Comparison (RQ1 Extension)",
        _md_table(
            df[df["section"] == "5.3.2"][
                ["dataset", "model", "intent", "method", "best_reward", "prediction_shift", "hit", "ndcg", "status"]
            ]
        ),
        "",
        "## 5.4 Cross-Market Effectiveness (RQ2)",
        _md_table(
            df[df["section"] == "5.4"][
                ["dataset", "market", "model", "intent", "best_reward", "prediction_shift", "hit", "ndcg", "status"]
            ]
        ),
        "",
        "## 5.5 Stealthiness (RQ3)",
        _md_table(stealth_df),
        "",
        "## 5.6 Ablation",
        _md_table(
            ablation_df[
                ["dataset", "model", "method", "best_reward", "objective", "prediction_shift", "hit", "ndcg", "status"]
            ]
        ),
        "",
        "## RQ4 Vulnerability Summary",
        _md_table(vuln_df),
        "",
        "## Artifact Files",
        "- `paper_order_all_results.csv`",
        "- `5_5_stealthiness.csv`",
        "- `rq4_vulnerability_summary.csv`",
    ]
    (out_root / "paper_order_report.md").write_text("\n".join(report), encoding="utf-8")

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(out_root),
        "args": vars(args),
        "epoch_cfg": epoch_cfg,
        "market_mode": "provided" if args.xmarket_root else "pseudo_from_ml1m",
    }
    (out_root / "paper_order_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[paper-order] done: {out_root}")
    print(f"[paper-order] report: {out_root / 'paper_order_report.md'}")


if __name__ == "__main__":
    main()
