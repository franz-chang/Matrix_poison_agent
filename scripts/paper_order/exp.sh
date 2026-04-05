#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${ROOT_DIR}/scripts/paper_order"

if [[ ! -d "${SCRIPT_DIR}" ]]; then
  echo "[exp] ERROR: scripts/paper_order not found" >&2
  exit 1
fi

# You can override these when running:
# QUICK=0 OUT_BASE=/path MODELS_CSV="bpr,neumf,lightgcn,sasrec" \
# BATCH_SIZE=4096 SASREC_BATCH_SIZE=3072 bash exp.sh
: "${QUICK:=1}"
: "${OUT_BASE:=${ROOT_DIR}/outputs/paper_shell_$(date +%Y%m%d_%H%M%S)}"
: "${MODELS_CSV:=bpr,neumf,lightgcn,sasrec}"
: "${COMPARISON_MODELS_CSV:=lightgcn}"
: "${COMPARISON_INTENTS_CSV:=nuke}"
: "${CROSS_MODELS_CSV:=lightgcn}"
: "${ABLATION_MODEL:=lightgcn}"
: "${BATCH_SIZE:=4096}"
: "${SASREC_BATCH_SIZE:=3072}"

export QUICK OUT_BASE MODELS_CSV COMPARISON_MODELS_CSV COMPARISON_INTENTS_CSV CROSS_MODELS_CSV ABLATION_MODEL BATCH_SIZE SASREC_BATCH_SIZE

echo "[exp] ROOT_DIR=${ROOT_DIR}"
echo "[exp] OUT_BASE=${OUT_BASE}"
echo "[exp] QUICK=${QUICK}"
echo "[exp] BATCH_SIZE=${BATCH_SIZE} SASREC_BATCH_SIZE=${SASREC_BATCH_SIZE}"

echo "[exp] 5.3.1 Single-Market Targeted Attack"
bash "${SCRIPT_DIR}/01_run_5_3_1_single_market.sh"

echo "[exp] 5.3.2 Baseline Comparison"
bash "${SCRIPT_DIR}/02_run_5_3_2_baseline_compare.sh"

echo "[exp] 5.4 Cross-Market Effectiveness"
bash "${SCRIPT_DIR}/03_run_5_4_cross_market.sh"

echo "[exp] 5.5 Stealthiness Evaluation"
bash "${SCRIPT_DIR}/04_run_5_5_stealth_eval.sh"

echo "[exp] 5.6 Ablation Study"
bash "${SCRIPT_DIR}/05_run_5_6_ablation.sh"

echo "[exp] RQ4 Vulnerability Summary"
bash "${SCRIPT_DIR}/06_run_rq4_vulnerability_summary.sh"

python - <<PY
from pathlib import Path
import pandas as pd
base = Path("${OUT_BASE}")
parts = []
for rel in [
    "5_3_1_single_market_targeted/summary_5_3_1.csv",
    "5_3_2_baseline_comparison/summary_5_3_2.csv",
    "5_4_cross_market/summary_5_4.csv",
    "5_5_stealthiness.csv",
    "5_6_ablation/summary_5_6.csv",
    "rq4_vulnerability_summary.csv",
]:
    p = base / rel
    if p.exists():
        df = pd.read_csv(p)
        df.insert(0, "source_file", rel)
        parts.append(df)
if parts:
    out = pd.concat(parts, ignore_index=True, sort=False)
    out_path = base / "paper_order_all_results.csv"
    out.to_csv(out_path, index=False)
    print(f"[exp] merged table: {out_path}")
else:
    print("[exp] WARNING: no summary csv files found to merge")
PY

echo "[exp] DONE"
