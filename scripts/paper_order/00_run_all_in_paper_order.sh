#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

echo "[paper_order] OUT_BASE=${OUT_BASE}"
echo "[paper_order] QUICK=${QUICK} CLEAN=${CLEAN_EPOCHS} PROBE=${PROBE_EPOCHS} JUDGE=${JUDGE_EPOCHS} MAX_ITERS=${MAX_ITERS}"
echo "[paper_order] CAMEL_ENABLED=${CAMEL_ENABLED} CAMEL_MODEL=${CAMEL_MODEL}"

bash "${SCRIPT_DIR}/01_run_5_3_1_single_market.sh"
bash "${SCRIPT_DIR}/02_run_5_3_2_baseline_compare.sh"
bash "${SCRIPT_DIR}/03_run_5_4_cross_market.sh"
bash "${SCRIPT_DIR}/04_run_5_5_stealth_eval.sh"
bash "${SCRIPT_DIR}/05_run_5_6_ablation.sh"
bash "${SCRIPT_DIR}/06_run_rq4_vulnerability_summary.sh"

"${PYTHON_BIN}" - <<PY
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
    out.to_csv(base / "paper_order_all_results.csv", index=False)
    print(f"[paper_order] merged -> {base / 'paper_order_all_results.csv'}")
else:
    print("[paper_order] no summary files to merge")
PY

echo "[paper_order] DONE"
