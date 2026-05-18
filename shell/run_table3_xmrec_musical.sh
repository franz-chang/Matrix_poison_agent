#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "[table3-xmrec] ERROR: python interpreter not found" >&2
    exit 1
  fi
fi

SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
TARGET_CATEGORY="${TARGET_CATEGORY:-Musical Instruments}"
TARGET_ITEM_POLICY="${TARGET_ITEM_POLICY:-head}"
XMARKET_ROOT="${XMARKET_ROOT:-${ROOT_DIR}/data/xmrec_musical/markets}"
RAW_ROOT="${RAW_ROOT:-${ROOT_DIR}/data/xmrec_musical/raw}"
OUT_BASE="${OUT_BASE:-${ROOT_DIR}/outputs/table3_xmrec_musical_seed${SEED}}"
PREP_FORCE="${PREP_FORCE:-0}"

BASELINE_OUT="${OUT_BASE}/baseline"
MATRIX_OUT="${OUT_BASE}/matrix"
FINAL_OUT="${OUT_BASE}/final"
mkdir -p "${BASELINE_OUT}" "${MATRIX_OUT}" "${FINAL_OUT}"

echo "[table3-xmrec] ROOT_DIR=${ROOT_DIR}"
echo "[table3-xmrec] PYTHON_BIN=${PYTHON_BIN}"
echo "[table3-xmrec] SEED=${SEED} DEVICE=${DEVICE}"
echo "[table3-xmrec] RAW_ROOT=${RAW_ROOT}"
echo "[table3-xmrec] XMARKET_ROOT=${XMARKET_ROOT}"

prep_args=()
if [[ "${PREP_FORCE}" == "1" ]]; then
  prep_args+=(--force)
fi

"${PYTHON_BIN}" scripts/prepare_xmrec_musical.py  --raw-root "${RAW_ROOT#${ROOT_DIR}/}"  --output-root "${XMARKET_ROOT#${ROOT_DIR}/}"  --category-name "${TARGET_CATEGORY}"  "${prep_args[@]}"

"${PYTHON_BIN}" scripts/run_table4_cross_market_baseline.py  --xmarket-root "${XMARKET_ROOT}"  --output-dir "${BASELINE_OUT}"  --models "bpr,neumf,lightgcn,sasrec"  --target-category "${TARGET_CATEGORY}"  --target-item-policy "${TARGET_ITEM_POLICY}"  --seed "${SEED}"  --device "${DEVICE}"  --min-positive-rating 1.0  --positive-threshold 1.0

"${PYTHON_BIN}" scripts/run_table4_cross_market_matrix_poison.py  --xmarket-root "${XMARKET_ROOT}"  --baseline-csv "${BASELINE_OUT}/table4_baseline_seed${SEED}.csv"  --output-dir "${MATRIX_OUT}"  --models "bpr,neumf,lightgcn,sasrec"  --intents "nuke,push"  --target-category "${TARGET_CATEGORY}"  --target-item-policy "${TARGET_ITEM_POLICY}"  --seed "${SEED}"  --device "${DEVICE}"  --min-positive-rating 1.0  --positive-threshold 1.0

ROOT_DIR="${ROOT_DIR}" BASELINE_OUT="${BASELINE_OUT}" MATRIX_OUT="${MATRIX_OUT}" FINAL_OUT="${FINAL_OUT}" SEED="${SEED}" "${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

root = Path(os.environ["ROOT_DIR"])
baseline_out = Path(os.environ["BASELINE_OUT"])
matrix_out = Path(os.environ["MATRIX_OUT"])
final_out = Path(os.environ["FINAL_OUT"])
seed = int(os.environ["SEED"])

baseline_csv = baseline_out / f"table4_baseline_seed{seed}.csv"
matrix_csv = matrix_out / f"table4_with_matrix_poison_seed{seed}.csv"

bdf = pd.read_csv(baseline_csv)
mdf = pd.read_csv(matrix_csv)

method_order = ["Baseline", "Matrix Poison-Nuke", "Matrix Poison-Push"]
model_order = ["BPR", "NeuMF", "LightGCN", "SASRec"]
cols = [
    "Methods",
    "Model",
    "US Hit@10", "US NDCG@10", "US P. Shift",
    "JP Hit@10", "JP NDCG@10", "JP P. Shift",
    "UK Hit@10", "UK NDCG@10", "UK P. Shift",
]

rows = []
for method in method_order:
    for model in model_order:
        if method == "Baseline":
            sub = bdf[(bdf["Methods"] == method) & (bdf["Model"] == model)]
        else:
            sub = mdf[(mdf["Methods"] == method) & (mdf["Model"] == model)]
        if sub.empty:
            raise ValueError(f"missing row: method={method}, model={model}")
        rows.append({c: sub.iloc[0][c] for c in cols})

out_df = pd.DataFrame(rows, columns=cols)
out_csv = final_out / f"table3_xmrec_musical_seed{seed}.csv"
out_df.to_csv(out_csv, index=False)

md_lines = [
    "Table 3: Comprehensive Performance Metrics with Baseline Methods in Cross-Market Scenarios in Musical Instruments.",
    "",
    "| " + " | ".join(cols) + " |",
    "| " + " | ".join(["---"] * len(cols)) + " |",
]
for _, r in out_df.iterrows():
    md_lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
(final_out / f"table3_xmrec_musical_seed{seed}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

meta = {
    "seed": seed,
    "baseline_csv": str(baseline_csv),
    "matrix_csv": str(matrix_csv),
    "table3_csv": str(out_csv),
}
(final_out / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

print("=== Table 3 (XMRec Musical) ===")
print(out_df.to_string(index=False))
print(f"saved: {out_csv}")
PY

echo "[table3-xmrec] DONE"
echo "[table3-xmrec] Baseline: ${BASELINE_OUT}/table4_baseline_seed${SEED}.csv"
echo "[table3-xmrec] Matrix:   ${MATRIX_OUT}/table4_with_matrix_poison_seed${SEED}.csv"
echo "[table3-xmrec] Table3:   ${FINAL_OUT}/table3_xmrec_musical_seed${SEED}.csv"
