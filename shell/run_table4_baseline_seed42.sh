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
    echo "[table4-baseline] ERROR: python interpreter not found" >&2
    exit 1
  fi
fi

SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
CLEAN_EPOCHS="${CLEAN_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
SASREC_BATCH_SIZE="${SASREC_BATCH_SIZE:-4096}"
SASREC_RECENT_STEPS="${SASREC_RECENT_STEPS:-0}"
EVAL_USER_LIMIT="${EVAL_USER_LIMIT:-0}"
TARGET_CATEGORY="${TARGET_CATEGORY:-Action}"
TARGET_ITEM_POLICY="${TARGET_ITEM_POLICY:-head}"
TARGET_EVAL_SIZE="${TARGET_EVAL_SIZE:-1}"
XMARKET_ROOT="${XMARKET_ROOT:-}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs/table4_baseline_seed42}"

# Better reproducibility on CUDA.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONHASHSEED="${SEED}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "[table4-baseline] ROOT_DIR=${ROOT_DIR}"
echo "[table4-baseline] PYTHON_BIN=${PYTHON_BIN}"
echo "[table4-baseline] SEED=${SEED} DEVICE=${DEVICE}"
echo "[table4-baseline] BATCH_SIZE=${BATCH_SIZE} SASREC_BATCH_SIZE=${SASREC_BATCH_SIZE} SASREC_RECENT_STEPS=${SASREC_RECENT_STEPS}"
echo "[table4-baseline] TARGET_CATEGORY=${TARGET_CATEGORY} TARGET_ITEM_POLICY=${TARGET_ITEM_POLICY} EVAL_USER_LIMIT=${EVAL_USER_LIMIT}"
echo "[table4-baseline] OUT_DIR=${OUT_DIR}"
if [[ -n "${XMARKET_ROOT}" ]]; then
  echo "[table4-baseline] XMARKET_ROOT=${XMARKET_ROOT}"
else
  echo "[table4-baseline] XMARKET_ROOT not set; will generate pseudo US/JP/UK from ML-1M"
fi

"${PYTHON_BIN}" scripts/run_table4_cross_market_baseline.py \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --clean-epochs "${CLEAN_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --sasrec-batch-size "${SASREC_BATCH_SIZE}" \
  --sasrec-recent-steps "${SASREC_RECENT_STEPS}" \
  --target-category "${TARGET_CATEGORY}" \
  --target-item-policy "${TARGET_ITEM_POLICY}" \
  --target-eval-size "${TARGET_EVAL_SIZE}" \
  --min-positive-rating 1.0 \
  --positive-threshold 1.0 \
  --eval-user-limit "${EVAL_USER_LIMIT}" \
  --output-dir "${OUT_DIR}" \
  ${XMARKET_ROOT:+--xmarket-root "${XMARKET_ROOT}"}

echo "[table4-baseline] DONE"
echo "[table4-baseline] CSV:   ${OUT_DIR}/table4_baseline_seed42.csv"
echo "[table4-baseline] LaTeX: ${OUT_DIR}/table4_baseline_seed42.tex"

