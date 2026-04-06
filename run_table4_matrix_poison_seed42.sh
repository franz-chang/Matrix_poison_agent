#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "[table4-matrix-poison] ERROR: python interpreter not found" >&2
    exit 1
  fi
fi

SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"

CLEAN_EPOCHS="${CLEAN_EPOCHS:-20}"
PROBE_EPOCHS="${PROBE_EPOCHS:-4}"
JUDGE_EPOCHS="${JUDGE_EPOCHS:-8}"
MAX_ITERS="${MAX_ITERS:-5}"

BATCH_SIZE="${BATCH_SIZE:-8192}"
SASREC_BATCH_SIZE="${SASREC_BATCH_SIZE:-4096}"

TARGET_CATEGORY="${TARGET_CATEGORY:-Action}"
TARGET_ITEM_POLICY="${TARGET_ITEM_POLICY:-head}"
TARGET_EVAL_SIZE="${TARGET_EVAL_SIZE:-1}"

NUKE_EPSILON="${NUKE_EPSILON:-0.02}"
PUSH_EPSILON="${PUSH_EPSILON:-0.02}"

XMARKET_ROOT="${XMARKET_ROOT:-}"
BASELINE_OUT="${BASELINE_OUT:-${ROOT_DIR}/outputs/table4_baseline_seed42}"
BASELINE_CSV="${BASELINE_CSV:-${BASELINE_OUT}/table4_baseline_seed42.csv}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs/table4_nuke_push_seed42}"

# Better reproducibility on CUDA.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONHASHSEED="${SEED}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "[table4-matrix-poison] ROOT_DIR=${ROOT_DIR}"
echo "[table4-matrix-poison] PYTHON_BIN=${PYTHON_BIN}"
echo "[table4-matrix-poison] SEED=${SEED} DEVICE=${DEVICE}"
echo "[table4-matrix-poison] EPOCHS clean=${CLEAN_EPOCHS} probe=${PROBE_EPOCHS} judge=${JUDGE_EPOCHS} max_iters=${MAX_ITERS}"
echo "[table4-matrix-poison] BATCH_SIZE=${BATCH_SIZE} SASREC_BATCH_SIZE=${SASREC_BATCH_SIZE}"
echo "[table4-matrix-poison] TARGET_CATEGORY=${TARGET_CATEGORY} TARGET_ITEM_POLICY=${TARGET_ITEM_POLICY}"
echo "[table4-matrix-poison] NUKE_EPSILON=${NUKE_EPSILON} PUSH_EPSILON=${PUSH_EPSILON}"
echo "[table4-matrix-poison] BASELINE_CSV=${BASELINE_CSV}"
echo "[table4-matrix-poison] OUT_DIR=${OUT_DIR}"
if [[ -n "${XMARKET_ROOT}" ]]; then
  echo "[table4-matrix-poison] XMARKET_ROOT=${XMARKET_ROOT}"
else
  echo "[table4-matrix-poison] XMARKET_ROOT not set; will use pseudo US/JP/UK from ML-1M"
fi

if [[ ! -f "${BASELINE_CSV}" ]]; then
  echo "[table4-matrix-poison] Baseline csv missing; running baseline first..."
  SEED="${SEED}" \
  DEVICE="${DEVICE}" \
  CLEAN_EPOCHS="${CLEAN_EPOCHS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  SASREC_BATCH_SIZE="${SASREC_BATCH_SIZE}" \
  TARGET_CATEGORY="${TARGET_CATEGORY}" \
  TARGET_ITEM_POLICY="${TARGET_ITEM_POLICY}" \
  TARGET_EVAL_SIZE="${TARGET_EVAL_SIZE}" \
  XMARKET_ROOT="${XMARKET_ROOT}" \
  OUT_DIR="${BASELINE_OUT}" \
  bash "${ROOT_DIR}/run_table4_baseline_seed42.sh"
fi

"${PYTHON_BIN}" scripts/run_table4_cross_market_matrix_poison.py \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --clean-epochs "${CLEAN_EPOCHS}" \
  --probe-epochs "${PROBE_EPOCHS}" \
  --judge-epochs "${JUDGE_EPOCHS}" \
  --max-iters "${MAX_ITERS}" \
  --batch-size "${BATCH_SIZE}" \
  --sasrec-batch-size "${SASREC_BATCH_SIZE}" \
  --target-category "${TARGET_CATEGORY}" \
  --target-item-policy "${TARGET_ITEM_POLICY}" \
  --target-eval-size "${TARGET_EVAL_SIZE}" \
  --nuke-epsilon "${NUKE_EPSILON}" \
  --push-epsilon "${PUSH_EPSILON}" \
  --min-positive-rating 1.0 \
  --positive-threshold 1.0 \
  --baseline-csv "${BASELINE_CSV}" \
  --output-dir "${OUT_DIR}" \
  ${XMARKET_ROOT:+--xmarket-root "${XMARKET_ROOT}"}

echo "[table4-matrix-poison] DONE"
echo "[table4-matrix-poison] Table CSV:   ${OUT_DIR}/table4_with_matrix_poison_seed42.csv"
echo "[table4-matrix-poison] Table LaTeX: ${OUT_DIR}/table4_with_matrix_poison_seed42.tex"

