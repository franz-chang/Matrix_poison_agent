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
    echo "[push-table3] ERROR: python interpreter not found" >&2
    exit 1
  fi
fi

SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
SASREC_BATCH_SIZE="${SASREC_BATCH_SIZE:-4096}"
DEVICE="${DEVICE:-auto}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs/sota_seed42_matrix_push_tuned}"
BASELINE_TABLE="${BASELINE_TABLE:-outputs/baseline_seed42_run1/table_baseline_only.csv}"
NUKE_TABLE="${NUKE_TABLE:-outputs/sota_seed42_matrix_nuke_final_best/table_matrix_poison_nuke_seed42_best.csv}"

# Improve determinism on CUDA runs.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONHASHSEED="${SEED}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "[push-table3] ROOT_DIR=${ROOT_DIR}"
echo "[push-table3] PYTHON_BIN=${PYTHON_BIN}"
echo "[push-table3] SEED=${SEED} DEVICE=${DEVICE}"
echo "[push-table3] BATCH_SIZE=${BATCH_SIZE} SASREC_BATCH_SIZE=${SASREC_BATCH_SIZE}"
echo "[push-table3] OUT_DIR=${OUT_DIR}"

"${PYTHON_BIN}" scripts/search_matrix_poison_push_fast.py \
  --output-dir "${OUT_DIR}" \
  --baseline-table "${BASELINE_TABLE}" \
  --nuke-table "${NUKE_TABLE}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --sasrec-batch-size "${SASREC_BATCH_SIZE}" \
  --min-positive-rating 1.0

echo "[push-table3] DONE"
echo "[push-table3] Push row:   ${OUT_DIR}/table_matrix_poison_push_seed42_best.csv"
echo "[push-table3] Full Table3: ${OUT_DIR}/table3_with_matrix_poison_push_seed42.csv"

