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
    echo "[table4-ablation] ERROR: python interpreter not found" >&2
    exit 1
  fi
fi

SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"

MODEL="${MODEL:-bpr}"
INTENT="${INTENT:-nuke}"
TARGET_CATEGORY="${TARGET_CATEGORY:-Action}"
TARGET_ITEM_POLICY="${TARGET_ITEM_POLICY:-auto}"
TARGET_EVAL_SIZE="${TARGET_EVAL_SIZE:-1}"

EPSILON="${EPSILON:-0.005}"
CLEAN_EPOCHS="${CLEAN_EPOCHS:-50}"
PROBE_EPOCHS="${PROBE_EPOCHS:-4}"
JUDGE_EPOCHS="${JUDGE_EPOCHS:-12}"
MAX_ITERS="${MAX_ITERS:-5}"
BATCH_SIZE="${BATCH_SIZE:-4096}"

MIN_POSITIVE_RATING="${MIN_POSITIVE_RATING:-1.0}"
POSITIVE_RATING_THRESHOLD="${POSITIVE_RATING_THRESHOLD:-4.0}"

ENABLE_CAMEL="${ENABLE_CAMEL:-0}"
CAMEL_MODEL="${CAMEL_MODEL:-openai:gpt-4o-mini}"
CAMEL_NO_STUB_FALLBACK="${CAMEL_NO_STUB_FALLBACK:-0}"

INTERACTIONS="${INTERACTIONS:-data/ml1m/interactions.csv}"
ITEMS="${ITEMS:-data/ml1m/items.csv}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs/table4_ablation_seed42}"

# Better reproducibility on CUDA.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONHASHSEED="${SEED}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

LLM_EFFECTIVE="disabled (--disable-camel)"
if [[ "${ENABLE_CAMEL}" == "1" ]]; then
  if printenv OPENAI_API_KEY >/dev/null; then
    LLM_EFFECTIVE="CAMEL active via ${CAMEL_MODEL}"
  elif [[ "${CAMEL_NO_STUB_FALLBACK}" == "1" ]]; then
    echo "[table4-ablation] ERROR: ENABLE_CAMEL=1 but OPENAI_API_KEY is missing and CAMEL_NO_STUB_FALLBACK=1" >&2
    exit 1
  else
    LLM_EFFECTIVE="CAMEL STUB fallback (OPENAI_API_KEY missing)"
  fi
fi

echo "[table4-ablation] ROOT_DIR=${ROOT_DIR}"
echo "[table4-ablation] PYTHON_BIN=${PYTHON_BIN}"
echo "[table4-ablation] model=${MODEL} intent=${INTENT} seed=${SEED} device=${DEVICE}"
echo "[table4-ablation] epochs clean=${CLEAN_EPOCHS} probe=${PROBE_EPOCHS} judge=${JUDGE_EPOCHS} max_iters=${MAX_ITERS}"
echo "[table4-ablation] epsilon=${EPSILON} batch_size=${BATCH_SIZE}"
echo "[table4-ablation] target_category=${TARGET_CATEGORY} target_item_policy=${TARGET_ITEM_POLICY} target_eval_size=${TARGET_EVAL_SIZE}"
echo "[table4-ablation] output=${OUT_DIR}"
echo "[table4-ablation] LLM validation => ${LLM_EFFECTIVE}"

cmd=(
  "${PYTHON_BIN}" "scripts/run_table4_ablation.py"
  --python-bin "${PYTHON_BIN}"
  --interactions "${INTERACTIONS}"
  --items "${ITEMS}"
  --model "${MODEL}"
  --intent "${INTENT}"
  --target-category "${TARGET_CATEGORY}"
  --target-item-policy "${TARGET_ITEM_POLICY}"
  --target-eval-size "${TARGET_EVAL_SIZE}"
  --epsilon "${EPSILON}"
  --clean-epochs "${CLEAN_EPOCHS}"
  --probe-epochs "${PROBE_EPOCHS}"
  --judge-epochs "${JUDGE_EPOCHS}"
  --max-iters "${MAX_ITERS}"
  --batch-size "${BATCH_SIZE}"
  --min-positive-rating "${MIN_POSITIVE_RATING}"
  --positive-rating-threshold "${POSITIVE_RATING_THRESHOLD}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --output-dir "${OUT_DIR}"
)

if [[ "${ENABLE_CAMEL}" == "1" ]]; then
  cmd+=(--enable-camel --camel-model "${CAMEL_MODEL}")
  if [[ "${CAMEL_NO_STUB_FALLBACK}" == "1" ]]; then
    cmd+=(--camel-no-stub-fallback)
  fi
fi

"${cmd[@]}"

echo "[table4-ablation] DONE"
echo "[table4-ablation] CSV:    ${OUT_DIR}/table4_ablation_seed42.csv"
echo "[table4-ablation] LaTeX:  ${OUT_DIR}/table4_ablation_seed42.tex"
echo "[table4-ablation] Meta:   ${OUT_DIR}/table4_ablation_seed42_meta.json"
