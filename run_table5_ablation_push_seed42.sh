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
    echo "[table5-push-ablation] ERROR: python interpreter not found" >&2
    exit 1
  fi
fi

SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"

MODEL="${MODEL:-bpr}"
INTENT="push"
TARGET_CATEGORY="${TARGET_CATEGORY:-Action}"
TARGET_ITEM_POLICY="${TARGET_ITEM_POLICY:-auto}"
TARGET_EVAL_SIZE="${TARGET_EVAL_SIZE:-1}"

# Keep the same ablation settings as Table 4 by default.
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
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs/table5_ablation_push_seed42}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-table5_ablation_push_seed42}"
TABLE_CAPTION="${TABLE_CAPTION:-Performance Metrics for Method Variants (Table 5 Push Ablation, Seed=42).}"
TABLE_LABEL="${TABLE_LABEL:-tab:table5-push-ablation-seed42}"

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
    echo "[table5-push-ablation] ERROR: ENABLE_CAMEL=1 but OPENAI_API_KEY is missing and CAMEL_NO_STUB_FALLBACK=1" >&2
    exit 1
  else
    LLM_EFFECTIVE="CAMEL STUB fallback (OPENAI_API_KEY missing)"
  fi
fi

echo "[table5-push-ablation] ROOT_DIR=${ROOT_DIR}"
echo "[table5-push-ablation] PYTHON_BIN=${PYTHON_BIN}"
echo "[table5-push-ablation] model=${MODEL} intent=${INTENT} seed=${SEED} device=${DEVICE}"
echo "[table5-push-ablation] epochs clean=${CLEAN_EPOCHS} probe=${PROBE_EPOCHS} judge=${JUDGE_EPOCHS} max_iters=${MAX_ITERS}"
echo "[table5-push-ablation] epsilon=${EPSILON} batch_size=${BATCH_SIZE}"
echo "[table5-push-ablation] target_category=${TARGET_CATEGORY} target_item_policy=${TARGET_ITEM_POLICY} target_eval_size=${TARGET_EVAL_SIZE}"
echo "[table5-push-ablation] output=${OUT_DIR} prefix=${OUTPUT_PREFIX}"
echo "[table5-push-ablation] LLM validation => ${LLM_EFFECTIVE}"

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
  --output-prefix "${OUTPUT_PREFIX}"
  --table-caption "${TABLE_CAPTION}"
  --table-label "${TABLE_LABEL}"
)

if [[ "${ENABLE_CAMEL}" == "1" ]]; then
  cmd+=(--enable-camel --camel-model "${CAMEL_MODEL}")
  if [[ "${CAMEL_NO_STUB_FALLBACK}" == "1" ]]; then
    cmd+=(--camel-no-stub-fallback)
  fi
fi

"${cmd[@]}"

echo "[table5-push-ablation] DONE"
echo "[table5-push-ablation] CSV:    ${OUT_DIR}/${OUTPUT_PREFIX}.csv"
echo "[table5-push-ablation] LaTeX:  ${OUT_DIR}/${OUTPUT_PREFIX}.tex"
echo "[table5-push-ablation] Meta:   ${OUT_DIR}/${OUTPUT_PREFIX}_meta.json"
