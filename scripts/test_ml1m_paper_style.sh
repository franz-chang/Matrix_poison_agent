#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"

# Expected preprocessed files:
#   DATA_DIR/interactions.csv
#   DATA_DIR/items.csv
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/ml1m}"
TARGET_CATEGORY="${TARGET_CATEGORY:-Thriller}"
TARGET_EVAL_SIZE="${TARGET_EVAL_SIZE:-5}"
MIN_POSITIVE_RATING="${MIN_POSITIVE_RATING:-1.0}"
MODEL="${MODEL:-lightgcn}"
INTENT="${INTENT:-nuke}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
PROFILE_MAX_LEN="${PROFILE_MAX_LEN:-120}"
PUSH_PROFILE_MAX_LEN="${PUSH_PROFILE_MAX_LEN:-24}"
POSITIVE_RATING_THRESHOLD="${POSITIVE_RATING_THRESHOLD:-4.0}"
PUSH_NEG_SAMPLE_BIAS="${PUSH_NEG_SAMPLE_BIAS:-0.0}"
NUKE_NEG_SAMPLE_BIAS="${NUKE_NEG_SAMPLE_BIAS:-0.8}"
PUSH_MIN_SHIFT="${PUSH_MIN_SHIFT:-0.0001}"
PUSH_NEGATIVE_PENALTY="${PUSH_NEGATIVE_PENALTY:-3.0}"
PUSH_TARGET_SHIFT="${PUSH_TARGET_SHIFT:-0.8}"
PUSH_EPSILON_MAX="${PUSH_EPSILON_MAX:-0.5}"
PUSH_USE_EXPLICIT_NEG="${PUSH_USE_EXPLICIT_NEG:-0}"
EPSILON="${EPSILON:-0.005}"
if [[ "${INTENT}" == "push" && "${EPSILON}" == "0.005" ]]; then
  EPSILON=0.02
fi
if [[ "${MODEL}" == "sasrec" && -z "${BATCH_SIZE_OVERRIDE:-}" ]]; then
  BATCH_SIZE=2048
fi

OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs/ml1m_${MODEL}_${INTENT}_${STAMP}}"

if [[ ! -f "${DATA_DIR}/interactions.csv" || ! -f "${DATA_DIR}/items.csv" ]]; then
  echo "[ml1m] ERROR: missing ${DATA_DIR}/interactions.csv or items.csv" >&2
  echo "[ml1m] If you have raw ml-1m files, run:" >&2
  echo "  ${PYTHON_BIN} scripts/preprocess_movielens1m.py --ratings-dat /path/ratings.dat --movies-dat /path/movies.dat --output-dir ${DATA_DIR}" >&2
  exit 1
fi

cd "${ROOT_DIR}"
mkdir -p "${OUT_DIR}"

echo "[ml1m] data_dir=${DATA_DIR}"
echo "[ml1m] target_category=${TARGET_CATEGORY} model=${MODEL} intent=${INTENT}"
echo "[ml1m] target_eval_size=${TARGET_EVAL_SIZE} min_positive_rating=${MIN_POSITIVE_RATING} batch_size=${BATCH_SIZE}"
echo "[ml1m] epsilon=${EPSILON} profile_max_len=${PROFILE_MAX_LEN} push_profile_max_len=${PUSH_PROFILE_MAX_LEN}"
echo "[ml1m] out_dir=${OUT_DIR}"

EXTRA_ARGS=()
if [[ "${PUSH_USE_EXPLICIT_NEG}" == "1" ]]; then
  EXTRA_ARGS+=(--push-use-explicit-neg)
fi

"${PYTHON_BIN}" scripts/run_matrix_poison.py \
  --interactions "${DATA_DIR}/interactions.csv" \
  --items "${DATA_DIR}/items.csv" \
  --target-category "${TARGET_CATEGORY}" \
  --target-eval-size "${TARGET_EVAL_SIZE}" \
  --min-positive-rating "${MIN_POSITIVE_RATING}" \
  --intent "${INTENT}" \
  --model "${MODEL}" \
  --max-iters 5 \
  --epsilon "${EPSILON}" \
  --clean-epochs 50 \
  --probe-epochs 4 \
  --judge-epochs 12 \
  --profile-max-len "${PROFILE_MAX_LEN}" \
  --push-profile-max-len "${PUSH_PROFILE_MAX_LEN}" \
  --positive-rating-threshold "${POSITIVE_RATING_THRESHOLD}" \
  --push-neg-sample-bias "${PUSH_NEG_SAMPLE_BIAS}" \
  --nuke-neg-sample-bias "${NUKE_NEG_SAMPLE_BIAS}" \
  --push-min-shift "${PUSH_MIN_SHIFT}" \
  --push-negative-penalty "${PUSH_NEGATIVE_PENALTY}" \
  --push-target-shift "${PUSH_TARGET_SHIFT}" \
  --push-epsilon-max "${PUSH_EPSILON_MAX}" \
  --batch-size "${BATCH_SIZE}" \
  --output-dir "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}"

echo "[ml1m] DONE: ${OUT_DIR}/run_summary.json"
