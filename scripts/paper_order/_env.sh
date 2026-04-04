#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_BASE="${OUT_BASE:-${ROOT_DIR}/outputs/paper_shell_${STAMP}}"
mkdir -p "${OUT_BASE}"

ML1M_DIR="${ML1M_DIR:-${ROOT_DIR}/data/ml1m}"
ML1M_INTERACTIONS="${ML1M_INTERACTIONS:-${ML1M_DIR}/interactions.csv}"
ML1M_ITEMS="${ML1M_ITEMS:-${ML1M_DIR}/items.csv}"

TARGET_CATEGORY="${TARGET_CATEGORY:-Action}"
PUSH_TARGET_CATEGORY="${PUSH_TARGET_CATEGORY:-Thriller}"
EPSILON_NUKE="${EPSILON_NUKE:-0.005}"
EPSILON_PUSH="${EPSILON_PUSH:-0.02}"

SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
QUICK="${QUICK:-1}" # 1=quick, 0=paper-scale

CAMEL_ENABLED="${CAMEL_ENABLED:-1}"
CAMEL_MODEL="${CAMEL_MODEL:-openai:gpt-4o-mini}"
CAMEL_NO_STUB_FALLBACK="${CAMEL_NO_STUB_FALLBACK:-0}"

if [[ "${QUICK}" == "1" ]]; then
  CLEAN_EPOCHS="${CLEAN_EPOCHS:-3}"
  PROBE_EPOCHS="${PROBE_EPOCHS:-1}"
  JUDGE_EPOCHS="${JUDGE_EPOCHS:-2}"
  MAX_ITERS="${MAX_ITERS:-3}"
else
  CLEAN_EPOCHS="${CLEAN_EPOCHS:-50}"
  PROBE_EPOCHS="${PROBE_EPOCHS:-4}"
  JUDGE_EPOCHS="${JUDGE_EPOCHS:-12}"
  MAX_ITERS="${MAX_ITERS:-5}"
fi

function ensure_ml1m_data() {
  if [[ ! -f "${ML1M_INTERACTIONS}" || ! -f "${ML1M_ITEMS}" ]]; then
    echo "[paper_order] ERROR: missing ML-1M files" >&2
    echo "  interactions: ${ML1M_INTERACTIONS}" >&2
    echo "  items:        ${ML1M_ITEMS}" >&2
    exit 1
  fi
}

function init_common_args() {
  COMMON_ARGS=(
    --max-iters "${MAX_ITERS}"
    --clean-epochs "${CLEAN_EPOCHS}"
    --probe-epochs "${PROBE_EPOCHS}"
    --judge-epochs "${JUDGE_EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --seed "${SEED}"
    --device "${DEVICE}"
    --camel-model "${CAMEL_MODEL}"
  )
  if [[ "${CAMEL_ENABLED}" != "1" ]]; then
    COMMON_ARGS+=(--disable-camel)
  fi
  if [[ "${CAMEL_NO_STUB_FALLBACK}" == "1" ]]; then
    COMMON_ARGS+=(--camel-no-stub-fallback)
  fi
}

function run_case() {
  local interactions="$1"
  local items="$2"
  local model="$3"
  local intent="$4"
  local category="$5"
  local epsilon="$6"
  local out_dir="$7"
  shift 7
  local extra=("$@")

  mkdir -p "${out_dir}"
  echo "[paper_order] running model=${model} intent=${intent} category=${category} out=${out_dir}"
  if [[ ${#extra[@]} -gt 0 ]]; then
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_matrix_poison.py" \
      --interactions "${interactions}" \
      --items "${items}" \
      --model "${model}" \
      --intent "${intent}" \
      --target-category "${category}" \
      --epsilon "${epsilon}" \
      --output-dir "${out_dir}" \
      "${COMMON_ARGS[@]}" \
      "${extra[@]}"
  else
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_matrix_poison.py" \
      --interactions "${interactions}" \
      --items "${items}" \
      --model "${model}" \
      --intent "${intent}" \
      --target-category "${category}" \
      --epsilon "${epsilon}" \
      --output-dir "${out_dir}" \
      "${COMMON_ARGS[@]}"
  fi
}
