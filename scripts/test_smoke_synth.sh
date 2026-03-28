#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"

DATA_DIR="${ROOT_DIR}/examples/synth_smoke_${STAMP}"
OUT_DIR="${ROOT_DIR}/outputs/smoke_${STAMP}"

MODEL="${MODEL:-bpr}"
TARGET_CATEGORY="${TARGET_CATEGORY:-C1}"
INTENT="${INTENT:-nuke}"

echo "[smoke] data_dir=${DATA_DIR}"
echo "[smoke] out_dir=${OUT_DIR}"
echo "[smoke] model=${MODEL} intent=${INTENT} category=${TARGET_CATEGORY}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" scripts/make_synthetic_dataset.py \
  --output-dir "${DATA_DIR}" \
  --num-users 140 \
  --num-items 260 \
  --num-categories 8 \
  --avg-interactions 24 \
  --seed 42

"${PYTHON_BIN}" scripts/run_matrix_poison.py \
  --interactions "${DATA_DIR}/interactions.csv" \
  --items "${DATA_DIR}/items.csv" \
  --target-category "${TARGET_CATEGORY}" \
  --intent "${INTENT}" \
  --model "${MODEL}" \
  --max-iters 2 \
  --clean-epochs 2 \
  --probe-epochs 1 \
  --judge-epochs 1 \
  --batch-size 256 \
  --output-dir "${OUT_DIR}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
summary = Path("${OUT_DIR}/run_summary.json")
obj = json.loads(summary.read_text(encoding="utf-8"))
print("[smoke] best_iteration:", obj["best_iteration"])
print("[smoke] best_reward:", obj["best_reward"])
print("[smoke] best_prediction_shift:", obj["best_metrics"].get("prediction_shift"))
print("[smoke] best_weights:", obj["best_weights"])
print("[smoke] summary_file:", summary)
PY

echo "[smoke] DONE"
