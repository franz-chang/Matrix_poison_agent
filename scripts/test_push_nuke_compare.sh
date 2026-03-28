#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"

DATA_DIR="${ROOT_DIR}/examples/synth_push_nuke_${STAMP}"
OUT_BASE="${ROOT_DIR}/outputs/push_nuke_${STAMP}"
MODEL="${MODEL:-lightgcn}"
TARGET_CATEGORY="${TARGET_CATEGORY:-C1}"

cd "${ROOT_DIR}"
mkdir -p "${OUT_BASE}"

"${PYTHON_BIN}" scripts/make_synthetic_dataset.py \
  --output-dir "${DATA_DIR}" \
  --num-users 220 \
  --num-items 420 \
  --num-categories 10 \
  --avg-interactions 26 \
  --seed 42

for intent in push nuke; do
  out_dir="${OUT_BASE}/${intent}"
  echo "[push_nuke] running intent=${intent}, model=${MODEL}"
  "${PYTHON_BIN}" scripts/run_matrix_poison.py \
    --interactions "${DATA_DIR}/interactions.csv" \
    --items "${DATA_DIR}/items.csv" \
    --target-category "${TARGET_CATEGORY}" \
    --intent "${intent}" \
    --model "${MODEL}" \
    --max-iters 3 \
    --clean-epochs 3 \
    --probe-epochs 1 \
    --judge-epochs 2 \
    --batch-size 256 \
    --output-dir "${out_dir}"
done

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
base = Path("${OUT_BASE}")
print("[push_nuke] Comparison")
print("intent\tbest_reward\tprediction_shift\tobjective")
for intent in ["push", "nuke"]:
    p = base / intent / "run_summary.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    bm = obj.get("best_metrics", {})
    print(
        f"{intent}\t{obj.get('best_reward', 0):.6f}\t"
        f"{bm.get('prediction_shift', 0):.6e}\t{bm.get('objective', 0):.6e}"
    )
print("[push_nuke] Output base:", base)
PY

echo "[push_nuke] DONE"
