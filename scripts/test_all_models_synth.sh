#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"

DATA_DIR="${ROOT_DIR}/examples/synth_all_models_${STAMP}"
OUT_BASE="${ROOT_DIR}/outputs/all_models_${STAMP}"
TARGET_CATEGORY="${TARGET_CATEGORY:-C1}"
INTENT="${INTENT:-nuke}"
MODELS=("bpr" "neumf" "lightgcn" "sasrec")

cd "${ROOT_DIR}"
mkdir -p "${OUT_BASE}"

echo "[all_models] building synthetic dataset..."
"${PYTHON_BIN}" scripts/make_synthetic_dataset.py \
  --output-dir "${DATA_DIR}" \
  --num-users 220 \
  --num-items 420 \
  --num-categories 10 \
  --avg-interactions 26 \
  --seed 42

for model in "${MODELS[@]}"; do
  out_dir="${OUT_BASE}/${model}"
  echo "[all_models] running model=${model} out_dir=${out_dir}"
  "${PYTHON_BIN}" scripts/run_matrix_poison.py \
    --interactions "${DATA_DIR}/interactions.csv" \
    --items "${DATA_DIR}/items.csv" \
    --target-category "${TARGET_CATEGORY}" \
    --intent "${INTENT}" \
    --model "${model}" \
    --max-iters 2 \
    --clean-epochs 2 \
    --probe-epochs 1 \
    --judge-epochs 1 \
    --batch-size 256 \
    --output-dir "${out_dir}"
done

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
base = Path("${OUT_BASE}")
print("[all_models] Summary")
print("model\tbest_reward\tpred_shift\thit@10\tndcg@10")
for p in sorted(base.glob("*/run_summary.json")):
    obj = json.loads(p.read_text(encoding="utf-8"))
    m = p.parent.name
    bm = obj.get("best_metrics", {})
    print(
        f"{m}\t{obj.get('best_reward', 0):.6f}\t"
        f"{bm.get('prediction_shift', 0):.6e}\t"
        f"{bm.get('hit', 0):.4f}\t{bm.get('ndcg', 0):.4f}"
    )
print("[all_models] Output base:", base)
PY

echo "[all_models] DONE"
