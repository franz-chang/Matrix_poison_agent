#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
ensure_ml1m_data
init_common_args

MODELS_CSV="${MODELS_CSV:-bpr,neumf,lightgcn,sasrec}"
IFS=',' read -r -a MODELS <<< "${MODELS_CSV}"
OUT_DIR="${OUT_BASE}/5_3_1_single_market_targeted"
mkdir -p "${OUT_DIR}"

for model in "${MODELS[@]}"; do
  model="$(echo "${model}" | xargs)"
  [[ -z "${model}" ]] && continue
  run_case "${ML1M_INTERACTIONS}" "${ML1M_ITEMS}" "${model}" "nuke" "${TARGET_CATEGORY}" "${EPSILON_NUKE}" \
    "${OUT_DIR}/${model}_nuke"
  run_case "${ML1M_INTERACTIONS}" "${ML1M_ITEMS}" "${model}" "push" "${PUSH_TARGET_CATEGORY}" "${EPSILON_PUSH}" \
    "${OUT_DIR}/${model}_push"
done

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
import pandas as pd
base = Path("${OUT_DIR}")
rows = []
for p in sorted(base.glob("*/run_summary.json")):
    obj = json.loads(p.read_text(encoding="utf-8"))
    bm = obj.get("best_metrics", {})
    rows.append({
        "model": p.parent.name.split("_")[0],
        "intent": p.parent.name.split("_")[1],
        "best_reward": obj.get("best_reward"),
        "prediction_shift": bm.get("prediction_shift"),
        "objective": bm.get("objective"),
        "hit": bm.get("hit"),
        "ndcg": bm.get("ndcg"),
        "output_dir": str(p.parent),
    })
df = pd.DataFrame(rows).sort_values(["model", "intent"])
df.to_csv(base / "summary_5_3_1.csv", index=False)
print(df.to_string(index=False))
print(f"[5.3.1] summary: {base / 'summary_5_3_1.csv'}")
PY
