#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
ensure_ml1m_data
init_common_args

COMPARISON_MODELS_CSV="${COMPARISON_MODELS_CSV:-lightgcn}"
COMPARISON_INTENTS_CSV="${COMPARISON_INTENTS_CSV:-nuke}"
IFS=',' read -r -a COMPARISON_MODELS <<< "${COMPARISON_MODELS_CSV}"
IFS=',' read -r -a COMPARISON_INTENTS <<< "${COMPARISON_INTENTS_CSV}"
METHODS=(random average bandwagon segmented matrix_poison)

OUT_DIR="${OUT_BASE}/5_3_2_baseline_comparison"
mkdir -p "${OUT_DIR}"

for model in "${COMPARISON_MODELS[@]}"; do
  model="$(echo "${model}" | xargs)"
  [[ -z "${model}" ]] && continue
  for intent in "${COMPARISON_INTENTS[@]}"; do
    intent="$(echo "${intent}" | xargs)"
    [[ -z "${intent}" ]] && continue
    if [[ "${intent}" == "push" ]]; then
      epsilon="${EPSILON_PUSH}"
      category="${PUSH_TARGET_CATEGORY}"
    else
      epsilon="${EPSILON_NUKE}"
      category="${TARGET_CATEGORY}"
    fi

    for method in "${METHODS[@]}"; do
      out_sub="${OUT_DIR}/${model}_${intent}_${method}"
      if [[ "${method}" == "matrix_poison" ]]; then
        run_case "${ML1M_INTERACTIONS}" "${ML1M_ITEMS}" "${model}" "${intent}" "${category}" "${epsilon}" "${out_sub}"
      else
        run_case "${ML1M_INTERACTIONS}" "${ML1M_ITEMS}" "${model}" "${intent}" "${category}" "${epsilon}" "${out_sub}" \
          --max-iters 1 \
          --disable-analysis-phase \
          --disable-planning-phase \
          --disable-judgment-phase \
          --disable-self-evolving \
          --fixed-attack-method "${method}"
      fi
    done
  done
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
    name = p.parent.name
    model, intent, method = name.split("_", 2)
    rows.append({
        "model": model,
        "intent": intent,
        "method": method,
        "best_reward": obj.get("best_reward"),
        "prediction_shift": bm.get("prediction_shift"),
        "objective": bm.get("objective"),
        "hit": bm.get("hit"),
        "ndcg": bm.get("ndcg"),
        "output_dir": str(p.parent),
    })
df = pd.DataFrame(rows).sort_values(["model", "intent", "method"])
df.to_csv(base / "summary_5_3_2.csv", index=False)
print(df.to_string(index=False))
print(f"[5.3.2] summary: {base / 'summary_5_3_2.csv'}")
PY
