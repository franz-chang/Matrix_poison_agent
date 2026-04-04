#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
ensure_ml1m_data
init_common_args

ABLATION_MODEL="${ABLATION_MODEL:-lightgcn}"
OUT_DIR="${OUT_BASE}/5_6_ablation"
mkdir -p "${OUT_DIR}"

# Full
run_case "${ML1M_INTERACTIONS}" "${ML1M_ITEMS}" "${ABLATION_MODEL}" "nuke" "${TARGET_CATEGORY}" "${EPSILON_NUKE}" \
  "${OUT_DIR}/${ABLATION_MODEL}_full"

# w/o Analysis
run_case "${ML1M_INTERACTIONS}" "${ML1M_ITEMS}" "${ABLATION_MODEL}" "nuke" "${TARGET_CATEGORY}" "${EPSILON_NUKE}" \
  "${OUT_DIR}/${ABLATION_MODEL}_wo_analysis" \
  --disable-analysis-phase

# w/o Planning
run_case "${ML1M_INTERACTIONS}" "${ML1M_ITEMS}" "${ABLATION_MODEL}" "nuke" "${TARGET_CATEGORY}" "${EPSILON_NUKE}" \
  "${OUT_DIR}/${ABLATION_MODEL}_wo_planning" \
  --disable-planning-phase

# w/o Judgment
run_case "${ML1M_INTERACTIONS}" "${ML1M_ITEMS}" "${ABLATION_MODEL}" "nuke" "${TARGET_CATEGORY}" "${EPSILON_NUKE}" \
  "${OUT_DIR}/${ABLATION_MODEL}_wo_judgment" \
  --disable-judgment-phase

# w/o Self-evolving
run_case "${ML1M_INTERACTIONS}" "${ML1M_ITEMS}" "${ABLATION_MODEL}" "nuke" "${TARGET_CATEGORY}" "${EPSILON_NUKE}" \
  "${OUT_DIR}/${ABLATION_MODEL}_wo_self_evolving" \
  --disable-self-evolving

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
import pandas as pd
base = Path("${OUT_DIR}")
rows = []
for p in sorted(base.glob("*/run_summary.json")):
    obj = json.loads(p.read_text(encoding="utf-8"))
    bm = obj.get("best_metrics", {})
    variant = p.parent.name.split("_", 1)[1]
    rows.append({
        "model": "${ABLATION_MODEL}",
        "variant": variant,
        "best_reward": obj.get("best_reward"),
        "objective": bm.get("objective"),
        "prediction_shift": bm.get("prediction_shift"),
        "hit": bm.get("hit"),
        "ndcg": bm.get("ndcg"),
        "output_dir": str(p.parent),
    })
df = pd.DataFrame(rows).sort_values("variant")
df.to_csv(base / "summary_5_6.csv", index=False)
print(df.to_string(index=False))
print(f"[5.6] summary: {base / 'summary_5_6.csv'}")
PY
