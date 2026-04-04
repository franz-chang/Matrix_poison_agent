#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
ensure_ml1m_data
init_common_args

CROSS_MODELS_CSV="${CROSS_MODELS_CSV:-lightgcn}"
IFS=',' read -r -a CROSS_MODELS <<< "${CROSS_MODELS_CSV}"
OUT_DIR="${OUT_BASE}/5_4_cross_market"
mkdir -p "${OUT_DIR}"

# Optional real cross-market root:
#   ${XMARKET_ROOT}/US/{interactions.csv,items.csv}
#   ${XMARKET_ROOT}/JP/{interactions.csv,items.csv}
#   ${XMARKET_ROOT}/UK/{interactions.csv,items.csv}
XMARKET_ROOT="${XMARKET_ROOT:-}"
MARKET_ROOT=""
if [[ -n "${XMARKET_ROOT}" ]] \
  && [[ -f "${XMARKET_ROOT}/US/interactions.csv" && -f "${XMARKET_ROOT}/US/items.csv" ]] \
  && [[ -f "${XMARKET_ROOT}/JP/interactions.csv" && -f "${XMARKET_ROOT}/JP/items.csv" ]] \
  && [[ -f "${XMARKET_ROOT}/UK/interactions.csv" && -f "${XMARKET_ROOT}/UK/items.csv" ]]; then
  MARKET_ROOT="${XMARKET_ROOT}"
  echo "[5.4] use provided XMARKET_ROOT=${MARKET_ROOT}"
else
  MARKET_ROOT="${OUT_BASE}/generated_cross_market"
  mkdir -p "${MARKET_ROOT}"
  echo "[5.4] XMARKET_ROOT not complete, generating pseudo markets from ML-1M -> ${MARKET_ROOT}"
  "${PYTHON_BIN}" - <<PY
import pandas as pd
from pathlib import Path
inter = pd.read_csv("${ML1M_INTERACTIONS}")
items = pd.read_csv("${ML1M_ITEMS}")
users = sorted(inter["user_id"].astype(str).unique().tolist())
mk = {"US": set(), "JP": set(), "UK": set()}
for i, u in enumerate(users):
    if i % 3 == 0:
        mk["US"].add(u)
    elif i % 3 == 1:
        mk["JP"].add(u)
    else:
        mk["UK"].add(u)
root = Path("${MARKET_ROOT}")
for name, uids in mk.items():
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    sub = inter[inter["user_id"].astype(str).isin(uids)].copy()
    sub.to_csv(d / "interactions.csv", index=False)
    items.to_csv(d / "items.csv", index=False)
print("[5.4] pseudo markets generated")
PY
fi

for market in US JP UK; do
  inter_path="${MARKET_ROOT}/${market}/interactions.csv"
  items_path="${MARKET_ROOT}/${market}/items.csv"
  [[ ! -f "${inter_path}" || ! -f "${items_path}" ]] && continue

  for model in "${CROSS_MODELS[@]}"; do
    model="$(echo "${model}" | xargs)"
    [[ -z "${model}" ]] && continue
    run_case "${inter_path}" "${items_path}" "${model}" "nuke" "${TARGET_CATEGORY}" "${EPSILON_NUKE}" \
      "${OUT_DIR}/${market}_${model}_nuke"
    run_case "${inter_path}" "${items_path}" "${model}" "push" "${PUSH_TARGET_CATEGORY}" "${EPSILON_PUSH}" \
      "${OUT_DIR}/${market}_${model}_push"
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
    market, model, intent = name.split("_", 2)
    rows.append({
        "market": market,
        "model": model,
        "intent": intent,
        "best_reward": obj.get("best_reward"),
        "prediction_shift": bm.get("prediction_shift"),
        "objective": bm.get("objective"),
        "hit": bm.get("hit"),
        "ndcg": bm.get("ndcg"),
        "output_dir": str(p.parent),
    })
df = pd.DataFrame(rows).sort_values(["market", "model", "intent"])
df.to_csv(base / "summary_5_4.csv", index=False)
print(df.to_string(index=False))
print(f"[5.4] summary: {base / 'summary_5_4.csv'}")
PY
