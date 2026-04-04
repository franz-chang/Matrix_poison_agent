#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

OUT_FILE="${OUT_BASE}/5_5_stealthiness.csv"
OUT_MD="${OUT_BASE}/5_5_stealthiness.md"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
import pandas as pd

base = Path("${OUT_BASE}")
rows = []
for p in sorted(base.glob("**/run_summary.json")):
    rel = str(p.relative_to(base))
    # Keep matrix-poison dominant runs for stealth analysis.
    # In baseline comparison, only *_matrix_poison should be included.
    if "5_3_2_baseline_comparison" in rel and not rel.endswith("_matrix_poison/run_summary.json"):
        continue
    obj = json.loads(p.read_text(encoding="utf-8"))
    bm = obj.get("best_metrics", {}) or {}
    clean = obj.get("clean_metrics", {}) or {}
    hit = float(bm.get("hit", 0.0))
    ndcg = float(bm.get("ndcg", 0.0))
    clean_hit = float(clean.get("hit", 0.0))
    clean_ndcg = float(clean.get("ndcg", 0.0))
    utility_drop = ((max(0.0, clean_hit - hit) + max(0.0, clean_ndcg - ndcg)) / 2.0)
    rows.append({
        "run": rel,
        "hit": hit,
        "ndcg": ndcg,
        "clean_hit": clean_hit,
        "clean_ndcg": clean_ndcg,
        "utility_drop": utility_drop,
        "rating_kl": float(bm.get("rating_kl", 0.0)),
        "profile_shift": float(bm.get("profile_shift", 0.0)),
        "interaction_ratio": float(bm.get("interaction_ratio", 1.0)),
        "stealth_penalty": float(bm.get("stealth_penalty", 0.0)),
    })

df = pd.DataFrame(rows).sort_values("run")
df.to_csv("${OUT_FILE}", index=False)

lines = []
lines.append("# 5.5 Stealthiness Evaluation")
lines.append("")
if df.empty:
    lines.append("_No run_summary.json found under OUT_BASE._")
else:
    cols = list(df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
Path("${OUT_MD}").write_text("\\n".join(lines), encoding="utf-8")
print(df.to_string(index=False))
print(f"[5.5] csv: ${OUT_FILE}")
print(f"[5.5] md : ${OUT_MD}")
PY
