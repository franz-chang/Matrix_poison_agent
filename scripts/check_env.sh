#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

echo "[check_env] ROOT_DIR=${ROOT_DIR}"
echo "[check_env] PYTHON_BIN=${PYTHON_BIN}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[check_env] ERROR: Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${ROOT_DIR}"

"${PYTHON_BIN}" --version
"${PYTHON_BIN}" -m pip --version

"${PYTHON_BIN}" - <<'PY'
import importlib
mods = ["numpy", "pandas", "torch", "tqdm"]
for m in mods:
    mod = importlib.import_module(m)
    print(f"[check_env] {m}={getattr(mod, '__version__', 'unknown')}")
import torch
print(f"[check_env] cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[check_env] gpu_count={torch.cuda.device_count()}")
    print(f"[check_env] gpu_name={torch.cuda.get_device_name(0)}")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "[check_env] nvidia-smi not found (CPU mode may still work)."
fi

echo "[check_env] OK"
