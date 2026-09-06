#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXPS_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-${EXPS_DIR}/.venv}"
READY_FILE="${VENV_DIR}/.slearn_experiment_deps_ready"

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  if ! "${VENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
    echo "Existing ${VENV_DIR} has a broken pip; rebuilding it."
    rm -rf "${VENV_DIR}"
  fi
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  rm -rf "${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${REPO_ROOT}"
python -m pip install -r "${REPO_ROOT}/requirements-experiments.txt"

if [[ "${INSTALL_RWKV_TRAINER:-0}" == "1" ]]; then
  python -m pip install rwkv-trainer
fi

python - <<'PYCHECK'
import torch
from minGRU_pytorch import minGRU
from linear_attention_transformer import LinearAttentionTransformer
from performer_pytorch import Performer

print(f"Experiment environment ready. torch={torch.__version__}")
print("Verified: minGRU_pytorch, linear_attention_transformer, performer_pytorch")
PYCHECK

date -u +%Y-%m-%dT%H:%M:%SZ > "${READY_FILE}"
