#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_PATH="${VENV_PATH:-.venv}"
PORT="${PORT:-8000}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
	echo "Python interpreter '$PYTHON_BIN' not found. Install it (e.g. via pyenv) or set PYTHON_BIN=/path/to/python." >&2
	exit 1
fi

if [ ! -d "$VENV_PATH" ]; then
	echo "Creating virtual env in $VENV_PATH using $PYTHON_BIN"
	"$PYTHON_BIN" -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"

echo "Installing dependencies (requirements.txt)…"
pip install --upgrade pip >/dev/null
pip install -r requirements.txt >/dev/null

export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "Model path: artifacts/efficientnet_b0_best.pt"
echo "Starting Uvicorn on 0.0.0.0:${PORT}"

exec uvicorn api.main:app --host 0.0.0.0 --port "$PORT" --reload
