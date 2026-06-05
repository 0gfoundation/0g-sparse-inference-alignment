#!/usr/bin/env bash
# Create a venv for the Qwen3-14B SIA configuration (vllm 0.10.1.1, b2 inproc).
#
# Usage:
#   scripts/setup_venv_qwen14b.sh /path/to/venv-qwen14b
#
# Will:
#   1. python3 -m venv <path>
#   2. activate, upgrade pip
#   3. pip install -r requirements/qwen14b-b2-inproc.txt
#   4. pip install -e . (registers sia_rm vllm plugin via pyproject.toml)
set -euo pipefail

VENV_PATH="${1:-}"
if [[ -z "$VENV_PATH" ]]; then
  echo "Usage: $0 /path/to/venv-qwen14b" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[setup] venv → $VENV_PATH"
python3 -m venv "$VENV_PATH"
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

echo "[setup] upgrading pip"
pip install --upgrade pip wheel

echo "[setup] installing requirements/qwen14b-b2-inproc.txt"
pip install -r "$REPO_ROOT/requirements/qwen14b-b2-inproc.txt"

echo "[setup] installing project (pip install -e .)"
pip install -e "$REPO_ROOT"

echo "[setup] done. activate with: source $VENV_PATH/bin/activate"
