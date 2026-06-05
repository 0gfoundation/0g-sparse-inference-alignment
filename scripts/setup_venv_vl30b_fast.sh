#!/usr/bin/env bash
# Create a venv for Qwen3-VL-30B-A3B SIA, b2 inproc fast path (vllm 0.17.1).
#
# This is the validated speedup configuration — see
# doc/vl30b-b2-inproc-speedup-20260605.md.
#
# Usage:
#   scripts/setup_venv_vl30b_fast.sh /path/to/venv-vl30b-fast
set -euo pipefail

VENV_PATH="${1:-}"
if [[ -z "$VENV_PATH" ]]; then
  echo "Usage: $0 /path/to/venv-vl30b-fast" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[setup] venv → $VENV_PATH"
python3 -m venv "$VENV_PATH"
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

echo "[setup] upgrading pip"
pip install --upgrade pip wheel

echo "[setup] installing requirements/vl30b-b2-inproc.txt (vllm 0.17.1)"
pip install -r "$REPO_ROOT/requirements/vl30b-b2-inproc.txt"

echo "[setup] installing project (pip install -e .)"
pip install -e "$REPO_ROOT"

echo "[setup] done. activate with: source $VENV_PATH/bin/activate"
echo "[setup] launch example:"
cat <<'EOF'
  SIA_LLM_CUDAGRAPH=piecewise SIA_RM_CUDAGRAPH=piecewise \
  SIA_RM_MULTIPROCESS=0 \
  python src/sia_vllm_server.py \
    --llm /path/to/Qwen3-VL-30B-A3B-Instruct \
    --rm_backend b2 \
    --rm_model /path/to/VM-Qwen3-4B-merged-for-vllm \
    --rm_b2_gpu_mem 0.15 --llm_gpu_mem 0.55 \
    --topk 10 --weight 1.0 --entropy_threshold 1.0 \
    --max_model_len 4096 --port 8000
EOF
