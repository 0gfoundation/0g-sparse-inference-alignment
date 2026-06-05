#!/usr/bin/env bash
# Create a venv for 0GM-1.0-35B-A3B SIA (vllm 0.19.0, HTTP RM only).
# Also works for Qwen3-VL-30B HTTP RM fallback.
#
# Usage:
#   scripts/setup_venv_0gm35b_http.sh /path/to/venv-0gm35b
set -euo pipefail

VENV_PATH="${1:-}"
if [[ -z "$VENV_PATH" ]]; then
  echo "Usage: $0 /path/to/venv-0gm35b" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[setup] venv → $VENV_PATH"
python3 -m venv "$VENV_PATH"
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

echo "[setup] upgrading pip"
pip install --upgrade pip wheel

echo "[setup] installing requirements/0gm35b-or-vl30b-http.txt (vllm 0.19.0)"
pip install -r "$REPO_ROOT/requirements/0gm35b-or-vl30b-http.txt"

echo "[setup] installing project (pip install -e .)"
pip install -e "$REPO_ROOT"

echo "[setup] done. activate with: source $VENV_PATH/bin/activate"
echo "[setup] launch example (two processes):"
cat <<'EOF'
  # Terminal A — RM
  vllm serve /path/to/VM-Qwen3-4B-merged-for-vllm \
    --runner pooling --convert classify \
    --enable-prefix-caching --gpu-memory-utilization 0.3 --port 8001

  # Terminal B — main LLM
  python src/sia_vllm_server.py --llm /path/to/0GM-1.0-35B-A3B \
    --rm_url http://localhost:8001 --rm_backend vllm \
    --rm_model /path/to/VM-Qwen3-4B-merged-for-vllm \
    --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --no_think_prompt --ban_think_token --disable_thinking \
    --top_p 0.95 --top_k 20 --port 8000
EOF
