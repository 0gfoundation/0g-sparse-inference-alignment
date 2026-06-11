#!/usr/bin/env bash
# Docker-friendly install for 0GM-35B SIA (b2 inproc fast path, vllm 0.18.0).
#
# Tested on base image: pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel
#
# What this script does:
#   1. apt-get install git + minimal system deps (skipped if already present)
#   2. Create a fresh Python venv (isolated from the docker's conda env's
#      pre-installed torch 2.11 — vllm 0.18.0 pulls its own torch 2.10.0)
#   3. pip install requirements/0gm35b-b2-inproc.txt + the editable project
#   4. Verify vllm + torch + CUDA all import + see GPU
#
# Usage (run inside the docker container, with this repo cloned/mounted):
#   bash scripts/docker_install_0gm35b.sh                # default venv path
#   bash scripts/docker_install_0gm35b.sh /opt/venv-0gm35b
#
# Default venv path: /opt/venv-0gm35b
# Repo path: auto-detected from this script's location.

set -euo pipefail

VENV_PATH="${1:-/opt/venv-0gm35b}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "================================================================"
echo "[docker-install] 0GM-35B SIA (b2 inproc, vllm 0.18.0) setup"
echo "[docker-install] repo : ${REPO_ROOT}"
echo "[docker-install] venv : ${VENV_PATH}"
echo "================================================================"

# ---------- Step 1: apt deps ----------------------------------------------
need_pkgs=()
for pkg in git ca-certificates; do
  dpkg -s "$pkg" >/dev/null 2>&1 || need_pkgs+=("$pkg")
done

if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  py_minor=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
  need_pkgs+=("python${py_minor}-venv")
  echo "[docker-install] python3 (=${py_minor}) lacks ensurepip; will install python${py_minor}-venv"
fi

if [[ ${#need_pkgs[@]} -gt 0 ]]; then
  echo "[docker-install] installing system deps via apt-get: ${need_pkgs[*]}"
  apt-get update -qq
  apt-get install -y --no-install-recommends "${need_pkgs[@]}" curl
  rm -rf /var/lib/apt/lists/*
else
  echo "[docker-install] system deps already present, skipping apt-get"
fi

# ---------- Step 2: create venv -------------------------------------------
echo "[docker-install] creating venv at ${VENV_PATH}"
python3 -m venv "${VENV_PATH}"

# ---------- Step 3: install requirements ----------------------------------
echo "[docker-install] upgrading pip + wheel"
"${VENV_PATH}/bin/pip" install --upgrade --no-cache-dir pip wheel

echo "[docker-install] installing requirements/0gm35b-b2-inproc.txt (vllm 0.18.0)"
"${VENV_PATH}/bin/pip" install --no-cache-dir -r "${REPO_ROOT}/requirements/0gm35b-b2-inproc.txt"

echo "[docker-install] installing project (pip install -e .)"
"${VENV_PATH}/bin/pip" install --no-cache-dir -e "${REPO_ROOT}"

# ---------- Step 4: verify -----------------------------------------------
echo
echo "[docker-install] verifying install..."
"${VENV_PATH}/bin/python" - <<'PYEOF'
import sys
print(f"python: {sys.version.split()[0]}  ({sys.executable})")

import vllm
print(f"vllm  : {vllm.__version__}")
assert vllm.__version__.startswith("0.18."), f"expected vllm 0.18.x, got {vllm.__version__}"

import torch
print(f"torch : {torch.__version__}")
print(f"CUDA  : available={torch.cuda.is_available()}  device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU 0 : {torch.cuda.get_device_name(0)}")
    print(f"        bf16 = {torch.cuda.is_bf16_supported()}")

# Verify vllm has the unlock_workspace API (required for MoE main LLM)
from vllm.v1.worker.workspace import unlock_workspace  # noqa: F401
print("unlock_workspace API: ✓ available")

# Verify Qwen3_5MoeForConditionalGeneration is in the model registry
from vllm.model_executor.models.registry import ModelRegistry
mods = ModelRegistry.get_supported_archs()
assert any("Qwen3_5Moe" in m for m in mods), \
    f"Qwen3_5MoeForConditionalGeneration not registered: {mods[:5]}..."
print("Qwen3_5MoE arch     : ✓ registered")

# Verify the sia_rm plugin is discoverable
from importlib.metadata import entry_points
plugins = [ep for ep in entry_points(group="vllm.general_plugins") if "sia" in ep.name.lower()]
print(f"sia_rm plugin       : {'✓' if plugins else '✗'} ({[p.name for p in plugins]})")
PYEOF

echo
echo "================================================================"
echo "[docker-install] DONE — install verified"
echo "================================================================"
echo
echo "Next steps:"
echo "  1. Activate venv:  source ${VENV_PATH}/bin/activate"
echo "  2. Make sure model weights are available:"
echo "     - <YOUR_MODELS_DIR>/0GM-1.0-35B-A3B-0427                  (main LLM, ~70 GB)"
echo "     - <YOUR_MODELS_DIR>/VM-Qwen3-4B-merged-for-vllm      (RM, ~11 GB)"
echo "  3. Launch SIA server (requires H200/141GB+ GPU):"
cat <<'LAUNCH'
        SIA_RM_CUDAGRAPH=none \
        SIA_RM_MULTIPROCESS=0 \
        python src/sia_vllm_server.py \
          --llm <YOUR_MODELS_DIR>/0GM-1.0-35B-A3B-0427 \
          --rm_backend b2 \
          --rm_model <YOUR_MODELS_DIR>/VM-Qwen3-4B-merged-for-vllm \
          --rm_b2_gpu_mem 0.15 --llm_gpu_mem 0.55 \
          --topk 10 --weight 1.0 --entropy_threshold 1.0 \
          --max_model_len 2048 --port 8000
LAUNCH
echo "  4. Smoke test (once 'Uvicorn running on http://0.0.0.0:8000' appears):"
echo "     curl -s -X POST http://localhost:8000/v1/chat/completions \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"model\":\"<YOUR_MODELS_DIR>/0GM-1.0-35B-A3B-0427\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":50}'"
