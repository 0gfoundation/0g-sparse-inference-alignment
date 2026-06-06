#!/usr/bin/env bash
# Docker-friendly install for VL-30B SIA (b2 inproc fast path, vllm 0.17.1).
#
# Tested on base image: pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel
#
# What this script does:
#   1. apt-get install git + minimal system deps (skipped if already present)
#   2. Create a fresh Python venv (isolated from the docker's conda env's
#      pre-installed torch 2.11 — vllm 0.17.1 pulls its own torch 2.10.0)
#   3. pip install requirements/vl30b-b2-inproc.txt + the editable project
#   4. Verify vllm + torch + CUDA all import + see GPU
#
# Usage (run inside the docker container, with this repo cloned/mounted):
#   bash scripts/docker_install_vl30b.sh                # default venv path
#   bash scripts/docker_install_vl30b.sh /opt/venv-vl30b
#
# Default venv path: /opt/venv-vl30b
# Repo path: auto-detected from this script's location.

set -euo pipefail

VENV_PATH="${1:-/opt/venv-vl30b}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "================================================================"
echo "[docker-install] VL-30B SIA (b2 inproc, vllm 0.17.1) setup"
echo "[docker-install] repo : ${REPO_ROOT}"
echo "[docker-install] venv : ${VENV_PATH}"
echo "================================================================"

# ---------- Step 1: apt deps ----------------------------------------------
# pytorch/pytorch:*-devel ships system Python from apt (typically 3.12 on
# Ubuntu 24.04 base) but WITHOUT the matching python3-venv package, so
# `python3 -m venv` fails with "ensurepip is not available". Detect that
# and add the right venv package to the apt install list.
need_pkgs=()
for pkg in git ca-certificates; do
  dpkg -s "$pkg" >/dev/null 2>&1 || need_pkgs+=("$pkg")
done

# Check if python3 -m venv works (don't actually create anything; just probe
# whether the venv module's ensurepip support is wired up).
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

# ---------- Step 2-3: create venv + install requirements ------------------
# Delegates to scripts/setup_venv_vl30b_fast.sh which:
#   - python3 -m venv $VENV_PATH
#   - pip install --upgrade pip wheel
#   - pip install -r requirements/vl30b-b2-inproc.txt  (vllm 0.17.1 + base deps)
#   - pip install -e .  (register the sia_rm vllm plugin)
bash "${REPO_ROOT}/scripts/setup_venv_vl30b_fast.sh" "$VENV_PATH"

# ---------- Step 4: verify -----------------------------------------------
echo
echo "[docker-install] verifying install..."
"${VENV_PATH}/bin/python" - <<'PYEOF'
import sys
print(f"python: {sys.version.split()[0]}  ({sys.executable})")

import vllm
print(f"vllm  : {vllm.__version__}")
assert vllm.__version__.startswith("0.17."), f"expected vllm 0.17.x, got {vllm.__version__}"

import torch
print(f"torch : {torch.__version__}")
print(f"CUDA  : available={torch.cuda.is_available()}  device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU 0 : {torch.cuda.get_device_name(0)}")
    print(f"        bf16 = {torch.cuda.is_bf16_supported()}")

# Verify vllm has the unlock_workspace API (required for MoE main LLM)
from vllm.v1.worker.workspace import unlock_workspace  # noqa: F401
print("unlock_workspace API: ✓ available")

# Verify Qwen3VLMoe is in the model registry
from vllm.model_executor.models.registry import ModelRegistry
mods = ModelRegistry.get_supported_archs()
assert any("Qwen3VLMoe" in m for m in mods), f"Qwen3VLMoe arch not registered: {mods[:5]}..."
print("Qwen3VLMoe arch     : ✓ registered")

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
echo "     - <YOUR_MODELS_DIR>/Qwen3-VL-30B-A3B-Instruct        (main LLM, ~60 GB)"
echo "     - <YOUR_MODELS_DIR>/VM-Qwen3-4B-merged-for-vllm      (RM, ~11 GB)"
echo "  3. Launch SIA server (uses ~80 GB GPU mem on H100/H200):"
cat <<'LAUNCH'
        SIA_LLM_CUDAGRAPH=piecewise \
        SIA_RM_CUDAGRAPH=piecewise \
        SIA_RM_MULTIPROCESS=0 \
        python src/sia_vllm_server.py \
          --llm <YOUR_MODELS_DIR>/Qwen3-VL-30B-A3B-Instruct \
          --rm_backend b2 \
          --rm_model <YOUR_MODELS_DIR>/VM-Qwen3-4B-merged-for-vllm \
          --rm_b2_gpu_mem 0.15 --llm_gpu_mem 0.55 \
          --topk 10 --weight 1.0 --entropy_threshold 1.0 \
          --max_model_len 4096 --port 8000
LAUNCH
echo "  4. Smoke test (once 'Uvicorn running on http://0.0.0.0:8000' appears):"
echo "     curl -s -X POST http://localhost:8000/v1/chat/completions \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"model\":\"<YOUR_MODELS_DIR>/Qwen3-VL-30B-A3B-Instruct\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":50}'"
