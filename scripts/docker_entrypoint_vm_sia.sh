#!/usr/bin/env bash
# Entrypoint: starts two SIA servers in one container.
#
# Both servers use Qwen3-4B-Base as the main LLM (no 30B model).
# Each uses a different VM reward head:
#   Port 8001 — scalar head         (VM-Qwen3-4B-merged-for-vllm)
#   Port 8002 — vocab_lowrank head  (VM-Qwen3-4B-vocab-lowrank-bt-20260704-merged)
#
# Sequential startup: scalar starts first; we wait for its /health before
# starting vocab — avoids GPU memory races during CUDA graph compilation.
#
# Signal handling: bash stays as PID 1 (no exec), trapping SIGTERM/SIGINT and
# forwarding to both server processes.

set -euo pipefail

REPO=/workspace/sia-repo/0g-sparse-inference-alignment
VENV_PIP=/opt/venv-vl30b/bin/pip
VENV_PYTHON=/opt/venv-vl30b/bin/python

# ── Register sia_rm vllm plugin ──────────────────────────────────────────────
if [[ -f "$REPO/pyproject.toml" ]]; then
  echo "[entrypoint] registering sia_rm vllm plugin (pip install -e $REPO)"
  "$VENV_PIP" install --quiet --no-deps -e "$REPO"
else
  echo "[entrypoint] WARNING: pyproject.toml not found at $REPO" >&2
  echo "[entrypoint] b2 backend will not work" >&2
fi

LLM=/workspace/sia-repo/models/Qwen3-4B-Base
VM_SCALAR=/workspace/models/VM-Qwen3-4B-merged-for-vllm
VM_VOCAB=/workspace/sia-repo/models/VM-Qwen3-4B-vocab-lowrank-frozen-20260706-merged

# 2 × (LLM 0.15 + RM 0.15) × 80 GB H200 = 48 GB total — safe margin.
# RM_GPU_MEM=0.15: gives RM b2 ~4 GiB KV cache (was 1 GiB at 0.08),
# preventing silent RM failures on long sequences.
# MAX_MODEL_LEN stays at 2048 — 4096 was confirmed to suppress intervention rate.
LLM_GPU_MEM=0.15
RM_GPU_MEM=0.15
MAX_MODEL_LEN=2048

# ── Server 1: scalar head ─────────────────────────────────────────────────────
echo "[entrypoint] Starting scalar-head SIA server on port 8001 ..."

SIA_RM_CUDAGRAPH=piecewise \
SIA_RM_MULTIPROCESS=0 \
SIA_DEBUG_HIST=1 \
SIA_LOG_LEVEL=detail \
SIA_DETAIL_STEPS=10 \
"$VENV_PYTHON" "$REPO/src/sia_vllm_server.py" \
  --llm "$LLM" \
  --model_id "Qwen3-4B-SIA-scalar" \
  --rm_backend b2 \
  --rm_model "$VM_SCALAR" \
  --llm_gpu_mem $LLM_GPU_MEM \
  --rm_b2_gpu_mem $RM_GPU_MEM \
  --topk 10 --weight 1.0 --entropy_threshold 1.0 \
  --max_model_len $MAX_MODEL_LEN \
  --enable_prefix_caching \
  --host 0.0.0.0 \
  --port 8001 \
  > /var/log/vm_sia_scalar.log 2>&1 &
PID_SCALAR=$!
echo "[entrypoint] scalar server PID=$PID_SCALAR — waiting for /health (up to 10 min) ..."

# Wait up to 600s for server 1 to be ready before starting server 2.
waited=0
while [ $waited -lt 600 ]; do
  sleep 5; waited=$((waited + 5))
  if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
    echo "[entrypoint] scalar server ready after ${waited}s"
    break
  fi
  if ! kill -0 $PID_SCALAR 2>/dev/null; then
    echo "[entrypoint] ERROR: scalar server process died. Last 40 lines:" >&2
    tail -40 /var/log/vm_sia_scalar.log >&2
    exit 1
  fi
done

if ! curl -sf http://localhost:8001/health > /dev/null 2>&1; then
  echo "[entrypoint] ERROR: scalar server did not come up in ${waited}s" >&2
  tail -40 /var/log/vm_sia_scalar.log >&2
  exit 1
fi

# ── Server 2: vocab_lowrank head ──────────────────────────────────────────────
echo "[entrypoint] Starting vocab_lowrank SIA server on port 8002 ..."

SIA_RM_CUDAGRAPH=piecewise \
SIA_RM_MULTIPROCESS=0 \
SIA_DEBUG_HIST=1 \
SIA_LOG_LEVEL=detail \
SIA_DETAIL_STEPS=10 \
"$VENV_PYTHON" "$REPO/src/sia_vllm_server.py" \
  --llm "$LLM" \
  --model_id "Qwen3-4B-SIA-vocab" \
  --rm_backend b2 \
  --rm_model "$VM_VOCAB" \
  --vm_head_type vocab_lowrank \
  --vm_head_rank 64 \
  --llm_gpu_mem $LLM_GPU_MEM \
  --rm_b2_gpu_mem $RM_GPU_MEM \
  --topk 10 --weight 1.0 --entropy_threshold 1.0 \
  --max_model_len $MAX_MODEL_LEN \
  --enable_prefix_caching \
  --host 0.0.0.0 \
  --port 8002 \
  > /var/log/vm_sia_vocab.log 2>&1 &
PID_VOCAB=$!
echo "[entrypoint] vocab server PID=$PID_VOCAB"

# ── Signal handler — bash stays as PID 1 and forwards to both children ────────
cleanup() {
  echo "[entrypoint] shutting down both servers ..."
  kill "$PID_SCALAR" "$PID_VOCAB" 2>/dev/null || true
  wait "$PID_SCALAR" "$PID_VOCAB" 2>/dev/null || true
  echo "[entrypoint] done"
}
trap cleanup SIGTERM SIGINT

echo "[entrypoint] both servers running. scalar=$PID_SCALAR vocab=$PID_VOCAB"
echo "[entrypoint] logs: /var/log/vm_sia_scalar.log  /var/log/vm_sia_vocab.log"

# Monitor: exit (and let Docker restart the container) if either server crashes.
while sleep 15; do
  if ! kill -0 "$PID_SCALAR" 2>/dev/null; then
    echo "[entrypoint] ERROR: scalar server ($PID_SCALAR) exited" >&2
    tail -20 /var/log/vm_sia_scalar.log >&2
    kill "$PID_VOCAB" 2>/dev/null || true
    exit 1
  fi
  if ! kill -0 "$PID_VOCAB" 2>/dev/null; then
    echo "[entrypoint] ERROR: vocab server ($PID_VOCAB) exited" >&2
    tail -20 /var/log/vm_sia_vocab.log >&2
    kill "$PID_SCALAR" 2>/dev/null || true
    exit 1
  fi
done
