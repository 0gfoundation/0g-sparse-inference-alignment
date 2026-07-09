#!/bin/bash
# 0GM-35B SIA with vocab_lowrank head VM and token mapping table
# Usage: MODEL_DIR=/path/to/models bash scripts/start_0gm35b_vocab_lowrank.sh
set -e

MODEL_DIR=${MODEL_DIR:-/workspace/sia-repo/models}
REPO_DIR=$(dirname "$(realpath "$0")")/..

SIA_RM_CUDAGRAPH=none \
SIA_RM_MULTIPROCESS=0 \
python "$REPO_DIR/src/sia_vllm_server.py" \
  --llm "$MODEL_DIR/0GM-1.0-35B-A3B" \
  --rm_backend b2 \
  --rm_model "$MODEL_DIR/VM-Qwen3-4B-vocab-lowrank-merged" \
  --rm_b2_gpu_mem 0.13 --llm_gpu_mem 0.75 \
  --topk 10 --weight 1.0 --entropy_threshold 1.0 \
  --max_model_len 2048 --mamba_cache_mode align --port 8000 \
  --vm_head_type vocab_lowrank \
  --mapping_table "$REPO_DIR/token_mapping_ogm35b_to_qwen3_4b.npy"
