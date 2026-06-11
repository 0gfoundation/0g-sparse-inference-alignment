#!/bin/bash
set -e

EXP_DIR=/workspace/git/0g-sparse-inference-alignment/exp/alpaca-0gm35b-b2-inproc-20260609
REPO=/workspace/git/0g-sparse-inference-alignment
VENV=/workspace/SIA/venv6

SIA_SERVER_PID=841203
SIA_EVAL_PID=843709

echo "[$(date)] 等待 SIA eval (PID=$SIA_EVAL_PID) 完成..."
while kill -0 $SIA_EVAL_PID 2>/dev/null; do
    sleep 30
done
echo "[$(date)] SIA eval 已完成"

echo "[$(date)] 停止 SIA server (PID=$SIA_SERVER_PID)..."
kill $SIA_SERVER_PID 2>/dev/null || true
sleep 5
kill -9 $SIA_SERVER_PID 2>/dev/null || true

# 等待 GPU 释放
sleep 15

echo "[$(date)] 启动 noSIA FULL fix server..."
source $VENV/bin/activate
cd $REPO

SIA_RM_CUDAGRAPH=none SIA_RM_MULTIPROCESS=0 \
python src/sia_vllm_server.py \
  --llm /workspace/SIA/models/0GM-1.0-35B-A3B-0427 \
  --rm_backend b2 \
  --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
  --llm_gpu_mem 0.80 --rm_b2_gpu_mem 0.15 \
  --topk 10 --weight 0 --entropy_threshold 9999 \
  --max_model_len 4096 --port 8000 \
  > $EXP_DIR/b2_08_nosia_fixed_server.log 2>&1 &

NOSIA_SERVER_PID=$!
echo "[$(date)] noSIA server PID=$NOSIA_SERVER_PID，等待就绪..."

# 等待 server 就绪（最多3分钟）
for i in $(seq 1 36); do
    sleep 5
    if curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "[$(date)] Server 就绪 (${i}×5s)"
        break
    fi
    if ! kill -0 $NOSIA_SERVER_PID 2>/dev/null; then
        echo "[$(date)] ERROR: server 进程已退出，检查 b2_08_nosia_fixed_server.log"
        exit 1
    fi
done

echo "[$(date)] 启动 noSIA eval..."
python eval/alpaca_eval.py \
  --base_url http://localhost:8000/v1 \
  --model 0GM-1.0-35B-A3B-0427 \
  --dataset data/alpaca_eval/alpaca_eval.json \
  --output $EXP_DIR/alpaca_0gm35b_b2_08_nosia_fixed.json \
  --limit 200 \
  --max_tokens 2048 \
  --temperature 1.0 \
  --top_p 0.95 \
  --top_k 20 \
  --repetition_penalty 1.0 \
  2>&1 | tee $EXP_DIR/alpaca_0gm35b_b2_08_nosia_fixed_gen.log

echo "[$(date)] noSIA eval 完成"
echo "[$(date)] 停止 noSIA server (PID=$NOSIA_SERVER_PID)..."
kill $NOSIA_SERVER_PID 2>/dev/null || true
