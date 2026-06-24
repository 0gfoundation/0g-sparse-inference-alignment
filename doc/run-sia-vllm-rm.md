# Command Reference for Running SIA + vLLM RM Evaluation

A complete set of ready-to-use commands covering the vLLM RM + SIA LLM server + MMLU evaluation, three processes in total.

## 0. One-time preparation: convert RM to vLLM-compatible checkpoint

Only needs to be done once. Skip this section if already converted.

```bash
python scripts/convert_rm_for_vllm.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --output /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm
```

Output directory is approximately 7.6 GB, standard HF `Qwen3ForSequenceClassification` checkpoint.

---

## 1. Fully automated startup (recommended)

Cleanup → start vLLM RM → wait for ready → start LLM server → wait for ready → run evaluation, all in one go.

```bash
# === Kill any lingering processes ===
pkill -f "sia_vllm_server"; pkill -f "sia_rm_server"; pkill -f "mmlu_eval"
pkill -f "vllm serve"; sleep 3

TS=$(date +%Y%m%d%H%M)

# === 1) vLLM RM server (port 8001) ===
nohup vllm serve /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --runner pooling \
    --convert classify \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.3 \
    --max-model-len 4096 \
    --port 8001 \
    > log_vllm_rm_${TS}.txt 2>&1 &

echo "Waiting for vLLM RM to start (~60s for CUDA graph capture)..."
until curl -s --max-time 2 http://localhost:8001/v1/models > /dev/null 2>&1; do
    sleep 5
done
echo "vLLM RM ready ✓"

# === 2) SIA LLM server (port 8000) ===
nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --rm_backend vllm \
    --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --llm_gpu_mem 0.6 \
    --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > log_llm_vllmrm_${TS}.txt 2>&1 &

echo "Waiting for LLM server to start..."
until curl -s --max-time 2 http://localhost:8000/v1/models > /dev/null 2>&1; do
    sleep 5
done
echo "LLM server ready ✓"

# === 3) Run MMLU evaluation ===
nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 \
    --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_vllmrm_${TS}.json \
    --limit 20 \
    > log_SIA_vllmrm_${TS}.txt 2>&1 &

echo ""
echo "================================="
echo "All started, TS=${TS}"
echo "  RM log:    log_vllm_rm_${TS}.txt"
echo "  LLM log:   log_llm_vllmrm_${TS}.txt"
echo "  Eval log:  log_SIA_vllmrm_${TS}.txt"
echo "  Output:    results/test_vllmrm_${TS}.json"
echo "================================="
```

---

## 2. Step-by-step startup (for manual control)

### 2.1 Start vLLM RM server

```bash
TS=$(date +%Y%m%d%H%M)
nohup vllm serve /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --runner pooling \
    --convert classify \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.3 \
    --max-model-len 4096 \
    --port 8001 \
    > log_vllm_rm_${TS}.txt 2>&1 &
```

Ready indicator: `Application startup complete` appears in the log and CUDA graph capture is finished.

Verify:
```bash
curl -s http://localhost:8001/v1/models | python -m json.tool
```

### 2.2 Start SIA LLM server (vLLM RM backend)

```bash
TS=$(date +%Y%m%d%H%M)
nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --rm_backend vllm \
    --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --llm_gpu_mem 0.6 \
    --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > log_llm_vllmrm_${TS}.txt 2>&1 &
```

**Critical parameters** (missing any one will cause failure):

| Parameter | Purpose |
|------|------|
| `--rm_backend vllm` | Switch to vLLM `/classify` backend (default is pytorch) |
| `--rm_model <path>` | Model path loaded by vLLM, must match the `vllm serve` path |
| `--rm_url http://localhost:8001` | Points to vLLM RM port |

On successful startup, the beginning of the log should show:
```
RM mode  : vllm  model=/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm
```

### 2.3 Run evaluation

```bash
TS=$(date +%Y%m%d%H%M)
nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 \
    --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_vllmrm_${TS}.json \
    --limit 20 \
    > log_SIA_vllmrm_${TS}.txt 2>&1 &
```

The eval command is **identical** to the pytorch backend version — eval only communicates with the LLM server (port 8000), and the LLM server internally decides which RM backend to call.

---

## 3. Monitoring / Debugging commands

### 3.1 Check startup status

```bash
ps aux | grep -E "vllm serve|sia_vllm_server|mmlu_eval" | grep -v grep
```

### 3.2 Check GPU usage

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
nvidia-smi --query-gpu=memory.free --format=csv,noheader
```

Expected:
- vLLM RM (Qwen3-4B) → ~43 GiB
- LLM server engine (Qwen3-14B) → ~80 GiB
- Total usage ~123 GiB, leaving ~20 GiB safety margin on a 143 GiB GPU

### 3.3 Monitor evaluation progress in real time

```bash
tail -f log_SIA_vllmrm_*.txt | grep -E "correct=|Throughput|Q[0-9]"
```

### 3.4 Check for errors

```bash
echo "RM call errors:"
grep -cE "RM error|500 Internal" log_llm_vllmrm_${TS}.txt

echo "Server crashes:"
grep -cE "Traceback|RuntimeError" log_llm_vllmrm_${TS}.txt log_vllm_rm_${TS}.txt
```

Normal: 0 errors.

### 3.5 View results summary

```bash
tail -10 log_SIA_vllmrm_*.txt
```

Expected output:
```
Avg token length: ~640 tokens/q
Throughput      : ~27 tokens/s
Results saved to: ...
```

---

## 4. Cleanup / Shutdown

### 4.1 Kill all processes

```bash
pkill -f "mmlu_eval"; pkill -f "sia_vllm_server"; pkill -f "vllm serve"
sleep 3
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
# Should output nothing — GPU fully released
```

⚠️ **Important: vLLM serve uses a subprocess (EngineCore); after killing the main process, the subprocess may linger**. If nvidia-smi still shows vllm processes, kill manually:

```bash
# Find lingering EngineCore subprocess
ps aux | grep -E "EngineCore|vllm" | grep -v grep
# Kill manually
kill -9 <PID>
```

### 4.2 Restart only the LLM server, keep vLLM RM running

```bash
pkill -f "sia_vllm_server"
# Kill the LLM's EngineCore subprocess
ps aux | grep "EngineCore" | grep -v grep | awk '{print $2}' | while read pid; do
    # Skip the vLLM RM's EngineCore (the one occupying ~43 GiB per nvidia-smi)
    mem=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep "^$pid," | awk -F'[ ,]+' '{print $2}')
    if [ "$mem" -gt 50000 ] 2>/dev/null; then
        echo "Killing LLM EngineCore PID=$pid (was using $mem MiB)"
        kill -9 $pid
    fi
done
```

---

## 5. Comparison with pytorch backend

If you want to use the older pytorch RM (`src/sia_rm_server.py`) for A/B comparison:

```bash
# Start pytorch RM (replaces the vLLM RM above)
nohup python src/sia_rm_server.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --rm_device cuda:0 \
    --port 8001 > log_rm_pytorch_${TS}.txt 2>&1 &

# Start LLM server (without --rm_backend, defaults to pytorch)
nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --llm_gpu_mem 0.6 \
    --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > log_llm_pytorchrm_${TS}.txt 2>&1 &

# eval command is identical (same as above)
```

The vLLM backend is expected to be ~35% faster than the pytorch backend (see `doc/vllm-rm-backend.md` for details).

---

## 6. Common Issues

### Q: LLM server startup fails with OOM `Free memory ... is less than desired GPU memory utilization`

A: Most likely there is a lingering old process. Run `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader` to see what is still occupying memory. A common culprit is the `EngineCore` subprocess of vllm serve — it does not exit automatically when the main process is killed; manually run `kill -9 <PID>`.

### Q: LLM gets `Connection refused` or `404 Not Found` when calling RM

A: Check `--rm_backend` and `--rm_url`:
- With `--rm_backend vllm`, the RM should be a `vllm serve` process (providing `/classify` on the port)
- With `--rm_backend pytorch` (default), the RM should be a `sia_rm_server.py` process (providing `/score` on the port)
- Do not mix them up; doing so will cause the LLM to call the wrong endpoint

### Q: Evaluation throughput is low or NaN

A: Run `grep "RM mode" log_llm_*.txt` to confirm the RM mode at LLM server startup. If it shows `RM mode : pytorch` but the port is serving vllm, it will call the wrong endpoint. When restarting, add `--rm_backend vllm --rm_model <path>`.

### Q: Log file was overwritten and traceback is lost

A: Use `TS=$(date +%Y%m%d%H%M)` to add a timestamp to log filenames. Do not reuse the same filename — nohup restart will overwrite it.

---

## 7. File References

- `scripts/convert_rm_for_vllm.py` — RM conversion script
- `scripts/bench_vllm_rm_realistic.py` — 1000-call latency benchmark
- `src/sia_vllm_server.py` — LLM server with `--rm_backend` parameter
- `src/sia_vllm_RM.py` — RM client adapter layer (pytorch + vllm dual backend)
- `doc/vllm-rm-backend.md` — vLLM backend architecture and performance data
- `doc/cuda-graph-debugging-journal.md` — previous pytorch CUDA graph debugging experience
- `doc/rm-profiling-guide.md` — RM server profiling tool documentation
