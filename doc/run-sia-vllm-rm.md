# 跑 SIA + vLLM RM 评测的命令手册

直接复制即可运行的完整命令集合，覆盖 vLLM RM + SIA LLM server + MMLU 评测三个进程。

## 0. 一次性准备：转换 RM 到 vLLM 兼容 checkpoint

只需做一次。已经转过的话跳过本节。

```bash
python scripts/convert_rm_for_vllm.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --output /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm
```

输出目录约 7.6 GB，标准 HF `Qwen3ForSequenceClassification` checkpoint。

---

## 1. 全自动启动（推荐）

清理 → 启动 vLLM RM → 等就绪 → 启动 LLM server → 等就绪 → 跑评测，一气呵成。

```bash
# === 清理可能残留的进程 ===
pkill -f "sia_vllm_server"; pkill -f "sia_rm_server"; pkill -f "mmlu_eval"
pkill -f "vllm serve"; sleep 3

TS=$(date +%Y%m%d%H%M)

# === 1) vLLM RM server (端口 8001) ===
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

# === 2) SIA LLM server (端口 8000) ===
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

# === 3) 跑 MMLU 评测 ===
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

## 2. 分步骤启动（手动控制时用）

### 2.1 启动 vLLM RM server

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

等待标志：日志里出现 `Application startup complete` 且 CUDA graph capture 完成。

验证：
```bash
curl -s http://localhost:8001/v1/models | python -m json.tool
```

### 2.2 启动 SIA LLM server (vLLM RM 后端)

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

**关键参数**（漏一个就会失败）：

| 参数 | 用途 |
|------|------|
| `--rm_backend vllm` | 切换到 vLLM `/classify` 后端（默认 pytorch） |
| `--rm_model <path>` | vLLM 加载的 model 路径，必须和 `vllm serve` 路径一致 |
| `--rm_url http://localhost:8001` | 指向 vLLM RM 端口 |

启动正常应看到日志开头：
```
RM mode  : vllm  model=/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm
```

### 2.3 跑评测

```bash
TS=$(date +%Y%m%d%H%M)
nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 \
    --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_vllmrm_${TS}.json \
    --limit 20 \
    > log_SIA_vllmrm_${TS}.txt 2>&1 &
```

eval 命令和 pytorch 后端版**完全一样** —— eval 只跟 LLM server 通信（端口 8000），LLM server 内部决定调用哪种 RM 后端。

---

## 3. 监控 / 调试命令

### 3.1 看启动状态

```bash
ps aux | grep -E "vllm serve|sia_vllm_server|mmlu_eval" | grep -v grep
```

### 3.2 看 GPU 占用

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
nvidia-smi --query-gpu=memory.free --format=csv,noheader
```

期望：
- vLLM RM (Qwen3-4B) → ~43 GiB
- LLM server engine (Qwen3-14B) → ~80 GiB
- 总占用 ~123 GiB，143 GiB GPU 剩 ~20 GiB safety margin

### 3.3 实时监控评测进度

```bash
tail -f log_SIA_vllmrm_*.txt | grep -E "correct=|Throughput|Q[0-9]"
```

### 3.4 检查错误

```bash
echo "RM call errors:"
grep -cE "RM error|500 Internal" log_llm_vllmrm_${TS}.txt

echo "Server crashes:"
grep -cE "Traceback|RuntimeError" log_llm_vllmrm_${TS}.txt log_vllm_rm_${TS}.txt
```

正常情况：0 errors。

### 3.5 看结果摘要

```bash
tail -10 log_SIA_vllmrm_*.txt
```

期望输出：
```
Avg token length: ~640 tokens/q
Throughput      : ~27 tokens/s
Results saved to: ...
```

---

## 4. 清理 / 关闭

### 4.1 关闭所有进程

```bash
pkill -f "mmlu_eval"; pkill -f "sia_vllm_server"; pkill -f "vllm serve"
sleep 3
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
# 应该输出空 —— GPU 完全释放
```

⚠️ **重要：vLLM serve 用了 subprocess (EngineCore)，pkill 主进程后 subprocess 可能残留**。如果上面 nvidia-smi 还看到 vllm 进程，手动 kill：

```bash
# 找出残留的 EngineCore subprocess
ps aux | grep -E "EngineCore|vllm" | grep -v grep
# 手动 kill
kill -9 <PID>
```

### 4.2 只重启 LLM server，保留 vLLM RM

```bash
pkill -f "sia_vllm_server"
# 杀掉 LLM 的 EngineCore subprocess
ps aux | grep "EngineCore" | grep -v grep | awk '{print $2}' | while read pid; do
    # 跳过 vLLM RM 的 EngineCore（用 nvidia-smi 看占用 43 GiB 的那个）
    mem=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep "^$pid," | awk -F'[ ,]+' '{print $2}')
    if [ "$mem" -gt 50000 ] 2>/dev/null; then
        echo "Killing LLM EngineCore PID=$pid (was using $mem MiB)"
        kill -9 $pid
    fi
done
```

---

## 5. 与 pytorch 后端对比

如果想用旧版 pytorch RM（`src/sia_rm_server.py`）做 A/B 对比：

```bash
# 启动 pytorch RM（替代上面的 vLLM RM）
nohup python src/sia_rm_server.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --rm_device cuda:0 \
    --port 8001 > log_rm_pytorch_${TS}.txt 2>&1 &

# 启动 LLM server（不加 --rm_backend，默认 pytorch）
nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --llm_gpu_mem 0.6 \
    --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > log_llm_pytorchrm_${TS}.txt 2>&1 &

# eval 命令完全一样（同上）
```

预期 vLLM 后端比 pytorch 后端快 ~35%（详见 `doc/vllm-rm-backend.md`）。

---

## 6. 常见问题

### Q: LLM server 启动报 OOM `Free memory ... is less than desired GPU memory utilization`

A: 大概率有旧进程残留。`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader` 看谁还在占。常见的是 vLLM serve 的 `EngineCore` subprocess —— pkill 主进程后它不会自动退出，要手动 `kill -9 <PID>`。

### Q: LLM 调 RM 时报 `Connection refused` 或 `404 Not Found`

A: 检查 `--rm_backend` 和 `--rm_url`：
- `--rm_backend vllm` 时 RM 应该是 `vllm serve` 进程（端口提供 `/classify`）
- `--rm_backend pytorch` (默认) 时 RM 应该是 `sia_rm_server.py` 进程（端口提供 `/score`）
- 两个不能搞混，否则 LLM 会调错 endpoint

### Q: 评测 throughput 偏低或 NaN

A: 查 `grep "RM mode" log_llm_*.txt` 确认 LLM server 启动时的 RM 模式。如果是 `RM mode : pytorch` 但端口上是 vLLM serve，就会调错。重新启动时加上 `--rm_backend vllm --rm_model <path>`。

### Q: 日志文件被覆盖丢失了 traceback

A: 用 `TS=$(date +%Y%m%d%H%M)` 给日志名加时间戳。不要重用同名文件 —— nohup 重启会覆盖。

---

## 7. 文件引用

- `scripts/convert_rm_for_vllm.py` — RM 转换脚本
- `scripts/bench_vllm_rm_realistic.py` — 1000-call latency benchmark
- `src/sia_vllm_server.py` — LLM server，含 `--rm_backend` 参数
- `src/sia_vllm_RM.py` — RM client 适配层（pytorch + vllm 双后端）
- `doc/vllm-rm-backend.md` — vLLM 后端架构与性能数据
- `doc/cuda-graph-debugging-journal.md` — 之前 pytorch CUDA graph 调试经验
- `doc/rm-profiling-guide.md` — RM server profiling 工具说明
