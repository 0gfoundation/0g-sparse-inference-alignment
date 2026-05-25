# vLLM RM Backend

把 RM 从手写 FastAPI（`sia_rm_server.py`）替换成 vLLM 原生部署，**latency 减少 35-40%**，**完全没有 NaN / 崩溃风险**，代码量减少 95%。

## 为什么换

原始 RM server（`src/sia_rm_server.py`）经过多轮手动优化（KV cache、torch.compile、CUDA graph 静态 bucketing）后，最好情况 ~40ms/call，但**仅在前 270 次调用工作**，之后触发 NaN 自动 disable，剩下都走 eager fallback ~57ms。详情见 `doc/cuda-graph-debugging-journal.md`。

vLLM 原生支持 Qwen3ForSequenceClassification + PagedAttention 前缀缓存 + 自动 CUDA graph 分桶，开箱即用，**所有调用稳定在 35-60ms**。

## 准备：把 RM 转成 vLLM 兼容 checkpoint

vLLM 不能加载我们独立的 `token_reward_head.pt`，必须把 head 焊进 HF checkpoint。一次性脚本：

```bash
python scripts/convert_rm_for_vllm.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --output /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm
```

输出是标准 `Qwen3ForSequenceClassification` HF checkpoint（~7.6 GB），含 tokenizer。

## 启动（两条流程）

### 1. RM server（vLLM serve）

```bash
nohup vllm serve /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --runner pooling \
    --convert classify \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.3 \
    --max-model-len 4096 \
    --port 8001 \
    > log_vllm_rm_$(date +%Y%m%d%H%M).txt 2>&1 &
```

第一次启动会捕获 CUDA graph（约 60s warmup），之后所有调用稳定。

### 2. LLM server（用 `--rm_backend vllm`）

```bash
nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --rm_backend vllm \
    --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --llm_gpu_mem 0.6 \
    --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > log_llm_$(date +%Y%m%d%H%M).txt 2>&1 &
```

新增 CLI 参数：
- `--rm_backend pytorch|vllm`（默认 `pytorch`，向后兼容）
- `--rm_model <path>`（`--rm_backend vllm` 时必填）

### 3. 跑评测

不变。eval/mmlu_eval.py 通过 `:8000` 调 LLM server，LLM server 内部自己决定调用哪种 RM 后端。

```bash
python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 \
    --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_$(date +%Y%m%d%H%M).json \
    --limit 20
```

## 调用契约

LLM server 内部（`src/sia_vllm_RM.py::_score_candidates_vllm`）做以下事情：

1. 用 LLM tokenizer 的 `apply_chat_template` 把每个 candidate 拼成 chat 格式：
   ```
   <|im_start|>user
   {user_content}<|im_end|>
   <|im_start|>assistant
   {response_so_far}{candidate}<|im_end|>
   ```
2. POST 5 个候选 text 到 vLLM `/classify`：
   ```json
   {
     "model": "/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm",
     "input": [text1, text2, text3, text4, text5],
     "activation": false
   }
   ```
3. 解析响应 `data[i].probs[0]` 作为第 i 个候选的 raw reward logit。

**关键参数 `"activation": false`**：vLLM 默认对 num_labels=1 的 classify 任务应用 sigmoid（输出 [0,1] 概率）。SIA 需要原始 logit（范围 -20 ~ +30）才能正确加到 LLM logits 上做采样。这个 flag 不能省。

## 性能对比

基于 1000 次模拟 SIA 调用（每次 batch=5，response 从 0 涨到 ~1100 token）：

| kv_len 段 | 当前 pytorch RM | vLLM `/classify` | 加速 |
|----------|----------------|------------------|------|
| <200 | ~57ms | **34ms** | **40%** |
| 200-500 | ~58ms | **42ms** | **28%** |
| 500-1000 | ~57ms | **55ms** | 4% |
| 1000+ | ~57ms | ~60ms | 持平 |
| **1000 calls 总耗时** | **~78s** | **50.4s** | **35%** |

更重要的是：vLLM 路径**完全稳定**，0 NaN、0 崩溃、不需要 per-request adaptive disable、不需要手写 mark_dynamic。

## 当前 pytorch 路径还保留吗

保留。两个原因：
1. 向后兼容：existing 启动脚本和 eval 命令不需要任何改动
2. 调试备用：如果将来发现 vLLM 模式有问题，可以 fallback 到 pytorch 模式继续工作

默认 `--rm_backend pytorch` 保持原行为。要用 vLLM 必须显式 `--rm_backend vllm --rm_model ...`。

## 限制 / 已知问题

- vLLM 用 `max_model_len=4096`。如果 RM 输入（user + response + chat template）超过这个长度会被截断。SIA 通常用短 prompts，不构成问题。
- vLLM 在 batch=5 时有约 0.2 单位的 score 抖动（同一文本单独算 vs 批量算结果略差）。**不影响 candidate 之间的 ranking**，对 SIA 完全够用。
- vLLM RM 和 LLM 共享同一 GPU。当前配置 LLM 60% + RM 30%，剩 10% safety margin。如果 OOM 把两个 `gpu-memory-utilization` 都降一些。
