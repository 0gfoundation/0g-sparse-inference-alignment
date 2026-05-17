# SIA 推理性能测试报告

## 实验目标

量化 SIA（Sparse Inference-time Alignment）的推理性能开销：

> **核心问题**：加了 Value Model 打分之后，端到端吞吐量下降了多少？瓶颈在哪里？

---

## 实验配置

| 组件 | 模型 |
|------|------|
| LLM | Qwen3-14B（vLLM 部署，`--llm_gpu_mem 0.6`） |
| Value Model | Qwen3-4B + VM-Qwen3-4B-Base LoRA（`cuda:0`） |

**SIA 干预参数：**

| 参数 | 值 | 说明 |
|------|----|------|
| `--topk` | 5 | 每个干预步骤评分的候选 token 数 |
| `--weight` | 1.0 | Value Model score 叠加到 logits 的权重 |
| `--entropy_threshold` | 1.0 | 熵低于此值时跳过干预（稀疏策略） |

**评测集：** MMLU-Redux，每科取 20 题，共 600 题。

---

## 一、端到端吞吐量

三种跑法在相同评测集下的整体吞吐量：

| 跑法 | 总 tokens | 吞吐量（tokens/s） |
|------|-----------|--------------------|
| SIA（`sia_vllm_server.py`，干预开启） | 239,389 | **11.3** |
| noSIA（`sia_vllm_server.py`，干预关闭） | 401,598 | **84.3** |
| noSIA（原生 vLLM，不经过 `sia_vllm_server`） | 430,908 | **86.8** |

**结论：**

- noSIA 两种跑法几乎相同（84.3 vs 86.8 tokens/s，差距 < 3%），说明 `sia_vllm_server.py` 本身的 HTTP 封装 overhead 可以忽略不计。
- SIA 开启后吞吐量降至 11.3 tokens/s，约为 noSIA 的 **1/7.5**，瓶颈完全来自 Value Model 逐步打分的开销。

---

## 二、逐步延迟分析

SIA 的稀疏干预策略将每个生成步骤分为两类：

| 步骤类型 | 步数 | 占比 | 平均延迟/step | 吞吐量（tokens/s） |
|---------|------|------|-------------|-------------------|
| SKIP（熵低，直接跳过 Value Model） | 174,226 | **72.7%** | 16.1 ms | **62.2** |
| INTERVENE（调用 Value Model 打分） | 65,503 | **27.3%** | 281.0 ms | **3.6** |
| 倍数 | — | — | **17.5x** | — |

**加权验证：**

```
加权平均步骤延迟 = 72.7% × 16.1ms + 27.3% × 281.0ms = 88.5ms
对应吞吐量 = 1000 / 88.5 = 11.3 tokens/s  ✓（与实测完全吻合）
```

---

## 三、性能瓶颈根本原因

有干预步骤慢 17.5 倍的根本原因：`sia_rm_server.py` 的 `/score` 端点对 topk=5 个候选 token **串行**执行了 5 次独立的 Value Model forward pass，每次都包含完整的 prefix（system prompt + question + 已生成内容），GPU 利用率极低：

```
当前实现（串行，5 次独立 forward）：
  候选 token_1 → Value Model forward → score_1   ╗
  候选 token_2 → Value Model forward → score_2   ║  各自包含完整 prefix
  候选 token_3 → Value Model forward → score_3   ║  重复计算 >> 无法复用
  候选 token_4 → Value Model forward → score_4   ║
  候选 token_5 → Value Model forward → score_5   ╝
  耗时 ≈ 281 ms
```

相比之下，SKIP 步骤（16.1 ms）只需一次正常的 vLLM forward，与原生推理速度接近，说明 LLM 本身的推理速度并不是瓶颈。

---

## 四、优化方向

以下优化方案均可在**不重训练**的前提下实施：

| 方案 | 核心思路 | 预期 INTERVENE 延迟 | 实现难度 |
|------|---------|---------------------|---------|
| **1. 批量 forward** | 将 k 个候选组成 batch，一次 forward 完成 | ~70–100 ms | 极低（约 20 行改动） |
| **2. KV cache prefix 共享** | k 个候选共享相同 prefix 的 KV states，仅对最后 1 token 做 decode | ~40–60 ms | 中 |
| **3. SGLang 作为 RM Server** | SGLang RadixAttention 自动复用共享 prefix KV，无需手写缓存逻辑 | ~50 ms | 中 |

方案 1 为最优先项：改动最小，预期可将 INTERVENE 延迟从 281 ms 降至 70–100 ms，整体吞吐量有望从 11.3 提升至 20–30 tokens/s（约 2–3x 提升）。

---

## 结论

1. **`sia_vllm_server.py` 封装 overhead 可忽略**：与原生 vLLM 吞吐量差距 < 3%，HTTP server 层不是瓶颈。

2. **SIA 开启后吞吐量下降约 7.5 倍**：从 84.3 降至 11.3 tokens/s，瓶颈完全来自 Value Model 串行 forward pass。

3. **稀疏策略有效减少了干预频率**：在 `entropy_threshold=1.0` 下，约 72.7% 的 token 生成步骤直接跳过 Value Model（仅 27.3% 触发干预），否则全量干预的吞吐量将进一步降低至约 3.6 tokens/s。

4. **性能优化空间明确**：将 `/score` 端点改为批量 forward 是最低成本的优化方向，预期整体吞吐量可提升 2–3 倍，且无需重训练。

---

## Appendix：实验运行命令

### SIA 推理

```bash
nohup python src/sia_rm_server.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --rm_device cuda:0 \
    --port 8001 > log_rm_server_SIA_202605142111.txt 2>&1 &

nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --llm_gpu_mem 0.6 \
    --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 > log_llm_server_SIA_202605142111.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 \
    --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_SIA_202605142111.json \
    --limit 20 > log_SIA_202605142111.txt 2>&1 &
```

### noSIA 推理（经 `sia_vllm_server.py`）

```bash
nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --llm_gpu_mem 0.6 \
    --topk 5 --weight 0.0 --entropy_threshold 999999 \
    --host 0.0.0.0 --port 8000 > log_llm_server_noSIA_202605142111.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 \
    --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_noSIA_selfServer_202605142111.json \
    --limit 20 > log_noSIA_selfServer_202605142111.txt 2>&1 &
```

### noSIA 推理（原生 vLLM）

```bash
nohup python -m vllm.entrypoints.openai.api_server \
    --model /workspace/SIA/models/Qwen3-14B \
    --gpu-memory-utilization 0.6 \
    --max-model-len 4096 \
    --override-generation-config '{"repetition_penalty": 1.3}' \
    --host 0.0.0.0 --port 8000 > log_llm_server_noSIA_vllm_202605142111.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 \
    --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_noSIA_selfServer_vllm_202605142111.json \
    --limit 20 > log_noSIA_selfServer_vllm_202605142111.txt 2>&1 &
```
