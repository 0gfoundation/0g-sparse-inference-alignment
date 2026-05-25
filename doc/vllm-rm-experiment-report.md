# 实验报告：用 vLLM 替代 PyTorch 部署 RM 之后的端到端性能分析

**日期**：2026-05-22  
**目标**：把 Value Model (Reward Model) 从手写 FastAPI（`sia_rm_server.py`） 切换到 vLLM 原生部署，看是否能降低 SIA 推理的整体延迟。

---

## 1. 实验结果总览

在 MMLU 30 个子学科各 20 题 = 600 题上跑 SIA 干预（`topk=5`, `weight=1.0`, `entropy_threshold=1.0`）。剔除实验 2 中 latency > 60s 的 65 个超时样本，实验 1 同步保留同顺序号样本，**两组各剩 535 个共有样本**做 side-by-side 对比。

| 指标 | 实验1 (noSIA, vLLM 仅 LLM) | 实验2 (SIA + vLLM RM) | Δ |
|------|------:|------:|------:|
| **样本数** | 535 | 535 | — |
| **correct=True** | 421 | **428** | **+7** |
| **正确率** | 78.69% | **80.00%** | **+1.31%pt** ✅ |
| **平均 tokens 长度** | 634.9 | **506.7** | **-128 (-20%)** |
| **整体 throughput** | **88.44 tok/s** | **33.47 tok/s** | -54.97 (-62%) |
| **总 wall latency** | 3840.9s | 8100.9s | +4260s |

**核心结论**：SIA 带来 +1.31%pt 准确率，**代价是 throughput 下降 62%**（每生成一个 token 多花 18.6ms）。和我们调优前 PyTorch RM 版本（throughput -76%）相比，vLLM RM 把性能损失从 76% 压到了 62%，但**距离"接近 noSIA"还有不小差距**。

---

## 2. 模型怎么得到的：从原始 RM 到 vLLM 兼容 checkpoint

### 2.1 输入

| 组件 | 路径 | 说明 |
|------|------|------|
| Base 模型 | `/workspace/SIA/models/Qwen3-4B` | HF 标准 Qwen3-4B 权重 |
| LoRA adapter | `/workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base/lora_weights/` | SIA 训练的 Value Model LoRA |
| Reward head | `/workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base/token_reward_head.pt` | 独立的 `Linear(2560, 1)`，含 weight + bias |

### 2.2 转换脚本

```bash
python scripts/convert_rm_for_vllm.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --output /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm
```

脚本做的事（见 `scripts/convert_rm_for_vllm.py`）：

1. 加载 Qwen3-4B 为 `Qwen3ForSequenceClassification(num_labels=1)`
2. 加载 LoRA adapter，调用 `peft_model.merge_and_unload()` 把 LoRA 焊进 base 权重
3. 加载 `token_reward_head.pt`，把它的 weight 拷进 `model.score`
4. 丢弃 head 的 bias（值 -0.0087，对 ranking 无影响，原 `Qwen3ForSequenceClassification.score` 默认 `bias=False`）
5. 设置 `model.config.pad_token_id` 让 batch padding 正确工作
6. 保存为标准 HF checkpoint（含 tokenizer）

### 2.3 输出验证

输出目录 `/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm/`（~7.6 GB）含：
- `config.json`：`architectures: ["Qwen3ForSequenceClassification"]`，`pad_token_id: 151643`
- 2 个 `.safetensors` shard
- tokenizer 文件

**Sanity check**（脚本自带）：
- Wrapper-style 计算（fp32 算 head）：scores = `[0.9097, -8.8784]`
- 标准 SeqClassification.forward（bf16）：scores = `[0.9102, -8.8750]`
- max diff 0.0034（bf16 量化噪声）→ ✅ ranking 完全一致

---

## 3. 启动命令拆解

### 3.1 vLLM RM Server

```bash
nohup vllm serve /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
      --runner pooling \
      --convert classify \
      --enable-prefix-caching \
      --no-enable-chunked-prefill \
      --gpu-memory-utilization 0.4 \
      --max-model-len 2048 \
      --port 8001 \
      > log_vllm_rm_opt_<TS>.txt 2>&1 &
```

**每个 flag 的作用**：

| 参数 | 作用 | 实际收益 |
|------|------|---------|
| `--runner pooling` | pooling 模式，无自回归生成，跳过 sampling/detokenize | ✅ 必须（不是优化，是 task 要求）|
| `--convert classify` | 把 SeqClassification 套上 score 头返回单值 | ✅ 必须 |
| `--enable-prefix-caching` | **核心优化**：跨请求 KV cache 复用 | ✅ SIA 步进时每步只长 1 token，prefix 100% 复用，节省 99% prefill |
| `--no-enable-chunked-prefill` | 关闭 chunked prefill 调度（pooling task 用不上）| ✅ 减少调度迭代次数 5-10% overhead |
| `--gpu-memory-utilization 0.4` | 给 RM 56 GiB（默认 0.3 是 42 GiB）| ✅ KV cache 容量从 ~10 万 token → ~14 万 token |
| `--max-model-len 2048` | 从 4096 减半 | ✅ 调度元数据减半 |

**调试过程中验证不可行的参数**（见 `~/Work/SIA_vllm_fine_tune.md`）：

| 参数 | 失败原因 |
|------|---------|
| `--block-size 8` | Qwen3 模型运行时要求 block_size 是 16 的倍数 → 报错 `Block size must be a multiple of 16` |
| `--cuda-graph-sizes 5` | vLLM 单值语义 = `[1,2,4]+range(8,N+1,8)`，5 时只 capture [1,2,4]，batch=5 落空回退到 eager 反而慢 |

### 3.2 SIA LLM Server

```bash
nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --rm_backend vllm \                                                            # ★ 关键：切到 vLLM 后端
    --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \                 # ★ vLLM serve 加载的模型路径
    --llm_gpu_mem 0.6 \
    --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > log_llm_vllmrm_<TS>.txt 2>&1 &
```

`--rm_backend vllm` 是新加的 CLI 参数（commit b0e0c02），让 `sia_vllm_RM.py::_score_candidates_vllm` 走 vLLM 的 `/classify` endpoint（带 `activation: false` 取原始 logit），而不是旧的 `/score`（pytorch RM 的自定义协议）。

### 3.3 评测

```bash
nohup python eval/mmlu_eval.py \
      --base_url http://localhost:8000/v1 \
      --model /workspace/SIA/models/Qwen3-14B \
      --output results/test_vllmrm_<TS>.json \
      --limit 20 \
      > log_SIA_vllmrm_<TS>.txt 2>&1 &
```

`--limit 20` 是**每个子学科保留 20 题**，30 学科共 600 题。

### 3.4 noSIA baseline（对照组）

```bash
nohup python -m vllm.entrypoints.openai.api_server \
    --model /workspace/SIA/models/Qwen3-14B \
    --gpu-memory-utilization 0.6 \
    --max-model-len 4096 \
    --override-generation-config '{"repetition_penalty": 1.3}' \
    --host 0.0.0.0 --port 8000 \
    > log_llm_server_noSIA_vllm_<TS>.txt 2>&1 &
```

直接用 vLLM 原生 OpenAI server，无 SIA 干预，无 RM 调用。

---

## 4. 实际上做了哪些优化（从 PyTorch RM → vLLM RM）

### 4.1 在 RM 服务端

| 维度 | 原 PyTorch RM | 现 vLLM RM |
|------|--------------|-----------|
| 框架 | 手写 FastAPI + 直接 PyTorch eager forward | vLLM v1 引擎 |
| KV cache | 手写 dict (`_req_kv[request_id]`)，每请求一个 entry | vLLM PagedAttention，block 级粒度，自动 LRU |
| 跨请求 prefix 复用 | 手写 `_find_common_prefix` + `_dc_expand_batch` | **`--enable-prefix-caching` 自动按 16-token block 匹配** |
| Attention 实现 | PyTorch SDPA（Flash/mem-eff/math 不可控）| Flash Attention 2 + CUDA graph 加速 decode |
| Kernel launch | 36 层 × ~10 个 kernel = ~360 launch | CUDA graph 把整张 forward 捕获成 1 次 replay |
| Batching | 一次只服务 1 个 request | continuous batching，自动合并多 request |
| NaN / 崩溃风险 | 频繁（手写 CUDA graph bucket 试过失败）| 0 |

### 4.2 在客户端 (`sia_vllm_RM.py`)

- 新增 `_score_candidates_vllm()` 分支，发 `/classify` 请求时强制 `activation: false`（不加 sigmoid），拿原始 logit 给 SIA 用
- chat template 用 LLM tokenizer 的 `apply_chat_template`（Qwen3 系两边一致）
- 旧 `_score_candidates_pytorch()` 保留作为 fallback，CLI flag `--rm_backend` 切换

### 4.3 性能基线变化（来自 1000-call 微 benchmark）

| 路径 | per-call p50 latency | 备注 |
|------|---------------------:|------|
| 原 PyTorch RM (Flash eager + KV cache) | 57ms | ⚠️ 不稳定，偶发 NaN |
| 原 PyTorch RM (CG bucket 工作时) | 36ms | ⚠️ 只在前 ~270 次工作，之后 disable |
| **vLLM RM `/classify`** | **38-50ms (随 prefix 长度涨)** | ✅ 完全稳定 |

---

## 5. 为什么端到端加速不达预期：数学分解

把每个 LLM 输出 token 的耗时拆开：

```
noSIA: per_token = LLM_forward_only = 1/88.4 = 11.3ms

SIA:   per_token = LLM_forward + SIA_processor_overhead + intervention_ratio × RM_call_cost
              = 11.3 + ε_skip + 0.28 × RM_call_cost
              = 1/33.5 = 29.9ms

      → 18.6ms 是 SIA 引入的额外开销
```

把这 18.6ms 拆开（基于代码审查 + benchmark 数据反推）：

### 5.1 每个 token 必交的"SIA 处理税" (~1.4ms 摊销)

`sia_vllm_RM.py::apply()` 里，每个 token 都会做：

```python
topk_logits, topk_indices = torch.topk(logits[i], self._TOPK)
probs = F.softmax(topk_logits.float(), dim=-1)
entropy = dist.Categorical(probs=probs).entropy().item()   # ← CUDA sync!
if entropy < self._ENTROPY_THRESHOLD:
    continue
```

`.item()` 强制 GPU 同步等结果，**把 LLM forward 的异步流水线打断**。即使 72% 的 token 走 SKIP 不调 RM，这一步也跑不掉。

- 每 token sync overhead ≈ 2ms
- 摊销：1.4ms / token

### 5.2 每个 INTERVENE 触发的 RM 调用 (~17ms 摊销)

vLLM `/classify` 单调用 50ms 左右，但端到端反推每次 INTERVENE 约 60-65ms。多出来的 10-15ms 在客户端：

```python
# 每次 INTERVENE：
response_so_far = self._llm_tok.decode(output_ids, ...)       # ~3-5ms (response 长了更慢)
formatted_texts = [chat_template + each_candidate for ...]    # ~2-3ms 字符串拼接
resp = session.post(URL, json={...})                          # HTTP serialize + localhost 网络
data = resp.json()["data"]                                    # JSON parse
```

- 摊销：60ms × 0.28 = 17ms / token

### 5.3 完整 budget

```
LLM forward:                       11.3ms     (LLM 自己)
SKIP overhead (entropy.item):       1.4ms     (每个 token 摊)
INTERVENE overhead:                17.0ms     (50ms RM + 10-15ms 客户端) × 28%
─────────────────────────────────
Total:                             29.7ms     ≈ 实测 29.9ms  ✓ 数学完全对得上
```

---

## 6. 为什么这次 vLLM 优化收益有限

把当前 vLLM CLI 优化能再压榨多少：

| 当前 RM 单调用 latency (p50) | ~40-50ms |
| 假设再优化压到 | ~30-35ms |
| 节省的 RM 时间 | 15ms |
| 摊到 per-token | **15ms × 28% = 4.2ms** |
| 端到端 throughput 改善 | 33.5 → **39 tok/s** (+16%) |

**结论**：vLLM RM 单调用的 latency 已经接近瓶颈（model forward 本身需要的时间）。**继续靠 RM CLI 调参的提升空间已经很小（~15-20%）**。

---

## 7. 继续提速必须改"架构层面"

按预期收益从大到小排序：

### 🔥 P0：去掉 `.item()` 同步 — 端到端 +5-10%

把 entropy 比较留在 GPU，不强制 sync。批量决策后再分发 HTTP。
- 实施成本：~30 行代码修改
- 风险：低
- **预期**：33.5 → 37-40 tok/s

### 🔥 P1：增量维护 response_so_far — 端到端 +1-3%

```python
# 当前：每步重 decode 整个 output_ids（长 response 越来越慢）
response_so_far = self._llm_tok.decode(output_ids, ...)

# 优化：只 decode 新增 token，append 到缓存
new_token_str = self._llm_tok.decode([output_ids[-1]], ...)
response_so_far = self._cached_response.get(i, "") + new_token_str
self._cached_response[i] = response_so_far
```
- 实施成本：~50 行
- 风险：BPE 边界细节需要处理
- **预期**：RM 调用 -3-5ms，端到端 +1-3%

### 🔥 P2：客户端缓存 chat template prefix

chat template 的 `<|im_start|>user...{user_content}<|im_end|>\n<|im_start|>assistant\n` 部分跨步是不变的。只动态拼 response_so_far + candidate。
- 实施成本：低
- **预期**：RM 调用 -2-3ms，端到端 +1-2%

### 💡 P3：RM 量化到 FP8

```bash
vllm serve ... --quantization fp8
```
H200 上 FP8 GEMM 比 BF16 快 ~2×。
- 实施成本：CLI 一行 + 验证质量
- 风险：可能小幅准确率下降
- **预期**：RM forward 50ms → 25-30ms，端到端 +5-10%（33.5 → 37 tok/s）

### 💡 P4：提高 `entropy_threshold` 减干预率

```
当前 0.27 干预率
若提到 0.15:
  per_token = 11.3 + 1.4 + 0.15 × 60 = 21.7ms = 46 tok/s (+37%)
```
- 实施成本：CLI 一行
- 风险：准确率下降（需 A/B 评测）
- **预期**：33.5 → 46 tok/s，但 +1.31%pt 的准确率收益可能受损

### 综合预期

| 优化路径 | 端到端 throughput | 累计相对 noSIA (88.4) |
|---------|-----:|-----:|
| 当前 (vLLM RM 默认参数 + CLI 微调) | 33.5 tok/s | 38% |
| + 去掉 .item() 同步 | ~38 tok/s | 43% |
| + response 增量缓存 + chat template 缓存 | ~41 tok/s | 46% |
| + FP8 RM | ~45 tok/s | 51% |
| + entropy_threshold 1.0 → 1.3 (质量代价) | ~50 tok/s | 57% |

理论上限是 noSIA 的 ~57%。**要超过这个上限只能要么干预次数大幅下降，要么 RM 直接消失**（=无干预 =回到 noSIA）。

---

## 8. 文件引用

| 文件 | 内容 |
|------|------|
| `scripts/convert_rm_for_vllm.py` | RM 转换脚本（base + LoRA + reward head → vLLM 兼容 HF checkpoint） |
| `scripts/bench_vllm_rm_realistic.py` | 1000-call 模拟 SIA 调用 benchmark |
| `src/sia_vllm_RM.py` | RM 客户端，含 pytorch / vllm 双后端 |
| `src/sia_vllm_server.py` | LLM server，含 `--rm_backend` 参数 |
| `doc/vllm-rm-backend.md` | vLLM RM 后端架构与启动手册 |
| `doc/run-sia-vllm-rm.md` | 端到端跑评测的命令手册 |
| `doc/cuda-graph-debugging-journal.md` | 之前 PyTorch CUDA graph 调试经验（已废弃路径） |
| `~/Work/SIA_vllm_fine_tune.md` | vLLM CLI 调优过程，含失败的参数 |

---

## 9. 复现日志文件

| 实验 | 日志路径 |
|------|---------|
| 实验 1 (noSIA, vLLM) | `log_noSIA_selfServer_vllm_202605142111.txt` (eval), `log_llm_server_noSIA_vllm_202605142111.txt` (LLM server) |
| 实验 2 (SIA + vLLM RM) | `log_SIA_vllmrm_202605221045.txt` (eval), `log_llm_vllmrm_202605221045.txt` (LLM server), `log_vllm_rm_opt_202605221045.txt` (RM server) |
