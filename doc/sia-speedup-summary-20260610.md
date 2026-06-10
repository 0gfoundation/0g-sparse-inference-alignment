# SIA 推理加速优化总结

**适用模型**：Qwen3-VL-30B-A3B-Instruct、0GM-1.0-35B-A3B  
**Value Model**：VM-Qwen3-4B（两个模型共用）  
**撰写日期**：2026-06-10

---

## 一、速度基线

| 模型 | noSIA 起点 | SIA 起点（最初） | SIA 终点（当前） |
|------|:---:|:---:|:---:|
| VL-30B | 80.8 tok/s | 59.2 tok/s | **78.3 tok/s** |
| 0GM-35B | ~57 tok/s | ~32 tok/s（HTTP RM）| **65.2 tok/s** |

"起点"指引入所有以下优化之前的实测吞吐量。"终点"为当前生产最优配置下的实测值。

---

## 二、两个模型共用的优化

### 2.1 RM 后端从 PyTorch 迁移到 vLLM（HTTP 模式）

**优化内容**

原始方案直接用 HuggingFace `AutoModel` 加载 Reward Model，在推理循环内调用 `model.forward()`，无 batching 优化、无 prefix caching。

优化后改用 vLLM 以 `--runner pooling --convert classify` 模式托管 RM，SIA 通过 vLLM 的 HTTP `/classify` 接口打分。vLLM 内置 APC（Automatic Prefix Caching）、动态 batching、CUDA graph，RM 推理性能大幅提升。

**实测数据（RM 单次调用延迟，topk=5/10）**

| 后端 | <200 token | 200–500 token | 1000+ token |
|------|:---:|:---:|:---:|
| PyTorch 原始 | ~78 ms | ~78 ms | ~78 ms |
| vLLM HTTP | ~34 ms | ~42 ms | ~57 ms |
| **降幅** | **-56%** | **-46%** | **-27%** |

---

### 2.2 RM 从 HTTP 模式切换到 b2 inproc 模式

**优化内容**

HTTP 模式下，SIA 每个 token 都需要通过 TCP 调用远端 RM server，承担网络往返 + 序列化开销，且 HTTP server 与 SIA 运行在独立进程，无法共享 GPU 显存时序。

b2 inproc 模式将 RM 作为嵌入式 `vllm.LLM` 实例，在 SIA 的 EngineCore 子进程内直接调用，消除全部网络开销，RM 与 LLM 共享同一 GPU。

**实测数据**

| 模型 | HTTP RM 端到端 SIA | b2 inproc SIA | 加速比 |
|------|:---:|:---:|:---:|
| VL-30B（AlpacaEval 200Q）| ~40 tok/s | 59.2 tok/s | **~1.5×** |
| 0GM-35B（AlpacaEval 200Q）| ~32 tok/s | ~37.8 tok/s | **~1.2×** |

> 注：0GM-35B 的 b2 inproc 起点偏低（37.8 tok/s），是因为彼时 `SIA_LLM_CUDAGRAPH=piecewise` 配置仍在，主 LLM 速度被人为压低。该问题由下一项优化修复。

**b2 inproc 为何减少延迟**

HTTP 模式每次打分经过：Python HTTP client → TCP → HTTP server → vLLM 内部队列 → RM forward → HTTP 响应 → Python 解析。b2 inproc 将整条链缩减为：直接 `llm.generate()` 调用 → RM forward，往返延迟从 ~70ms 降至 ~11-30ms。

---

### 2.3 主 LLM CUDA Graph：去掉 PIECEWISE-only 限制

**优化内容**

早期配置了 `SIA_LLM_CUDAGRAPH=piecewise`，将主 LLM 强制限定在 PIECEWISE 模式。该模式下 vllm 不使用 FULL graph（整个 model forward 捕获为一张完整 CUDA graph），decode 步的每个 kernel 仍需 CPU 逐一调度，GPU 流水线无法连续。

去掉此环境变量后，vllm 默认使用 FULL_AND_PIECEWISE：decode 步走完整 FULL graph，CPU 只需 launch 一次即可驱动完整 forward pass，kernel 之间无 CPU 介入，GPU 可做跨 kernel 的流水线调度。

**实测数据**

| 模型 | 配置 | noSIA | SIA |
|------|------|:---:|:---:|
| VL-30B | 旧（piecewise only）| 80.8 tok/s | 59.2 tok/s |
| VL-30B | 新（FULL_AND_PIECEWISE）| **122.8 tok/s (+52%)** | **78.3 tok/s (+32%)** |
| 0GM-35B | 旧（piecewise only）| ~57 tok/s | 37.8 tok/s |
| 0GM-35B | 新（FULL_AND_PIECEWISE）| **107.0 tok/s (+88%)** | **54.1 tok/s (+43%)** |

**为何 FULL graph 更快**

每次 decode step，模型有数百至数千个 CUDA kernel（matmul、layernorm、rope、attention 等，64 层 × N 个算子）。eager 模式下 CPU 每个 kernel 各自 launch，每次 launch 有 ~5–10 μs 的调度延迟，累积起来不可忽视。FULL graph 将整个 forward 录制后，replay 时 CPU 只发一个信号，GPU 按预录序列连续执行全部 kernel，中间无停顿，且 GPU command processor 可预取下一个 kernel 的参数，实现无缝流水。

---

## 三、VL-30B 独有的优化

### 3.1 RM 也使用 CUDA Graph（SIA_RM_CUDAGRAPH=piecewise）

**优化内容**

VL-30B 的 RM（VM-Qwen3-4B）同样使用 PIECEWISE CUDA graph 模式，RM forward 单次调用从 ~71ms 降至 ~11ms（6.4×）。

**原理**

vllm 0.17.1 的 PIECEWISE 走 AOT（Ahead-of-Time）capture 路径：在模型初始化阶段一次性录制所有常见 batch shape 的 CUDA graph，推理时只需查表 replay，不会在推理期间触发新的 capture。这使得 RM 的 forward 也完全走 CUDA graph，消除了 eager 模式的 kernel dispatch overhead。

**为何 0GM-35B 不能直接复用此优化**

0GM-35B 使用 vllm 0.18.0，该版本的 PIECEWISE capture 行为与 0.17.1 不同，存在两个问题：

1. **前缀缓存触发 runtime capture**：开启 APC（Automatic Prefix Caching）后，推理过程中出现 warmup 阶段未见过的新 `batch_descriptor`，vllm 试图在推理时动态 capture 新 graph → 触发 `RuntimeError: CUDA graph capturing detected at an inappropriate time`。

2. **RM 是 prefill-heavy 工作负载**：RM 每步处理 topk=10 条完整序列（无 decode 阶段，序列长度随步骤线性增长）。CUDA graph 的加速主要针对 decode 阶段（固定 token 数量的小 batch）；对于 prefill（序列长度不固定的大 batch），PIECEWISE 帮助有限。实测 0GM-35B RM 在 piecewise 模式下比 eager 模式慢 2-3×（115–140ms vs 30ms），已放弃。

---

## 四、0GM-35B 独有的优化

### 4.1 Stable Prefix 跨分词器优化

**背景：为何存在跨分词器问题**

VL-30B 和 RM 使用相同的 Qwen3 分词器（151K 词表），RM token ID 序列与 LLM 完全一致，APC 可直接复用 KV cache。

0GM-35B 使用 Qwen3.5 分词器（248K 词表），而 RM 仍使用 Qwen3 分词器（151K 词表）。每次 RM 打分必须经过：LLM token IDs → 解码为文本 → 用 RM 分词器重新编码 → RM token IDs。

**旧代码的问题**

旧代码对每个候选 token 分别编码完整序列（前缀文本 + 候选文本）：

```python
for 候选 in [A, B, ..., J]:          # topk=10 个候选
    full_text = prefix_text + cand_text
    rm_ids = rm_tok.encode(full_text)  # 整段重新编码
```

BPE 分词在拼接边界处会对常见字符组合做 merge（如 "quick" + "ly" → "quickly"，1 个 token）。由于 10 个候选的 `cand_text` 各不相同，边界 merge 的结果也各不相同，导致 10 条序列的前缀 RM token ID 序列在末尾 1-2 个 token 处产生差异。

vllm APC 按每 16 个 token 一个 block 计算 hash 决定是否复用 KV。前缀末尾 token 不同 → 最后一个 block 的 hash 在 10 个候选上各不相同 → 该 block 对每个候选都 cache miss，重算 10 次，而非共享 1 次。随序列增长，每步需重算的 token 数线性增加，RM 调用延迟从 34ms（短序列）增长到 55ms（长序列）。

**优化方案**

将前缀编码与候选编码分离：

```python
stable_rm_prefix_ids = rm_tok.encode(prefix_text)  # 前缀只编码一次

for 候选 in [A, B, ..., J]:
    cand_rm_ids = rm_tok.encode(cand_text)          # 只编码候选 suffix
    rm_ids = stable_rm_prefix_ids + cand_rm_ids     # 直接拼接 ID
```

10 个候选共享完全相同的前缀 token ID 序列，所有前缀 block 的 hash 完全一致，APC 命中率接近 100%，RM 每步只需 forward ~20 个 suffix token，而非 ~600+ 个完整序列。

**实测数据**

| 指标 | 优化前 | 优化后 |
|------|:---:|:---:|
| b2_score_call p50（短序列）| 34 ms | 30 ms |
| b2_score_call p50（长序列）| ~55 ms（持续增长）| **~30 ms（稳定不增长）** |
| SIA tok/s | 54.1 | **65.2（+20%）** |

**为何 VL-30B 不需要此优化**

VL-30B 的 LLM 和 RM 使用相同分词器，token ID 可直接传递，无跨分词器转换，不存在 BPE 边界 merge 问题，APC 天然 100% 命中前缀。

---

## 五、后续可能的优化方向

以下方向尚未实施，不提供加速比预估。

**通用方向（两个模型均适用）**

- **RM 并行化**：在主 LLM forward 期间同时准备 RM 输入（独立 CUDA stream），减少 RM 等待时间
- **更小的 VM**：Qwen3-1.7B 替代 Qwen3-4B 作为 Value Model，减少 RM forward 计算量
- **RM FP8 量化**：减少 RM 显存占用和推理延迟（需验证质量无回归）

**0GM-35B 专属方向**

- **训练与 0GM-35B 相同词表的 VM**：从根本消除跨分词器转换开销，b2_score_call 延迟有望接近 VL-30B 的 11ms 水平

---

## Appendix：小优化清单

以下优化已实施，单项提升 ≤5%，不单独展开。

| 优化项 | 适用模型 | 说明 | 加速比（端到端）|
|-------|---------|------|:---:|
| `SIA_RM_MULTIPROCESS=0`（inproc 单进程）| 两者 | 取消 RM 的 multiprocessing executor，减少 ZMQ IPC 开销 | ~+3–5% |
| RM APC 默认开启 | 两者 | b2 inproc 模式下保持 RM prefix caching 开启 | ~+2–4% |
| `max_num_batched_tokens` 修复（0GM-35B）| 0GM-35B | 防止 topk×length 超限导致 chunked prefill 错误 reward，消除每步跳过干预的 bug | 间接（修 bug）|
| WeakSet 隔离修复（0GM-35B）| 0GM-35B | 防止主 LLM profiling 阶段 `clear_all_graphs()` 清空 RM CUDA graph，消除 RuntimeError | 间接（修 bug）|
