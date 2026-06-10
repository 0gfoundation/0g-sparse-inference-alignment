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

**为什么 PyTorch 方案慢**

SIA 每生成一个 token，都要调用 RM 对 topk=10 个候选打分。原始方案直接用 HuggingFace `AutoModel` 加载 RM，每次打分就调用一次 `model.forward()`。

PyTorch 的 `forward()` 没有任何记忆——每次调用都从第 1 个 token 算到最后，完整计算整条序列的 KV（键值向量，Transformer attention 计算的中间结果，下文§4.1有详细解释）。假设目前已生成 300 个 token：

- **第 300 步**：PyTorch 对 10 个候选各算一次，每次把前 300 个 token 的 KV 全算一遍。前 300 个 token 的 KV 在 10 个候选里完全相同，却被重算了 10 次。
- **第 301 步**：前 301 个 token 的 KV 又从头算一遍，完全不记得第 300 步刚算过。

**vLLM 如何解决**

vLLM 有两个关键机制：

1. **APC（Automatic Prefix Caching，自动前缀缓存）**：模型 forward 时，每个 token 位置计算出的 KV 会被缓存下来。只要某段前缀的 token ID 序列没有变化，这段 KV 就永远不需要重算——无论是同一步的 10 个候选共享，还是下一步沿用上一步的缓存，都是同一套机制。

   同样是第 300 步打分：vLLM 发现前 300 个 token 的前缀 KV 上一步已经算过，直接从缓存读取。10 个候选只需各自计算自己那 1-2 个候选 token 的新 KV。前缀越长，这个节省越显著。

2. **动态 batching**：10 个候选在一次 GPU forward 里并行处理，而不是串行。

**实测数据（RM 单次调用延迟）**

| 后端 | 序列 <200 token | 序列 200–500 token | 序列 1000+ token |
|------|:---:|:---:|:---:|
| PyTorch 原始 | ~78 ms | ~78 ms | ~78 ms |
| vLLM HTTP | ~34 ms | ~42 ms | ~57 ms |
| **降幅** | **-56%** | **-46%** | **-27%** |

> 序列越长，vLLM 的绝对延迟也在上涨（34 ms → 57 ms），原因：即使命中缓存，attention 计算仍要读取所有已缓存的 KV，这部分 IO 随序列增长。PyTorch 的开销同样在增长，只是表中区间的测量点恰好显示 PyTorch 相对更"平稳"（它每次全算，没有缓存带来的额外管理开销）。

---

### 2.2 RM 从 HTTP 模式切换到 b2 inproc 模式

**为什么 HTTP 方案还是慢**

vLLM HTTP 模式下，RM 运行在一个独立的服务进程里，SIA 每次打分都需要发出 HTTP 请求：

```
SIA（进程A）→ TCP网络 → RM server（进程B）→ GPU计算 → TCP网络 → SIA
```

每次往返多出：序列化 JSON、TCP 收发、HTTP 解析，加起来约 20–30 ms 固定开销，与序列长短无关。

**b2 inproc 方案**

将 RM 直接嵌入 SIA 所在的进程内，变成一个内嵌的 vLLM 实例，打分变成函数调用：

```
SIA → 直接调用 RM.generate() → GPU计算 → 返回结果
```

网络往返全部消除。

**实测数据**

| 模型 | HTTP RM（SIA 端到端）| b2 inproc（SIA 端到端）| 加速比 |
|------|:---:|:---:|:---:|
| VL-30B（AlpacaEval 200Q）| ~40 tok/s | 59.2 tok/s | **~1.5×** |
| 0GM-35B（AlpacaEval 200Q）| ~32 tok/s | ~37.8 tok/s | **~1.2×** |

> 0GM-35B 的 b2 inproc 起点（37.8 tok/s）偏低，是因为当时还有另一个配置错误压制了主 LLM 速度（见下一节）。

---

### 2.3 主 LLM CUDA Graph：去掉 PIECEWISE-only 限制

**背景：GPU 执行模型需要 CPU 不断"发号施令"**

神经网络的 forward pass 由数百个 GPU 计算函数组成（每层的矩阵乘法、归一化、注意力……），每个函数称为一个 **kernel**。在普通（eager）模式下，CPU 要一个接一个地通知 GPU"现在执行这个 kernel"，每次通知本身有约 5–10 μs 的延迟。一个 35B 模型有 64 层，每层十余个 kernel，一次 decode step 累积下来 CPU 调度开销可达数毫秒。

**CUDA Graph 的思路**

CUDA Graph 相当于**排练 + 演出**：先走一遍 forward，把"执行哪些 kernel、按什么顺序、传什么参数"全部录制成一张"乐谱"（graph）。之后每次 decode step，CPU 只发一次"按谱演奏"的信号，GPU 自己按照乐谱从头跑到尾，中间不需要 CPU 再介入。

效果：
- **CPU 调度开销**：从每步数百次 kernel launch → 1 次 graph replay
- **GPU 流水线**：GPU command processor 拿到完整乐谱后，可以在执行当前 kernel 的同时预先加载下一个 kernel 的参数，实现无缝流水

**早期配置错误：强制 PIECEWISE-only**

早期代码设置了 `SIA_LLM_CUDAGRAPH=piecewise`，将主 LLM 锁定在 PIECEWISE 模式。该模式只对模型的部分算子（非 attention 部分）做 CUDA graph，**attention 仍走 eager 逐 kernel 执行**，而 attention 恰恰是计算量最大的部分。

去掉这个限制后，vllm 默认使用 FULL_AND_PIECEWISE：decode 步走完整 FULL graph，attention 也被录入图中，一次 replay 驱动整个 forward。

**实测数据**

| 模型 | 配置 | noSIA | SIA |
|------|------|:---:|:---:|
| VL-30B | 旧（PIECEWISE only）| 80.8 tok/s | 59.2 tok/s |
| VL-30B | 新（FULL_AND_PIECEWISE）| **122.8 tok/s（+52%）** | **78.3 tok/s（+32%）** |
| 0GM-35B | 旧（PIECEWISE only）| ~57 tok/s | 37.8 tok/s |
| 0GM-35B | 新（FULL_AND_PIECEWISE）| **107.0 tok/s（+88%）** | **54.1 tok/s（+43%）** |

noSIA 提升更大（+52–88%），是因为 noSIA 几乎全是 decode step，100% 受益于 FULL graph；SIA 每步还需等待 RM 打分，RM 耗时部分抵消了 LLM 加速的收益。

---

## 三、VL-30B 独有的优化

### 3.1 RM 也使用 CUDA Graph

VL-30B 的 RM（VM-Qwen3-4B）使用 vllm 0.17.1，在**模型初始化阶段**一次性把所有常见 batch 大小（1 条、2 条……到 topk=10 条）都录制成 CUDA graph，推理时直接查表 replay。

效果：RM 单次调用从 ~71ms（eager）→ ~11ms（CUDA graph），**6.4×**。

**为何 0GM-35B 不能复用**

0GM-35B 使用 vllm 0.18.0，尝试给 RM 也开启 CUDA graph，遇到了两个独立的问题：

**问题一：vllm 0.18.0 遇到未预录形状时崩溃**

vllm 初始化时会提前录制一批"固定形状"的 CUDA graph：比如 1 条序列各 forward N 个 token、2 条序列各 forward M 个 token……覆盖几组常见的（序列数 × 每条序列新 token 数）组合。运行时遇到匹配的形状，直接 replay；遇到没有录过的形状，vllm 0.18.0 会尝试**临时录制新 graph**，但录制只能在初始化阶段进行，运行时条件已不满足，因此抛出 RuntimeError。

vllm 0.17.1（VL-30B 所用版本）在遇到未预录的形状时，选择**直接退回 eager 逐 kernel 执行**，而不是尝试临时录制，所以 VL-30B 的 RM 不会崩溃，CUDA graph 也能正常发挥作用。

**问题二：RM 的工作模式与 CUDA graph 不匹配**

LLM 的 decode 阶段每步只生成 1 个 token，每次 forward 的 batch 大小固定（topk 个序列，每个只新增 1 个 token），非常适合 CUDA graph（固定形状，反复 replay）。

0GM-35B 的 RM 情况不同：每步需要对 topk=10 条**完整序列**（从第 1 个 token 到当前最新 token）做完整的 prefill forward，序列长度随着生成推进线性增长（从几十 token 增长到几百 token）。这叫做 **prefill-heavy** 工作模式——每次 forward 的序列长度都不一样，无法用固定形状的 CUDA graph 覆盖。实测 CUDA graph 模式比 eager 模式还慢 2-3×（115–140ms vs 30ms），已放弃。

---

## 四、0GM-35B 独有的优化

### 4.1 Stable Prefix 跨分词器优化

**先理解三个概念**

**① 分词器（Tokenizer）**：模型不直接处理文字，而是把文字切成"词片"（token），每个词片对应一个整数 ID。不同模型用不同的切法，同一段文字在不同分词器下会切出不同的 ID 序列。

**② KV Cache**：Transformer 在处理每个 token 时，会为它计算"键（K）"和"值（V）"两个向量，并把结果存起来。之后处理更长的序列时，前面已算过的 token 的 KV 可以直接读取，不用重算。这份缓存叫 KV Cache。

**③ APC（Automatic Prefix Caching）**：§2.1 已介绍原理（前缀 token ID 相同则共享 KV Cache）。关键实现细节：vllm 以每 **16 个 token** 为一个 block 计算 hash，block 内只要有 1 个 token ID 不同，整块 KV 都不能复用。

---

**为什么 0GM-35B 存在跨分词器问题**

VL-30B 和 RM 用的是同一套 Qwen3 分词器（151K 词表），LLM 生成的 token ID 可以直接喂给 RM，APC 自然生效。

0GM-35B 用的是 Qwen3.5 分词器（248K 词表），RM 用的是 Qwen3 分词器（151K 词表）。两套分词器对同一段文字的切法不同，token ID 完全不同。所以每次 RM 打分，都必须把 LLM 的 token ID 先解码回文字，再用 RM 的分词器重新切一遍：

```
LLM 生成的 token IDs  →  解码为文字  →  RM 分词器重新编码  →  RM token IDs
```

---

**旧代码的问题：前缀 KV Cache 被迫重算 10 次**

旧代码每个候选各自做一次完整的"文字 → RM token IDs"转换：

```python
for 候选 in [A, B, ..., J]:          # 10 个候选
    full_text = prefix_text + cand_text   # "已生成内容" + "这个候选token的文字"
    rm_ids = rm_tok.encode(full_text)     # 整段重新切词
```

关键问题在切词边界。BPE 切词算法（两个模型都在用的分词方式）有一个特性：当两段文字拼在一起编码时，拼接处可能发生"合并"——比如前缀末尾是 "quick"，候选 A 是 "ly"，合并后编成 "quickly"（1 个 token）；候选 B 是 " fox"，不合并，前缀末尾的 "quick" 保持原样。

结果：10 个候选拼出的 10 条序列，前缀部分大多相同，但**末尾 1-2 个 token 因为合并结果不同而各异**。

APC 以 16 个 token 为一个 block 做 hash。前缀末尾那个 block（包含受合并影响的 token）在 10 个候选里各有不同的 hash → **该 block 对 10 个候选全部 cache miss，各自重算一次**，即使它们 99% 的内容是相同的前缀。

随着生成推进，前缀越来越长，每步需要重算的 token 数也线性增加，RM 延迟从 34ms（短序列）一路涨到 55ms（长序列）。

---

**新方案：前缀只切一次，候选分开切**

```python
stable_rm_prefix_ids = rm_tok.encode(prefix_text)  # 前缀单独切，只切一次

for 候选 in [A, B, ..., J]:
    cand_rm_ids = rm_tok.encode(cand_text)           # 只切候选那 1-2 个 token
    rm_ids = stable_rm_prefix_ids + cand_rm_ids      # 直接拼 ID，不再整段重切
```

前缀不参与拼接再切词，10 个候选的 `stable_rm_prefix_ids` **完全相同，一个 ID 不差**。APC 看到 10 条序列有相同的前缀 hash，前缀 KV 只计算一次，剩余 9 个候选直接复用。每步 RM 实际只 forward 10 × 1-2 个候选 token，而非 10 条完整序列。

---

**实测数据**

| 指标 | 优化前 | 优化后 |
|------|:---:|:---:|
| RM 调用延迟（序列短）| 34 ms | 30 ms |
| RM 调用延迟（序列长）| ~55 ms（持续增长）| **~30 ms（稳定不增长）** |
| SIA tok/s | 54.1 | **65.2（+20%）** |

新方案延迟稳定不增长——APC 始终命中，每步 RM 工作量恒定。

**为何 VL-30B 不需要此优化**

VL-30B 的 LLM 和 RM 使用同一套分词器，LLM token ID 可以直接传给 RM，不存在跨分词器转换，也就没有 BPE 边界合并问题，APC 天然 100% 命中前缀。

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
