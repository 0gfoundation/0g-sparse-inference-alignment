# SIA 推理加速优化总结

**适用模型**：Qwen3-VL-30B-A3B-Instruct、0GM-1.0-35B-A3B  
**Value Model**：VM-Qwen3-4B（两个模型共用）  
**撰写日期**：2026-06-10

---

## 术语说明

| 术语 | 含义 |
|------|------|
| **SIA** | Sparse Inference-time Alignment。每生成一个 token，用 VM 对候选打分并干预 logit 分布，使输出偏向更高奖励的方向。`--entropy_threshold` 控制只在高熵（模型不确定）的步骤才干预，降低计算开销。 |
| **VM（Value Model）** | 对候选 token 打分的小模型（本项目用 VM-Qwen3-4B），估计"沿此路径继续生成"的期望奖励。CLI 参数写作 `--rm` / `--rm_backend` 等，系历史命名遗留，含义相同。 |
| **noSIA** | 不启用 SIA 干预，主模型正常生成。"纯 vLLM noSIA"指不加载 VM 的原生 vLLM 推理，是速度上限基线。 |
| **b2 inproc** | 将 VM 嵌入主模型所在进程，打分变成进程内函数调用，消除 HTTP 网络往返开销。 |
| **APC** | Automatic Prefix Caching，vllm 的 KV 自动前缀缓存。前缀 token ID 相同时 KV 不重算，直接复用。以 16 token 为一个 block 计算 hash，block 内有 1 个 ID 不同则整块不能复用。 |
| **CUDA Graph** | GPU 执行计划。首次 dry-run 把"执行哪些 kernel、按什么顺序"编译成固定计划，之后每步只提交一次"按此计划执行"，消除 CPU 逐 kernel dispatch 开销。 |
| **topk** | 每个 decode step 送给 VM 打分的候选 token 数量（默认 10）。 |
| **entropy_threshold** | 熵阈值，低于此值说明主模型已很确信，跳过 VM 打分，只有高熵的"关键决策点"才干预。 |

---

## 一、速度基线

| 模型 | 纯 vLLM（无 VM）| SIA 起点（最初）| SIA 终点（当前）|
|------|:---:|:---:|:---:|
| VL-30B | 127.4 tok/s | ~40 tok/s | **78.3 tok/s** |
| 0GM-35B | 114.1 tok/s | ~32 tok/s | **66–68 tok/s** |

"纯 vLLM（无 VM）"为单独跑 `vllm serve`、不加载 Value Model 的实测吞吐量，是真正的上限基线。"SIA 起点"为 HTTP VM 后端下的实测吞吐量（两个模型最早的可比基准）。"终点"为当前生产最优配置下的实测值。

---

## 二、两个模型共用的优化

### 2.1 VM 后端从 PyTorch 迁移到 vLLM（HTTP 模式）

**为什么 PyTorch 方案慢**（详细分析见 [sia-original-implementation-efficiency-analysis.md](sia-original-implementation-efficiency-analysis.md)）

SIA 每生成一个 token，都要调用 VM 对 topk=10 个候选打分。原始方案直接用 HuggingFace `AutoModel` 加载 VM，每次打分就调用一次 `model.forward()`。

PyTorch 的 `forward()` 没有任何记忆——每次调用都从第 1 个 token 算到最后，完整计算整条序列的 KV（键值向量，Transformer attention 计算的中间结果，下文§4.1有详细解释）。假设目前已生成 300 个 token：

- **第 300 步**：PyTorch 对 10 个候选各算一次，每次把前 300 个 token 的 KV 全算一遍。前 300 个 token 的 KV 在 10 个候选里完全相同，却被重算了 10 次。
- **第 301 步**：前 301 个 token 的 KV 又从头算一遍，完全不记得第 300 步刚算过。

**vLLM 如何解决**

vLLM 有两个关键机制：

1. **APC（Automatic Prefix Caching，自动前缀缓存）**：模型 forward 时，每个 token 位置计算出的 KV 会被缓存下来。只要某段前缀的 token ID 序列没有变化，这段 KV 就永远不需要重算——无论是同一步的 10 个候选共享，还是下一步沿用上一步的缓存，都是同一套机制。

   同样是第 300 步打分：vLLM 发现前 300 个 token 的前缀 KV 上一步已经算过，直接从缓存读取。10 个候选只需各自计算自己那 1 个候选 token 的新 KV。前缀越长，这个节省越显著。

2. **动态 batching**：10 个候选在一次 GPU forward 里并行处理，而不是串行。

---

### 2.2 VM 从 HTTP 模式切换到 b2 inproc 模式

**为什么 HTTP 方案还是慢**

vLLM HTTP 模式下，VM 运行在一个独立的服务进程里，SIA 每次打分都需要发出 HTTP 请求：

```
SIA（进程A）→ TCP网络 → VM server（进程B）→ GPU计算 → TCP网络 → SIA
```

每次往返多出：序列化 JSON、TCP 收发、HTTP 解析，加起来约 20–30 ms 固定开销，与序列长短无关。

**b2 inproc 方案**

将 VM 直接嵌入 SIA 所在的进程内，变成一个内嵌的 vLLM 实例，打分变成函数调用：

```
SIA → 直接调用 VM.generate() → GPU计算 → 返回结果
```

网络往返全部消除。

**实测数据**（[VL-30B 详细报告](alpaca-eval-vl30b-b2-docker-20260608.md) · [0GM-35B 详细报告](0gm-35b-alpacaeval-200q-b2-inproc-thinking-20260609.md)）

| 模型 | HTTP VM（SIA 端到端）| b2 inproc（SIA 端到端）| 加速比 |
|------|:---:|:---:|:---:|
| VL-30B（AlpacaEval 200Q）| ~36 tok/s | 59.2 tok/s | **~1.6×** |
| 0GM-35B（AlpacaEval 200Q）| ~32 tok/s | ~37.8 tok/s | **~1.2×** |

> 0GM-35B 的 b2 inproc 起点（37.8 tok/s）偏低，是因为当时还有另一个配置错误压制了主 LLM 速度（见下一节）。

---

### 2.3 主 LLM CUDA Graph：去掉 PIECEWISE-only 限制

**背景：GPU 执行模型需要 CPU 不断"发号施令"**

神经网络的 forward pass 由数百个 GPU 计算函数组成（每层的矩阵乘法、归一化、注意力……），每个函数称为一个 **kernel**。在普通（eager）模式下，CPU 要一个接一个地通知 GPU"现在执行这个 kernel"，每次通知本身有约 5–10 μs 的延迟。一个 35B 模型有 64 层，每层十余个 kernel，一次 decode step 累积下来 CPU 调度开销可达数毫秒。

**CUDA Graph 的思路**

CUDA Graph 相当于数据库的**执行计划**：先 dry-run 一遍 forward，把"执行哪些 kernel、按什么顺序、传什么参数"编译成一份固定的执行计划（graph）。之后每次 decode step，CPU 只提交一次"按此计划执行"的指令，GPU 自己按计划跑完整个 forward，中间不需要 CPU 再介入。

效果：
- **CPU 调度开销**：从每步数百次 kernel launch → 1 次 graph replay
- **GPU 流水线**：GPU command processor 拿到完整执行计划后，可以在执行当前 kernel 的同时预先加载下一个 kernel 的参数，实现无缝流水

**早期配置错误：强制 PIECEWISE-only**

早期代码设置了 `SIA_LLM_CUDAGRAPH=piecewise`，将主 LLM 锁定在 PIECEWISE 模式。该模式只对模型的部分算子（非 attention 部分）做 CUDA graph，**attention 仍走 eager 逐 kernel 执行**，而 attention 恰恰是计算量最大的部分。

去掉这个限制后，vllm 默认使用 FULL_AND_PIECEWISE：decode 步走完整 FULL graph，attention 也被录入图中，一次 replay 驱动整个 forward。

**实测数据**（[VL-30B 详细报告](alpaca-eval-vl30b-b2-docker-20260608.md) · [0GM-35B 详细报告](0gm-35b-alpacaeval-200q-b2-inproc-thinking-20260609.md)）

| 模型 | 配置 | SIA（tok/s）|
|------|------|:---:|
| VL-30B | 旧（PIECEWISE only）| 59.2 |
| VL-30B | 新（FULL_AND_PIECEWISE）| **78.3（+32%）** |
| 0GM-35B | 旧（PIECEWISE only）| 37.8 |
| 0GM-35B | 新（FULL_AND_PIECEWISE）| **54.1（+43%）** |

---

## 三、VL-30B 独有的优化

### 3.1 VM 也使用 CUDA Graph

VL-30B 的 VM（VM-Qwen3-4B）使用 vllm 0.17.1，在**模型初始化阶段**一次性把所有常见 batch 大小（1 条、2 条……到 topk=10 条）都录制成 CUDA graph，推理时直接查表 replay。

效果：VM 单次调用从 ~71ms（eager）→ ~11ms（CUDA graph），**6.4×**。（[详细报告](vl30b-b2-inproc-speedup-20260605.md)）

> **注**：~11ms 为该报告中短序列场景的测量值。AlpacaEval 长序列实验（§五）的稳态 b2_score_call p50 约 17ms——序列越长、每步候选文字越多，VM forward 时间略有增加。

**为何 0GM-35B 不能复用**：见 [Appendix B](#appendix-b0gm-35b-vm-无法使用-cuda-graph-的原因)。

---

## 四、0GM-35B 独有的优化

### 4.1 Stable Prefix 跨分词器优化

**先理解三个概念**

**① 分词器（Tokenizer）**：模型不直接处理文字，而是把文字切成"词片"（token），每个词片对应一个整数 ID。不同模型用不同的切法，同一段文字在不同分词器下会切出不同的 ID 序列。

**② KV Cache**：Transformer 在处理每个 token 时，会为它计算"键（K）"和"值（V）"两个向量，并把结果存起来。之后处理更长的序列时，前面已算过的 token 的 KV 可以直接读取，不用重算。这份缓存叫 KV Cache。

**③ APC（Automatic Prefix Caching）**：§2.1 已介绍原理（前缀 token ID 相同则共享 KV Cache）。关键实现细节：vllm 以每 **16 个 token** 为一个 block 计算 hash，block 内只要有 1 个 token ID 不同，整块 KV 都不能复用。

---

**为什么 0GM-35B 存在跨分词器问题**

VL-30B 和 VM 用的是同一套 Qwen3 分词器（151K 词表），LLM 生成的 token ID 可以直接喂给 VM，APC 自然生效。

0GM-35B 用的是 Qwen3.5 分词器（248K 词表），VM 用的是 Qwen3 分词器（151K 词表）。两套分词器对同一段文字的切法不同，token ID 完全不同。所以每次 VM 打分，都必须把 LLM 的 token ID 先解码回文字，再用 VM 的分词器重新切一遍：

```
LLM 生成的 token IDs  →  解码为文字  →  VM 分词器重新编码  →  VM token IDs
```

---

**旧代码的问题：前缀 KV Cache 被迫重算 10 次**

旧代码每个候选各自做一次完整的"文字 → VM token IDs"转换：

```python
for 候选 in [A, B, ..., J]:          # 10 个候选
    full_text = prefix_text + cand_text   # "已生成内容" + "这个候选token的文字"
    rm_ids = rm_tok.encode(full_text)     # 整段重新切词
```

关键问题在切词边界。BPE 切词算法（两个模型都在用的分词方式）有一个特性：当两段文字拼在一起编码时，拼接处可能发生"合并"——比如前缀末尾是 "quick"，候选 A 是 "ly"，合并后编成 "quickly"（1 个 token）；候选 B 是 " fox"，不合并，前缀末尾的 "quick" 保持原样。

结果：10 个候选拼出的 10 条序列，前缀部分大多相同，但**末尾 1-2 个 token 因为合并结果不同而各异**。

APC 以 16 个 token 为一个 block 做 hash。前缀末尾那个 block（包含受合并影响的 token）在 10 个候选里各有不同的 hash → **该 block 对 10 个候选全部 cache miss，各自重算一次**，即使它们 99% 的内容是相同的前缀。

随着生成推进，前缀越来越长，每步需要重算的 token 数也线性增加，VM 延迟从 34ms（短序列）一路涨到 55ms（长序列）。

---

**新方案：前缀只切一次，候选分开切**

```python
stable_rm_prefix_ids = rm_tok.encode(prefix_text)  # 前缀单独切，只切一次

for 候选 in [A, B, ..., J]:
    cand_rm_ids = rm_tok.encode(cand_text)           # 只切候选那 1-2 个 token
    rm_ids = stable_rm_prefix_ids + cand_rm_ids      # 直接拼 ID，不再整段重切
```

前缀不参与拼接再切词，10 个候选的 `stable_rm_prefix_ids` **完全相同，一个 ID 不差**。APC 看到 10 条序列有相同的前缀 hash，前缀 KV 只计算一次，剩余 9 个候选直接复用。每步 VM 实际只 forward 10 × 1-2 个候选 token，而非 10 条完整序列。

---

**实测数据**（[详细报告](0gm-35b-sia-perf-breakdown-20260609.md)）

| 指标 | 优化前 | 优化后 |
|------|:---:|:---:|
| VM 调用延迟（序列短）| 34 ms | 30 ms |
| VM 调用延迟（序列长）| ~55 ms（持续增长）| **~30 ms（稳定不增长）** |
| SIA tok/s | 54.1 | **66–68（+22%）** |

新方案延迟稳定不增长——APC 始终命中，每步 VM 工作量恒定。

**为何 VL-30B 不需要此优化**

VL-30B 的 LLM 和 VM 使用同一套分词器，LLM token ID 可以直接传给 VM，不存在跨分词器转换，也就没有 BPE 边界合并问题，APC 天然 100% 命中前缀。

---

## 五、当前优化态耗时分析

数据来源：`exp/alpaca-vl30b-b2-docker-20260610/docker_log.txt`（VL-30B）、`exp/alpaca-0gm35b-stable-prefix-20260610/server.log`（0GM-35B），均取稳态阶段 pf-summary 数据（步骤数 @15000–25000 / @5000–15000，各取 100 个采样点中位数）。

### 5.1 单步耗时分布

SIA 在每个 decode step 调用一次 `SIALogitsProcessor.apply()` 回调。下图展示每步各阶段的发生顺序（横轴为时间，纵轴为执行主体）：

```
── 非干预步（skip step，占绝大多数 token）────────────────────────────────

         t=0              t≈3.6ms  t≈4.1ms
          │                  │        │
LLM GPU  [████ FULL graph decode ████]·· (空闲，等 SIA 回调返回) ··
SIA CPU       [←─ cpu_sync ─→][topk/ent] → 回调返回，vLLM 提交下步
              │  等 GPU 完成  │ ~0.5ms
VM GPU   ──────────── 不调用 ────────────────────────────────────

              ←──── skip_step ~4.1ms ────►

── 干预步（intervened step，占 ~12%/20%）──────────────────────────────────────────────────────────

         t=0              t≈3.6ms  t≈4.1ms  t≈4.2ms         t≈4.2+17ms(VL)/30ms(0GM)   t末
          │                  │        │        │                       │                   │
LLM GPU  [████ FULL graph decode ████]·····LLM 空闲（等待 SIA 回调返回）·················[下步▶]
SIA CPU       [←─ cpu_sync ─→][topk/ent][prep][←──── b2_score_call（等 VM）────────────►][appl]
              │  等 GPU 完成  │ ~0.5ms  │0.07m│                                           │0.5m│
VM GPU   ─────────────────────────────────────[████ VM forward（APC 复用前缀 KV cache）████]
                                               ←──── b2_score_call ~17ms(VL)/~30ms(0GM) ──►
```

各指标含义说明（`pf-summary` 统计 p50，单位 ms）：

| 指标 | 说明 | VL-30B | 0GM-35B |
|------|------|:---:|:---:|
| **b2_score_call** | VM GPU 计算时间（含分词 + forward + 取分）；仅干预步调用 | **~17 ms** | **~30 ms** |
| **apply_cpu_sync** | SIA 回调开始时等待 LLM GPU 完成 FULL graph（即 LLM 单步 decode 耗时）；是 skip_step 的主体 | ~3.6 ms | ~3.9 ms |
| **intv_prepare** | 取出候选文字、组装 VM 请求对象（Python 侧，不含分词） | ~0.07 ms | ~0.07 ms |
| **intv_apply_logits** | 将 VM 分值写回 logit 分布 | ~0.46 ms | ~0.42 ms |
| **skip_step** | 非干预步 SIA 回调总耗时 = cpu_sync + topk/entropy；SIA 自身引入的额外计算（topk/entropy）仅 ~0.5ms | ~4.1 ms | ~4.3 ms |
| **干预率（intv_rate）** | 实际调用 VM 的步骤比例（以 token 计）| 11.7% | 20.1% |

**各实验实测吞吐**（同一实验组内 SIA vs noSIA 对比）：

| 模型 | noSIA（VM 加载，weight=0）| SIA | 比值 |
|------|:---:|:---:|:---:|
| VL-30B | 122.8 tok/s | 78.3 tok/s | **64%** |
| 0GM-35B | 112.3 tok/s | 66.2 tok/s | **59%** |

> **与§一基线的差值**：§一"纯 vLLM（无 VM）"为完全不加载 VM 的原生 vLLM 推理（VL-30B 127.4 / 0GM-35B 114.1 tok/s）。本表 noSIA 是 VM 已加载但 `weight=0`，SIA 回调仍在每步触发（含 apply_cpu_sync 等开销），因此比纯 vLLM 略低 3–5 tok/s。两者均可作为 SIA 的对比基准，§一选纯 vLLM 体现理论上限，本表选同组实验确保比较条件一致。

---

### 5.2 瓶颈分析

**① VM forward（b2_score_call）是干预步的主要开销**

b2_score_call 占干预步总 SIA 耗时的约 80%（VL-30B：17ms / ~21ms）和 87%（0GM-35B：30ms / ~34ms），是 SIA 性能瓶颈的核心。

以干预率加权后，每步平均 VM 开销为：

- VL-30B：11.7% × 17 ms ≈ **2.0 ms/step**
- 0GM-35B：20.1% × 30 ms ≈ **6.0 ms/step**

0GM-35B 干预率更高（20% vs 12%）且单次 VM 更慢（30ms vs 17ms），双重叠加导致其 SIA/noSIA 比值（59%）低于 VL-30B（64%）。

**② 为何 VL-30B b2_score_call 更短（~17ms vs ~30ms）**

VL-30B 的 VM 使用 vllm 0.17.1，开启了 CUDA graph（§3.1），VM forward 从 ~71ms（eager）降至约 11–17ms 水平。较低的 b2_score_call 值还受益于 VL-30B 与 VM 共用同一套 Qwen3 分词器——无需跨分词器转换，LLM token ID 直接传给 VM，节省约 2ms 的 CPU 编码时间。

0GM-35B 的 VM 无法使用 CUDA graph（见 Appendix B），只能逐 kernel 的 eager 执行。§4.1 的 stable prefix 优化已将 APC miss 引起的延迟波动消除，把长序列延迟从 ~55ms 稳定压回至 ~30ms。但相比 VL-30B 的 CUDA graph 路径，0GM-35B 的 VM forward 仍慢约 1.8×。

**③ apply_cpu_sync（~3.6–3.9ms）**

SIA 回调在每步开始时调用 `torch.cuda.synchronize()`，等待 LLM FULL graph 在 GPU 上执行完毕后，才能读取 logits。这约 3.6–3.9ms 的等待在两个模型上几乎相同，主要反映 LLM decode 的 GPU 执行时长——是 SIA 与 LLM 之间必要的时序屏障，并非 SIA 引入的额外计算。

**④ 非干预步开销（skip_step ~4.1–4.3ms）**

非干预步不调用 VM，只做 cpu_sync（~3.6ms）+ topk/entropy 计算（~0.46ms）。其中 cpu_sync 大半属于等待 GPU 完成 LLM forward 的"必要等待"，SIA 自身额外引入的计算仅 0.5ms 左右。

---

## 六、后续可能的优化方向

以下方向尚未实施，不提供加速比预估。

### 6.1 效果无损优化

纯工程改进，不改变候选打分逻辑，生成内容的 Reward 分数理论上不受影响。

- **VM 并行化**：在主 LLM forward 期间同时在独立 CUDA stream 上准备 VM 输入，将 intv_prepare 的 ~0.07ms 与 LLM decode 重叠执行，消除串行等待
- **VM FP8 量化**：将 VM 权重从 BF16 量化为 FP8，减少 GPU 显存占用和 forward 延迟；BF16→FP8 量化损耗极小，但仍需实验验证打分质量无回归

### 6.2 效果有损优化（速度–质量权衡）

以下参数调整或替换可提升速度，但会不同程度地影响 SIA 的干预质量，需根据实际任务评估可接受的权衡点。

- **减少 `--topk`**：每步候选从 10 降到 5 或 3，VM batch 缩小，b2_score_call 时间线性缩短；代价是候选池变小，可能错过更优 token
- **增大 `--entropy_threshold`**：提高跳过干预的熵阈值，降低干预率（当前 VL-30B 11.7%、0GM-35B 20.1%）；代价是部分高熵关键决策点被跳过，不再受 VM 引导
- **更小的 VM（Qwen3-1.7B）**：替代 Qwen3-4B，b2_score_call 显著缩短；代价是 VM 打分精度降低
- **VM INT8/INT4 量化**：比 FP8 更激进的量化，速度和显存收益更大；代价是 VM 打分精度损失明显，需要较充分的质量评估

### 6.3 模型专属方向

- **0GM-35B：训练与 0GM-35B 相同词表的 VM**：从根本消除跨分词器转换开销，并有望恢复 CUDA graph（参见 Appendix B），b2_score_call 延迟有望接近 VL-30B 当前实测的 ~17ms 水平；该方向属于效果无损的工程优化

---

## Appendix A：小优化清单

以下优化已实施，单项提升 ≤5%，不单独展开。

| 优化项 | 适用模型 | 说明 | 加速比（端到端）|
|-------|---------|------|:---:|
| `SIA_RM_MULTIPROCESS=0`（inproc 单进程）| 两者 | 取消 VM 的 multiprocessing executor，减少 ZMQ IPC 开销 | ~+3–5% |
| VM APC 默认开启 | 两者 | b2 inproc 模式下保持 VM prefix caching 开启 | ~+2–4% |
| `max_num_batched_tokens` 修复（0GM-35B）| 0GM-35B | 防止 topk×length 超限导致 chunked prefill 错误 reward，消除每步跳过干预的 bug | 间接（修 bug）|
| WeakSet 隔离修复（0GM-35B）| 0GM-35B | 防止主 LLM profiling 阶段 `clear_all_graphs()` 清空 VM CUDA graph，消除 RuntimeError | 间接（修 bug）|

---

## Appendix B：0GM-35B VM 无法使用 CUDA Graph 的原因

0GM-35B 使用 vllm 0.18.0，尝试给 VM 也开启 CUDA graph，遇到了两个独立的问题：

**问题一：vllm 0.18.0 遇到未预录形状时崩溃**

vllm 初始化时会提前录制一批"固定形状"的 CUDA graph：比如 1 条序列各 forward N 个 token、2 条序列各 forward M 个 token……覆盖几组常见的（序列数 × 每条序列新 token 数）组合。运行时遇到匹配的形状，直接 replay；遇到没有录过的形状，vllm 0.18.0 会尝试**临时录制新 graph**，但录制只能在初始化阶段进行，运行时条件已不满足，因此抛出 RuntimeError。

vllm 0.17.1（VL-30B 所用版本）在遇到未预录的形状时，选择**直接退回 eager 逐 kernel 执行**，而不是尝试临时录制，所以 VL-30B 的 VM 不会崩溃，CUDA graph 也能正常发挥作用。

**问题二：VM 的工作模式与 CUDA graph 不匹配**

LLM 的 decode 阶段每步只生成 1 个 token，每次 forward 的 batch 大小固定（topk 个序列，每个只新增 1 个 token），非常适合 CUDA graph（固定形状，反复 replay）。

0GM-35B 的 VM 情况不同：每步需要对 topk=10 条**完整序列**（从第 1 个 token 到当前最新 token）做完整的 prefill forward，序列长度随着生成推进线性增长（从几十 token 增长到几百 token）。这叫做 **prefill-heavy** 工作模式——每次 forward 的序列长度都不一样，无法用固定形状的 CUDA graph 覆盖。实测 CUDA graph 模式比 eager 模式还慢 2-3×（115–140ms vs 30ms），已放弃。
