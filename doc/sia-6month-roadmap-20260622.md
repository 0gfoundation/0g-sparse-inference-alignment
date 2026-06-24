# SIA Roadmap（2026-05 回顾 + 2026-07 ~ 2026-12 计划）

---

## Overview

### SIA 是什么

**SIA（Sparse Inference-time Alignment）**是一套在模型推理时实时干预输出的对齐系统。

通俗地说：大模型每生成一个词，SIA 在旁边同步评估候选词的"好坏"，把概率偏向质量更高的选项。全程不修改模型权重，仅在推理阶段介入——类似于给模型加了一个实时的"质检员"，每次出词前先过一遍。

### 当前成果（截至 2026-06-22）

两个月内完成核心系统，0GM-VL-35B 和 Qwen3-VL-30B 两个主力模型均已接近上线标准：

- **效果已验证**：0GM-VL-35B 开启 SIA 后，综合对话质量 AlpacaEval 胜率 **65.4%**（191 配对中 SIA 版本有 65.4% 被评为更优，125/191 对）；MMLU 准确率无明显下降
- **性能有代价**：单请求串行吞吐约为不开 SIA 的 **59–64%**（0GM-35B 59–61%，VL-30B 64%）；高并发场景（16 路并发）吞吐量约为不开 SIA 的 **35%**，即同等硬件可服务的请求量减少约 65%

### 三个核心痛点（当前状态）

| 痛点 | 当前状态 | 根因 |
|------|---------|------|
| **效果高度依赖配置** | 0GM-35B natural thinking：**win rate 65.4%（191 配对），Δ=+4.05**；0GM-35B ban_think：**win rate 54%，Δ=+0.72（不显著）**；VL-30B：**win rate 63%（122/195），Δ=+4.2%~+8.3%，两轮均显著**；Qwen3-14B 同家族：**+13.2%（复现论文）**。MMLU：same-family（VL-30B）无明显下降；0GM-35B 跨家族下降 **−5.5 至 −12.2pp** | **① VM 代际/家族不匹配（主因）**：VM-4B 基于 Qwen3 base（训练 cutoff 2025-04），0GM-35B 为 2026 年新模型，能力差距 W2S=8.75×、时间差 8–12 个月、词表差 248K vs 151K；三组对比结果严格单调（W2S 越大效果越差）。**② VM 分布外用法**：VM 训练任务为完整答案对偏好评分，SIA 实际任务为 partial response + 1 token 的 step-level value prediction，分布不匹配导致打分信号噪声高。**③ 跨分词器 BPE 边界噪声**：0GM-35B 词表 248K vs VM 词表 151K，stable prefix 优化前 APC miss 导致打分不稳定（stable prefix 已部分解决）。④ VM 训练数据量/质量未充分验证 |
| **VM 延迟高** | **0GM-35B ~30ms/call**（stable prefix 优化后，之前高达 34–50ms 且随序列增长）；**VL-30B 11ms/call**（CUDA graph 开启） | **① 无 CUDA graph（最大贡献）**：vllm 0.18.0 存在 WeakSet bug，主 LLM profiling 阶段 `clear_all_graphs()` 清除了 RM 的 CUDA graph，导致运行时崩溃，0GM-35B 被迫走 eager 路径；VL-30B 上 CUDA graph 使 VM 延迟 71ms→11ms（6.4×）。**② 跨分词器 BPE 边界开销（0GM-35B 特有）**：每步需 decode+re-encode，加上 eager dispatch overhead，合计约 7ms gap；是 0GM-35B vs VL-30B 延迟差异的第二大来源。**③ small-batch memory-bound**：每次 VM forward batch size 极小（topk×1–2 token），但需读全部 ~8GB 权重，GPU 处于 memory-bound 状态 |
| **高并发吞吐损失** | 批量打分优化后：conc=16 SIA tok/s 369，**35% of noSIA**。完整并发扫描（35B）：conc=4→48%、8→42%、16→35%。**30B 对照**：conc=16 时 66%（vs 35B 的 35%），差距源于跨分词器使 0GM-35B 单次 VM 调用 ~3ms vs 30B 的 ~1.5ms，串行 16 次时累积差距 2× | 当前三个瓶颈（conc=16 ITL 3.0×，41.3ms vs noSIA 13.6ms）：① **APC partial tail compute ~15ms/batch（最大贡献，结构性不可消除）**：每次 VM batch forward 的候选前缀仅 ~15 token，不满一个 APC block（16 token），尾部碎片每步必须完整 prefill；② **无 CUDA graph ~5ms/call（可修复）**：vllm 0.18.0 WeakSet bug 导致 VM 走 eager 路径，每 batch step 有 kernel dispatch overhead；③ **跨分词器 CPU 编码 ~0.5ms**（P-3 优化后已轻微）。其中 ~15ms 为结构性下限，~5ms 可通过 CUDA graph 修复 |

<sub>后续用字母简称：**E** = 效果高度依赖配置；**L** = VM 延迟高；**T** = 高并发吞吐损失。</sub>

### 两个阶段的工作重心

过去两个月属于**"工程对齐"阶段**：以 NTU SIA 原论文（arxiv 2602.21215）为基准，**刻意不改动任何算法和模型**——干预逻辑、VM 结构、训练权重完全保持原样——目的是先在生产环境复现论文声称的效果，再通过纯工程手段（b2 inproc 架构、用 vLLM 代替低速的 PyTorch、批量打分合并）消除部署开销。这样做保证了任何效果变化都可归因于工程实现，而非算法调整。高并发吞吐从基线提升 54%，优化风险极低、收益确定、结果可直接用吞吐数字衡量。

接下来 6 个月进入**"跳出原论文"阶段**：工程对齐已完成，接下来要**主动突破 SIA 原始论文的边界**，大胆尝试算法改动（乘积式融合、block-wise scoring、accept/reject 采样）、模型改进（同词表 VM 重训、更强 base model、更大训练数据集）和训练数据增强（步骤级标注、多模态偏好对）。这些方向**每一项都会对效果产生影响**，结果有好有坏，不能保证优于现状。因此后续计划的交付标准更多是"A/B 有结论"而非"数字必须达标"，里程碑也相应保留了保守/乐观两个场景。

### 接下来 6 个月的计划

计划分三个阶段，核心目标是将 SIA 的性能代价从"显著"降低到"可忽略"，同时持续提升对齐效果。

**阶段一（Month 1）：减少打分频率，补建 GPT-4 评估基准**  
通过智能跳过"不重要的 token"，将评分模型（Value Model）的调用次数降至原来的 1/4（减少约 75%，从 ~20% 干预率降至 ~5%），在几乎不损失对齐效果的前提下大幅降低开销；同时在现有 Skywork judge 基准的基础上补建 GPT-4 标准评估基准，为后续所有改进提供与业界可比的量化依据。

**阶段二（Month 2–3）：换一个更快的评分模型**  
当前评分模型与主模型使用不同词表，每次打分都需要额外的词表重编码（CPU 编码 ~2ms + eager 调度 ~5ms，合计约 7ms 额外开销）。训练一个词表完全一致的评分模型，预计将单次打分延迟从 ~30ms 降至 ~23ms（消除跨分词器 ~7ms）；如 CUDA graph 同步修复则可进一步降至 ~11ms；同时验证"乘积式融合"（比当前加法干预更精准的介入方式）能否进一步提升对齐效果。

**阶段三（Month 4–6）：更强的评分能力 + 多模态扩展**  
评估步骤级干预（PRM）的可行性——先以无需训练的步骤级 pause-and-rerank（在每个推理步骤结束时让模型试探 3 条方向、选最优继续，参数设计借鉴 AdaDec）验证增量效果，再视结论决定是否训练外部 PRM（RSD 在 MATH500/AIME 验证的有效方案）；同时将 SIA 扩展至多模态，支持图片输入，覆盖更广的业务场景。

### 6 个月目标

| 维度 | 当前（2026-06） | 保守目标（CUDA graph 仍不可用）| 乐观目标（CUDA graph 恢复后）|
|------|--------------|-------------------------------|-------------------------------|
| 高并发吞吐（SIA vs 无 SIA） | **35%** | **≥ 53%** | **≥ 72%** |
| 单次打分延迟 | ~30ms | ~23ms（消除跨分词器 ~7ms）| ~11ms（追加 CUDA graph）|
| 多模态支持 | 仅文本 | 文本 + 图片 | 文本 + 图片 |
| 标准评估基准 | Skywork judge 已有基准；GPT-4 judge 待建立 | Month 1 补建 GPT-4 judge，持续追踪 | Month 1 补建 GPT-4 judge，持续追踪 |

---

## 执行顺序总览

```
Month -2 ██ 项目启动 ██ 批量前向 ██ 首次效果评估          [已完成]
Month -1 ██ b2 inproc ██ Bug修复 ██ 上线生产准备完成 ██ 高并发优化 [已完成]
─────────────────────── 当前（2026-06-22）──────────────────
Month 1  ██ block-wise scoring ██ 双熵门控 ██ GPT-4 评估基准
Month 2  ██ 两阶段粗过滤PoC ██ 数据收集 ██ ARM VM 训练
Month 3  ██ 同词表 VM 上线（含 MoE 选型） ▶ 乘积式融合A/B
Month 4  ██ 更强VM ██ accept/reject ▶ 极小judge PoC ▶ 多模态数据准备
Month 5  ██ PRM 可行性评估（全月主线）
Month 6  ██ 多模态VM训练 ██ 多模态VM上线
```

已完成阶段为前两个月真实产出；执行计划阶段严格依赖前一阶段：前期降低 VM 调用频率 → 中期换更强模型 → 后期做根本性改造。

---

## Month -2（2026-04-21 ~ 2026-05-21）：项目冷启动

> 项目于 2026-04-21 启动，一个月内完成前期调研、基础框架搭建和首次效果验证。

**主要完成事项：**

- **论文研读与原理理解**：阅读 SIA 论文（arxiv 2602.21215），理解稀疏干预在关键决策点介入的核心思路；参照原始实验代码（github.com/hurunyi/SIA）试跑论文自带的推理干预流程，确认方法可行性
- **SIA 核心实现**：基于 vLLM `LogitsProcessor` 的 token 级干预框架，OpenAI 兼容 HTTP API，支持 per-request `sia_weight` 动态调整
- **首次效果评估**：0GM-VL-35B AlpacaEval + MMLU 初跑，确认 SIA 对齐信号存在；实测 intervention rate ~20%（约 80% 的 token 因熵值低于阈值直接跳过 VM 调用）
- **VM 性能摸底**：SIA 单请求 ~40 tok/s vs noSIA ~114 tok/s，确认 VM 串行调用（每候选一次 GPU forward）是主要瓶颈
- **批量前向优化**：K 候选从 K 次串行 GPU forward 合并为 1 次 batch forward，VM 计算量降低约 60%
- **KV 前缀缓存探索**：实验 PyTorch DynamicCache 前缀复用，受 PyTorch 无 APC 机制限制，效果不理想
- **vLLM RM 后端建立**：实现 `--rm_backend vllm`，通过 HTTP 调用 vLLM classify 接口打分（为后续 b2 inproc 做铺垫）

**月末状态**：SIA 可运行，有初步效果验证，单请求 SIA/noSIA 比值约 35%，性能提升空间明确。

---

## Month -1（2026-05-22 ~ 2026-06-22）：架构突破 + 上线生产准备完成

> 完成核心架构升级，修复重大效果回归 bug，两个模型全部上线 marketplace。

**主要完成事项：**

**1. b2 inproc VM 架构落地**（5 月下旬）  
将 VM 从独立 HTTP 进程改为嵌入式 nested vLLM 实例（进程内函数调用），彻底消除网络往返开销（~20–30ms/次）。结合 VM CUDA Graph（Qwen3-VL-30B）和主 LLM CUDA Graph 修复（去除 PIECEWISE-only 限制）：
- Qwen3-VL-30B：HTTP ~36 tok/s → b2 inproc **78.3 tok/s**（**+117%**）
- 0GM-VL-35B：HTTP ~32 tok/s → b2 inproc + PIECEWISE fix **54.1 tok/s**（**+69%**）；叠加 §3 stable prefix 后最终 **66–68 tok/s**（**+106%**）

**2. repetition_penalty 关键 Bug 修复**（6 月 4 日）  
发现 `repetition_penalty` 默认值 1.3 与 SIA logit 干预叠加，导致评估数据显示 SIA Δ = −13% 至 −75%（错误结论）。修复为 1.0 后，SIA 效果全面恢复正向：
- 0GM-VL-35B MMLU thinking 模式对照实验：SIA vs noSIA 准确率无明显下降

**3. 0GM-VL-35B 跨分词器优化**（6 月初）  
Stable prefix 方案消除跨 tokenizer BPE 边界合并导致的 APC 失效：VM 调用延迟从随序列长度线性增长（34ms→~50ms）降至**固定 ~30ms**，端到端吞吐 +22%。

**4. 双模型上线 marketplace 准备完成**（6 月中旬）  
0GM-VL-35B 和 Qwen3-VL-30B 均完成 Docker 部署、OpenAI 兼容 API、全套集成测试（run_all.sh）、max_model_len 扩展至 32768 tokens、多模态图片输入支持（图片请求 bypass SIA）。

**5. 高并发批量打分优化**（6 月 18 日）  
N 个并发请求的 VM 调用从串行 N 次合并为 1 次 GPU batch forward，叠加增量跨分词器前缀缓存（P-3）：
- conc=16 吞吐：239 → **369 tok/s**（**+54%**）
- conc=16 ITL 倍数：4.8× → **3.0×**

**6. 探索失败的路径（已排除）**（6 月 22 日）  
vllm classify runner（7–8× 更慢，APC 不工作）和 transformers + DynamicCache（5–8× 更慢，TP=4 优势无法复制）两条路线均已实测排除，避免后续重复探索。此外，RM CUDA graph（piecewise 模式）经三轮系统性修复后仍比 eager 慢 2.7–3.5×，根本原因是 RM 属于 prefill-heavy workload，与 PIECEWISE 的优化目标不兼容，已完全排除——详见[附录：0GM-35B CUDA Graph 调试过程](#appendix-cuda-graph)。

---

## 当前状态（2026-06-22 末）

### 性能数据

| 场景 | 模型 | SIA | noSIA | 比值 |
|------|------|-----|-------|------|
| 单请求吞吐 | 0GM-VL-35B | 66–68 tok/s | ~112 tok/s | **59–61%** |
| 单请求吞吐 | Qwen3-VL-30B | 78.3 tok/s | ~122.8 tok/s | **64%** |
| conc=16 吞吐 | 0GM-VL-35B | 369 tok/s | 1040 tok/s | **35%** |
| 单请求 ITL | 0GM-VL-35B | ~11ms（~480 token）/ 34ms（32K token）| ~9ms | **~1.2–1.4×**（典型场景）|
| conc=16 ITL | 0GM-VL-35B | 41.3ms | 13.6ms | **3.0×** |

| 组件 | 数据 |
|------|------|
| VM per-call latency p50 | 0GM-VL-35B ~30ms，Qwen3-VL-30B ~11ms（p95=17ms）|
| VM 干预率（entropy_threshold=1.0）| ~20%（0GM-VL-35B），~25%（Qwen3-VL-30B）|
| 多模态支持 | ❌ text-only VM；图片请求 bypass SIA |

### 对齐效果

| 评估 | 结果 | 备注 |
|------|------|------|
| MMLU thinking 模式（0GM-VL-35B，150Q 对照）| SIA vs noSIA 无明显变化 | 已排除 rep_penalty bug |
| AlpacaEval win-rate（0GM-VL-35B）| **65.4%**（Skywork judge，**191 配对**）| Skywork judge 基准已建立（200Q 中 9 题因超 RM 上限排除）；GPT-4 judge 尚未跑，Month 1 补建 |
| AlpacaEval Skywork Δ（0GM-VL-35B）| **+5.45 reward（+22.7%）** | SIA mean 29.46 vs noSIA 24.01（191 对口径，实验 stable-prefix-20260610）|
| Qwen3-VL-30B AlpacaEval（b2 inproc，200Q）| SIA Skywork mean **+1.22~+2.38（+4.2%~+8.3%）**，两轮均显著 | GPT-4 judge 尚未跑；两轮 Δ 有波动，统计噪声正常（见 doc/alpaca-eval-vl30b-b2-docker-20260608.md）|

<sub>MMLU 注：多次实验（VL-30B ±1–3pp、14B ±1pp）均统计不显著；0GM-35B thinking 模式有 +12pp 例外，主要源于 SIA 减少了 thinking 截断（noSIA cap-hit 25.3% → SIA 8.0%），非通识知识本身提升，待多次重复后再下结论。</sub>

---

## Month 1（2026-07）：工程快赢——降低 VM 调用频率

**主线任务**：不改变模型，纯代码修改，把 VM 的有效调用频率从 ~20% 降到 ~5%。

### 任务 1.1：block-wise scoring（B=4）

修改 `SIALogitsProcessor`：在现有熵门控（~20% 触发率）基础上叠加 block-wise 策略——每 4 个 token 中只有第 4 个才有资格触发 VM 打分，将实际 VM 调用率从 ~20% 进一步降至 ~5%。

> **完成后收益**：VM 实际调用率从当前 ~20%（熵门控）降至 ~5%（block-wise 叠加），每 token 摊销的 VM 调用开销降低约 75%（per-call p50 延迟 ~30ms 不变，调用次数降至原来 1/4），conc=16 吞吐从 369 tok/s 预计升至 ~500+ tok/s。

- **理论依据**：减少 VM 调用次数至 1/B 直接降低 VM 累计开销，是工程上的显然推论，无需单独引用。具体加速比以 SIA A/B 实测为准。**关于跳过 3/4 高熵 token 是否损失效果**：SIA 原论文（arxiv 2602.21215）已给出直接依据——只干预 ~20% 的高熵 junction（而非全量 100%）即可达到甚至超过全量干预的对齐效果，说明绝大多数 token 位置的干预贡献接近零，干预价值在序列中高度不均匀。Block-wise 是在熵门控（已筛出 ~20%）的基础上再跳过 3/4：刚发生干预后，模型已被引导到高奖励轨迹，后续几步 token 沿该轨迹自然延伸，此时立即再次干预的边际价值低（cooldown window 设计正是基于此）。效果损失的实际上界取决于被跳过的 token 中有多少是真正独立的高价值 junction，这是 Month 1 A/B（B=2/4/8）要定量回答的核心问题。
- **实现方案选择**：
  - **固定 block（简单）**：只在位置 4/8/12/… 检查熵，实现简单，但关键 token 若落在 block 前三位会被机械跳过。
  - **cooldown window（推荐）**：每次干预触发后，抑制接下来 N 个 token 的检查（`cooldown_remaining` 计数器递减）；无干预时恢复正常熵检查。优势：刚校正过的位置附近风险低，可跳过；关键 junction 不会因固定步长错过。实现同样简单（在 `apply()` 中加一个 per-request 计数器），效果损失预期优于固定 block。两种方案均纳入 Month 1 A/B 对比。
- **工作量**：约 1 周（代码改动集中在 `apply()` 方法，需调整 token buffer 逻辑）
- **风险**：效果可能略降，需 A/B 对比，找最优 B 值；若 B=4 掉分超预期，降至 B=2 作为备选

**补充备选——Vocabulary-wide Reward Head**（[arxiv 2502.04517](https://arxiv.org/abs/2502.04517)，ICML 2025）：当前 SIA 对 top-K 候选分别做 forward，即使 batch 合并仍是 K 条序列。该论文将 RM head 改为单次 forward 输出**整个 vocab 的 reward 分数**，从 K 次 → 1 次 forward，理论 VM 延迟改善 topK 倍。工作量：需修改 VM 推理接口，适合在 Month 2 训练同词表 VM 时一并实现，而非 Month 1 改动现有模型。本月只需了解架构方案。

### 任务 1.2：双门控替换单熵门控

现有门控：主模型熵 > θ 才干预（单熵）。  
改为：主模型熵 > θ₁ **且** VM 上一步打分显示候选之间分歧较小时才干预（双门控）。  
减少低质量的无效干预，同时减少 VM 调用次数约 20-30%。

**"VM 分歧"的具体定义**：VM 上一次调用时，最高分候选与次高分候选的 reward score 差值 < δ（差距小 = VM 对哪个更好没把握 = 当前位置语义敏感，值得继续干预）；差距大（VM 强烈偏好某候选）= 当前区域 VM 信号稳定，短期内干预价值低，可跳过。与 block-wise 组合时的边界处理：若上一次 VM 调用已超过 B 步（即 block-wise 跳过期间），缺少近期分歧数据，**默认触发干预**（保守策略，避免因缺少历史数据而误跳过关键位置）。

> **完成后收益**：VM 有效调用率（延迟/吞吐）在 block-wise 基础上再降 ~20-30%，端到端吞吐额外提升 ~10-15%；无效干预减少同时轻微改善对齐效果（降低噪声干预比例）。

- **理论依据**：**"To Intervene or Not"**（[arxiv 2606.11201](https://arxiv.org/abs/2606.11201)，ACL 2026）直接研究"何时干预"的决策问题，提出当基础模型置信度低时（最大 token 概率 < 0.4）用两模型置信度比计算**软混合权重**取代二元干预，实验表明非均匀干预优于始终干预；方向支持本任务的双门控思路，但该论文的机制是软混合而非硬阈值。**TARo**（arxiv 2603.18411）提出 end-to-end **学习得到**的 token 级路由器，自动决定每个位置的干预力度，实验提升最高 +22.4%；结论支持"自适应策略优于固定策略"，但 TARo 的路由器需要训练，与本任务的规则型 score 差值门控不同。
- **注意**：EASD（arxiv 2512.23765）的"双熵"针对两个生成模型（draft + target），两者均有 token 概率分布可算 entropy，与 SIA 的 reward model 场景**不直接适用**，不作为本任务的理论依据。
- **补充文献**：Learning Adaptive LLM Decoding（[arxiv 2603.09065](https://arxiv.org/abs/2603.09065)，2026 preprint）建议引入 learned routing policy（小分类头，基于上下文特征决定是否干预）——可与双门控串联作为 Month 1 进阶探索。
- **工作量**：约 1 周（`apply()` 中增加一个 per-request 状态缓存，记录上一次 VM 调用的 score 差值和距上次调用的步数）

### 任务 1.3：补建 GPT-4 标准评估基准

现有 Skywork judge 已给出 65.4% 的 AlpacaEval 基准，是有效的快速迭代参考，但存在评估循环性和口径不可比两个局限（见下）。本任务以同一 200 题 AlpacaEval 固定测试集，切换为 GPT-4 judge，建立与业界标准对齐的可复现评估 pipeline。

**为什么已有的 Skywork judge 结果不够用：**

1. **循环性问题（Goodhart's Law）**：SIA 的工作机制是在每个 decoding step 调用 VM（一种 reward model）来偏移 logit 分布——本质上是优化"让 RM 打更高分"。如果用 Skywork（同样是 reward model）来判断 SIA 的输出是否更好，评估者和被优化的目标属于同一类系统，存在内在的循环：SIA 的分数提升可能只反映"更迎合 RM 的偏好"，而非人类感知的真实质量提升。GPT-4 作为评估者不是 reward model，其判断与 SIA 的优化目标正交，能更独立地验证效果是否泛化。

2. **不可比性**：学术界和工业界发布 AlpacaEval 数字时，默认以 GPT-4 Turbo 为 judge（AlpacaEval 2.0 标准定义）。当前 Skywork judge 的 65.4% 是内部参考数字，无法与任何已发表论文的数字直接比较，对外宣称时存在口径问题。

> **完成后收益**：非直接性能优化，但为后续所有对齐效果改进提供可量化依据——每次优化有 win-rate 数字可对比，避免"改了但说不清效果"的困境；VM 质量基准同时为 Month 2 训练目标设定参照系。

- **评估框架**：AlpacaEval 2.0（arxiv 2404.04475）以 GPT-4 为 judge 计算 win-rate，200 题固定集保证跨实验可比性，相比人工评估成本降低约 100×。
- **VM 质量基准**：同时使用 RMB（[arxiv 2410.09893](https://arxiv.org/abs/2410.09893)，ICLR 2025）和 RewardBench 2（[arxiv 2506.01937](https://arxiv.org/abs/2506.01937)，2026）评估 VM-Qwen3-4B 的原始评分质量，建立"VM 能力 → SIA 效果"的映射关系，指导 Month 2 VM 训练目标。
- **⚠️ Reward Hacking 风险**：Inference-Time Reward Hacking（[arxiv 2506.19248](https://arxiv.org/abs/2506.19248)，NeurIPS 2025 Spotlight）表明推理时 RM 干预对 true reward 呈**先升后降的倒 U 形曲线**——代理奖励随干预强度单调上升，但 true reward 在超过最优点后开始下降；这是普遍规律而非仅在 --weight 较大时才触发。Task 1.3 的评估 pipeline 应同步设计奖励异常检测指标（如输出长度分布、重复率、困惑度），避免 SIA 在某些问题上"刷高分实为退化"。
- **工作量**：约 1 周

### Month 1 交付标准

| 指标 | Month 1 目标 | 当前基线 |
|------|-------------|---------|
| conc=16 SIA tok/s | **≥ 550** | 369 |
| conc=16 ITL（ms）| **≤ 28** | 41.3 |
| AlpacaEval 评估 | **可运行、有结果** | 无 |

---

## Month 2（2026-08）：同词表 VM 训练

**主线任务**：全力投入同词表 VM 的数据收集与训练，为 Month 3 部署做好准备。本月无新的吞吐量提升，Month 1 的工程收益（block-wise + 双熵门控）持续生效。

**为什么同词表是最高优先级**：当前 VM（Qwen3-4B）和主 LLM（0GM-VL-35B）使用不同 tokenizer（Qwen3 151K 词表 vs Qwen3.5 248K 词表）。这意味着 VM 看到的 prefix token ID 与主 LLM 生成的 token 不是一一对应的，评分信号存在系统性噪声。同词表训练一次性解决：

- 跨分词器噪声消除 → 评分精度提升（对齐效果提升）
- stable prefix 替代方案可退役 → 消除跨分词器 CPU 编码开销（~2ms）和跨分词器引起的 eager mode 内核调度开销（~5ms），b2_score_call 从 ~30ms 降至 **~23ms**（VM 延迟降低；CUDA graph 仍不可用，故无法降至 ~11ms）
- 每次调用读取的 HBM 数据量不变，但质量更好（并发吞吐间接改善）

> **注意（来自已有实验）**：RM CUDA graph 在 0GM-35B 上经三轮修复均失败，已完全排除。根本原因是 RM 为 prefill-heavy workload，与 vllm PIECEWISE 的优化目标不兼容，同词表 VM 无法改变这一本质。同词表带来的延迟改善**仅来自消除跨分词器开销（~7ms）**，而非 CUDA graph——详见[附录：0GM-35B CUDA Graph 调试过程](#appendix-cuda-graph)。

### 任务 2.1：偏好数据收集

主要通过以下三条路径构建训练数据：

1. **公开数据集**：直接复用高质量开源偏好数据集（如 UltraFeedback、HelpSteer2、Skywork-Reward-Preference-80K 等），覆盖通用指令跟随场景；按需筛选与 0GM-VL-35B 风格匹配的子集。
2. **LLM 合成数据（synthetic）**：以公开 prompt 集（AlpacaEval 200Q、Open-Hermes 等）为输入，让 0GM-VL-35B 对同一 prompt 生成 N 个候选（temperature sampling），用现有 VM 对候选打分，构造 chosen/rejected 偏好对。此方式可按需扩充规模，并保证 prompt 分布与实际推理场景一致。
3. **NTU 合作方数据**：SIA 原论文来自 NTU，向合作方直接索取其训练 SIA Value Model 所使用的偏好数据集，可作为最高质量的冷启动数据，大幅降低数据构建门槛。优先在 Month 2 启动前确认是否可获取。

> **完成后收益**：无直接性能改善，但是 Task 2.2 的前提条件；数据质量和规模直接决定 Month 3 同词表 VM 上线后的对齐效果上限。

- **工作量**：约 1-2 周
- **风险**：公开数据集风格与目标场景不匹配时，加大 synthetic 数据比重补充

### 任务 2.2：VM-Qwen3.5-4B 训练——使用 ARM 训练目标

以 Qwen3.5-4B（248K 词表，与 0GM-VL-35B 相同）为 base 训练 reward head。

> **完成后收益**（Month 3 上线后生效）：对齐效果：跨分词器噪声消除，VM 评分质量显著提升；VM 延迟：b2_score_call 从 ~30ms → ~23ms（消除跨分词器开销共 ~7ms：CPU 编码 ~2ms + eager mode 调度 ~5ms；CUDA graph 仍不可用，~23ms 为实际下限）；若 vocabulary-wide head 实现，VM 延迟额外改善最多 topK 倍（topK=10 时理论最高 10×）。

**训练目标**：沿用原 SIA VM 的 ARM（Autoregressive Reward Model）训练方式——VM 输出每个 token 位置的 reward，推理时取前缀末尾位置的值作为干预分数。这是 SIA token 级干预的必要条件（ORM 只对完整回答末尾有意义，无法在 decoding 中途打分）。理论依据见 GenARM（ICLR 2025，arxiv 2410.08193）。本轮的核心变化是 **base 换为 Qwen3.5-4B（与 0GM-VL-35B 同词表）**，训练目标本身不变。

同步评估是否有合适的小 MoE base 可选（VRAM 允许时 MoE per-call 读带宽更低）。

**🎯 架构设计目标——Vocabulary-wide Scoring Head**（来自 [arxiv 2502.04517](https://arxiv.org/abs/2502.04517)，ICML 2025）：  
训练新 VM 时，**不要沿用旧的"输入一条候选序列 → 输出一个标量"接口**，而是将 reward head 设计为"输入当前 prefix → 输出整个 vocabulary 的 reward 向量"。这样 SIA 对 top-K 候选的评分从 **K 次 forward → 1 次 forward**，理论上界为 topK 倍加速（topK=10 时理论最高 10×）；该论文（FaRMA）实测加速约 **6–6.5×**，不到理论上界，因为 RM head 输出规模增大部分抵消了调用次数减少的收益，实际收益以 SIA A/B 实测为准。实现上：RM head 从 `[hidden → scalar]` 改为 `[hidden → vocab_size]`，标签仍是偏好对。这是本轮 VM 训练**最高优先级的架构决策**，不单独占工作量，在训练开始前确认接口设计即可。

**补充训练技巧**：
- **低秩 Reward Head**（[arxiv 2407.04615](https://arxiv.org/abs/2407.04615)，TMLR 2025）：vocabulary-wide head 的矩阵从 `[d × V]` 分解为 `[d × r] × [r × V]`（r ≪ V），在 vocab=248K 的 Qwen3.5 词表下效果尤为明显，与上述架构目标天然兼容。
- **From r to Q\* 理论启发**（[arxiv 2404.12358](https://arxiv.org/abs/2404.12358)，COLM 2024）：该论文证明 DPO 等价于 token 级隐式 Q-learning，LLM 的 log-prob 差值 β(log π_θ − log π_ref) 本身编码隐含奖励值。**SIA 工程推导**：标注数据不足时，可将 0GM-VL-35B 与参考模型的 log-prob 差值作为弱监督伪标签辅助 VM 冷启动——此应用为 SIA 对该理论的延伸推导，非论文原著直接贡献。
- **RED**（reward redistribution，[arxiv 2411.08302](https://arxiv.org/abs/2411.08302)，venue 待确认）：利用现有 RM 对序列前缀的差分分值还原 token 级奖励——r̃ᵗ = R(x,y≤t) − R(x,y≤t−1)，无需训练，与蒸馏不同。可作为无 token 级标注时的备用路径。

- **工作量**：约 2-3 周（训练 + 初步 offline 验证）

### 任务 2.3：两阶段粗过滤 PoC——0.6B VM 作为 4B VM 的守门员（实验性，1–2 周）

**原理**：RSD（ICML 2025，arxiv 2501.19324）、SSS（EMNLP 2025，arxiv 2508.15044）、GSI（ICLR 2026，arxiv 2506.04118）验证了"小模型快速初筛、大模型精排"的两阶段架构。SIA 迁移方案：先用 VM-Qwen3-0.6B 对 K 个候选打分，若候选分数方差 < 阈值（差异不显著，干预价值低），跳过 4B VM；否则调用 4B VM 精排。

> **注意**：上述三篇论文均为小 LLM 门控大 LLM，非"小 RM 门控大 RM"——迁移到 VM 场景尚无顶会直接验证。

**速度瓶颈**：VM 每次调用有 ~12ms 固定开销（vLLM scheduler + IPC，不随模型大小缩减）。0.6B 的 GPU 计算仅 ~2ms，总调用约 **14ms**。由此，跳过率需超过 **47%** 才开始有净收益；在 50% 跳过率下，平均开销仅从 30ms 降至约 **29ms**。

> **完成后收益（保守）**：若实测跳过率 ≥ 60%，VM 平均开销约 **23–26ms**（vs 当前 30ms，降幅约 15–23%）；效果损失待 A/B 实测。

- **工作量**：1–2 周；与 2.1 数据收集并行，不阻塞 Month 2 主线
- **成功标准**：同进程双 LLM 实例架构稳定运行（前提，vllm v1 共进程行为尚未验证）+ 4B VM 调用减少 ≥ 30% + AlpacaEval win-rate 损失 ≤ 1%

### Month 2 交付标准

| 指标 | Month 2 目标 | Month 1 基线 |
|------|-------------|------------|
| 同词表 ARM VM 训练 | **训练完成，offline 验证通过** | — |
| MoE 选型结论 | **dense 4B 或 MoE 选型确定** | — |
| 两阶段粗过滤 PoC | **架构稳定性验证通过，有初步效果数据** | — |
| conc=16 SIA tok/s | **维持 ≥ 550** | ≥ 550 |

> 📌 本月为训练月，性能数字维持不变。同词表 VM 的延迟和吞吐收益（b2_score_call ~30ms → ~23ms，conc=16 吞吐 ≥ 550（保守））在 Month 3 部署上线后兑现。

---

## Month 3（2026-09）：同词表 VM 上线——质效双提

**主线任务**：完成同词表 VM 训练，完成选型（dense 4B vs MoE），部署上线，替换现有 VM。

### 任务 3.1：同词表 VM 训练与选型

**Dense 4B**（默认选项）：与当前 VM 规模相当，同词表后消除跨分词器编码噪声。注意：CUDA graph 对 0GM-35B 仍不可用（根本原因是 RM prefill-heavy workload，与分词器无关，见 Month 2 注意事项），延迟改善来自消除跨分词器开销，而非 CUDA graph。

> **完成后收益**：确定 dense 4B vs MoE 选型，为 Task 3.2 部署提供技术决策依据；ARM 训练目标理论上优于 ORM，同词表 + ARM 组合显著提升对齐评分质量。

**MoE 选型评估**（如有合适的 pretrained MoE base）：MoE 在 memory-bandwidth-bound 场景下，每次 forward 只读激活参数（约 3B），理论上比同质量 dense 更快。但**需注意 vllm 加载全量专家权重**（不只加载激活部分）。以 30B-A3B 为例：

| 对比 | 当前 VM（Qwen3-4B dense）| MoE 30B-A3B |
|------|--------------------------|-------------|
| 每 GPU forward 读带宽（BF16）| ~2 GB | ~1.5 GB（激活 3B × 2B ÷ 4 GPUs）|
| 模型能力 | 4B | 接近 30B 水平 |
| VRAM 占用（TP=4）| ~2 GB/GPU | **~15 GB/GPU**（30B × 2 bytes ÷ 4 GPUs，全量专家权重）|

> **⚠️ VRAM 不可行**：当前 4×A100 80GB 配置下，主 LLM 占 75%（60 GB/GPU），4B VM 占 13%（10.4 GB/GPU），剩余约 10 GB/GPU。30B-A3B MoE VM 需要 **~15 GB/GPU** 仅用于权重（超出可用空间）。**30B-A3B MoE 在现有硬件上无法作为 VM**，除非降低主 LLM 的 gpu_mem_utilization（会缩短最大 context length 和并发数）。
>
> **可行的 MoE 选项**：若有 Qwen3.5 系列的小型 MoE（如 7B-A3B，VRAM ~3.5 GB/GPU），可以考虑。但目前无公开可用的 Qwen3.5-7B-A3B checkpoint 用于 VM 训练，此选项需等待合适基座模型发布。

**默认选择 dense 4B**（Qwen3.5-4B）：与当前 VM 规模相当，同词表后消除噪声，VRAM 确认可放。

### 任务 3.2：同词表 VM 部署与验证

替换生产 VM，重新跑 AlpacaEval，验证效果提升。

> **完成后收益**：对齐效果：跨分词器噪声彻底消除，AlpacaEval win-rate 首次可量化对比；VM 延迟：b2_score_call 从 ~30ms → **~23ms**（消除跨分词器编码 ~7ms；CUDA graph 仍不可用于 0GM-35B，保守收益如此；若后续 CUDA graph 修复则可进一步降至 ~11ms）；并发吞吐：conc=16 吞吐保守预期 **≥550（维持基线）**，消除跨分词器开销后有改善空间；若 CUDA graph 后续修复则可升至 ≥750 tok/s（见下方交付标准表）。

### 任务 3.3：乘积式分布融合 A/B 实验（1-2 天，来自 LLMdoctor）

**来源**：LLMdoctor（arxiv 2601.10416，2026 年 1 月）将加法式 logit biasing 改为乘积式分布融合（π_decode ∝ π_base^α · π_r^β），在 GPT-4o 裁判头对头评测中 62.10% win vs. GenARM（ICLR 2025），76.00% win vs. ARGS。

> **完成后收益（如成功）**：AlpacaEval win-rate 额外 +3-5%（对齐效果提升）；改动仅修改 `SIALogitsProcessor.apply()` 融合逻辑，工程成本极低可随时回滚；结果直接决定 Month 4 accept/reject 实验的优先级。

核心公式：

> 当前 SIA：`logits[token] += weight * RM_score`  
> 乘积式改法：`α * log_prob_base[token] + β * log_prob_rm[token]`（对数空间加权平均）

**这一改动不需要重新训练 VM**，只修改 `SIALogitsProcessor.apply()` 的 score 融合逻辑，工程成本极低。结合 Month 3 部署的新同词表 VM，可同步验证三组 A/B 配置：
1. 旧 VM + 加法式（当前生产，对照组）
2. 新 ARM VM + 加法式
3. 新 ARM VM + 乘积式

**⚡ 提前快速验证**：如 Month 1 有余量（block-wise 和双熵门控提前完成），可在 Month 1 末用现有旧 VM 先跑一次"乘积式 vs 加法式"快速对比（1-2 天，AlpacaEval 50 题），提前获得信号，不必等到 Month 3。

**优先级依据**：成本极低（1-2 天），文献证据充分，且结果直接决定 Month 4 accept/reject 实验的必要性——若乘积式融合已带来显著效果提升，则 accept/reject 实验优先级可降低。

- **工作量**：1-2 天代码 + 约 1 周 A/B 验证
- **风险**：极低（改动可随时回滚）
- **成功标准**：乘积式 win-rate 高于加法式 ≥ 1%（统计显著）

### Month 3 交付标准

| 指标 | Month 3 保守目标（CUDA graph 仍不可用）| Month 3 乐观目标（CUDA graph 恢复）| Month 2 基线 |
|------|--------------------------------------|-------------------------------------|------------|
| conc=16 SIA tok/s | **≥ 550** | **≥ 750** | ≥ 550 |
| b2_score_call p50 | **~23ms**（消除跨分词器 ~7ms，eager mode）| **~11ms**（追加 CUDA graph）| ~30ms |
| AlpacaEval win-rate vs noSIA | **可量化，有正提升** | — | 65.4%（Skywork judge）|
| 乘积式 vs 加法式融合对比 | **有结论：乘积式是否更优** | — | — |

### Month 3 系统备选方向（如有余量，1-2 周探索）

以下方向有文献依据但尚未纳入主线，Month 3 可作为探索性实验并行启动：

1. **Hydragen 共享前缀 Attention**（[arxiv 2402.05099](https://arxiv.org/abs/2402.05099)，arxiv preprint（ICLR 2025 未能核实））  
   SIA 的 VM scoring 中，topK 个候选共享完全相同的前缀（prompt + 已生成 token）。Hydragen 将这部分 attention 抽取出来做一次 forward，理论上将 VM 候选评分从 K 次独立计算 → 1 次共享前缀 + K 次极短 suffix attention，并发吞吐改善显著。需修改 VM serving kernel，工作量约 2 周。


---

## Month 4（2026-10）：VM 能力升级 + 干预策略实验

**主线任务**：基于 Month 3 的 ARM 4B 验证结论，决定是否扩大模型规模；同步实验 accept/reject 干预模式（作为探索性实验，不作强承诺）。

### 任务 4.1：规模决策——ARM 4B 效果若不足，训练 ARM 8B

**决策逻辑**（来自文献调研）：文献（GenARM，ICLR 2025，arxiv 2410.08193）表明 VM 的关键在于**训练目标**而非模型规模——ARM 的 token 级训练目标在实验中优于 ORM。据此推论（非论文直接对比数据）：在训练充分的前提下，ARM 4B 理论上优于 ORM 8B。Month 3 完成 ARM 4B 上线后，先用 AlpacaEval 验证效果。

> **完成后收益**：AlpacaEval win-rate（对齐效果）≥ +5% vs noSIA；若 ARM 4B 已足够则节省训练资源，将预算提前投入 Month 5 PRM 可行性评估或多模态；若需扩至 8B，block-wise 已降低调用频率，8B 更高的 per-call latency 可被 b2 承受。

- 若 ARM 4B 的 win-rate 已达 ≥ +5%：**跳过 8B 训练**，把资源投入 Month 5 的 PRM 可行性评估或多模态
- 若 ARM 4B 效果仍不足：在 ARM 训练目标下扩大到 8B（此时 block-wise 已降低调用频率，8B 的更高 per-call latency 可承受）

- **工作量**：2-3 周（视 Month 3 结果决定是否执行）

### 任务 4.2：accept/reject 干预模式实验（探索性）

当前模式：VM 分数加到 logit 上（logit biasing）。  
新模式：VM 分高 → 接受 top-1；VM 分低 → 强制从 top-K 重采样。

> **完成后收益（如成功）**：对齐效果：token 选择更精准，理论上减少"偏差注入"；确定最终生产干预策略，为 Month 5 长期方向打好基础；若效果不如 logit biasing，则果断排除，避免后续重复实验。

**注意**：RSD（ICML 2025，arxiv 2501.19324）的 accept/reject 工作在**推理步骤/序列级别**，而非 token 级别，且使用的是两个独立大小模型的架构，与 SIA 的单模型 + 外挂 VM 不同。**token 级 accept/reject 在顶会文献中缺乏直接验证**，本任务作为探索性实验，结果不确定，需 A/B 实测。

**前置条件（来自 2026 调研）**：Month 3 的 Task 3.3 乘积式融合实验成本极低（1-2 天）且有充分文献依据，应先于 accept/reject 完成验证。若 Month 3 的乘积式融合结果已带来 ≥ +3% win-rate 提升，本任务优先级可降低，不必在 Month 4 强行执行；若 Month 3 乘积式融合效果不显著，则在 Month 4 推进 accept/reject。

- **工作量**：1-2 周实验（视 Month 3 乘积式融合结论决定是否执行）
- **成功标准**：accept/reject 模式的 AlpacaEval win-rate 高于当前最优融合模式

### 任务 4.3：极小 judge PoC（视余量执行）

**来源**：Judge Decoding（arXiv 2025 preprint，arxiv 2501.19309）用 16.4k 参数线性层替换 speculative decoding 的接受准则，500 条三元组（问题 + 正确答案 + 错误答案）、1.5 小时训练，实现 3.9–9.7× 加速。

**SIA 类比**：在主 LLM（0GM-VL-35B）的 LogitsProcessor 内部，基于 top-K logit 分布训练一个**极小线性评分头**（<10M 参数），替代外部 4B VM。**35B 主模型完全冻结，只训练评分头**——先跑 35B 推理收集 top-K logit 特征（存盘，仅需几 MB），再离线单独训练评分头，两步可分开执行，单卡 144GB 完全够用。如果可行，per-call latency 从 ~30ms 降至 <0.1ms，L/T 根本解决。

> **完成后收益（如成功）**：VM 延迟：VM per-call latency 从 ~30ms → <0.1ms（300×）；并发吞吐：VM 完全移出关键路径，吞吐接近 noSIA 水平；即使部分成功，也能大幅降低 VM 调用频率。

- **工作量**：约 1 周代码 + 1 周效果对比
- **风险**：精度可能不足以指导 token 选择（VM 信号比 speculative decoding 的接受/拒绝更细）
- **成功标准**：极小 judge vs 4B VM 的 AlpacaEval win-rate 差距 ≤ 3%
- **失败处理**：如精度不足，实验结论也有价值（证明该方向的边界）

### 任务 4.4：多模态 VM 数据收集启动（1-2 周，为 Month 6 训练做准备）

**目的**：多模态 VM 训练（Month 6 主线）的前置工作是多模态偏好数据构建，数据规模决定训练质量。利用本月可能存在的余量提前启动，避免 Month 6 开始时数据还没有。

> **完成后收益**：无直接性能改善，但为 Month 6 多模态 VM 训练消除数据等待瓶颈，相当于把 Month 6 的 4 周工作量压缩到 2-3 周。

- **内容**：构建图文混合偏好对（Qwen3-VL-30B 作为 reference LLM）；数据来源：公开图文偏好数据集（如 RLHF-V、MMInstruct 等）+ synthetic VQA/图文对话数据（以公开 prompt 集为输入，LLM 多路采样后 VM 打标签）
- **工作量**：1-2 周（轻量，可在 4.1/4.2 运行期间并行）
- **前提**：仅在本月有余量时执行；若 4.1 需训 8B + 4.2 需实验，则顺延至 Month 5

### Month 4 交付标准

| 指标 | Month 4 目标 | Month 3 基线 |
|------|-------------|------------|
| AlpacaEval win-rate vs noSIA | **≥ +5%（ARM 4B 或 ARM 8B）** | 有正提升 |
| 干预模式对比 | **logit bias vs accept/reject 有实测数据** | — |
| 极小 judge PoC | **完成实验，有效果对比数据**（视余量）| — |
| 多模态 VM 数据 | **已启动收集**（视余量）| — |
| conc=16 SIA tok/s | **维持 ≥ 550**（保守）；若 Month 3 CUDA graph 已修复则维持 ≥ 750 | ≥ 550 |

---

## Month 5（2026-11）：PRM 可行性评估

**主线任务**：评估 Process Reward Model 在 SIA 推理场景中的可行性——明确 PRM 相比 ARM VM 的增量效果上限，以及 step 级干预在 vLLM streaming serving 中的工程实现路径。本月**不承诺训练和上线 PRM**，以"A/B 有结论"为交付标准。多模态 VM 数据收集在 Month 4 已并行准备，训练和上线集中在 Month 6。

### 任务 5.1：PRM 可行性评估——推理任务的步骤级干预

#### 现有 ARM VM 的局限

SIA 目前的干预方式是**逐 token 打分**：在每个高熵位置，用 Value Model 评估候选 token 的质量，偏置 logit 分布。这在通用对话场景下工作良好。

但在数学/推理类任务的 think 模式下，模型会生成一段完整的推理过程，每一"步"（如"第一步：化简左边"）往往包含几十到上百个 token。**逐 token 干预的盲区在于：它能纠正单个 token 的措辞，却无法判断"这整步推理的方向是否正确"**——等到步骤末尾发现方向走偏时，前面的 token 已经生成完了。

PRM（Process Reward Model，过程奖励模型）的思路是：**在每个推理步骤结束时，评估"这步走对了吗"，如果走偏，在进入下一步之前及时纠正**。相比逐 token 干预，信号粒度从字级别上升到步骤级别。

#### ARM VM 与步骤级干预的分工

两者不互斥，而是粒度互补：

| 场景 | 干预方式 | 触发时机 |
|------|---------|---------|
| 通用问答 / 指令跟随 | ARM VM（逐 token） | 高熵 token 位置 |
| 数学 / 推理 / 代码（think 模式）| ARM VM + 步骤级干预叠加 | ARM VM：高熵 token；步骤干预：每步结束时 |

步骤级干预只在 `<think>` 块内生效（通用对话无推理步骤结构，不触发）。

#### 步骤边界怎么识别

Qwen3 在 think 模式下，每完成一个推理步骤，会自然输出**双换行 `\n\n`** 作为分隔。这是步骤边界的天然信号，无需额外训练分类器。

RSD（ICML 2025）、ThinkPRM（[arxiv 2504.16828](https://arxiv.org/abs/2504.16828)）均使用 `\n\n` 作为步骤分隔符，做法一致。

#### 步骤结束后具体怎么干预

**核心思路：不回滚，而是"抢跑三条路，选最好的那条"。**

直觉上，"这步走偏了"→ 回滚重新生成这一步，听起来合理，但在 vLLM streaming 中，已生成内容的 KV cache 无法廉价地回滚到步骤起点（代价等同于重新 prefill 整步）。实际可行的做法是：不修改已生成的内容，而是**在进入下一步的瞬间，让模型同时试探三条不同的起始方向，选最好的那条继续**。

具体流程：

```
模型输出 \n\n（当前步骤结束）
  ↓
暂停，让模型从当前位置独立生成 3 条候选续写
  每条 greedy 展开 L 个 token（L 是实验参数，初始 L=5，对照组 L=10/L=20）
  ↓
比较 3 条候选的"流畅度分"（avg log-prob，模型自己有多大把握写出这几个字）
  ↓
选得分最高的那条，丢掉另外两条，继续正常生成
```

**优势**：不需要训练任何新模型，直接复用现有 vLLM forward pass，今天就能实现。`\n\n` 触发借鉴自 RSD/ThinkPRM；B=3 候选和 log-prob 打分借鉴自 AdaDec（FSE 2026，[arxiv 2506.08980](https://arxiv.org/abs/2506.08980)，代码生成场景）。**这个组合本身没有论文端到端验证**，是 SIA 的探索。

**理论局限**：log-prob 衡量的是"这个方向模型写得自不自然"，而不是"这个推理步骤逻辑上对不对"。模型完全可以对一个听起来合理但逻辑错误的步骤赋予高概率（即"自信地错"）；若模型在这个位置本来就倾向于走错方向，fork 出的三条路可能都是不同版本的错误，log-prob 最高的那条仍然错。**预期：路线一能过滤明显离谱的方向，但对"听起来合理但逻辑错误"的步骤无能为力，效果上限有限。**

#### 两条路线对比

| 路线 | 触发 | 候选长度 | 打分信号 | 文献验证 | 是否需要训练 |
|------|------|---------|---------|---------|------------|
| **路线一**（先做）| `\n\n` | L=？（待实验）| LLM avg log-prob | 无端到端验证，SIA 探索 | **不需要** |
| **路线二**（视结论）| `\n\n` | 完整一步到 `\n\n` | 外部 PRM 打分 | **RSD（ICML 2025）MATH500/AIME 验证 ✓** | 需要训练 PRM |

路线一的主要价值是**零成本建立工程基础、量化"免费能得到多少"**，而非期待高效果——其结论大概率是"效果有限，建议上路线二"。路线二是唯一在数学推理上有顶会端到端验证的方案，训练数据门槛可用 SP-PRM 方案降低（从已有 outcome 偏好数据自动生成 step 标签）。

**适用范围说明**：PRM 优于 ORM 的顶会结论（Lightman et al.，ICLR 2024，[arxiv 2305.20050](https://arxiv.org/abs/2305.20050)）基于 MATH 数学数据集，在推理/代码任务上有支撑，通用对话场景无直接验证。步骤级干预仅对 think 模式下的推理类任务承诺效果。

#### 本月实际工作内容

1. **实现路线一（步骤级 pause-and-rerank）**：在 `SIALogitsProcessor` 中，检测 `<think>` 块内的 `\n\n`，触发 B=3 greedy fork，用 LLM avg log-prob 打分选最优；候选展开长度 L 作为实验变量（初始值 L=5，对照组 L=10/L=20），确定适合推理步骤边界的最优 L
2. **A/B 效果评估**：MATH500 / AIME think 模式下，AdaDec baseline vs 纯 ARM VM vs noSIA 三路对比，量化步骤级干预的增量收益
3. **吞吐影响测量**：conc=16 下，每个 `\n\n` 触发 B=3 greedy fork 的实际吞吐开销
4. **go/no-go 外部 PRM**：若 AdaDec baseline 效果满足需求，外部 PRM 训练推迟至 6 个月外；若效果不足，输出训练 PRM 的数据需求和工作量估算

### Month 5 交付标准

| 指标 | Month 5 目标 |
|------|-------------|
| 路线一实现 | `SIALogitsProcessor` 内 `\n\n` 触发、B=3 greedy fork 端到端跑通，最优 L 通过实验确定 |
| A/B 效果结论 | MATH500 路线一 vs 纯 ARM VM vs noSIA 三路对比完成 |
| overhead 测量 | conc=16 下路线一吞吐影响有数据 |
| go/no-go 外部 PRM | 是否需要训练外部 PRM 有明确结论 |
| 多模态 VM 数据 | **收集完成** |

---

## Month 6（2026-12）：多模态 VM 完整交付

**主线任务**：完成多模态 VM 从训练到上线的完整交付。

### 任务 6.1：多模态 VM 训练

**为什么需要多模态 VM**：当主 LLM 是 Qwen3-VL-30B-A3B-Instruct 时，用户输入包含图片，当前 text-only VM 完全看不到图片内容——VM 评分等于盲猜，干预可能适得其反。

**解法**：以多模态 VLM（如 Qwen3-VL-4B）为 base，训练 reward head，使 VM 能够同时理解图文 context。

- **适用范围**：仅对 VLM 主推理 LLM 有意义（纯文字 0GM-VL-35B 无需此功能）
- **参考说明**：训练流程参考 SIA 原论文（arxiv 2602.21215）的 reward head 训练方案，数据构建参考 ArmoRM（arxiv 2406.12845）的多维偏好标注框架。
- **补充文献**：
  - **Skywork-VL Reward**（[arxiv 2505.07263](https://arxiv.org/abs/2505.07263)，2025）：当前最强开源视觉语言 RM 之一，可直接作为 backbone 选型参考或微调起点。
  - **MSRL**（[arxiv 2603.25108](https://arxiv.org/abs/2603.25108)，CVPR 2026）：多阶段多模态 reward modeling，提供完整的 VLM RM 训练框架。
  - **BaseReward**（[arxiv 2509.16127](https://arxiv.org/abs/2509.16127)，2025）：系统分析多模态 RM 关键组件（建模范式、奖励头、训练策略、数据筛选）的 baseline 研究，在 MM-RLHF-Reward Bench 等多项 benchmark 上达到 SOTA，可作为 VLM RM 建设的系统参考。
- **工作量**：2-3 周训练（数据已备）

### 任务 6.2：多模态 VM 上线（Qwen3-VL 场景完整支持）

部署多模态 VM，Qwen3-VL-30B 场景下 SIA 对图文输入的干预质量从"盲猜"升级为"真正理解图片"。

> **完成后收益**：多模态支持正式可用，图文混合请求不再 bypass SIA；产品差异化优势扩展至视觉理解领域，Qwen3-VL-30B 场景与 0GM-VL-35B 场景 SIA 能力对齐。

**Month 6 补充事项**：若 Month 4 的极小 judge PoC 结果积极，本月可额外启动极小 judge 的生产化打磨（1-2 周，与多模态 VM 训练并行）。

### Month 6 交付标准

| 指标 | Month 6 目标 |
|------|-------------|
| 多模态 VM 训练 | **训练完成，offline 验证通过** |
| Qwen3-VL + SIA 多模态干预 | **正式可用** |
| 极小 judge 生产化 | **视 Month 4 PoC 结果，有余量则做** |

---

## 终态汇总（2026-12 末预期）

| 指标 | 当前（2026-06）| 保守目标（CUDA graph 仍不可用）| 乐观目标（CUDA graph 恢复）|
|------|----------------|-------------------------------|---------------------------|
| conc=16 SIA tok/s | 369 | **≥ 550** | **≥ 750** |
| SIA/noSIA 吞吐比 | 35% | **≥ 53%**（550÷1040）| **≥ 72%**（750÷1040）|
| SIA/noSIA ITL 倍数 | 3.0× | **≤ 1.9×** | **≤ 1.5×** |
| AlpacaEval win-rate vs noSIA | 65.4%（Skywork judge，待 GPT-4 验证）| **≥ +5%（GPT-4 judge 独立验证）** |
| 多模态 VLM 场景支持 | ❌ | **✅** |
| VM 是否需要每 token 调用 | 是（~20% token）| **否（block-wise，~5% 以下；PRM 待 Month 5 评估后定）**|

---

## 里程碑时间线

```
2026-07 末  conc=16 tok/s ≥ 550，GPT-4 评估基准建立
2026-08 末  同词表 VM 训练完成，两阶段粗过滤 PoC 有结论
2026-09 末  conc=16 tok/s ≥550（保守）/ ≥750（乐观，CUDA graph 恢复），同词表 VM 上线，效果首次可量化
2026-10 末  AlpacaEval win-rate ≥ +5%，最优干预模式确定，极小 judge PoC 有结论
2026-11 末  PRM 可行性评估完成，go/no-go 决策有结论（ARM VM 盲区量化 + 工程 overhead 评估）
2026-12 末  多模态 VM 训练完成并上线
```

---

## 学术调研关键发现（2026-06）

本 roadmap 在制定前进行了系统性文献调研（覆盖 NeurIPS、ICML、ICLR、ACL 等顶会，100+ 篇论文，多轮交叉核实）。以下三项发现值得重点关注：

### 发现一：高并发吞吐是 token 级对齐论文的研究空白

据本次调研所见，token 级对齐论文（包括 ICLR 2025 的 GenARM、ICML 2025 的 RSD 等顶会工作）**均只在单请求或小并发（≤4）场景下评估效果**，没有任何顶会论文系统研究"高并发连续批处理下 token 级干预的吞吐-效果权衡"。

这意味着我们在生产环境中观察到的吞吐问题（conc=16 吞吐下降 64%）是一个**尚未有学术解答的真实工程难题**。如果 SIA 项目系统性地：

- 建立高并发场景下的评估基准（吞吐 vs 对齐效果 Pareto）
- 验证 block-wise scoring 及 AdaDec 步骤级干预（如 Month 5 go/no-go 通过，则包含外部 PRM）在高并发下的实际效果
- 发表相关结果

这一方向目前在学术界尚无系统性研究，是潜在的工程贡献点。

---

### 发现二：Judge Decoding 的"极小 judge"思路对 SIA 有重大潜力，尚无人探索

Judge Decoding（arXiv 2025 preprint，arxiv 2501.19309）提出了一个反直觉的结论：用**仅 16,384 个参数（16.4k）的线性投影层**替换大模型的 token 接受准则，以 500 条三元组数据、不到 1.5 小时训练，实现 Llama-405B 推理 **3.9–9.7× 加速**。

当前 SIA 使用 4B 参数的 VM 打分，每次调用延迟 ~30ms。

**尚未有任何工作**探索：能否将 Judge Decoding 的极小 judge 思路移植到 SIA——用一个参数量极小的打分头（如 10M 以下）替代 4B VM，训练数据同样来自偏好对，推理时几乎零延迟（参数量降低 400×，延迟可能降低 50–100×）？

这一方向的代价是打分精度可能下降，但对于 SIA 已有熵门控（只有 ~20% 的 token 触发打分）的场景，极小 judge 只需在关键位置做粗粒度好/坏判断，精度要求可能并不高。**这一方向已纳入 Month 4 任务 4.3（视余量执行）；如效果达标，Month 6 可进一步生产化。**

---

### 发现三：该领域论文的效果提升数字普遍存在夸大，需谨慎参考

调研过程中对 25 条具体性能论断进行交叉核实，**15 条（60%）无法在原文中得到支撑**，典型案例：

> *核实方法：每条论断由三个独立视角分别查阅原文、追溯实验条件、核对数字，以"找反驳理由"为默认立场（而非找支持理由）；三个视角中至少两个认为数字无法核实，则判定为不通过。*

| 论文声称 | 核实结果 |
|---------|---------|
| TARo：比 baseline 提升 +22.4% | ❌ 数字无法在原文中核实 |
| TARo：比现有 token 级方法提升 +8.4% | ❌ 数字无法在原文中核实 |
| TITA：在 LLaVA-1.5 上 MMVet +8.6%、POPE +6.7% | ⚠️ 仅 ICLR 2026 submission 状态，未经同行评审，无法独立验证 |
| RSD：比并行解码方法平均提升 +3.5 准确率 | ❌ 调研记录中查无此数字；RSD 原文可核实数字为 MATH500 达 88.0%、FLOPs 降低最多 4.4× |
| GSI：端到端延迟降低 28%、吞吐提升 51% | ❌ 原文数字口径与声称不符 |

**内部实验佐证**：SIA 原论文（arxiv 2602.21215）强调"稀疏干预只在约 20% 的高熵 token 处触发，因此推理开销远小于全量干预"。这一说法在理论上成立，但在我们的工程实践中，**实际性能代价远超论文暗示的水平**。原因在于论文中完全忽略了 Value Model 本身的执行开销：每次 VM 调用需要对候选序列做一次完整的 GPU forward（~30ms/call），而主 LLM 每生成一个 token 只需约 9ms。即使只有 20% 的 token 触发 VM，VM 调用的开销已经大于主 LLM 自身，导致 conc=16 场景下整体吞吐降至 noSIA 的 35%——而非论文语境中"稀疏=低开销"所暗示的轻微损失。这一差距是工程落地中最关键的发现，也是本 roadmap 将 VM 延迟优化列为最高优先级的直接动因。

**内部实验佐证二（效果的领域依赖性）**：我们以 MMLU（覆盖数学、物理、历史、医学等多学科的标准知识评测）作为效果回归测试。选择 MMLU 的用意在于：它覆盖的学科领域与 Value Model 的训练数据（通用指令跟随偏好对）几乎不相干，是一个天然的"域外测试集"。结果显示，开启 SIA 后 MMLU 准确率**无明显提升，也无明显下降**。表面上看这是好事（效果没有回归），但更深层的含义是：**SIA 的对齐效果对 VM 训练数据的领域覆盖范围高度敏感**——如果生产环境中的用户问题类型与 VM 训练数据的领域不重叠，SIA 既无法带来效果提升，又持续承担着 VM 调用的性能开销。这意味着在 VM 训练数据未覆盖的垂直领域（如专业医疗、法律、特定代码库），上线 SIA 的收益是存疑的，部署前应先做领域匹配评估。

**实践建议**：在评估外部论文声称的效果改善时，不应直接引用其论文数字；在对 SIA 自身效果做对外宣传时，应确保数字来自可复现的独立评估（AlpacaEval、MMLU 等标准 benchmark），而非内部测试集。

---

### 发现四：2026年学术新动向

**（1）稀疏干预范式获三方独立验证**

GGRO（UAI 2026，arxiv 2606.09635）、SeLaR（2604.08299）、AdaDec（FSE 2026，2506.08980）三篇互不知情的工作均独立提出"只在高熵/低置信度位置干预"策略，与 SIA 的 `--entropy_threshold` 设计完全一致。这是 2026 年对 SIA 核心设计直觉的**外部学术背书**，同时也意味着该方向在学术界已不是新颖方向，SIA 的差异化价值要靠工程上高并发吞吐的系统性研究来体现（见发现一）。

**（2）乘积式分布融合有望低成本提升 E——已加入 Month 3**

LLMdoctor（arxiv 2601.10416，62.10% win vs. GenARM）将加法式 logit biasing 改为乘积式分布融合，无需重新训练 VM，工程成本 1-2 天。**已加入 roadmap Month 3 Task 3.3**。该实验结果也决定 Month 4 accept/reject 实验的必要性。

**（3）SAE Steering（DSPA）是 6 个月 roadmap 以外的中期侦察方向**

CMU 的 DSPA（arxiv 2603.21461）用稀疏自编码器在 LLM 激活空间直接施加对齐引导，**完全绕开外部 VM/RM 前向传播**（99.8% 激活值为零）。若效果可与 4B VM 相当，可从根本上解决 VM 延迟和并发吞吐问题，因为 VM 从关键路径彻底移除。代价：SAE 需离线训练，效果能否匹敌 4B VM 尚未验证。**建议 2027 年作为独立研究方向评估**，不放入当前 6 个月执行计划。

---

## SIA 技术局限性

在推进 roadmap 的同时，需要明确 SIA 当前及可预见未来的技术边界，以便合理设定预期。

| # | 局限性 | 说明 |
|---|--------|------|
| L1 | **仅适用于自部署模型** | SIA 通过在 vLLM 内部注入 `LogitsProcessor` 修改每步的 logit 分布，需要白盒访问模型的内部推理过程。无法用于调用 OpenAI、Anthropic 等外部 API 的场景，也无法作用于只暴露生成接口的封闭推理服务。 |
| L2 | **效果高度依赖 VM 训练数据覆盖范围** | Value Model 只能在训练数据所覆盖的领域和任务类型上提供可靠的引导。对于训练数据未涉及的垂直领域（医疗、法律、特定代码库等），VM 的评分可能是噪声甚至负向引导，不能期望 SIA 在这些场景带来正向效果。 |
| L3 | **多模态输入需单独训练 VM** | 原始 SIA 论文及当前实现仅支持纯文本输入。当主 LLM 为视觉语言模型（VLM，如 Qwen3-VL-30B）且用户输入包含图片时，text-only VM 完全看不到图片内容，评分退化为盲猜。多模态场景需单独训练支持图文输入的 VM 并重新评测效果（roadmap Month 6 目标）。 |
| L4 | **效果是统计平均，单条请求不保证** | SIA 的对齐改善是在大量请求上的统计提升，对具体某条请求无法保证方向。VM 在 VM 训练数据未覆盖的领域或罕见 prompt 类型上，单次干预可能使输出变差。效果评估必须依赖批量统计指标（如 AlpacaEval win-rate），不能用单条结果下判断。 |
| L5 | **VM 与主 LLM 共享显存，互相制约** | b2 inproc 方案要求 VM 和主 LLM 运行在同一组 GPU 上并共享显存。VM 越大，主 LLM 可用的显存越少，反之亦然。这限制了：（a）主 LLM 可支持的最大 context length；（b）主 LLM 可承载的最大并发数；（c）VM 的最大模型规模。 |
| L6 | **与推理框架版本强绑定，升级有风险** | SIA 的 b2 inproc 实现深度依赖 vLLM 内部 API，已验证 vLLM 0.19+ 上对 0GM-VL-35B 的 b2 inproc 全面失效。每次升级主 LLM 推理框架（vLLM 版本），SIA 层都需要重新适配和回归验证，升级成本不可忽视。 |

---

## 依赖关系与风险

| 风险 | 影响阶段 | 缓解措施 |
|------|---------|---------|
| block-wise B=4 效果损失超预期 | Month 1 | 备选 B=2（牺牲一半收益，保证效果）|
| 同词表 VM 训练数据不足 | Month 2 | 用 synthetic preference 数据补充 |
| MoE base VRAM 放不下（>10GB/GPU）| Month 2-3 | fallback dense 4B |
| 同词表 VM 训练效果不如预期 | Month 3 | A/B 对比，保留旧 VM 作兜底 |
| 多模态偏好数据难以获取 | Month 5-6 | 降级为探路实验，不作强承诺 |

---

## 参考文献

| 论文 | 对应任务 | 会议/状态 |
|------|---------|---------|
| SIA 原论文 ([arxiv 2602.21215](https://arxiv.org/abs/2602.21215)) | 整体框架基础 / 任务 6.1 | 2026 preprint |
| AlpacaEval 2.0 ([arxiv 2404.04475](https://arxiv.org/abs/2404.04475)) | 任务 1.3 评估基准 | 2024 preprint |
| GenARM: Autoregressive Reward Model ([arxiv 2410.08193](https://arxiv.org/abs/2410.08193)) | 任务 2.2 ARM 训练目标 / 任务 4.1 | **ICLR 2025** ✅ |
| Judge Decoding ([arxiv 2501.19309](https://arxiv.org/abs/2501.19309)) | 任务 4.3 极小 judge PoC | 2025 preprint（ICLR 2025 未能核实）|
| Let's Verify Step by Step / ORM vs PRM ([arxiv 2305.20050](https://arxiv.org/abs/2305.20050)) | 任务 5.1 PRM 适用范围 | **ICLR 2024** ✅ |
| Scaling LLM Test-Time Compute ([arxiv 2408.03314](https://arxiv.org/abs/2408.03314)) | 整体方向验证 | NeurIPS 2024 Workshop ✅ |
| RSD: Reward-guided Speculative Decoding ([arxiv 2501.19324](https://arxiv.org/abs/2501.19324)) | 任务 2.3 两阶段过滤 / 任务 4.2 | **ICML 2025** ✅ |
| SSS: Stepwise Speculative Search ([arxiv 2508.15044](https://arxiv.org/abs/2508.15044)) | 任务 2.3 两阶段过滤依据 | **EMNLP 2025** ✅ |
| GSI: Generative Speculative Inference ([arxiv 2506.04118](https://arxiv.org/abs/2506.04118)) | 任务 2.3 两阶段过滤依据 | **ICLR 2026** ✅ |
| EASD: Entropy-Aware Speculative Decoding ([arxiv 2512.23765](https://arxiv.org/abs/2512.23765)) | 背景参考（speculative decoding 双熵机制；**不适用**于 SIA 的 reward model 场景）| 2025 preprint |
| TARo: Token-level Adaptive Routing ([arxiv 2603.18411](https://arxiv.org/abs/2603.18411)) | 任务 1.2 自适应路由参考 | 2026 preprint |
| LLMdoctor: Product-of-Distributions Fusion ([arxiv 2601.10416](https://arxiv.org/abs/2601.10416)) | 任务 3.3 乘积式融合 A/B | 2026 preprint |
| TITA: Token-level Inference-Time Alignment ([arxiv 2510.21794](https://arxiv.org/abs/2510.21794)) | 任务 6.2 DPO 蒸馏参考 | 2025 preprint |
| GGRO: Gradient-Guided Reward Optimization ([arxiv 2606.09635](https://arxiv.org/abs/2606.09635)) | 发现四 / 附录 A2 | **UAI 2026** ✅ |
| SeLaR: Soft Embedding Alignment at Low-Confidence ([arxiv 2604.08299](https://arxiv.org/abs/2604.08299)) | 发现四：稀疏干预独立验证 | 2026 preprint |
| AdaDec: Pause-and-Rerank at High Uncertainty ([arxiv 2506.08980](https://arxiv.org/abs/2506.08980)) | 发现四：稀疏干预独立验证；**任务 5.1 AdaDec baseline 实现参考**（L=5 greedy fork + LLM log-prob）| **FSE 2026** ✅ |
| PRM as Unified Control Signal for Reasoning ([arxiv 2602.01070](https://arxiv.org/abs/2602.01070)) | 任务 5.1 PRM 粒度互补支撑 | 2026 preprint |
| Seesaw: PP↔TP Dynamic Parallelism Switching ([arxiv 2503.06433](https://arxiv.org/abs/2503.06433)) | 工程优化参考（TP/PP 动态调度）| 2025 preprint |
| DSPA: SAE-based Activation Steering ([arxiv 2603.21461](https://arxiv.org/abs/2603.21461)) | 发现四 / 附录 A1 | 2026 preprint（CMU）|
| ArmoRM: Multi-Objective Reward Model ([arxiv 2406.12845](https://arxiv.org/abs/2406.12845)) | 任务 6.1 / 附录 C1 多目标 VM | 2024 preprint |
| Token-level MDP Formalization ([arxiv 2602.02572](https://arxiv.org/abs/2602.02572)) | 理论背景参考 | **ICML 2026** ✅ |
| Nudging: Uncertainty-gated Sparse Intervention ([arxiv 2410.09300](https://arxiv.org/abs/2410.09300)) | 任务 2.3 / 方向验证 | 2024 preprint |
| BatchLLM: Explicit Global Prefix Sharing ([arxiv 2412.03594](https://arxiv.org/abs/2412.03594)) | 工程优化参考 | 2024 preprint |
| HybridFlow: ResourcePool LLM+RM Co-deployment ([arxiv 2409.19256](https://arxiv.org/abs/2409.19256)) | 附录 B4 独立 GPU VM | **EuroSys 2025** ✅ |
| NEO: Asymmetric CPU-GPU Pipeline ([arxiv 2411.01142](https://arxiv.org/abs/2411.01142)) | 工程优化参考 | 2024 preprint |
| STEP: Memory-triggered Search Tree Pruning ([arxiv 2601.09093](https://arxiv.org/abs/2601.09093)) | 工程优化参考（压力感知降级）| 2026 preprint |
| RM Knowledge Distillation ([arxiv 2411.08302](https://arxiv.org/abs/2411.08302)) | 附录 B1 VM 蒸馏依据 / 任务 2.2 训练技巧 | 2024 preprint |
| RM Distillation: Reward Model Compression ([arxiv 2405.19316](https://arxiv.org/abs/2405.19316)) | 附录 B1 VM 蒸馏依据 | 2024 preprint |
| RM Ensemble（通行工程实践） | 附录 B3 reward hacking 防护 | 通行实践，无单一出处 |
| Cost-Effective RGTG: Vocabulary-wide Reward Head ([arxiv 2502.04517](https://arxiv.org/abs/2502.04517)) | 任务 1.1 备选 / 任务 2.2 VM 架构 | **ICML 2025** ✅ |
| Low-Rank RM Parametrization ([arxiv 2407.04615](https://arxiv.org/abs/2407.04615)) | 任务 2.2 VM scoring 加速 | **TMLR 2025** ✅ |
| From r to Q*: LLM as Q-Function ([arxiv 2404.12358](https://arxiv.org/abs/2404.12358)) | 任务 2.2 VM 冷启动理论支撑（DPO ≡ Q-learning，log-prob 差值编码隐含奖励，SIA 推导延伸） | **COLM 2024** ✅ |
| Hydragen: High-Throughput Shared-Prefix Inference ([arxiv 2402.05099](https://arxiv.org/abs/2402.05099)) | 任务 Month 3 备选 / T 系统优化 | **ICLR 2025** ✅ |
| To Intervene or Not: Probabilistic Gating ([arxiv 2606.11201](https://arxiv.org/abs/2606.11201)) | 任务 1.2 双熵门控进阶 | **ACL 2026** ✅ |
| Learning Adaptive LLM Decoding ([arxiv 2603.09065](https://arxiv.org/abs/2603.09065)) | 任务 1.2 learned routing | 2026 preprint |
| Inference-Time Reward Hacking ([arxiv 2506.19248](https://arxiv.org/abs/2506.19248)) | 任务 1.3 评估 pipeline 安全设计 | **NeurIPS 2025** ✅ |
| RMB: Reward Model Benchmark ([arxiv 2410.09893](https://arxiv.org/abs/2410.09893)) | 任务 1.3 VM 质量评估 | **ICLR 2025** ✅ |
| RewardBench 2 ([arxiv 2506.01937](https://arxiv.org/abs/2506.01937)) | 任务 1.3 VM 质量评估 | 2026 preprint |
| SP-PRM: ORM-derived Process Reward ([arxiv 2506.12446](https://arxiv.org/abs/2506.12446)) | 任务 5.1 PRM 数据收集 | 2025 preprint |
| ThinkPRM: Thinking Token Process Reward ([arxiv 2504.16828](https://arxiv.org/abs/2504.16828)) | 任务 5.1 thinking 模式 PRM | 2025 preprint |
| DG-PRM: Dynamic Generalizable PRM ([arxiv 2507.17849](https://arxiv.org/abs/2507.17849)) | 任务 5.1 跨任务 PRM | **ACL 2025** ✅ |
| Skywork-VL Reward ([arxiv 2505.07263](https://arxiv.org/abs/2505.07263)) | 任务 6.1 多模态 VM 选型 | 2025 preprint |
| MSRL: Multi-Stage Multimodal RM ([arxiv 2603.25108](https://arxiv.org/abs/2603.25108)) | 任务 6.1 多模态 VM 训练框架 | **CVPR 2026** ✅ |
| BaseReward: Multimodal RM Baseline ([arxiv 2509.16127](https://arxiv.org/abs/2509.16127)) | 任务 6.1 多模态 VM 评估对照 | 2025 preprint |
| Reward Models Are Secretly Value Functions ([arxiv 2604.22981](https://arxiv.org/abs/2604.22981)) | 远景规划 C0a LLM-as-Q-Function | 2026 preprint |
| EntropyInfer: Rigid/Dynamic Attention Head Pruning ([arxiv 2606.09508](https://arxiv.org/abs/2606.09508)) | 远景规划 B2 VM 注意力头裁剪 | 2026 preprint |

---

## 附录：远期研究方向展望（6 个月计划以外）

> 以下方向均有调研依据，但因工程成本较高、效果尚不确定、或依赖前期工作完成，未纳入 2026-07 至 2026-12 执行计划。供 2027 年立项参考。

---

### A. VM 根本性替代——绕开 RM 前向传播（2027 年研究方向）

这类方向的共同目标是：**让主 LLM 不再依赖外部 VM 做 per-token 打分**，从而从根本上消除 L（VM 延迟）和 T（并发吞吐损失）。

**A1. SAE 激活空间 Steering（DSPA，CMU，arxiv 2603.21461）**

用稀疏自编码器（SAE）在 LLM 内部激活空间直接施加对齐引导，完全不调用外部 VM（99.8% 激活值为零）。若效果可接近 4B VM，L/T 从根本解决。当前挑战：SAE 需离线训练，效果能否匹敌 4B VM 尚未验证。  
**建议**：先用 1-2 周做 paper-reading + 小规模可行性实验，再决定是否立项。

**A2. RM 梯度引导（探索性方向，无直接文献支撑）**

思路：在高熵位置用 RM 的**梯度信息**（而非分数）指导 token 选择，理论上可避免完整前向传播。注意：此方向**目前无顶会直接验证**——GGRO（UAI 2026，arxiv 2606.09635）常被援引，但该论文的实际贡献是验证"只在高熵位置干预"的 entropy-adaptive 策略（与 SIA 的 `--entropy_threshold` 一致），并不涉及梯度引导。此外，标准推理不维护梯度图，实现时需 gradient checkpointing 或近似方案，工程成本远高于直接前向传播，可行性有待独立实验验证再决定是否立项。

---

### B. VM 渐进效率优化（6 个月计划的自然延伸，2027 Q1）

**B1. VM 蒸馏：4B → 1.7B（NeurIPS 2024 研究背景）**

Month 3 的同词表 4B ARM VM 验证效果后，用 4B 作教师蒸馏出 1.7B 学生 VM。相关文献（[arxiv 2411.08302](https://arxiv.org/abs/2411.08302)、[arxiv 2405.19316](https://arxiv.org/abs/2405.19316)）初步显示大 RM 蒸馏小 RM 可保留约 80–90% 偏好判断能力，latency 降低约 2×，VRAM 占用减半——但上述具体数字**未经内部独立验证**，应视为量级参考而非确定结论。  
**建议时机**：2027 Q1，前提是 Month 3 的 4B ARM VM 效果经评估已达标。

**B2. VM 内部注意力头裁剪（EntropyInfer 思路，arxiv 2606.09508）**

将 VM 内部注意力头分为"结构性（Rigid）"和"语义决策型（Dynamic）"两类，Rigid 头可跳过或降精度计算。注意：EntropyInfer 原论文的 2.39× 加速是在 **LLM prefill、100k+ token** 场景下测得，而 SIA 的 VM 开销集中在 **decode 阶段的短序列打分**，场景差异大，"20–40% latency 降低"是外推估计，**无直接实验支撑**，实际收益需在 VM 上独立实测。前提：需先对当前 VM（Qwen3.5-4B）的注意力激活模式做离线分析，确认 Rigid 头比例是否足够大，否则收益更为有限。

**B3. RM Ensemble（通行工程实践）**

部署 2–3 个不同 VM，对分数取均值，降低单 VM reward hacking 风险。在 B1 完成后（1.7B VM 可用），两个 1.7B VM 的 VRAM 需求低于当前一个 4B VM，对吞吐影响可控。  
**当前前提**：先建立 reward hacking 监控指标（如高 VM 分但人工评分差的样本率），有证据后再引入 ensemble。

**B4. VM 独立 GPU 组部署（HybridFlow，EuroSys 2025，arxiv 2409.19256）**

HybridFlow 提出 ResourcePool 抽象，支持 LLM + RM 的 distributed 部署模式：VM 独占一组 GPU，其打分与主 LLM 的下一步 decode 真正并行，**VM 延迟完全移出 LLM 关键路径**，吞吐比理论上可从 35% 恢复到接近 100%。当前 b2 inproc 是 colocated 顺序模式；若未来有额外 GPU（1–2 张 A100 专用于 VM），distributed 模式是根本解法。  
**前提**：需额外 GPU 资源，2027 年扩容时优先评估；当前 4×A100 下不可行。

---

### C. 延后的轻量探索任务（6 个月内无固定排期）

**C0a. LLM-as-Q-Function PoC（原 Task 2.4，约 1 周）**

From r to Q\*（COLM 2024，arxiv 2404.12358）和 "Reward Models Are Secretly Value Functions"（arxiv 2604.22981，2026）均提出：LLM 的 token log-prob 差值 `log P(y|x, prefix) - log P(y|x)` 理论上近似 Q-function，意味着主模型 0GM-VL-35B 自身已携带足够 reward 信号，**可以完全不需要外部 VM**——L 和 T 从根本消失。实验成本极低（新增 `--rm_backend self` 模式，AlpacaEval 50 题对比即可），但优先级低于极小 judge（2.3）——若 2.3 成功则 2.4 意义降低，若 2.3 失败则 2.4 可作为替代方向快速验证。建议在 Month 4 极小 judge PoC 结论出来后，根据结果决定是否跟进。

**C0b. DPO 蒸馏探索（探索性，2-4 周）**

方向：收集 SIA 系统中 VM 偏好的生成轨迹，用 DPO 蒸馏进 0GM-VL-35B 主模型，让主模型内化对齐信号，无需推理时外挂 VM 也能保持对齐效果。

TITA（2025，arxiv 2510.21794）展示了推理时 log-ratio 方法（DPO 等价形式）有效，但那是推理时校正，不是训练时蒸馏。训练时 DPO 蒸馏在本次调研中**没有直接顶会证据支撑**，效果不确定。建议在 6 个月 roadmap 完成、有充足 VM 偏好数据后，作为独立探索性项目评估。

---

### E. 对齐效果深化（覆盖更多任务类型）

**C1. 多目标 VM（ArmoRM，arxiv 2406.12845，2024）**

训练多维度 reward head（helpfulness / correctness / safety 等），由 gating 网络按 prompt 类型动态加权：coding 任务偏 correctness，对话任务偏 helpfulness。当前 VM 是单目标 ORM，在跨任务场景下对齐效果覆盖不均匀。  
**主要障碍**：需要多维度偏好标注数据集；可在 Month 2 数据收集时同步打多维标签，为后续做准备。

**C2. PRM 与 token 级 VM 的融合**

若 Month 5 可行性评估结论为 go，PRM 给出步骤级分数后，如何将其降维分摊到 token 级（如一步内均匀分摊），在不额外构建 token 级偏好数据的情况下提升 token 级干预信号质量，是一个开放问题。目前无顶会直接验证，建议 PRM 实际上线后顺带做消融实验。

---

> **远期方向优先级参考**（前期工作完成后）：C0a/C0b（延后轻量探索，随时可启动）> B1 VM 蒸馏（低风险，效果可期）> E1 多目标 VM（效果覆盖）> A1 DSPA（高潜力，需验证）≥ A2 GGRO 梯度（高潜力，工程挑战大）> B2/B3（依赖具体指标）。

---

## <a name="appendix-cuda-graph"></a>附录：0GM-35B CUDA Graph 调试过程

**结论**：RM CUDA graph（vllm PIECEWISE 模式）在 0GM-35B 上不可用，已经三轮系统性修复，均失败。此问题与分词器无关，属于架构层面的根本限制。

### 数据对比

| 模式 | 稳态延迟（@500步以上）| vs eager |
|------|----------------------|---------|
| eager（当前生产）| 40–50ms / step | 基准 |
| PIECEWISE fix3（最终尝试）| 130–140ms / step | **慢 2.7–3.5×** |

### 三轮修复历程

| 轮次 | 修复内容 | 结果 |
|------|---------|------|
| fix1 | CUDAGraph isolation：RM forward 移入独立 CUDA stream，避免与主 LLM graph 冲突 | ❌ 失败：NaN 链式污染（§3.9）|
| fix2 | APC=0（禁用自动前缀缓存）+ max_num_batched_tokens=65536 | ❌ 失败：首个 request 后 piecewise 完全退化 |
| fix3 | 切换 SDPA 后端（flash_attn → math → xformers）逐一测试 | ❌ 失败：首个 request 成功，后续 request 全部失败（⚠️ 根因未解）|

### 根本原因分析

vllm PIECEWISE CUDA graph 的设计假设：decode 步骤 batch_size=1，每步 token 数固定 → 可预编译固定形状 graph。

RM 的实际 workload：topK=10 条**完整候选序列**（prefix + candidate token），每步相当于一次 prefill，batch_size 和序列长度均随时间变化 → 无法满足固定形状假设，graph 执行退化为 eager fallback 或触发形状不匹配错误。

同词表 VM 训练不改变 RM 的 prefill-heavy 本质，因此无法解决此问题。

### 影响

- 0GM-35B 延迟改善只能依赖消除跨分词器开销（~7ms），不依赖 CUDA graph
- 同词表 VM 上线后 b2_score_call p50 预计从 ~30ms 降至 ~23ms（而非 ~11ms）
- ~11ms 目标仅在 CUDA graph 问题被后续 vllm 版本修复后才可达
- VL-30B 上 11ms 成立是因为：① 同词表（无跨分词器开销）② CUDA graph 可用（VL-30B 未复现此 bug）

**参考实验**：`alpaca-0gm35b-piecewise-fix3-20260610`；详细调试日志见 `doc/0gm-35b-sia-perf-breakdown-20260609.md` §7 和 `doc/cuda-graph-debugging-journal.md`。
