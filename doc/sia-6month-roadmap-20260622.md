# SIA Roadmap（2026-05 回顾 + 2026-07 ~ 2026-12 计划）

**撰写日期**：2026-06-22  
**执笔**：工程侧  

---

## Overview

### SIA 是什么

**SIA（Sparse Inference-time Alignment）**是一套在模型推理时实时干预输出的对齐系统。

通俗地说：大模型每生成一个词，SIA 在旁边同步评估候选词的"好坏"，把概率偏向质量更高的选项。全程不修改模型权重，仅在推理阶段介入——类似于给模型加了一个实时的"质检员"，每次出词前先过一遍。

### 当前成果（截至 2026-06-22）

两个月内完成核心系统，0GM-VL-35B 和 Qwen3-VL-30B 两个主力模型均已接近上线标准：

- **效果已验证**：0GM-VL-35B 开启 SIA 后，综合对话质量 AlpacaEval 胜率 **65.4%**（200 道题中，SIA 版本有 65.4% 被评为更优）；MMLU 准确率无明显下降
- **性能有代价**：高并发场景（16 路并发）吞吐量约为不开 SIA 的 **35%**，即同等硬件可服务的请求量减少约 65%

### 接下来 6 个月的计划

计划分三个阶段，核心目标是将 SIA 的性能代价从"显著"降低到"可忽略"，同时持续提升对齐效果。

**阶段一（Month 1）：减少打分频率，建立评估基准**  
通过智能跳过"不重要的 token"，将评分模型（Value Model）的调用次数减少 50% 以上，在几乎不损失对齐效果的前提下大幅降低开销；同时建立 GPT-4 标准评估基准，为后续所有改进提供可比较的量化依据。

**阶段二（Month 2–3）：换一个更快的评分模型**  
当前评分模型与主模型使用不同词表，每次打分都需要额外的"词表翻译"（约损耗 17ms）。训练一个词表完全一致的评分模型，预计将单次打分延迟从 ~30ms 降至 ~13ms；同时验证"乘积式融合"（比当前加法干预更精准的介入方式）能否进一步提升对齐效果。

**阶段三（Month 4–6）：更强的评分能力 + 多模态扩展**  
引入能理解"推理过程"的过程奖励模型（PRM），在关键决策点精准干预而非盲目打分；同时将 SIA 扩展至多模态，支持图片输入，覆盖更广的业务场景。

### 6 个月目标

| 维度 | 当前（2026-06） | 目标（2026-12） |
|------|--------------|--------------|
| 高并发吞吐（SIA vs 无 SIA） | **35%** | **≥ 72%** |
| 单次打分延迟 | ~30ms | ≤ 15ms |
| 多模态支持 | 仅文本 | 文本 + 图片 |
| 标准评估基准 | 尚未建立（GPT-4 judge）| Month 1 建立，持续追踪 |

---

## Month -2（2026-04-21 ~ 2026-05-21）：项目冷启动

> 项目于 2026-04-21 启动，一个月内完成前期调研、基础框架搭建和首次效果验证。

**主要完成事项：**

- **论文研读与原理理解**：阅读 SIA 论文（arxiv 2602.21215），理解稀疏干预在关键决策点介入的核心思路；参照原始实验代码（github.com/hurunyi/SIA）试跑论文自带的推理干预流程，确认方法可行性
- **SIA 核心实现**：基于 vLLM `LogitsProcessor` 的 token 级干预框架，OpenAI 兼容 HTTP API，支持 per-request `sia_weight` 动态调整
- **首次效果评估**：0GM-VL-35B AlpacaEval + MMLU 初跑，确认 SIA 对齐信号存在；发现 SIA 干预下主要在高熵位置（intervention rate ~20%）
- **VM 性能摸底**：SIA 单请求 ~40 tok/s vs noSIA ~114 tok/s，确认 VM 串行调用（每候选一次 GPU forward）是主要瓶颈
- **批量前向优化**：K 候选从 K 次串行 GPU forward 合并为 1 次 batch forward，VM 计算量降低约 60%
- **KV 前缀缓存探索**：实验 PyTorch DynamicCache 前缀复用，受 PyTorch 无 APC 机制限制，效果不理想
- **vLLM RM 后端建立**：实现 `--rm_backend vllm`，通过 HTTP 调用 vLLM classify 接口打分（为后续 b2 inproc 做铺垫）

**月末状态**：SIA 可运行，有初步效果验证，单请求 SIA/noSIA 比值约 35%，性能提升空间明确。

---

## Month -1（2026-05-22 ~ 2026-06-22）：架构突破 + 上线生产

> 完成核心架构升级，修复重大效果回归 bug，两个模型全部上线 marketplace。

**主要完成事项：**

**1. b2 inproc VM 架构落地**（5 月下旬）  
将 VM 从独立 HTTP 进程改为嵌入式 nested vLLM 实例（进程内函数调用），彻底消除网络往返开销（~20–30ms/次）。结合 VM CUDA Graph（Qwen3-VL-30B）和主 LLM CUDA Graph 修复（去除 PIECEWISE-only 限制）：
- Qwen3-VL-30B：HTTP ~36 tok/s → b2 inproc **78.3 tok/s**（**+117%**）
- 0GM-VL-35B：HTTP ~32 tok/s → b2 inproc **66–68 tok/s**（**+106%**）

**2. repetition_penalty 关键 Bug 修复**（6 月 4 日）  
发现 `repetition_penalty` 默认值 1.3 与 SIA logit 干预叠加，导致评估数据显示 SIA Δ = −13% 至 −75%（错误结论）。修复为 1.0 后，SIA 效果全面恢复正向：
- 0GM-VL-35B MMLU thinking 模式对照实验：SIA vs noSIA 准确率无明显下降

**3. 0GM-VL-35B 跨分词器优化**（6 月初）  
Stable prefix 方案消除跨 tokenizer BPE 边界合并导致的 APC 失效：VM 调用延迟从随序列长度线性增长（34ms→55ms）降至**固定 ~30ms**，端到端吞吐 +22%。

**4. 双模型上线 marketplace**（6 月中旬）  
0GM-VL-35B 和 Qwen3-VL-30B 均完成 Docker 部署、OpenAI 兼容 API、全套集成测试（run_all.sh）、max_model_len 扩展至 32768 tokens、多模态图片输入支持（图片请求 bypass SIA）。

**5. 高并发批量打分优化**（6 月 18 日）  
N 个并发请求的 VM 调用从串行 N 次合并为 1 次 GPU batch forward，叠加增量跨分词器前缀缓存（P-3）：
- conc=16 吞吐：239 → **369 tok/s**（**+54%**）
- conc=16 ITL 倍数：4.8× → **3.0×**

**6. 探索失败的路径（已排除）**（6 月 22 日）  
vllm classify runner（7–8× 更慢，APC 不工作）和 transformers + DynamicCache（5–8× 更慢，TP=4 优势无法复制）两条路线均已实测排除，避免后续重复探索。

---

## 当前状态（2026-06-22 末）

### 性能数据

| 场景 | 模型 | SIA | noSIA | 比值 |
|------|------|-----|-------|------|
| 单请求吞吐 | 0GM-VL-35B | 66–68 tok/s | ~112 tok/s | **59–61%** |
| 单请求吞吐 | Qwen3-VL-30B | 78.3 tok/s | ~122.8 tok/s | **64%** |
| conc=16 吞吐 | 0GM-VL-35B | 369 tok/s | 1040 tok/s | **35%** |
| 单请求 ITL | 0GM-VL-35B | ~34ms | ~9ms | **3.8×** |
| conc=16 ITL | 0GM-VL-35B | 41.3ms | 13.6ms | **3.0×** |

| 组件 | 数据 |
|------|------|
| VM per-call latency p50 | 0GM-VL-35B ~30ms，Qwen3-VL-30B ~17ms |
| VM 干预率（entropy_threshold=1.0）| ~20%（0GM-VL-35B），~25%（Qwen3-VL-30B）|
| 多模态支持 | ❌ text-only VM；图片请求 bypass SIA |

### 对齐效果

| 评估 | 结果 | 备注 |
|------|------|------|
| MMLU thinking 模式（0GM-VL-35B，150Q 对照）| SIA vs noSIA 无明显下降 | 已排除 rep_penalty bug |
| AlpacaEval win-rate（0GM-VL-35B）| **65.4%**（Skywork judge，200Q）| Skywork 作 judge 已有数字；GPT-4 judge 尚未跑，Month 1 建立标准评估基准 |
| AlpacaEval Skywork Δ（0GM-VL-35B）| **+5.45 reward（+22.7%）** | SIA mean 29.36 vs noSIA 24.01（191 对，实验 stable-prefix-20260610）|
| Qwen3-VL-30B AlpacaEval | 暂无结论（Skywork / GPT-4 均未正式跑）| Month 1 建立统一评估基准后补齐 |

### 三个核心痛点（当前状态）

| 痛点 | 当前状态 | 根因 |
|------|---------|------|
| **E 效果不显著** | MMLU 有信号，AlpacaEval 无基准 | VM 与 LLM 跨分词器噪声；无量化对比数据 |
| **L VM 延迟高** | 0GM-VL-35B ~30ms/call | dense 4B VM，memory-bound，无 CUDA graph |
| **T 高并发吞吐损失** | conc=16 仅 35% of noSIA | VM 调用仍占关键路径；每 token 都可能调用 |

---

## 执行顺序总览

```
Month -2 ██ 项目启动 ██ 批量前向 ██ 首次效果评估                    [已完成]
Month -1 ██ b2 inproc ██ Bug修复 ██ 上线生产 ██ 高并发优化           [已完成]
─────────────────────────── 当前（2026-06-22）──────────────────────
Month 1  ██ block-wise scoring ██ 双熵门控 ██ 评估基准（三项独立工程任务）
Month 2  ██ 两阶段粗过滤PoC ██ 数据收集 ████████ ARM VM 训练 ████████
Month 3  ████████ 同词表 VM 上线（含 MoE 选型）████████  ▶ 乘积式融合A/B
Month 4  ████ 更强VM（视M3结果）████  ██ accept/reject ██  ▶ 极小judge PoC ▶ 多模态数据准备
Month 5  ████████████████ PRM 训练（全月主线）████████████████
Month 6  ████ 多模态VM训练 ████  ████████ 多模态VM上线 ████████
```

已完成阶段为前两个月真实产出；执行计划阶段严格依赖前一阶段：前期降低 VM 调用频率 → 中期换更强模型 → 后期做根本性改造。

---

## Month 1（2026-07）：工程快赢——降低 VM 调用频率

**主线任务**：不改变模型，纯代码修改，把 VM 的有效调用频率从 ~20% 降到 ~5%。

### 任务 1.1：block-wise scoring（B=4）

修改 `SIALogitsProcessor`：每生成 4 个 token 才触发一次 VM 打分，而非每个 token 都触发。

> **完成后收益**：L/T VM 实际调用次数降至 ~25%（每 4 token 打分一次），b2_score_call 平均耗时降低 ~75%，conc=16 吞吐从 369 tok/s 预计升至 ~500+ tok/s。

- **理论依据**：arxiv 2503.02368 实测 per-token RM 比 block-wise(4) 慢 2.7-4.4×
- **工作量**：约 1 周（代码改动集中在 `apply()` 方法，需调整 token buffer 逻辑）
- **风险**：效果可能略降，需 A/B 对比，找最优 B 值

**补充备选——Vocabulary-wide Reward Head**（[arxiv 2502.04517](https://arxiv.org/abs/2502.04517)，ICML 2025）：当前 SIA 对 top-K 候选分别做 forward，即使 batch 合并仍是 K 条序列。该论文将 RM head 改为单次 forward 输出**整个 vocab 的 reward 分数**，从 K 次 → 1 次 forward，理论 L 改善 topK 倍。工作量：需修改 VM 推理接口，适合在 Month 2 训练同词表 VM 时一并实现，而非 Month 1 改动现有模型。本月只需了解架构方案。

### 任务 1.2：双熵门控替换单熵门控

现有门控：主模型熵 > θ 才干预（单熵）。  
改为：主模型熵 > θ₁ **且** VM 上一步评分也显示高不确定性时才干预（双熵）。  
减少低质量的无效干预，同时减少 VM 调用次数约 20-30%。

> **完成后收益**：L/T 在 block-wise 基础上叠加，VM 有效调用率再降 ~20-30%，端到端吞吐额外提升 ~10-15%；无效干预减少同时轻微改善 E（降低噪声干预比例）。

- **理论依据**：EASD（arxiv 2512.23765）验证分层熵门控在 speculative decoding 中比单阈值减少约 25% 无效干预；TARo（arxiv 2603.18411）进一步表明基于当前步不确定性的自适应路由优于固定比例干预。
- **补充文献**：Learning Adaptive Decoding（[arxiv 2603.09065](https://arxiv.org/abs/2603.09065)，2026 preprint）表明仅用熵判断不够，建议引入 learned routing policy（小分类头，基于 token 上下文特征决定是否干预）——可与双熵门控串联作为 Month 1 进阶探索。"To Intervene or Not"（[arxiv 2606.11201](https://arxiv.org/abs/2606.11201)，ACL 2026）提供概率性干预框架，可在固定熵阈值之上叠加不确定性置信度，预期再减少 ~30% 无效调用。
- **工作量**：约 1 周（`apply()` 中增加一个状态缓存）

### 任务 1.3：建立评估基准

选定 200 题 AlpacaEval 作为固定测试集，建立可复现的评估 pipeline。这是后续所有优化的衡量标准，不建基准则无法证明改进。

> **完成后收益**：非直接性能优化，但为后续所有 E 改进提供可量化依据——每次优化有 win-rate 数字可对比，避免"改了但说不清效果"的困境；VM 质量基准同时为 Month 2 训练目标设定参照系。

- **评估框架**：AlpacaEval 2.0（arxiv 2404.04475）以 GPT-4 为 judge 计算 win-rate，200 题固定集保证跨实验可比性，相比人工评估成本降低约 100×。
- **VM 质量基准**：同时使用 RMB（[arxiv 2410.09893](https://arxiv.org/abs/2410.09893)，ICLR 2025）和 RewardBench 2（[arxiv 2506.01937](https://arxiv.org/abs/2506.01937)，2026）评估 VM-Qwen3-4B 的原始评分质量，建立"VM 能力 → SIA 效果"的映射关系，指导 Month 2 VM 训练目标。
- **⚠️ Reward Hacking 风险**：Inference-Time Reward Hacking（[arxiv 2506.19248](https://arxiv.org/abs/2506.19248)，NeurIPS 2025 Spotlight）表明推理时 RM 干预会导致分布外样本触发奖励黑客，尤其在 --weight 较大时。Task 1.3 的评估 pipeline 应同步设计奖励异常检测指标（如输出长度分布、重复率、困惑度），避免 SIA 在某些问题上"刷高分实为退化"。
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

**为什么同词表是最高优先级**：当前 VM（Qwen3-4B）和主 LLM（0GM-VL-35B）使用不同 tokenizer（Qwen3 32K 词表 vs Qwen3.5 248K 词表）。这意味着 VM 看到的 prefix token ID 与主 LLM 生成的 token 不是一一对应的，评分信号存在系统性噪声。同词表训练一次性解决：

- 跨分词器噪声消除 → 评分精度提升（E）
- stable prefix 替代方案可退役 → 消除跨分词器 CPU 编码开销（~2ms）和 BPE 边界 APC miss（~5ms），b2_score_call 从 ~30ms 降至 ~13ms（L）
- 每次调用读取的 HBM 数据量不变，但质量更好（T 间接改善）

> **注意（来自已有实验）**：RM CUDA graph（piecewise）在 0GM-VL-35B 上已完整试验并彻底失败——三层修复后 piecewise 比 eager 慢 2-3×（115-140ms vs 30ms）。根本原因：RM 是 prefill-heavy workload（每步 topk=10 条完整序列），vllm PIECEWISE 只优化 decode 步骤（固定 batch=1），对 prefill 无效。同词表 VM 的 L 改善**不依赖 CUDA graph**，而是靠消除跨分词器开销（实验 `alpaca-0gm35b-piecewise-fix3-20260610` 已确认 CUDA graph 死路，见 `doc/0gm-35b-sia-perf-breakdown-20260609.md` §7）。

### 任务 2.1：偏好数据收集

从现有生产流量中收集 SIA 干预前后的 token 偏好对，或使用 synthetic 方法（让主 LLM 对同一 prompt 生成 N 个候选，用现有 VM 打标签）。

> **完成后收益**：无直接性能改善，但是 Task 2.2 的前提条件；数据质量和规模直接决定 Month 3 同词表 VM 上线后的 E 效果上限。

- **工作量**：约 1-2 周
- **风险**：数据量不足时用 synthetic 补充

### 任务 2.2：VM-Qwen3.5-4B 训练——使用 ARM 训练目标

以 Qwen3.5-4B（248K 词表，与 0GM-VL-35B 相同）为 base 训练 reward head。

> **完成后收益**（Month 3 上线后生效）：E 跨分词器噪声消除，VM 评分质量显著提升；L b2_score_call 从 ~30ms → ~13ms（消除跨分词器编码开销 ~7ms）；若 vocabulary-wide head 实现，L 额外改善最多 topK 倍（topK=10 时理论最高 10×）。

**关键：训练目标选 ARM（Autoregressive Reward Model），而非传统 ORM（Outcome Reward Model）。**  
理由来自 GenARM（ICLR 2025，arxiv 2410.08193）：ARM 参数化**在理论上可将 frozen LLM 引导至 KL 正则 RL 框架内任意传统 RM 可实现的分布**，而 ORM 只对最终输出打分，逐 token 干预时信号粗糙、位置错位。同等 4B 规模下，ARM 目标比 ORM 目标理论信号质量更高，先验证这一点，再决定是否需要扩大模型规模。

同步评估是否有合适的小 MoE base 可选（VRAM 允许时 MoE per-call 读带宽更低）。

**🎯 架构设计目标——Vocabulary-wide Scoring Head**（来自 [arxiv 2502.04517](https://arxiv.org/abs/2502.04517)，ICML 2025）：  
训练新 VM 时，**不要沿用旧的"输入一条候选序列 → 输出一个标量"接口**，而是将 reward head 设计为"输入当前 prefix → 输出整个 vocabulary 的 reward 向量"。这样 SIA 对 top-K 候选的评分从 **K 次 forward → 1 次 forward**，L 理论改善 topK 倍（topK=10 时最多 10×）。实现上：RM head 从 `[hidden → scalar]` 改为 `[hidden → vocab_size]`，标签仍是偏好对。这是本轮 VM 训练**最高优先级的架构决策**，不单独占工作量，在训练开始前确认接口设计即可。

**补充训练技巧**：
- **低秩 Reward Head**（[arxiv 2407.04615](https://arxiv.org/abs/2407.04615)，TMLR 2025）：vocabulary-wide head 的矩阵从 `[d × V]` 分解为 `[d × r] × [r × V]`（r ≪ V），在 vocab=248K 的 Qwen3.5 词表下效果尤为明显，与上述架构目标天然兼容。
- **From r to Q\* 训练初始化**（[arxiv 2404.12358](https://arxiv.org/abs/2404.12358)，COLM 2024）：标注数据不足时，用主模型 0GM-VL-35B 的 log-prob 差值作为伪标签初始化 VM，降低冷启动门槛。
- **RED 蒸馏**（[arxiv 2411.08302](https://arxiv.org/abs/2411.08302)，EMNLP 2025）：从 holistic feedback 蒸馏 token 级奖励，可作为无 token 级标注时的备用路径。

- **工作量**：约 2-3 周（训练 + 初步 offline 验证）

### 任务 1.4：两阶段粗过滤 PoC——0.6B VM 作为 4B VM 的守门员（L，1–2 周）

> 本任务原计划 Month 1，因 Month 1 已有三项工程快赢任务，且双 LLM 实例架构具有不确定性，延后至本月工程工作（与数据收集并行）执行。

**理论依据**：RSD（ICML 2025，arxiv 2501.19324）、SSS（EMNLP 2025，arxiv 2508.15044）、GSI（ICLR 2026，arxiv 2506.04118）三篇独立工作验证了两阶段粗过滤模式：用极小模型做第一阶段快速筛选，只在不确定位置调用大模型。在 δ=0.7 阈值下，**约 48% 的调用可完全跳过大模型**。

**SIA 实现方案**：利用官方已有的 VM-Qwen3-0.6B-Base checkpoint，在 `SIALogitsProcessor.apply()` 中增加第一阶段：先对当前所有候选用 0.6B VM 打分，计算 N 个候选分数的**方差**；若方差 < 阈值 δ（所有候选分数接近，说明 0.6B 无法区分），则跳过 4B VM，直接用 0.6B 分数（或 0 偏置）作为干预信号；若方差 ≥ δ，则正常调用 4B VM 做精细打分。

> **完成后收益**：L 4B VM 调用减少约 40–50%，VM 平均开销从 30ms → ~17ms/step（0.6B 预筛约 3–5ms）；与 block-wise B=4 叠加后，VM 有效调用占比从 ~20% → ~3–5%；T conc=16 吞吐在 Month 1 基础上再提升。

- **工作量**：1–2 周（加载 0.6B VM + 修改 apply() 逻辑 + A/B 对比验证）；与 2.1 数据收集并行执行
- **架构前置验证（PoC 第一步）**：同进程内同时运行两个 `LLM()` 实例（0.6B + 4B）并配合 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 是**尚未验证的架构**。vllm v1 的 EngineCore 以 spawn 模式启动，两个 InprocClient 共进程的行为未经实验确认，有潜在冲突风险。**PoC 第一天应先只验证两个 LLM() 实例能否在同进程内稳定并存**，再进行效果实验，避免 3-5 天白费。
- **补充优化——cross-request 前缀 pin**（与 PoC 并行，约 3-5 天）：BatchLLM（[arxiv 2412.03594](https://arxiv.org/abs/2412.03594)）发现 vLLM APC 在高并发长前缀场景下因 LRU 驱逐导致命中率仅 6.3%（vs 显式 pin 的 92.6%）。SIA VM 的每条输入以 `system_prompt + partial_response` 为前缀，在 conc=16 时多请求共享同一 system_prompt 前缀块，当前 stable-prefix 优化已解决 per-request 内 KV 复用，但 **cross-request 维度**的前缀块仍可能被 LRU 驱逐。通过向 vLLM b2 后端注入前缀 pin 标记，可在 L+T 维度获得额外收益，预期 conc=16 VM per-call 延迟再降 10–20%。
- **风险**：① 双 LLM 实例架构冲突（优先验证）；② 0.6B 预筛准确率不足，导致部分高价值干预被错误跳过；需 A/B 对比效果损失 vs 速度收益
- **成功标准**：双 LLM 实例架构稳定运行（前提）+ 4B VM 调用减少 ≥ 30% + AlpacaEval win-rate 损失 ≤ 1%

### Month 2 交付标准

| 指标 | Month 2 目标 | Month 1 基线 |
|------|-------------|------------|
| 同词表 ARM VM 训练 | **训练完成，offline 验证通过** | — |
| MoE 选型结论 | **dense 4B 或 MoE 选型确定** | — |
| 两阶段粗过滤 PoC | **架构稳定性验证通过，有初步效果数据** | — |
| conc=16 SIA tok/s | **维持 ≥ 550**（本月无新工程优化）| ≥ 550 |

---

## Month 3（2026-09）：同词表 VM 上线——质效双提

**主线任务**：完成同词表 VM 训练，完成选型（dense 4B vs MoE），部署上线，替换现有 VM。

### 任务 3.1：同词表 VM 训练与选型

**Dense 4B**（默认选项）：与当前 VM 规模相当，同词表后消除噪声，可恢复 CUDA graph。

> **完成后收益**：确定 dense 4B vs MoE 选型，为 Task 3.2 部署提供技术决策依据；ARM 训练目标理论上优于 ORM，同词表 + ARM 组合显著提升 E 评分质量。

**MoE 选型评估**（如有合适的 pretrained MoE base）：MoE 在 memory-bandwidth-bound 场景下，每次 forward 只读激活参数（约 3B），理论上比同质量 dense 更快。但**需注意 vllm 加载全量专家权重**（不只加载激活部分）。以 30B-A3B 为例：

| 对比 | 当前 VM（Qwen3-4B dense）| MoE 30B-A3B |
|------|--------------------------|-------------|
| 每 GPU forward 读带宽 | ~0.5 GB | ~0.375 GB（激活参数 ÷TP=4）|
| 模型能力 | 4B | 接近 30B 水平 |
| VRAM 占用（TP=4）| ~2 GB/GPU | **~15 GB/GPU**（30B × 2 bytes ÷ 4 GPUs，全量专家权重）|

> **⚠️ VRAM 不可行**：当前 4×A100 80GB 配置下，主 LLM 占 75%（60 GB/GPU），4B VM 占 13%（10.4 GB/GPU），剩余约 10 GB/GPU。30B-A3B MoE VM 需要 **~15 GB/GPU** 仅用于权重（超出可用空间）。**30B-A3B MoE 在现有硬件上无法作为 VM**，除非降低主 LLM 的 gpu_mem_utilization（会缩短最大 context length 和并发数）。
>
> **可行的 MoE 选项**：若有 Qwen3.5 系列的小型 MoE（如 7B-A3B，VRAM ~3.5 GB/GPU），可以考虑。但目前无公开可用的 Qwen3.5-7B-A3B checkpoint 用于 VM 训练，此选项需等待合适基座模型发布。

**默认选择 dense 4B**（Qwen3.5-4B）：与当前 VM 规模相当，同词表后消除噪声，VRAM 确认可放。

### 任务 3.2：同词表 VM 部署与验证

替换生产 VM，重新跑 AlpacaEval，验证效果提升。

> **完成后收益**：E 跨分词器噪声彻底消除，AlpacaEval win-rate 首次可量化对比；L b2_score_call 从 ~30ms → ~13ms（−57%）；T conc=16 吞吐从 ≥550 升至 ≥750 tok/s。

### 任务 3.3：乘积式分布融合 A/B 实验（1-2 天，来自 LLMdoctor）

**来源**：LLMdoctor（arxiv 2601.10416，2026 年 1 月）将加法式 logit biasing 改为乘积式分布融合，在 AlpacaEval 类对比中 62.10% win vs. GenARM（ICLR 2025），76.00% win vs. ARGS。

> **完成后收益（如成功）**：E AlpacaEval win-rate 额外 +3-5%；改动仅修改 `SIALogitsProcessor.apply()` 融合逻辑，工程成本极低可随时回滚；结果直接决定 Month 4 accept/reject 实验的优先级。

核心公式：

> 当前 SIA：`logits[token] += weight * RM_score`  
> 乘积式改法：`α * log_prob_base[token] + β * log_prob_rm[token]`（对数空间加权平均）

**这一改动不需要重新训练 VM**，只修改 `SIALogitsProcessor.apply()` 的 score 融合逻辑，工程成本极低。结合 Month 3 部署的新同词表 VM，可同步验证三组 A/B 配置：
1. 旧 VM + 加法式（当前生产，对照组）
2. 新 ARM VM + 加法式
3. 新 ARM VM + 乘积式

**⚡ 提前快速验证**：如 Month 1 有余量（block-wise 和双熵门控提前完成），可在 Month 1 末用现有旧 VM 先跑一次"乘积式 vs 加法式"快速对比（1-2 天，AlpacaEval 50 题），提前获得信号，不必等到 Month 3。

**优先级依据**：成本极低（1-2 天），证据强（3-0 验证），且结果直接决定 Month 4 accept/reject 实验的必要性——若乘积式融合已带来显著效果提升，则 accept/reject 实验优先级可降低。

- **工作量**：1-2 天代码 + 约 1 周 A/B 验证
- **风险**：极低（改动可随时回滚）
- **成功标准**：乘积式 win-rate 高于加法式 ≥ 1%（统计显著）

### Month 3 交付标准

| 指标 | Month 3 目标 | Month 2 基线 |
|------|-------------|------------|
| conc=16 SIA tok/s | **≥ 750** | ≥ 550 |
| b2_score_call p50 | **≤ 15ms**（同词表消除跨分词器开销，非 CUDA graph）| ~30ms |
| AlpacaEval win-rate vs noSIA | **可量化，有正提升** | 难以量化 |
| 乘积式 vs 加法式融合对比 | **有结论：乘积式是否更优** | — |

### Month 3 系统备选方向（如有余量，1-2 周探索）

以下方向有文献依据但尚未纳入主线，Month 3 可作为探索性实验并行启动：

1. **Hydragen 共享前缀 Attention**（[arxiv 2402.05099](https://arxiv.org/abs/2402.05099)，ICLR 2025）  
   SIA 的 VM scoring 中，topK 个候选共享完全相同的前缀（prompt + 已生成 token）。Hydragen 将这部分 attention 抽取出来做一次 forward，理论上将 VM 候选评分从 K 次独立计算 → 1 次共享前缀 + K 次极短 suffix attention，T 改善显著。需修改 VM serving kernel，工作量约 2 周。

2. **AsyncSpade 异步 VM 调用**（[arxiv 2510.07486](https://arxiv.org/abs/2510.07486)，arXiv 2025）  
   将 `SIALogitsProcessor` 的同步 VM 调用改为异步（VM 调用与主 LLM decoding 并行），实测 20-50% TPOT 减少。改造 `apply()` 的 blocking call 为 prefetch/overlap 模式，约 2 周工作量。

---

## Month 4（2026-10）：VM 能力升级 + 干预策略实验

**主线任务**：基于 Month 3 的 ARM 4B 验证结论，决定是否扩大模型规模；同步实验 accept/reject 干预模式（作为探索性实验，不作强承诺）。

### 任务 4.1：规模决策——ARM 4B 效果若不足，训练 ARM 8B

**决策逻辑**（来自文献调研）：文献（GenARM，ICLR 2025，arxiv 2410.08193）表明 VM 的关键在于**训练目标**而非模型规模——ARM（Autoregressive RM）4B 理论上优于 ORM 8B。Month 3 完成 ARM 4B 上线后，先用 AlpacaEval 验证效果。

> **完成后收益**：E AlpacaEval win-rate ≥ +5% vs noSIA；若 ARM 4B 已足够则节省训练资源，将预算提前投入 Month 5 PRM 或多模态；若需扩至 8B，block-wise 已降低调用频率，8B 更高的 per-call latency 可被 b2 承受。

- 若 ARM 4B 的 win-rate 已达 ≥ +5%：**跳过 8B 训练**，把资源投入 Month 5 的 PRM 或多模态
- 若 ARM 4B 效果仍不足：在 ARM 训练目标下扩大到 8B（此时 block-wise 已降低调用频率，8B 的更高 per-call latency 可承受）

- **工作量**：2-3 周（视 Month 3 结果决定是否执行）

### 任务 4.2：accept/reject 干预模式实验（探索性）

当前模式：VM 分数加到 logit 上（logit biasing）。  
新模式：VM 分高 → 接受 top-1；VM 分低 → 强制从 top-K 重采样。

> **完成后收益（如成功）**：E token 选择更精准，理论上减少"偏差注入"；确定最终生产干预策略，为 Month 5 长期方向打好基础；若效果不如 logit biasing，则果断排除，避免后续重复实验。

**注意**：RSD（ICML 2025，arxiv 2501.19324）的 accept/reject 工作在**推理步骤/序列级别**，而非 token 级别，且使用的是两个独立大小模型的架构，与 SIA 的单模型 + 外挂 VM 不同。**token 级 accept/reject 在顶会文献中缺乏直接验证**，本任务作为探索性实验，结果不确定，需 A/B 实测。

**前置条件（来自 2026 调研）**：Month 3 的 Task 3.3 乘积式融合实验成本极低（1-2 天）且有 3-0 证据支持，应先于 accept/reject 完成验证。若 Month 3 的乘积式融合结果已带来 ≥ +3% win-rate 提升，本任务优先级可降低，不必在 Month 4 强行执行；若 Month 3 乘积式融合效果不显著，则在 Month 4 推进 accept/reject。

- **工作量**：1-2 周实验（视 Month 3 乘积式融合结论决定是否执行）
- **成功标准**：accept/reject 模式的 AlpacaEval win-rate 高于当前最优融合模式

### 任务 2.3：极小 judge PoC（延后至本月，利用条件性余量）

> 本任务原计划 Month 2 并行执行，因 Month 2 数据收集+训练主线已满，延后至本月。Month 4 的 4.1/4.2 均为条件性任务，若 ARM 4B 已达标且乘积式融合已充分，本月有余量恰好完成此 PoC。

**来源**：Judge Decoding（ICLR 2025，arxiv 2501.19309）用 16.4k 参数线性层替换 speculative decoding 的接受准则，500 条偏好对、1.5 小时训练，实现 3.9–9.7× 加速。

**SIA 类比**：在主 LLM（0GM-VL-35B）的 LogitsProcessor 内部，基于 top-K logit 分布训练一个**极小线性评分头**（<10M 参数），替代外部 4B VM。如果可行，per-call latency 从 ~30ms 降至 <0.1ms，L/T 根本解决。

> **完成后收益（如成功）**：L VM per-call latency 从 ~30ms → <0.1ms（300×）；T VM 完全移出关键路径，吞吐接近 noSIA 水平；即使部分成功，也能大幅降低 VM 调用频率。

- **工作量**：约 1 周代码 + 1 周效果对比
- **风险**：精度可能不足以指导 token 选择（VM 信号比 speculative decoding 的接受/拒绝更细）
- **成功标准**：极小 judge vs 4B VM 的 AlpacaEval win-rate 差距 ≤ 3%
- **失败处理**：如精度不足，实验结论也有价值（证明该方向的边界）

### 任务 4.3：多模态 VM 数据收集启动（1-2 周，为 Month 6 训练做准备）

**目的**：多模态 VM 训练（Month 6 主线）的前置工作是多模态偏好数据构建，数据规模决定训练质量。利用本月可能存在的余量提前启动，避免 Month 6 开始时数据还没有。

> **完成后收益**：无直接性能改善，但为 Month 6 多模态 VM 训练消除数据等待瓶颈，相当于把 Month 6 的 4 周工作量压缩到 2-3 周。

- **内容**：收集图文混合偏好对（Qwen3-VL-30B 作为 reference LLM）；数据来源：生产流量中的图文请求 + synthetic VQA/图文对话数据
- **工作量**：1-2 周（轻量，可在 4.1/4.2 运行期间并行）
- **前提**：仅在本月有余量时执行；若 4.1 需训 8B + 4.2 需实验，则顺延至 Month 5

### Month 4 交付标准

| 指标 | Month 4 目标 | Month 3 基线 |
|------|-------------|------------|
| AlpacaEval win-rate vs noSIA | **≥ +5%（ARM 4B 或 ARM 8B）** | 有正提升 |
| 干预模式对比 | **logit bias vs accept/reject 有实测数据** | — |
| 极小 judge PoC | **完成实验，有效果对比数据**（视余量）| — |
| 多模态 VM 数据 | **已启动收集**（视余量）| — |
| conc=16 SIA tok/s | **维持 ≥ 750** | ≥ 750 |

---

## Month 5（2026-11）：PRM——推理任务专项优化

**主线任务**：全力完成 Process Reward Model 的数据准备、训练和上线，将 VM 在推理/数学/代码类任务中的调用频率从 O(token) 降至 O(step)。多模态 VM 数据收集在 Month 4 已并行准备，训练和上线集中在 Month 6。

### 任务 5.1：Process Reward Model（PRM）训练——聚焦推理/数学/代码任务

当前 VM 是 token 级 outcome reward model，信号粒度极细，噪声大。PRM 改为**推理步骤级**打分（在自然推理分隔点评估），调用频率从 O(token) 降至 O(step)，信号质量更高。

> **完成后收益**：E 推理/数学/代码类任务效果显著提升（step 级奖励信号比 token 级更精准）；L/T VM 调用频率从 O(token) → O(step)，降低 3-5×，吞吐损失大幅减少。

**适用范围说明**（基于文献）：ICLR 2024 对 ORM vs PRM 的系统对比（Lightman et al.，[arxiv 2305.20050](https://arxiv.org/abs/2305.20050)）实验全部在 MATH 数学数据集上进行。PRM 优于 ORM 的结论在**推理/数学/代码**类任务上有顶会支撑，在通用问答/对话类任务上缺乏直接验证。**本任务优先针对推理类任务，不宜对通用指令跟随场景过度承诺。**

**补充文献**：PRM as Unified Control Signal（[arxiv 2602.01070](https://arxiv.org/abs/2602.01070)，2026 preprint）将推理过程形式化为"迭代轨迹生成与选择"，用 step-level PRM 在生成中途剪枝低 reward 候选（支持 beam search 和 lookahead search），机制描述 3-0 验证通过。与 SIA 粒度互补：SIA 做 token 级局部干预，PRM 做 step 级轨迹选择，两者可叠加。

- **数据需求**：构建 step-level preference 数据集（数学/推理类，每步打好/坏标签）
- **补充文献**（来自综合调研）：
  - **SP-PRM**（[arxiv 2506.12446](https://arxiv.org/abs/2506.12446)，2025）：从 ORM 推导过程奖励，**大幅降低 PRM 训练数据收集门槛**——无需 step 级人工标注，利用已有 outcome 偏好数据自动生成 step 奖励信号，直接适合 SIA 资源约束。
  - **ThinkPRM**（[arxiv 2504.16828](https://arxiv.org/abs/2504.16828)，2025）：在 thinking token 级别应用 PRM，对应 Qwen3 的 `/think` 模式；为推理过程中的每个思维步骤打分，与 SIA 在 thinking 模式下的应用高度匹配。
  - **DG-PRM**（[arxiv 2507.17849](https://arxiv.org/abs/2507.17849)，ACL 2025）：动态 PRM，跨任务泛化性强，减少 SIA 对特定任务标注数据的依赖；适合从数学推理扩展到代码生成。
- **工作量**：4 周（数据准备 2 周 + 训练 + 验证 2 周）

### Month 5 交付标准

| 指标 | Month 5 目标 |
|------|-------------|
| PRM 上线（推理类任务）| 推理任务 VM 调用次数 ≤ 20% of token 数 |
| 多模态 VM 数据 | **收集完成**（在 Month 4 已启动，本月收尾）|

---

## Month 6（2026-12）：多模态 VM 完整交付

**主线任务**：完成多模态 VM 从训练到上线的完整交付。数据在 Month 4（Task 4.3）已并行准备，本月专注训练和部署。

### 任务 5.2：多模态 VM 训练（接续 Month 4 数据准备）

> 本任务数据收集在 Month 4 Task 4.3 已启动，本月完成训练和初步验证。

**为什么需要多模态 VM**：当主 LLM 是 Qwen3-VL-30B-A3B-Instruct 时，用户输入包含图片，当前 text-only VM 完全看不到图片内容——VM 评分等于盲猜，干预可能适得其反。

**解法**：以多模态 VLM（如 Qwen3-VL-4B）为 base，训练 reward head，使 VM 能够同时理解图文 context。

- **适用范围**：仅对 VLM 主推理 LLM 有意义（纯文字 0GM-VL-35B 无需此功能）
- **参考说明**：训练流程参考 SIA 原论文（arxiv 2602.21215）的 reward head 训练方案，数据构建参考 ArmoRM（arxiv 2406.12845）的多维偏好标注框架。
- **补充文献**：
  - **Skywork-VL Reward**（[arxiv 2505.07263](https://arxiv.org/abs/2505.07263)，2025）：当前最强开源视觉语言 RM 之一，可直接作为 backbone 选型参考或微调起点。
  - **MSRL**（[arxiv 2603.25108](https://arxiv.org/abs/2603.25108)，CVPR 2026）：多阶段多模态 reward modeling，提供完整的 VLM RM 训练框架。
  - **BaseReward**（[arxiv 2509.16127](https://arxiv.org/abs/2509.16127)，2025）：强 baseline 多模态 RM，用于评估训练质量的对照基准。
- **工作量**：2-3 周训练（数据已备）

### 任务 6.1：多模态 VM 上线（Qwen3-VL 场景完整支持）

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

| 指标 | 当前（2026-06）| 6 个月后目标 |
|------|----------------|------------|
| conc=16 SIA tok/s | 369 | **≥ 750** |
| SIA/noSIA 吞吐比 | 35% | **≥ 72%** |
| SIA/noSIA ITL 倍数 | 3.0× | **≤ 1.5×** |
| AlpacaEval win-rate vs noSIA | 难以量化 | **≥ +5%（可量化证明）** |
| 多模态 VLM 场景支持 | ❌ | **✅** |
| VM 是否需要每 token 调用 | 是（~20% token）| **否（block-wise + PRM，~5% 以下）**|

---

## 里程碑时间线

```
2026-07 末  conc=16 tok/s ≥ 550，评估基准建立
2026-08 末  同词表 VM 训练完成，两阶段粗过滤 PoC 有结论
2026-09 末  conc=16 tok/s ≥ 750，同词表 VM 上线，效果首次可量化
2026-10 末  AlpacaEval win-rate ≥ +5%，最优干预模式确定，极小 judge PoC 有结论
2026-11 末  PRM 上线，推理任务 VM 调用进一步降低
2026-12 末  多模态 VM 训练完成并上线
```

---

## 学术调研关键发现（2026-06）

本 roadmap 在制定前进行了系统性文献调研（覆盖 NeurIPS、ICML、ICLR、ACL 等顶会，108 个并行验证智能体，25 篇论文，121 条论断经三票对抗验证）。以下三项发现值得重点关注：

### 发现一：高并发吞吐是学术空白，SIA 有机会贡献原创研究

当前所有 token 级对齐论文（包括 ICLR 2025 的 GenARM、ICML 2025 的 RSD 等顶会工作）**均只在单请求或小并发（≤4）场景下评估效果**，没有任何顶会论文系统研究"高并发连续批处理下 token 级干预的吞吐-效果权衡"。

这意味着我们在生产环境中观察到的 T 问题（conc=16 吞吐下降 64%）是一个**尚未有学术解答的真实工程难题**。如果 SIA 项目系统性地：

- 建立高并发场景下的评估基准（吞吐 vs 对齐效果 Pareto）
- 验证 block-wise scoring / PRM 等方案在高并发下的实际效果
- 发表相关结果

**有可能成为该方向的第一批顶会投稿**，填补当前学术空白。

---

### 发现二：Judge Decoding 的"极小 judge"思路对 SIA 有重大潜力，尚无人探索

ICLR 2025 发表的 Judge Decoding（arxiv 2501.19309）提出了一个反直觉的结论：用**仅 16,384 个参数（16.4k）的线性投影层**替换大模型的 token 接受准则，以 500 条标注数据、不到 1.5 小时训练，实现 Llama-405B 推理 **3.9–9.7× 加速**。

当前 SIA 使用 4B 参数的 VM 打分，每次调用延迟 ~30ms。

**尚未有任何工作**探索：能否将 Judge Decoding 的极小 judge 思路移植到 SIA——用一个参数量极小的打分头（如 10M 以下）替代 4B VM，训练数据同样来自偏好对，推理时几乎零延迟（参数量降低 400×，延迟可能降低 50–100×）？

这一方向的代价是打分精度可能下降，但对于 SIA 已有熵门控（只有 ~20% 的 token 触发打分）的场景，极小 judge 只需在关键位置做粗粒度好/坏判断，精度要求可能并不高。**这是 6 个月 roadmap 之外、值得单独立项探索的研究方向。**

---

### 发现三：该领域论文的性能数字普遍存在夸大，需谨慎参考

调研过程中对 25 条具体性能论断进行了三票对抗验证，**15 条（60%）被否定**，典型案例：

| 论文声称 | 验证结果 |
|---------|---------|
| TARo：比 baseline 提升 +22.4% | ❌ 0-3 否定（数字无法在原文中核实）|
| TARo：比现有 token 级方法提升 +8.4% | ❌ 0-3 否定 |
| TITA：在 LLaVA-1.5 上 MMVet +8.6%、POPE +6.7% | ❌ 1-2 否定 |
| RSD：比并行解码方法平均提升 +3.5 准确率 | ❌ 0-3 否定 |
| GSI：端到端延迟降低 28%、吞吐提升 51% | ❌ 1-2 否定 |

**实践建议**：在评估外部论文声称的效果改善时，不应直接引用其论文数字；在对 SIA 自身效果做对外宣传时，应确保数字来自可复现的独立评估（AlpacaEval、MMLU 等标准 benchmark），而非内部测试集。

---

### 发现四：2026年补充调研的三项新信号

*（2026 年 6 月补充调研，针对 2026 年 1–6 月 arxiv，103 个验证智能体，15 条通过验证）*

**（1）稀疏干预范式获三方独立验证**

GGRO（UAI 2026，arxiv 2606.09635）、SeLaR（2604.08299）、AdaDec（FSE 2026，2506.08980）三篇互不知情的工作均独立提出"只在高熵/低置信度位置干预"策略，与 SIA 的 `--entropy_threshold` 设计完全一致。这是 2026 年对 SIA 核心设计直觉的**外部学术背书**，同时也意味着该方向在学术界已不是新颖方向，SIA 的差异化价值要靠工程上高并发吞吐的系统性研究来体现（见发现一）。

**（2）乘积式分布融合有望低成本提升 E——已加入 Month 3**

LLMdoctor（arxiv 2601.10416，62.10% win vs. GenARM，3-0 验证）将加法式 logit biasing 改为乘积式分布融合，无需重新训练 VM，工程成本 1-2 天。**已加入 roadmap Month 3 Task 3.3**。该实验结果也决定 Month 4 accept/reject 实验的必要性。

**（3）SAE Steering（DSPA）是 6 个月 roadmap 以外的中期侦察方向**

CMU 的 DSPA（arxiv 2603.21461，3-0 机制验证）用稀疏自编码器在 LLM 激活空间直接施加对齐引导，**完全绕开外部 VM/RM 前向传播**（99.8% 激活值为零）。若效果可与 4B VM 相当，可从根本上解决 L（VM 延迟）和 T（并发吞吐），因为 VM 从关键路径彻底移除。代价：SAE 需离线训练，效果能否匹敌 4B VM 尚未验证。**建议 2027 年作为独立研究方向评估**，不放入当前 6 个月执行计划。

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
| SIA 原论文 ([arxiv 2602.21215](https://arxiv.org/abs/2602.21215)) | 整体框架基础 / 任务 5.2 | 2026 preprint |
| AlpacaEval 2.0 ([arxiv 2404.04475](https://arxiv.org/abs/2404.04475)) | 任务 1.3 评估基准 | 2024 preprint |
| GenARM: Autoregressive Reward Model ([arxiv 2410.08193](https://arxiv.org/abs/2410.08193)) | 任务 2.2 ARM 训练目标 / 任务 4.1 | **ICLR 2025** ✅ |
| Judge Decoding ([arxiv 2501.19309](https://arxiv.org/abs/2501.19309)) | 任务 2.3 极小 judge PoC | **ICLR 2025** ✅ |
| Let's Verify Step by Step / ORM vs PRM ([arxiv 2305.20050](https://arxiv.org/abs/2305.20050)) | 任务 5.1 PRM 适用范围 | **ICLR 2024** ✅ |
| Scaling LLM Test-Time Compute ([arxiv 2408.03314](https://arxiv.org/abs/2408.03314)) | 整体方向验证 | NeurIPS 2024 Workshop ✅ |
| RSD: Reward-guided Speculative Decoding ([arxiv 2501.19324](https://arxiv.org/abs/2501.19324)) | 任务 1.4 两阶段过滤 / 任务 4.2 | **ICML 2025** ✅ |
| SSS: Stepwise Speculative Search ([arxiv 2508.15044](https://arxiv.org/abs/2508.15044)) | 任务 1.4 两阶段过滤依据 | **EMNLP 2025** ✅ |
| GSI: Generative Speculative Inference ([arxiv 2506.04118](https://arxiv.org/abs/2506.04118)) | 任务 1.4 两阶段过滤依据 | **ICLR 2026** ✅ |
| Iterative Value Function Optimization ([arxiv 2503.02368](https://arxiv.org/abs/2503.02368)) | 任务 1.1 block-wise scoring | 2025 preprint |
| EASD: Entropy-Aware Speculative Decoding ([arxiv 2512.23765](https://arxiv.org/abs/2512.23765)) | 任务 1.2 双熵门控 | 2025 preprint |
| TARo: Token-level Adaptive Routing ([arxiv 2603.18411](https://arxiv.org/abs/2603.18411)) | 任务 1.2 自适应路由参考 | 2026 preprint |
| LLMdoctor: Product-of-Distributions Fusion ([arxiv 2601.10416](https://arxiv.org/abs/2601.10416)) | 任务 3.3 乘积式融合 A/B | 2026 preprint |
| TITA: Token-level Inference-Time Alignment ([arxiv 2510.21794](https://arxiv.org/abs/2510.21794)) | 任务 6.2 DPO 蒸馏参考 | 2025 preprint |
| GGRO: Gradient-Guided Reward Optimization ([arxiv 2606.09635](https://arxiv.org/abs/2606.09635)) | 发现四：稀疏干预范式验证 / 附录 A2 | **UAI 2026** ✅ |
| SeLaR: Soft Embedding Alignment at Low-Confidence ([arxiv 2604.08299](https://arxiv.org/abs/2604.08299)) | 发现四：稀疏干预三方独立验证之二 | 2026 preprint |
| AdaDec: Pause-and-Rerank at High Uncertainty ([arxiv 2506.08980](https://arxiv.org/abs/2506.08980)) | 发现四：稀疏干预三方独立验证之三 | **FSE 2026** ✅ |
| PRM as Unified Control Signal for Reasoning ([arxiv 2602.01070](https://arxiv.org/abs/2602.01070)) | 任务 5.1 PRM 粒度互补支撑 | 2026 preprint |
| Seesaw: PP↔TP Dynamic Parallelism Switching ([arxiv 2503.06433](https://arxiv.org/abs/2503.06433)) | 工程优化参考（TP/PP 动态调度）| 2025 preprint |
| DSPA: SAE-based Activation Steering ([arxiv 2603.21461](https://arxiv.org/abs/2603.21461)) | 发现四 / 附录 A1 | 2026 preprint（CMU）|
| ArmoRM: Multi-Objective Reward Model ([arxiv 2406.12845](https://arxiv.org/abs/2406.12845)) | 任务 5.2 / 附录 C1 多目标 VM | 2024 preprint |
| Token-level MDP Formalization ([arxiv 2602.02572](https://arxiv.org/abs/2602.02572)) | 附录 D2 学术发表基础 | **ICML 2026** ✅ |
| Nudging: Uncertainty-gated Sparse Intervention ([arxiv 2410.09300](https://arxiv.org/abs/2410.09300)) | 任务 1.4 / 方向验证 | 2024 preprint |
| BatchLLM: Explicit Global Prefix Sharing ([arxiv 2412.03594](https://arxiv.org/abs/2412.03594)) | 工程优化参考 | 2024 preprint |
| HybridFlow: ResourcePool LLM+RM Co-deployment ([arxiv 2409.19256](https://arxiv.org/abs/2409.19256)) | 附录 B4 独立 GPU VM | **EuroSys 2025** ✅ |
| NEO: Asymmetric CPU-GPU Pipeline ([arxiv 2411.01142](https://arxiv.org/abs/2411.01142)) | 工程优化参考 | 2024 preprint |
| STEP: Memory-triggered Search Tree Pruning ([arxiv 2601.09093](https://arxiv.org/abs/2601.09093)) | 工程优化参考（压力感知降级）| 2026 preprint |
| RM Knowledge Distillation ([arxiv 2411.08302](https://arxiv.org/abs/2411.08302)) | 附录 B1 VM 蒸馏依据 / 任务 2.2 训练技巧 | 2024 preprint |
| RM Distillation: Reward Model Compression ([arxiv 2405.19316](https://arxiv.org/abs/2405.19316)) | 附录 B1 VM 蒸馏依据 | 2024 preprint |
| RM Ensemble / NeurIPS 2024 consensus | 附录 B3 reward hacking 防护 | NeurIPS 2024 ✅ |
| Cost-Effective RGTG: Vocabulary-wide Reward Head ([arxiv 2502.04517](https://arxiv.org/abs/2502.04517)) | 任务 1.1 备选 / 任务 2.2 VM 架构 | **ICML 2025** ✅ |
| Low-Rank RM Parametrization ([arxiv 2407.04615](https://arxiv.org/abs/2407.04615)) | 任务 2.2 VM scoring 加速 | **TMLR 2025** ✅ |
| From r to Q*: LLM as Q-Function ([arxiv 2404.12358](https://arxiv.org/abs/2404.12358)) | 任务 2.2 VM 训练初始化 | **COLM 2024** ✅ |
| Hydragen: High-Throughput Shared-Prefix Inference ([arxiv 2402.05099](https://arxiv.org/abs/2402.05099)) | 任务 Month 3 备选 / T 系统优化 | **ICLR 2025** ✅ |
| AsyncSpade: Asynchronous Sparse Decoding ([arxiv 2510.07486](https://arxiv.org/abs/2510.07486)) | 任务 Month 3 备选 / 异步 VM 调用 | 2025 preprint |
| To Intervene or Not: Probabilistic Gating ([arxiv 2606.11201](https://arxiv.org/abs/2606.11201)) | 任务 1.2 双熵门控进阶 | **ACL 2026** ✅ |
| Learning Adaptive LLM Decoding ([arxiv 2603.09065](https://arxiv.org/abs/2603.09065)) | 任务 1.2 learned routing | 2026 preprint |
| Inference-Time Reward Hacking ([arxiv 2506.19248](https://arxiv.org/abs/2506.19248)) | 任务 1.3 评估 pipeline 安全设计 | **NeurIPS 2025** ✅ |
| RMB: Reward Model Benchmark ([arxiv 2410.09893](https://arxiv.org/abs/2410.09893)) | 任务 1.3 VM 质量评估 | **ICLR 2025** ✅ |
| RewardBench 2 ([arxiv 2506.01937](https://arxiv.org/abs/2506.01937)) | 任务 1.3 VM 质量评估 | 2026 preprint |
| SP-PRM: ORM-derived Process Reward ([arxiv 2506.12446](https://arxiv.org/abs/2506.12446)) | 任务 5.1 PRM 数据收集 | 2025 preprint |
| ThinkPRM: Thinking Token Process Reward ([arxiv 2504.16828](https://arxiv.org/abs/2504.16828)) | 任务 5.1 thinking 模式 PRM | 2025 preprint |
| DG-PRM: Dynamic Generalizable PRM ([arxiv 2507.17849](https://arxiv.org/abs/2507.17849)) | 任务 5.1 跨任务 PRM | **ACL 2025** ✅ |
| Skywork-VL Reward ([arxiv 2505.07263](https://arxiv.org/abs/2505.07263)) | 任务 5.2 多模态 VM 选型 | 2025 preprint |
| MSRL: Multi-Stage Multimodal RM ([arxiv 2603.25108](https://arxiv.org/abs/2603.25108)) | 任务 5.2 多模态 VM 训练框架 | **CVPR 2026** ✅ |
| BaseReward: Multimodal RM Baseline ([arxiv 2509.16127](https://arxiv.org/abs/2509.16127)) | 任务 5.2 多模态 VM 评估对照 | 2025 preprint |
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

**A2. RM 梯度引导（GGRO 思路，UAI 2026，arxiv 2606.09635）**

在高熵位置用 RM 的**梯度信息**（而非分数）指导 token 选择，理论上可避免 RM 全量前向传播。当前挑战：标准推理不维护梯度图，需探索 gradient checkpointing 或近似梯度方案，工程可行性待研究。可与 A1 并行探索，代码实验成本低。

---

### B. VM 渐进效率优化（6 个月计划的自然延伸，2027 Q1）

**B1. VM 蒸馏：4B → 1.7B（NeurIPS 2024 研究背景）**

Month 3 的同词表 4B ARM VM 验证效果后，用 4B 作教师蒸馏出 1.7B 学生 VM。文献（[arxiv 2411.08302](https://arxiv.org/abs/2411.08302)、[arxiv 2405.19316](https://arxiv.org/abs/2405.19316)）表明大 RM 蒸馏小 RM 可保留约 80–90% 偏好判断能力，latency 降低约 2×，VRAM 占用减半。  
**建议时机**：2027 Q1，前提是 Month 3 的 4B ARM VM 效果经评估已达标。

**B2. VM 内部注意力头裁剪（EntropyInfer 思路，arxiv 2606.09508）**

将 VM 内部注意力头分为"结构性（Rigid）"和"语义决策型（Dynamic）"两类，Rigid 头可跳过或降精度计算，预期降低 VM per-call latency 20–40%。前提：需先对当前 VM（Qwen3.5-4B）的注意力激活模式做离线分析，确认 Rigid 头比例是否足够大，否则收益有限。

**B3. RM Ensemble（NeurIPS 2024 共识）**

部署 2–3 个不同 VM，对分数取均值，降低单 VM reward hacking 风险。在 B1 完成后（1.7B VM 可用），两个 1.7B VM 的 VRAM 需求低于当前一个 4B VM，对吞吐影响可控。  
**当前前提**：先建立 reward hacking 监控指标（如高 VM 分但人工评分差的样本率），有证据后再引入 ensemble。

**B4. VM 独立 GPU 组部署（HybridFlow，EuroSys 2025，arxiv 2409.19256）**

HybridFlow 提出 ResourcePool 抽象，支持 LLM + RM 的 distributed 部署模式：VM 独占一组 GPU，其打分与主 LLM 的下一步 decode 真正并行，**VM 延迟完全移出 LLM 关键路径**，T 吞吐比理论上可从 35% 恢复到接近 100%。当前 b2 inproc 是 colocated 顺序模式；若未来有额外 GPU（1–2 张 A100 专用于 VM），distributed 模式是根本解法。  
**前提**：需额外 GPU 资源，2027 年扩容时优先评估；当前 4×A100 下不可行。

---

### C. 延后的轻量探索任务（6 个月内无固定排期）

**C0a. LLM-as-Q-Function PoC（原 Task 2.4，约 1 周）**

From r to Q\*（COLM 2024，arxiv 2404.12358）和 "Reward Models Are Secretly Value Functions"（arxiv 2604.22981，2026）均提出：LLM 的 token log-prob 差值 `log P(y|x, prefix) - log P(y|x)` 理论上近似 Q-function，意味着主模型 0GM-VL-35B 自身已携带足够 reward 信号，**可以完全不需要外部 VM**——L 和 T 从根本消失。实验成本极低（新增 `--rm_backend self` 模式，AlpacaEval 50 题对比即可），但优先级低于极小 judge（2.3）——若 2.3 成功则 2.4 意义降低，若 2.3 失败则 2.4 可作为替代方向快速验证。建议在 Month 4 极小 judge PoC 结论出来后，根据结果决定是否跟进。

**C0b. DPO 蒸馏探索（原 Task 6.2，探索性，2-4 周）**

方向：收集 SIA 系统中 VM 偏好的生成轨迹，用 DPO 蒸馏进 0GM-VL-35B 主模型，让主模型内化对齐信号，无需推理时外挂 VM 也能保持对齐效果。

TITA（2025，arxiv 2510.21794）展示了推理时 log-ratio 方法（DPO 等价形式）有效，但那是推理时校正，不是训练时蒸馏。训练时 DPO 蒸馏在本次调研中**没有直接顶会证据支撑**，效果不确定。建议在 6 个月 roadmap 完成、有充足 VM 偏好数据后，作为独立探索性项目评估。

---

### E. 对齐效果深化（覆盖更多任务类型）

**C1. 多目标 VM（ArmoRM，arxiv 2406.12845，2024）**

训练多维度 reward head（helpfulness / correctness / safety 等），由 gating 网络按 prompt 类型动态加权：coding 任务偏 correctness，对话任务偏 helpfulness。当前 VM 是单目标 ORM，在跨任务场景下对齐效果覆盖不均匀。  
**主要障碍**：需要多维度偏好标注数据集；可在 Month 2 数据收集时同步打多维标签，为后续做准备。

**C2. PRM 与 token 级 VM 的融合**

Month 5 的 PRM 给出步骤级分数，如何将其降维分摊到 token 级（如一步内均匀分摊），在不额外构建 token 级偏好数据的情况下提升 token 级干预信号质量，是一个开放问题。目前无顶会直接验证，建议 Month 5 PRM 上线后顺带做消融实验。

---

### F. 原创学术发表机会

**F1. 高并发场景下 token 级对齐的吞吐-效果 Pareto（学术空白）**

所有现有顶会工作（包括 ICLR 2025 的 GenARM、ICML 2025 的 RSD）均在单请求或 conc ≤ 4 场景下评估，**无一研究高并发连续批处理场景**。SIA 若系统性对比 block-wise / PRM / 乘积式融合在 conc = 4/8/16/32 下的吞吐-效果 Pareto 并发表，有望成为该方向的首批顶会工作。  
**目标会议**：MLSys 2027 / ICML 2027 Systems Track；数据积累可从 Month 1 评估基准起步。

**F2. 基于 ICML 2026 MDP 框架形式化 SIA → 投稿 ICLR 2027**

Token-level 对齐已被 ICML 2026（arxiv 2602.02572）形式化为标准 MDP 框架，SIA 是该框架的一个具体实例（VM 估计 $Q(s_t, a_t)$，logit 空间加权）。以此框架重新表述 SIA 的理论保证——结合 ARM 训练目标 + 熵稀疏干预 + 乘积式融合的理论组合——加上 6 个月的工程实测数据，ICLR 2027（截稿约 2026-10）是可行的投稿窗口。

**F3. 极小 judge 作为独立学术贡献（条件触发：Month 4 PoC 结果积极）**

若 Month 4 Task 2.3（<10M 参数线性评分头替代 4B VM）的效果达标，效果-延迟权衡本身是原创贡献。可与 F1 合并：以极小 judge 作为效率方案、以高并发场景作为评估框架，合写一篇"SIA 高并发吞吐系统工作"，比分别发表更完整。

---

> **远期方向优先级参考**（前期工作完成后）：C0a/C0b（延后轻量探索，随时可启动）> B1 VM 蒸馏（低风险，效果可期）> E1 多目标 VM（效果覆盖）> A1 DSPA（高潜力，需验证）≥ A2 GGRO 梯度（高潜力，工程挑战大）> F1/F2/F3（学术发表，需 6 个月数据积累）> B2/B3（依赖具体指标）。
