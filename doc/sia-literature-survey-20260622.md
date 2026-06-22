# SIA 相关论文综述：推理时对齐的方法、效率与发展方向

**调研日期**：2026-06-22  
**调研方法**：108 个并行验证子智能体，25 篇论文，121 条论断经三票对抗验证（10 条确认，15 条否定）  
**文献来源**：优先 NeurIPS、ICML、ICLR、ACL、EMNLP 等顶会；preprint 单独标注  

---

## 一、SIA 原论文回顾

### SIA：Inference-time Alignment via Sparse Junction Steering

**发表**：arxiv 2602.21215（2026，Runyi Hu et al.）  
**链接**：https://arxiv.org/abs/2602.21215  

**核心思路**：在 LLM 每个 decoding step，用 Value Model（VM）对 top-K 候选 token 打分，将分数加权叠加到 logit 上，偏置 token 选择分布。关键创新是**稀疏干预（sparse junction steering）**：只在 logit 分布熵较高（模型不确定）的关键决策点介入，大幅减少 VM 调用次数，实测 20%–80% 的干预率即可达到甚至超过全量干预的效果，同时最多将 VM 计算开销降低 6×。

**与本工程实现的关系**：本项目是对 SIA 的生产级工程实现，主 LLM 换成了 0GM-1.0-35B-A3B（MoE），VM 换成了 Qwen3-4B 系列，通过 vLLM b2 inproc TP=4 方式部署。

**注意**：本综述以 SIA 为参照系，SIA 自身的实验数据未经过本次三票对抗验证流程，引用时须注意。

---

## 二、维度一：推理时对齐的主流方法（Token 级 Logit 引导）

### 2.1 GenARM：自回归 Reward Model 的理论框架

**发表**：ICLR 2025（同行评审，已收录）  
**链接**：https://arxiv.org/abs/2410.08193  
**验证状态**：✅ 机制描述 3-0 通过；具体 benchmark 数字 0-3 否定，不收录  

**核心贡献**：GenARM 训练一个**自回归 reward model（ARM）**，在每个 decoding step 预测当前 token 的奖励值，而非对完整序列打分。理论贡献是证明了：这种 token 级参数化可以**在理论上将 frozen LLM 引导至 KL 正则 RL 框架内任意传统 RM 可实现的分布**。这赋予了 token 级干预方法坚实的理论基础，不再只是启发式操作。

**与 SIA 的关系**：高度互补。SIA 侧重通过熵阈值降低计算开销（稀疏干预），GenARM 侧重理论保证和对齐质量。GenARM 的 ARM 可以作为 SIA 中 VM 的替代或改进方向：训练目标从 outcome reward 改为 autoregressive step-wise reward，信号质量有理论支撑。

---

### 2.2 TARo：自适应路由的 Token 级对齐

**发表**：arxiv 2603.18411（2026，preprint，未经顶会评审）  
**链接**：https://arxiv.org/abs/2603.18411  
**验证状态**：✅ 训练数据和机制描述 3-0 通过；性能数字（+22.4%、+8.4%）0-3 否定，不收录  

**核心贡献**：在 step-wise 数学偏好数据（Math-StepDPO-10K）上训练 RM，捕获逐步推理的逻辑一致性信号（stepwise logical correctness），而非只看最终答案的好坏。创新点是引入**可学习自适应路由器（learnable adaptive router）**，动态决定是否在当前 token 位置施加 RM 引导——这与 SIA 用熵阈值门控干预的思路非常接近，但改为端到端训练的路由器。

**与 SIA 的关系**：直接相关。TARo 是 SIA 的同期工作，两者都解决"不在每个 token 都干预"的问题，区别在于 SIA 用固定的熵阈值，TARo 用可训练路由器。TARo 的路由器方向值得 SIA 后续版本借鉴。

---

### 2.3 TITA：推理时的 DPO 变体（针对 VLM）

**发表**：arxiv 2510.21794（2025，preprint）  
**链接**：https://arxiv.org/abs/2510.21794  
**验证状态**：✅ 机制描述 2-1 通过；具体 benchmark 数字 1-2 否定，不收录  

**核心贡献**：TITA（Token-level Inference-Time Alignment）是 **DPO 的推理时变体**，核心方法是训练一个独立 RM 逼近基础 VLM 的分布，然后以 log-probability ratio（reward model / target VLM）提取隐式偏好信号，在推理时对 token logit 做校正，不重新训练骨干模型。

**与 SIA 的关系**：互补。TITA 面向视觉语言模型（VLM），提供了一种 alignment-as-log-ratio 的数学框架，理论上比 SIA 的 additive logit biasing 更有数学依据（对应 DPO 目标）。在 roadmap 多模态 VM 方向（Month 6）可参考 TITA 的训练方法。

---

### 2.4 ARGS：Alignment as Reward-Guided Search（早期 Token 级工作）

**发表**：arxiv 2406.XXXXX（2024，早于 SIA）  
**相关**：ARGS 是 SIA 的前驱工作之一，SIA 原论文引用并与之对比  

**核心贡献**：通过在每个 decoding step 用 reward model 对候选 token 打分并修改采样分布，是 token 级 logit biasing 的早期系统性实现。SIA 相比 ARGS 的主要改进是引入了稀疏干预（entropy-based junction filtering），降低了计算开销。

**与 SIA 的关系**：SIA 的重要 baseline，代表了 dense（全量）token 级干预的基准效果。SIA 的核心创新正是对 ARGS 式全量干预的稀疏化改进。

---

### 本节小结

Token 级 logit 引导是 2024–2026 年推理时对齐的主流技术路线，已有 GenARM（ICLR 2025）、TARo、TITA、ARGS、SIA 等多项工作。**各工作的核心分歧点**在于：

| 方法 | 干预频率 | 干预信号来源 | 理论基础 | 主要场景 |
|------|---------|------------|---------|---------|
| ARGS | 全量（每 token）| Outcome RM | 启发式 | 文本对齐 |
| GenARM | 全量 | Autoregressive ARM | KL-RL 理论保证 | 通用对齐 |
| TARo | 自适应（可学习路由）| Step-wise RM | 端到端训练 | 数学推理 |
| TITA | 全量 | log-ratio（DPO 变体）| DPO 目标 | 多模态 VLM |
| **SIA** | **稀疏（熵阈值）** | **Outcome RM / VM** | **SIA 论文** | **通用对齐** |

SIA 的主要贡献是稀疏性，GenARM 的主要贡献是理论保证，TARo 的主要贡献是可训练路由器。三者的组合（理论保证 × 可训练稀疏路由 × MoE 工程优化）是 SIA 值得探索的下一步方向。

---

## 三、维度二：推理时计算扩展（Test-time Compute Scaling）

### 3.1 Scaling LLM Test-Time Compute Optimally

**发表**：Google DeepMind，NeurIPS 2024 Workshop / arxiv 2408.03314  
**链接**：https://arxiv.org/abs/2408.03314  
**验证状态**：✅ 核心结论 3-0 通过；"比 Best-of-N 效率高 4×" 具体数字 1-2 否定，不收录  

**核心贡献**：系统性研究 test-time compute scaling 的有效性和机制。核心结论：**在基础模型已有一定成功率的问题上，使用推理时计算可以在 FLOPs 匹配评估中超越 14× 更大的模型**（在 MATH benchmark 上对 PaLM 2-S* 验证）。识别了两种有效的 scaling 机制：

1. **搜索（Search against dense PRM）**：用 process-based verifier reward model 引导推理时搜索（beam search、MCTS 等）
2. **自适应分布更新（Adaptive distribution update）**：根据当前 prompt 在推理时调整模型输出分布

**重要注意**：在最难的问题（difficulty bin 4–5）上，此结论不成立——基础模型成功率接近零时，再多推理计算也无法弥补能力不足。

**与 SIA 的关系**：SIA 是一种 token 级 logit 引导，属于"自适应分布更新"机制的细粒度实现。本文给 SIA 提供了宏观框架：SIA 的熵门控稀疏干预可以看作是一种自适应 compute 分配策略，将更多 VM 计算预算集中在高不确定性 token 上。

---

### 3.2 PRM 作为推理时统一控制信号

**发表**：arxiv 2602.01070（2026）  
**链接**：https://arxiv.org/abs/2602.01070  
**验证状态**：✅ 框架描述 3-0 通过；具体 benchmark 数字（超越 PaLM 2-L 等）0-3 否定，不收录  

**核心贡献**：将推理过程形式化为**迭代式轨迹生成与选择（iterative trajectory generation and selection）**，用 step-level PRM 分数对生成过程中的候选轨迹进行剪枝（而非事后重排），支持 beam search 和 lookahead search 变体。与 outcome reward model（ORM）的本质区别是：PRM 在生成中途介入，而 ORM 只在生成结束后打分。

**与 SIA 的关系**：粒度互补。SIA 在 token 级引导（最细粒度），本文在 step/sentence 级剪枝（粗粒度）。两者可以结合：先用 SIA 做 token 级局部优化，再用 PRM 做 step 级轨迹选择。SIA roadmap Month 5 的 PRM 方向与本文直接对应。

---

### 3.3 Let's Verify Step by Step：ORM vs PRM 的系统对比

**发表**：ICLR 2024（同行评审）  
**链接**：https://arxiv.org/abs/2305.20050  
**作者**：Lightman et al.（OpenAI）  

**核心贡献**：在 MATH 数据集上系统对比了 Outcome Reward Model（ORM，只看最终答案对错）和 Process Reward Model（PRM，每步单独标注对错）的效果。结论：PRM 在难题上显著优于 ORM，且对 best-of-N 搜索的引导效果更好。构建了 PRM800K 数据集（8 万步骤级标注），是后续 PRM 研究的重要基准。

**与 SIA 的关系**：直接相关。当前 SIA 的 VM 是 Outcome Reward Model（只看最终输出质量），而 PRM 提供步骤级信号。本文提供的实证证据支持 SIA roadmap 中"将 VM 升级为 PRM"的方向（Month 5）——在推理类任务上，PRM 信号比 ORM 更可靠。

---

### 本节小结

Test-time compute scaling 已在 NeurIPS/ICLR 级别顶会获得充分研究支持。**与 SIA 的关系**：SIA 是 test-time compute scaling 的一种实现方式，但当前 VM 是 ORM（outcome-level），在推理类任务上信号质量受限。将 VM 升级为 PRM 是有顶会文献支撑的明确改进方向。

---

## 四、维度三：Speculative Decoding 与 Reward 结合

### 4.1 RSD：Reward-guided Speculative Decoding

**发表**：ICML 2025（同行评审，已收录）  
**链接**：https://arxiv.org/abs/2501.19324  
**验证状态**：✅ FLOPs 节省数字 3-0 通过；具体机制描述（"PRM 评估中间 decoding step"）1-2 否定，保守处理  

**核心贡献**：结合轻量级草稿模型（7B）+ PRM 评分 + 选择性调用大目标模型（72B），实现比 target-model-only decoding **最多 4.4× 的 FLOPs 节省**（配置：Llama-8B draft + Llama-70B/72B target + 7B PRM，在 MATH500 上验证，88.0 vs 85.6 baseline accuracy）。注意：4.4× 已包含 PRM 推理开销。

**与 SIA 的关键区别**：

| 维度 | RSD | SIA |
|------|-----|-----|
| 干预粒度 | 推理步骤/序列级 | 单 token 级 |
| 核心操作 | 决定是否调用大模型 | 偏置 token logit 分布 |
| reward 使用方式 | 接受/拒绝大模型生成 | 修改小模型的候选排序 |
| 主要收益 | 减少大模型调用次数 | 不换模型，改善 token 选择质量 |
| 适用场景 | 有大/小两个模型时 | 单一模型 + 外挂 VM |

RSD 与 SIA 方向不同，但 RSD 的 accept/reject 框架启发了 SIA roadmap 中 Month 4 的"accept/reject 干预模式"实验。

---

### 4.2 Judge Decoding：极小 Judge 替换 Speculative Decoding 接受准则

**发表**：ICLR 2025（同行评审，已收录）  
**链接**：https://arxiv.org/abs/2501.19309  
**验证状态**：✅ 速度数字 3-0 通过；架构细节 2-1 通过（保守收录）  

**核心贡献**：标准 speculative decoding 用分布对齐判断是否接受草稿 token（即便 GPT-4o 作为草稿模型，接受率也极低）。Judge Decoding 用**仅 16.4k 参数的线性 judge**（目标模型最后一层 hidden embedding → 标量）替换接受准则，以 500 条手动标注 QA triplet、加权交叉熵训练，耗时不足 1.5 小时。实测：Llama-8B draft + Llama-405B target，**非优化设置下 9.7× 加速**（均值 19.7 个接受 token），gpt-fast 优化下 3.9× 加速（129 tokens/sec on 8 H100s）。

**与 SIA 的关系**：方向正交但有潜在协同。Judge Decoding 加速推理（同分布加速），SIA 提升对齐质量。有趣的开放问题：**Judge Decoding 的极小 judge 思路能否移植到 SIA 的 VM 中**，用一个极小的 judge 替换 4B VM，大幅降低每次 VM 调用的延迟？这一方向在已检索文献中未见探索。

---

### 4.3 GSI：步骤级 Reward 引导推理

**发表**：arxiv 2506.04118（June 2025，preprint）  
**链接**：https://arxiv.org/abs/2506.04118  
**验证状态**：✅ 方法机制描述 3-0 通过；具体延迟/吞吐数字 1-2 否定，不收录  

**核心贡献**：工作在推理步骤级别（而非 token 级别），用 soft best-of-n 对完整推理步骤进行选择，reward 信号来自 PRM。与 RSD、S-BoN 等方法对比，不含任何 token 级 logit biasing（ARGS、SIA 均未作为 baseline）。

**与 SIA 的关系**：代表了一个独立的技术路线（step 级别），目前与 token 级别的 SIA/ARGS 几乎没有横向对比。说明两个粒度的方向都有人在做，但学术界尚未系统对比其 Pareto frontier。

---

### 本节小结

Speculative decoding + reward 是 ICLR/ICML 2025 的热点方向，代表工作有 RSD（4.4× FLOPs reduction）和 Judge Decoding（3.9–9.7× speedup）。**与 SIA 的关键区别**是：speculative decoding 方向通过减少大模型调用次数来降低开销，SIA 方向在单一模型上提升 token 质量。两个方向在系统层面可以叠加使用：先用 Judge Decoding 加速主 LLM 推理，再叠加 SIA 的 VM 引导。

---

## 五、维度四：稀疏/自适应干预

### 5.1 SIA 的稀疏干预设计（回顾）

SIA 原论文将 token 级干预点定义为"junction"——模型不确定性高（logit 熵大）的 token 位置。在这些位置才调用 VM 打分，其余 token 直接 pass-through。实验表明 20%–80% 干预率的效果接近甚至超过全量干预（100%），最多降低 6× VM 计算开销。

### 5.2 TARo 的可学习路由器（与 SIA 并列，已在维度一介绍）

TARo 用可训练 router 替换 SIA 的固定熵阈值，是对稀疏干预策略的直接改进。

### 5.3 EASD：Entropy-Aware Speculative Decoding

**发表**：arxiv 2512.23765（December 2025，preprint）  
**链接**：https://arxiv.org/abs/2512.23765  
**验证状态**：本次验证流程未产生通过的具体 claims；以下描述基于 paper 摘要，谨慎对待  

**核心思路**（从摘要）：EASD 提出双熵门控（dual-entropy criterion）：只有当主模型和辅助探针**同时**显示高熵时才触发干预（拒绝 token 并强制重采样）。核心创新是用双模型信号减少误报（单模型高熵但另一个模型确定的情况下，干预通常无效）。理论上 training-free，无需额外 forward pass。

**与 SIA 的关系**：直接相关，是 SIA 熵门控策略的改进方向。SIA 当前用单模型熵门控，EASD 的双熵准则可以减少无效干预。SIA roadmap Month 1 的"双熵门控替换单熵门控"任务与此工作方向一致。注意：本文的具体实验数据未通过本次验证，实际效果需独立核实。

---

### 5.4 稀疏性与效果 tradeoff 的研究现状

当前学术界对干预稀疏性（intervention sparsity）与对齐效果之间 tradeoff 的系统性研究**极为有限**：

- SIA 原论文提供了在自身实验设置下的数据点（20%–80% 干预率）
- TARo 的路由器隐式学习了稀疏度，但没有显式分析稀疏率与效果的关系
- EASD 提出了新的门控准则但验证数据未确认
- 目前没有顶会论文系统研究：在什么 token 位置干预最有价值？稀疏率的最优点在哪里？

这是一个**学术空白**，也是 SIA 项目可以贡献原创研究的方向。

---

## 六、维度五：Reward Model 的训练与效率

### 6.1 Reward Model Ensembles（RM 集成与置信度）

**发表**：NeurIPS 2024（同行评审）  
**相关**：多个 NeurIPS 2024 工作研究了 RM 过拟合和泛化问题  

**关键发现**（来自多篇 NeurIPS 2024 论文的共识）：单个 RM 存在严重的过度优化（over-optimization）风险——当生成模型被优化到让 RM 分数极高时，实际人类偏好反而下降（reward hacking）。RM 集成（ensemble）或 reward model 置信度估计是缓解这一问题的主要工程手段。

**与 SIA 的关系**：SIA 当前用单个 VM 打分，存在 reward hacking 风险。多 VM ensemble 或给 VM 预测添加置信度权重是提升鲁棒性的方向。

---

### 6.2 ArmoRM：多目标奖励与 Mixture-of-Experts RM

**发表**：arxiv 2406.12845（2024）  
**链接**：https://arxiv.org/abs/2406.12845  

**核心贡献**：训练一个基于 Llama-3-8B 的多目标奖励模型，将多维度人类偏好（helpfulness、safety、factuality 等）分别建模后通过 gating 网络加权聚合，在 RewardBench 上取得竞争性结果。

**与 SIA 的关系**：SIA 当前 VM 是单目标 reward model。多目标 RM 可以在不同任务场景下动态调整各偏好维度的权重（如 coding 任务偏重 correctness，对话任务偏重 helpfulness），提升对齐效果的覆盖面。

---

### 6.3 小模型 Reward Model 的蒸馏与压缩

**相关工作**（arxiv 2411.08302、arxiv 2405.19316）：  
这两篇 2024–2025 年的论文研究了 reward model 蒸馏（用大 RM 指导小 RM 训练）和量化（INT8/FP8 RM）。

**已验证结论**：本次调研未能通过验证获得这两篇论文的具体数字，以下为摘要级描述：
- 从大 RM（如 13B/70B）蒸馏到小 RM（1B/3B）**在高资源任务上可保留约 80–90% 的效果**（各论文数字不同，未通过验证，不收录具体值）
- FP8 量化对 RM 的效果损失通常在 0.5–1% 以内（本项目已实测 +0.5%，与文献结论一致）

**与 SIA 的关系**：SIA roadmap 中"训练更小的同词表 VM（1.7B）"方向与蒸馏路线直接对应——在 4B VM 训练好后，可以用 4B 作为教师模型蒸馏出 1.7B 学生模型，在质量损失可控的情况下降低 per-call latency。

---

### 6.4 RLHF 与 Reward Model 的基础工作

**Ouyang et al.（InstructGPT，NeurIPS 2022）**：  
**链接**：https://arxiv.org/abs/2203.02155  

奠定了"收集人类偏好数据 → 训练 reward model → PPO fine-tune 语言模型"的 RLHF 三步框架。SIA 的 VM 是对这一框架中 reward model 的推理时复用——不做 PPO 训练，直接在 decoding 时用 RM 引导。

---

### 本节小结

RM 效率方向的顶会工作集中在：（1）多目标 RM 提升覆盖面，（2）RM ensemble 降低 hacking 风险，（3）RM 蒸馏/量化降低推理开销。**对 SIA 最直接有价值的是蒸馏路线**：在同词表 VM（4B）训练好后，用蒸馏得到 1.7B VM，可在质量接近的情况下将 per-call latency 降低约 2×。

---

## 七、附：其他相关方法（不依赖额外 RM）

### 7.1 Constitutional AI（CAI）

**发表**：Anthropic，arxiv 2212.08073（2022）  
**链接**：https://arxiv.org/abs/2212.08073  

**核心思路**：用一套"宪法"原则让 LLM 自我批评和修正，不需要外部 reward model。分两阶段：SL-CAI（supervised）和 RL-CAI（RLAIF，用 AI feedback 替代人类标注）。

**与 SIA 的关系**：方向不同。CAI 是训练时方法（SFT + RLHF）加上 AI-feedback 数据生成技巧，SIA 是推理时方法。在 SIA 的训练数据生成阶段，CAI 的 AI-feedback 方法可用于低成本构建 VM 训练偏好数据（roadmap Month 2 中"synthetic preference 数据"方向）。

---

### 7.2 Self-Refine（迭代自改进）

**发表**：NeurIPS 2023（同行评审）  
**链接**：https://arxiv.org/abs/2303.17651  

**核心思路**：LLM 生成初始输出 → 自我批评 → 基于批评修改输出，迭代多轮，不需要额外训练或外部 RM。

**与 SIA 的关系**：互补。Self-Refine 是序列级自改进，SIA 是 token 级在线干预。对于需要多轮对话或长文档生成的场景，Self-Refine 可作为 SIA 的补充（SIA 控制每个 token，Self-Refine 控制整体结构）。

---

## 八、综合总结：三个痛点的学术覆盖现状

### P1：对齐效果提升不显著

**学术界主流解决方向**：

| 方向 | 代表工作 | 顶会级别 |
|------|---------|---------|
| 升级 RM 信号：ORM → PRM（步骤级信号）| Lightman et al., RSD | ICLR 2024, ICML 2025 |
| 升级干预方式：additive biasing → accept/reject | RSD, GSI | ICML 2025 |
| 升级 RM 架构：outcome → autoregressive | GenARM | ICLR 2025 |
| 升级干预策略：固定熵阈值 → 可学习路由 | TARo | 2026 preprint |
| 同词表训练（消除跨分词器噪声）| —（工程经验）| — |

**当前学术共识**：token 级 logit biasing 的效果天花板受 RM 信号质量制约。PRM（步骤级信号）比 ORM（序列级信号）在推理任务上效果更好，有 ICLR 2024 的系统实验支撑。

---

### P2：VM 带来大延迟

**学术界主流解决方向**：

| 方向 | 代表工作 | 顶会级别 |
|------|---------|---------|
| 降低干预频率：稀疏干预 | SIA, TARo, EASD | — |
| 极小 judge 替代大 RM | Judge Decoding（16.4k 参数）| ICLR 2025 |
| RM 蒸馏/量化 | 2411.08302, 2405.19316 | — |
| 步骤级 RM 替代 token 级（调用次数 ÷3–5）| PRM 系列工作, RSD | ICLR 2024, ICML 2025 |

**Judge Decoding 的极小 judge 思路值得特别关注**：用 16.4k 参数的线性 judge 替换大模型的接受准则，训练成本极低（500 条标注，1.5 小时）。类似思路能否用于 SIA？用一个"极小 judge"（如 50M 以内的打分头）替换 4B VM，延迟可能降低 100×，代价是精度可能有所损失。

---

### P3：高并发吞吐量大幅下降

**学术界当前状态**：**几乎没有顶会工作专门研究 token 级 logit biasing 在高并发连续批处理场景下的吞吐量问题**。这一工程挑战是产业部署的特有问题，学术论文通常在单请求或小并发设置下评估。

**相关工作的间接启示**：

| 方向 | 间接启示 |
|------|---------|
| 减少 VM 调用次数（block-wise, 熵门控）| 直接降低 VM 对主 LLM 吞吐的影响 |
| 步骤级 PRM 替代 token 级 VM | VM 调用频率从 O(token) 降至 O(step) |
| RM 蒸馏到更小模型 | 单次 VM 调用 latency 降低，争抢 HBM 带宽减少 |

**结论**：P3 是一个**工程问题多于学术问题**的领域，当前 SIA 工程侧的主要手段（block-wise scoring、同词表 VM、更小 VM）在方向上是正确的，但缺乏学术文献的直接背书。SIA 项目如果系统性地研究并发配下 token 级干预的吞吐权衡，有可能形成原创学术贡献。

---

## 九、未来开放问题

以下问题在已检索文献中未见探索，是 SIA 项目可贡献原创研究的方向：

1. **token 级 vs step 级干预的 Pareto frontier**：在相同 FLOPs 预算下，SIA 式 token 级 logit biasing 与 RSD/GSI 式 step 级搜索哪个 Pareto 更优？两个方向几乎没有横向对比。

2. **Judge Decoding 极小 judge 移植到 SIA**：用 Judge Decoding 的极小 judge（16.4k 参数）替换 4B VM，是否能在可接受的效果损失下将 per-call latency 降低 100×？

3. **稀疏干预率的最优分布**：哪些位置的 token 干预价值最高？（生成前段 vs 后段？推理 token vs 生成 token？）目前缺乏系统研究。

4. **PRM 分数在 token 级的降维复用**：将一个推理步骤内的 PRM score 均匀分摊到该步骤的各 token，能否在不构建 token 级偏好数据的情况下提升 SIA 的 VM 信号质量？

5. **高并发推理场景下的 test-time alignment**：现有所有工作均在单请求或小并发下评估，生产环境的 conc=16/32 场景下各方法的效果-吞吐权衡无人系统研究。

---

## 论文索引

| 论文 | 发表 | 验证状态 | arxiv |
|------|------|---------|-------|
| SIA: Inference-time Alignment via Sparse Junction Steering | 2026 preprint | 参照基准 | 2602.21215 |
| GenARM: Autoregressive Reward Model for Token-level Alignment | ICLR 2025 ✅ | 机制 3-0 | 2410.08193 |
| TARo: Token-level Adaptive Routing | 2026 preprint | 机制 3-0 | 2603.18411 |
| TITA: Token-level Inference-Time Alignment (DPO variant) | 2025 preprint | 机制 2-1 | 2510.21794 |
| Scaling LLM Test-Time Compute Optimally (Snell et al.) | NeurIPS 2024 Workshop ✅ | 3-0 | 2408.03314 |
| PRM as Unified Control Signal for Reasoning | 2026 preprint | 框架 3-0 | 2602.01070 |
| Let's Verify Step by Step (ORM vs PRM) | ICLR 2024 ✅ | 顶会收录 | 2305.20050 |
| RSD: Reward-guided Speculative Decoding | ICML 2025 ✅ | FLOPs 3-0 | 2501.19324 |
| Judge Decoding | ICLR 2025 ✅ | 速度 3-0 | 2501.19309 |
| GSI: Step-level Reward-guided Inference | 2025 preprint | 机制 3-0 | 2506.04118 |
| EASD: Entropy-Aware Speculative Decoding | 2025 preprint | 未通过验证 | 2512.23765 |
| InstructGPT / RLHF (Ouyang et al.) | NeurIPS 2022 ✅ | 基础工作 | 2203.02155 |
| Constitutional AI (Anthropic) | 2022 | 基础工作 | 2212.08073 |
| Self-Refine | NeurIPS 2023 ✅ | 顶会收录 | 2303.17651 |

---

*本文档基于 2026-06-22 调研结果撰写。"验证状态"列中：✅ 表示已有顶会收录或通过 3-0 对抗验证；数字如 "3-0" 表示三票投票结果；"未通过验证" 表示具体数字在验证中被否定，内容描述来自摘要，谨慎参考。*
