# Alignment 方法综述：论文中的分类与对比

本文档整理 SIA 论文（Inference-time Alignment via Sparse Junction Steering）中对各类 alignment 方法的分类、评价与量化对比。

---

## 一、训练时方法（Training-time）

| 方法 | 机制 | 局限性 |
|------|------|--------|
| SFT（Supervised Fine-tuning） | 在高质量样本上直接微调 | 只能学习预先选定的行为，灵活性差 |
| RLHF | 训练 Reward Model，再用 RL 优化策略 | 需要大量人工标注，计算开销极高 |
| DPO | 直接用偏好对训练策略，省去显式 RM | 仍需离线训练阶段 |

论文总结：这三类方法都依赖"大量参数更新"，计算成本高，且一旦训练完成就固化，无法灵活适配新目标。SIA 作为推理时方法完全绕开了这一成本。

**适合场景**：一次性大规模对齐，对延迟不敏感、有充足算力和标注资源。

---

## 二、推理时方法——Prompt-based（提示词引导）

**代表**：Constitutional AI、in-context learning

**机制**：把对齐目标直接写进 prompt，靠语言指令引导模型行为。

**局限**（论文原文）：

> "effectiveness is limited by the expressiveness of natural language prompts"

天花板受限于自然语言能描述多复杂的目标，对复杂对齐需求难以精确表达。

**适合场景**：成本最低，适合简单的行为调整（"请礼貌回答"），不适合精细的价值观对齐。

---

## 三、推理时方法——Search-based（搜索式）

这是论文重点比较的对象，分四种：

| 方法 | 机制 | 论文指出的问题 |
|------|------|----------------|
| **Best-of-N (BoN)** | 生成 N 个完整候选，取 reward 最高的 | 要生成完整序列才能评分，效率最差；BoN-8 计算量是 SIA 的约 4 倍 |
| **ARGS** | 在解码每步用 trajectory reward 直接调整 logits | "noisy local signals"——中间步骤对 reward 不敏感，信号噪声大 |
| **Transfer Q\*** | 用多 token 前向模拟估算 Q 值 | 计算成本高、方差大 |
| **CBS（Chunk-based Search）** | 以 chunk 为单位搜索 | 比 BoN 更细粒度，但效率仍是 SIA 的 2–4 倍计算量 |

**SIA 的量化优势**（论文 Table 1 数据）：

- vs BoN-8：计算开销降低约 **4 倍**，同等或更好 reward
- vs CBS-8：效率提升 **2–4 倍**
- vs ARGS / Transfer Q\*：在全量 token 干预场景下 reward 明显更高

**适合场景**：算力充裕、只需最终结果最优的离线场景，不适合延迟敏感的在线推理。

---

## 四、推理时方法——Refine-based（迭代精炼）

**代表**：Aligner、DIFFPO、SEA

**机制**：先生成一个回答，再通过多轮迭代/反馈不断改进。

**局限**（论文原文）：

> "typically require additional inference rounds, leading to increased latency"

多轮生成让延迟成倍增加，吞吐量极低。

**适合场景**：对质量要求极高、延迟不敏感的场景（如文档生成、代码审查），不适合交互式对话。

---

## 五、推理时方法——Token-level Steering（逐 token 引导）

SIA 所属的类别。论文对该类已有方法的核心批评：现有逐 token 引导方法假设**每个 decoding step 对 alignment 的贡献相同**，对所有 token 一律干预（dense steering）。

论文原文：

> "Dense intervention creates an inherent conflict between the original LLM distribution and external reward signals, ultimately degrading overall generation quality."

**SIA 的改进**：只在高熵的"junction"关键决策点干预（稀疏），20%–80% 的干预率即可达到或超过全量干预效果，同时避免了对生成质量的损害。

---

## 六、总结对比

| 方法类别 | 代表 | 优点 | 缺点 | 适合场景 |
|----------|------|------|------|----------|
| 训练时 | RLHF / DPO / SFT | 效果扎实 | 训练成本高，固化后难调整 | 一次性大规模对齐 |
| Prompt-based | Constitutional AI | 零额外成本 | 精度有限，复杂目标难表达 | 简单行为调整 |
| Search-based | BoN / ARGS / CBS | 效果好 | 计算量倍增（2–4x） | 离线、算力充裕场景 |
| Refine-based | Aligner / SEA | 质量可迭代提升 | 延迟倍增，多轮推理 | 离线高质量生成 |
| Dense token steering | ARGS 全量版 | 精细干预 | 损害原始生成质量 | — |
| **SIA（稀疏 token steering）** | — | 低开销、不损质量、即插即用 | 低熵错误无法纠正 | 在线推理、延迟敏感场景 |
