# FaRMA 论文详细摘要（中文）

**标题：** Towards Cost-Effective Reward Guided Text Generation
**发表：** ICML 2025，arXiv:2502.04517
**整理日期：** 2026-07-01

---

## 一、摘要

现有的奖励引导文本生成（RGTG）方法在每个解码步骤需要多次调用奖励模型（RM），导致推理成本极高。FaRMA 提出了一种新的 RM 架构：**一次前向传播同时输出全词表所有候选 token 的分数**，将每步 RM 调用从 K 次降为 1 次。同时，论文从理论上证明了现有方法（PARGS、CD）在 partial sequence 打分上的缺陷，并提出了基于 Bradley-Terry loss + Temporal Difference 约束的训练目标，确保前缀打分的正确性。

---

## 二、引言（问题背景）

RGTG 的基本思路：冻结 LLM，用外部 RM 在每步解码时对候选 token 打分，将分数叠加到 logits 上，引导生成向高奖励方向走。

**现有方法的两个核心问题：**

**问题 1：解码成本过高。**
每步需要对 top-K 个候选 token 各调用一次 RM，产生 K 次 forward pass。ARGS 的实测数据：每次生成需要 **596 次 RM 调用**。总耗时是 FaRMA 的约 6×。

**问题 2：partial sequence 打分不准确。**
RM 在训练时只见过完整序列，用它对前缀（partial sequence）打分会产生系统性偏差。现有方法（PARGS、CD）通过不同近似手段绕开这个问题，但都有理论缺陷（论文 Theorem 1、2 给出了反例证明）。

---

## 三、方法

### 3.1 FaRMA 架构

**核心改变：** 将 RM 的输出头从 `[hidden → scalar]` 改为 `[hidden → |vocab|]`。

解码时的分数计算公式：

$$\text{score}(y_i | \mathbf{x}, \mathbf{y}_{1:i-1}) = \log \pi_{\text{ref}}(y_i | \mathbf{x}, \mathbf{y}_{1:i-1}) + \beta \cdot V_\theta(y_i | \mathbf{x}, \mathbf{y}_{1:i-1})$$

其中 $V_\theta$ 输出一个 $|\mathcal{D}|$ 维向量，覆盖整个词表。对任意 top-K 候选 token，直接通过 index 读出分数，**无需额外 RM 调用**。

**效率对比：**

| 方法 | 每步 RM 调用次数 |
|------|----------------|
| ARGS | K 次（top-K 逐一评分）|
| FaRMA | **1 次**（全词表，index 取分）|

### 3.2 Temporal Difference 约束（核心理论贡献）

FaRMA 引入了一个约束，确保前缀打分的正确性：

$$V_\theta(\mathbf{y}_{1:i} | \mathbf{x}) = \max_{y_{i+1}} V_\theta(\mathbf{y}_{1:i+1} | \mathbf{x})$$

**含义：** 一个前缀的价值，等于它能扩展到的最优下一步的价值。这与强化学习中的 Bellman 最优方程完全对应（确定性转移的 MDP 中，V = max Q）。

---

## 四、训练目标

训练使用两个 loss 交替优化：

**Loss (a)：Bradley-Terry loss（在完整序列上）**

$$\mathcal{L}^{(a)} = -\mathbb{E} \log \sigma\left(V_\theta(\mathbf{y}^w | \mathbf{x}) - V_\theta(\mathbf{y}^l | \mathbf{x})\right)$$

确保 chosen 序列的分数高于 rejected 序列。这是标准的偏好学习 loss。

**Loss (b)：TD 约束 loss（在 partial sequence 上）**

$$\mathcal{L}^{(b)} = \frac{1}{2}\left[V_\theta(\mathbf{y}_{1:i} | \mathbf{x}) - \max_{y_{i+1}} V_\theta(\mathbf{y}_{1:i+1} | \mathbf{x})\right]^2$$

确保前缀的价值估计满足 Bellman 方程。

两个 loss 交替训练（不同 batch 分别优化），最终同时保证：完整序列排序正确 + 前缀打分有理论保证。

---

## 五、理论分析

### Theorem 1：PARGS 的缺陷

PARGS 假设"前缀的偏好排序 = 完整序列的偏好排序"，但这个假设不成立。

**反例：** 如果前缀 $p$ 在训练数据中大多数时候后跟着低质量的完整序列（即便 $p$ 本身可以延伸到最优序列），PARGS 会给 $p$ 打低分——因为它没有区分"前缀本身的价值"和"从这个前缀实际生成出来的序列的质量"。

### Theorem 2：CD（Controlled Decoding）的缺陷

CD 用 base LLM（$\pi_{\text{ref}}$）做 rollout 来估计前缀价值，导致价值函数严重依赖 base LLM 的生成倾向。

**反例：** 即使前缀 $p$ 可以延伸到最优序列，如果 $\pi_{\text{ref}}$ 从 $p$ 出发倾向于生成低质量输出，CD 也会给 $p$ 打低分。

### Theorem 3：FaRMA 的保证

在无限训练数据和足够表达能力的前提下，FaRMA 保证：

$$V(\mathbf{y}^*_{1:i} | \mathbf{x}) \geq V(\mathbf{y}'_{1:j} | \mathbf{x}) \quad \forall i, j, \mathbf{y}'$$

即：能延伸到最优完整序列的前缀，其价值一定高于所有其他前缀。TD 约束从数学上保证了这一点。

---

## 六、实验设置

| 任务 | 数据集 | LLM |
|------|--------|-----|
| 摘要生成 | Reddit TL;DR | Llama-3.2-1B-Instruct |
| 对话 | Anthropic HH | Pythia-2.8B（测试了 400M/1B/2.8B）|
| 文本生成 | UltraFeedback | Zephyr-7B |

**对比方法：** ARGS、PARGS、CD、CARDS（RGTG 类）；PPO、DPO（RLHF 类）；$\pi_{\text{ref}}$（不干预基线）

---

## 七、实验结果

### 7.1 RM 调用次数对比（效率）

| 方法 | LLM 调用次数 | RM 调用次数 | 总计 |
|------|------------|-----------|------|
| ARGS（TL;DR）| 59.7 | **596.9** | 656.6 |
| FaRMA（TL;DR）| 53.3 | **53.3** | 106.5 |
| ARGS（HH）| 71.9 | **718.5** | 790.4 |
| FaRMA（HH）| 90.1 | **90.1** | 180.2 |

FaRMA 将 RM 调用次数减少约 **6×**。

### 7.2 摘要任务性能（TL;DR）

| 方法 | Reward（↑）| 生成时间 |
|------|-----------|---------|
| 不干预（$\pi_{\text{ref}}$）| 0.98 | 2 min |
| ARGS | 1.46 | 32 min |
| PARGS | 1.56 | 31 min |
| **FaRMA** | **2.05** | **5 min** |
| DPO | 2.08 | 2 min |
| PPO | 2.05 | 2 min |

FaRMA 达到与 PPO/DPO 相当的奖励，同时比 ARGS/PARGS **快 6×**，只比不干预基线慢 2.5×。

### 7.3 对话任务性能（Anthropic HH）

| 方法 | Reward（↑）| 时间 |
|------|-----------|------|
| 不干预 | 1.18 | 2 min |
| FaRMA-400M | 1.49 | 2 min |
| FaRMA-1B | 1.56 | 3 min |
| **FaRMA-2.8B** | **1.80** | **5 min** |
| CARDS | 1.92 | 20 min |
| PPO | 1.92 | 2 min |

小至 400M 的 RM 仍有显著提升。

### 7.4 生成多样性（ROUGE-L，越低越多样）

| 任务 | 方法 | ROUGE-L（↓）|
|------|------|------------|
| TL;DR | 不干预 | 0.20 |
| TL;DR | CARDS | 0.49 |
| TL;DR | FaRMA | 0.21 |
| HH | 不干预 | 0.29 |
| HH | FaRMA | **0.24** |

CARDS 因为 rejection sampling 会大量生成相似内容（ROUGE-L=0.49），FaRMA 的多样性与不干预基线相当，**不牺牲多样性**。

---

## 八、关键结论

1. **效率：** 平均生成时间 ARGS 需 19.4 秒，FaRMA 只需 **2.99 秒**（同等任务），节省约 6.5×。

2. **质量：** FaRMA 在奖励分数上与 PPO/DPO 持平，显著优于 ARGS/PARGS，而无需任何参数更新。

3. **可扩展性：** RM 可以缩小到 400M，仍有效果，且推理时间接近不干预基线。

4. **多样性：** 不像 rejection sampling 类方法（CARDS）会降低多样性，FaRMA 保持了生成多样性。

5. **理论保证：** TD 约束是目前唯一从理论上证明了 partial sequence 打分正确性的 RGTG 方法。

---

## 九、与 SIA 的直接关联

| 对比维度 | SIA（当前）| SIA + FaRMA（目标）|
|---------|-----------|-----------------|
| 每步 RM 调用 | K=10 次 | 1 次 |
| RM 输出头 | `Linear(hidden, 1)` | `Linear(hidden, vocab_size)` |
| 吞吐量（实测）| 322 tok/s | ~760 tok/s（proxy 验证）|
| 训练改动 | — | 新增 Loss(b)：TD 约束 loss |

FaRMA 的 TD 约束 loss 是论文的核心贡献，SIA 现有训练代码（只有 BT/MSE loss）还没有这部分。训练 FaRMA VM 时需要同时加入 Loss(b)。

---

## 十、需要修改的代码（SIA 适配）

### 10.1 model.py — 修改输出头

```python
# 当前（标量头）
self.token_reward_head = nn.Linear(self.hidden_size, 1)

# FaRMA 目标（全词表头）
self.token_reward_head = nn.Linear(self.hidden_size, vocab_size)
```

### 10.2 train.py — 新增 TD 约束 loss

在原有 Bradley-Terry / MSE loss 基础上，增加每个位置的 Bellman backup loss：

```python
# 对于 partial sequence，要求每个前缀的 value = 后续最大 value
# token_rewards: (batch, seq_len, vocab_size)
# 取每个位置实际下一个 token 的分数，对比当前位置最大分数
td_loss = 0.5 * (V_prefix - V_next.detach().max(dim=-1).values) ** 2
```

### 10.3 sia_vllm_RM.py — 推理时直接 index

```python
# 当前：K 次 RM forward，每次传入一个候选 token
# FaRMA：1 次 RM forward，输出 vocab_size 维向量，直接 index 取 K 个分数
vocab_scores = rm_model(prefix_input_ids)  # (batch, vocab_size)
candidate_scores = vocab_scores[:, candidate_token_ids]  # (batch, K)
```
