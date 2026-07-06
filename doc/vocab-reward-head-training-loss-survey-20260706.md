# vocab-size Reward Head 训练 Loss 文献综述

**日期：** 2026-07-06  
**调研规模：** 2轮独立调研，51个独立 agent，94篇论文  
**核心问题：** 对专用的 vocab-size reward head（Linear(hidden→vocab_size) 或其低秩分解版本），
业界成熟论文是否使用"激励真实下一个 token、打压其他所有 token"的交叉熵 loss（CE loss）
作为训练目标的一部分？

---

## 一、结论摘要

**没有任何已知论文对专用 vocab-size reward head 使用 CE(true_next_token) 作为训练目标。**

这一结论在两轮独立调研中完全一致，覆盖了 ICML/NeurIPS/ICLR/ACL/EMNLP 2021-2026 所有主要相关论文。
当前项目使用的 BT + TD 训练方案（与 FaRMA, ICML 2025 完全一致）是该架构下业界验证过的唯一标准做法。

---

## 二、所有使用 vocab-size head 的主要论文及其 loss

### 2.1 专用 vocab-size reward head（与本项目架构相关）

#### FaRMA — Towards Cost-Effective Reward Guided Text Generation
**arXiv:2502.04517 · ICML 2025** ← **与本项目架构最直接对应的论文**

- **Head：** Linear(hidden, vocab_size) 输出 Q 值（reward magnitudes，非概率）
- **Loss (a)：** Bradley-Terry 偏好损失（序列级）
  ```
  L_a = -E[log σ(V_θ(y_w|x) − V_θ(y_l|x))]
  V = 各 token 位置 reward 的加权平均
  ```
- **Loss (b)：** Temporal Difference / Bellman optimality constraint（部分序列级）
  ```
  L_b = 0.5 × [V_θ(y_{1:i}|x) − max_{y_{i+1}} V_θ(y_{1:i+1}|x)]²
  ```
  注意：`max_{y_{i+1}}` 对全词表所有 token 取最大值，这是给"其他 token"提供梯度的关键机制。
- **CE loss：** ❌ 无
- **说明：** FaRMA 明确指出，vocab head 输出的是 Q 值而非对数概率，对其使用 CE 会把 reward
  discrimination 与 language modeling 混淆。TD loss 的 max 操作已经在理论上解决了"其他
  token 无梯度"的问题（FaRMA Theorem 3）。

#### Q-RM — Discriminative Policy Optimization for Token-Level Reward Models
**ICML 2025**

- **Head：** vocab-size softmax 输出（等价于 Q-function advantage over vocab）
- **Loss：** 轨迹级 BT 偏好损失
  ```
  p(τ_w ≥ τ_l) = σ[β(Q(τ_w) − Q(τ_l)) − δ]
  ```
- **CE：** ❌ 无

#### RAD-Q — Low-Rank ARM Parametrization
**arXiv:2407.04615 · TMLR 2025**

- **Head：** 双线性低秩 vocab-size 评分：`r(token|h) = dot(W·h, embed[token])`（冻结 LM 输出 embedding）
- **Loss：** 累积 MSE 蒸馏（从预训练 scalar RM 蒸馏）+ L2 正则化
- **CE：** ❌ 无

---

### 2.2 全量 LM head 作为 reward head（间接相关）

#### GenARM / ARM — Reward Guided Generation with Autoregressive Reward Model
**arXiv:2410.08193 · ICLR 2025**

- **Head：** 完整因果 LM head（LM 本身即 reward model，`reward = sum_t log π_r(y_t|context)`）
- **Loss：** BT 偏好损失作用在全序列对数概率求和上
  ```
  L = -E[log σ(β_r·Σ_t log π_r(y_w,t|·) − β_r·Σ_t log π_r(y_l,t|·))]
  ```
- **CE：** ❌（虽然架构上是 LM head，但训练目标是对比偏好，不是 next-token prediction）
- **相关论文：** PARM (2505.06274), UniARM (2602.09538), TITA (2510.21794) 均采用相同范式

---

### 2.3 使用 CE 的论文（但不适用于本项目场景）

以下论文确实用了 CE loss 作用在 vocab-size head 上，但与本项目的 dedicated reward head 有本质区别：

| 论文 | CE 作用的目标 | 为何不适用 |
|---|---|---|
| **GenRM** (ICLR 2025, 2408.15240) | 词表上插入的 Yes/No 验证 token | CE 打的是二值标签，不是自然语言下一个 token；是 math reasoning verifier |
| **ThinkPRM** (2025) | CoT 中的 correct/incorrect 决策 token | 同上，二值 CE |
| **GeDi** (EMNLP 2021) | 完整 LM 的下一个 token | 目的是保持语言模型的生成能力，不是 reward 训练 |
| **Math-Shepherd** (ACL 2024) | 仅限 +/- 两个符号 | 本质是 binary CE，词表限制为两个符号 |
| **RPO** (NeurIPS 2024) | 偏好响应的 SFT CE | 作用在 policy LM head，不是 reward head；用途是防止 overoptimization |
| **Q-SFT** (ICLR 2025) | Bellman 加权 CE 作用在 LM head | 权重来自 Q-learning backup，非 one-hot；作用在 LM head，不是专用 reward head |

---

## 三、为什么 FaRMA 不用 CE——理论层面的解释

### 3.1 vocab head 输出 Q 值，不是概率

FaRMA 的 vocab head 输出 reward magnitudes（Q 值），语义是"该 token 对最终 response 质量的贡献"。
对这个 head 做 softmax + CE，等价于把 Q 值强行解释成分类 logits，在语义上存在冲突：

- CE 要求：`score(true_token) > score(所有其他 token)`（分类器语义）
- Q 值要求：`score(token) ∝ 该 token 路径的预期累积奖励`（Q-function 语义）

这两个约束不一定兼容——最佳下一个 token（Q 值最高）不一定就是"真实下一个 token"（训练数据里出现的那个）。

### 3.2 TD loss 的 max 操作已经解决了"其他 token 无梯度"的问题

当我们担心"其他 token 的 score 没有监督信号"时，FaRMA 的 TD loss 已经通过 Bellman optimality
来解决：

```
L_td = 0.5 × (V(prefix) − max_{y'} V(prefix + y'))²
```

计算 `max_{y'} V(prefix + y')` 时，需要对全部 vocab tokens 前向计算一遍，取最大值的那几个 token
会接收到梯度（被压低，因为 max 值应该等于 V(prefix)）。这是一种 softer 且更 principled 的全词表监督。

### 3.3 理论反对意见

**"Why is Your Language Model a Poor Implicit Reward Model?"（Razin et al., NeurIPS 2024）** 
专门从理论上分析了为什么用 CE 训练的 LM 作为 reward model 效果差：CE 会使 head 倾向于给
高频/流畅的 token（冠词、介词、常用动词）打高分，而这些 token 的出现频率与 response 质量
相关性很弱。这会导致 SIA 的 logits processor 向"流畅性"而不是"质量"方向偏转，
有可能反向降低 alignment 效果。

---

## 四、对当前项目的影响

### 4.1 当前训练方案的评估

| 项目 | 状态 | 说明 |
|---|---|---|
| Loss 类型（BT + TD） | ✅ 符合业界标准 | 与 FaRMA ICML 2025 完全一致 |
| 是否需要加 CE | ❌ 不建议（当前阶段） | 文献无先例，有理论反对意见 |
| "其他 token 无梯度"担忧 | ℹ️ TD loss 已解决 | `max_{y'} V` 提供了间接的全词表梯度信号 |

### 4.2 加 CE 的风险评估

**不加 CE 的风险：** 低。BT+TD 是同行评审、经过验证的方案，主要风险是 BT 信号稀疏（序列级），
需要更多数据才能泛化。

**加 naive CE 的风险：** 中高。具体失效场景：CE 训练使 vocab_lowrank head 对高频/流畅 token
（冠词、介词）打高分，导致 SIA 推理时向流畅性偏转而非质量提升，可能逆转 alignment 效果。
Razin et al. (NeurIPS 2024) 在大规模实验上记录了这一失效模式。

**加 Bellman 加权 CE（Q-SFT 风格）的风险：** 中低。理论上更 principled，但增加实现复杂度，
且在 vocab_lowrank 设定下无直接验证。

### 4.3 如果未来想实验 CE

可以参考 Q-SFT (ICLR 2025) 的做法——不是 naive CE，而是 Bellman 加权 CE：

```python
# 不是这个（naive CE）：
L_ce = F.cross_entropy(full_vocab_scores, true_next_token_ids)

# 而是这个（Q-SFT 风格，Bellman 加权）：
bellman_weight = compute_bellman_target(V_prefix, max_V_next)   # 来自 TD 计算
L_qce = bellman_weight * F.cross_entropy(full_vocab_scores, true_next_token_ids)
```

但这应作为**独立的对比实验**，不应替换当前已验证的 BT+TD。

---

## 五、参考文献

| 论文 | 场景 | arXiv |
|---|---|---|
| FaRMA (ICML 2025) | vocab-size reward head, BT+TD | 2502.04517 |
| GenARM (ICLR 2025) | full LM head as reward, BT | 2410.08193 |
| Q-RM (ICML 2025) | vocab-size Q-function, BT | — |
| RAD-Q (TMLR 2025) | low-rank vocab head, MSE distillation | 2407.04615 |
| Q-SFT (ICLR 2025) | LM head as Q-value, Bellman-weighted CE | — |
| GenRM (ICLR 2025) | generative verifier, SFT CE on Yes/No | 2408.15240 |
| PARM (2025) | multi-objective ARM, BT | 2505.06274 |
| UniARM (2025) | ARM family, BT | 2602.09538 |
| SIA (2025) | scalar head, MSE distillation | 2602.21215 |
| Razin et al. (NeurIPS 2024) | 理论：CE LM 作为 reward model 的局限性 | — |
| GeDi (EMNLP 2021) | LM 做生成判别器，CE 保持生成能力 | — |
| Math-Shepherd (ACL 2024) | PRM，binary CE on +/- tokens | — |
| RPO (NeurIPS 2024) | DPO + SFT CE 防 overoptimization | — |
