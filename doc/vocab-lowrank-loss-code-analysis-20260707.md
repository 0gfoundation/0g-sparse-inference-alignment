# vocab_lowrank head: 低秩分解原因 + 代码 Loss 实现详解与对比分析

**日期：** 2026-07-07
**适用代码：** `SIA/src/value_model/train.py` + `SIA/src/value_model/model.py`
**使用此 loss 的实验：** `VM-Qwen3-4B-vocab-lowrank-sia-backbone-frozen-20260706`（val R²=0.8159）

---

## 零、为什么用两个小矩阵代替一个大矩阵（低秩分解的原因）

### 0.1 全量矩阵的规模

vocab-size head 的完整形式是一个 Linear(hidden=2560, vocab=151936) 层：

    W: (151936, 2560)   参数量 = 151936 × 2560 = 389M

低秩分解（rank=64）把它拆成两个矩阵的乘积：

    W ≈ B · A

    A (score_A): (64, 2560)       参数量 =      163,840
    B (score_B): (151936, 64)     参数量 =    9,723,904
    合计:                                      9,887,744  ≈ 9.9M

压缩比：389M / 9.9M = 39.3 倍

### 0.2 原因一：训练激活显存（最重要）

推理时只需要对当前 1 个 token 位置打分，全量矩阵完全可以接受。
训练时的问题在于：如果把整批序列的 vocab 分数全部展开，需要一个 (B, L, V) 张量：

    batch=8, seq_len=1024, vocab=151936, fp32:
    8 × 1024 × 151936 × 4 = 4.98 GB（仅此一个中间激活）

低秩路径的训练技巧（dot-product trick）：每个位置只取 score_B 里实际 token 对应的那一行，
完全不展开 (B, L, V)，中间激活只需约 4 MB，节省 1250 倍。

    W_B = score_B.weight                  # (151936, 64)
    W_B[input_ids[:, 1:]]                 # 只取实际 token 行，(B, L-1, 64)
    scores = (h_r[:, :-1] * W_B[...]).sum(-1)   # (B, L-1) 标量

全量矩阵也可以用同样的 dot-product trick（直接取 W 对应行做点积）——
即使用全量矩阵，只要训练时不展开 (B, L, V)，激活显存同样可以很低。

### 0.3 原因二：参数量与过拟合风险

|  | 全量矩阵 | 低秩 (rank=64) |
|--|---------|---------------|
| 新增可训练参数 | 389M | 9.9M |
| bf16 权重显存 | 0.78 GB | 20 MB |
| Adam 优化器状态 fp32 | 3.1 GB | 79 MB |
| 过拟合风险 | 高（训练数据有限时） | 低 |

训练数据规模有限时（本项目约数万对），389M 参数的全量矩阵极易过拟合。
低秩分解把有效容量压缩到 9.9M，起到了类似 LoRA 的正则化效果。

### 0.4 原因三：隐含的低秩假设

低秩分解等价于假设：reward 信息可以被 64 个正交基向量完整表达。

    score_A 提取"当前 context 属于哪种 reward 模式"（64 维向量）
    score_B 存储"每个 token 在各 reward 模式下的得分权重"（151936 × 64）

如果 context 的 reward 信息确实在低维流形上（比如"是否回答了问题"、
"逻辑是否连贯"等少数维度），rank=64 就足够捕获，不需要 2560 维的全连接。

### 0.5 什么时候全量矩阵也可以训练

以下两个条件同时满足时，全量矩阵在显存上完全可行：

1. 冻结 backbone（只训练 head）：backward 不需要存 transformer 各层激活
2. 训练时使用 dot-product trick（不展开全量 (B, L, V)）

此时完整的显存需求：

    Backbone bf16（frozen）: 8 GB
    全量 head 权重 fp32:     1.56 GB
    全量 head 梯度 fp32:     1.56 GB
    Adam 优化器状态 fp32:    3.12 GB
    hidden states 激活:      0.08 GB
    ────────────────────────────────
    合计:                  ≈ 14.3 GB

80 GB 的卡完全可以训练全量 vocab head。主要阻力不是显存，而是过拟合风险
（389M 参数，训练数据有限）以及训练时间（参数多则收敛慢）。

---

## 一、大白话 + 公式：代码里 loss 到底在算什么

### 1.1 head 的结构（vocab_lowrank）

head 由两个线性层组成：

    score_A: Linear(hidden=2560, rank=64)     权重形状 (64, 2560)
    score_B: Linear(rank=64, vocab_size=151936)  权重形状 (151936, 64)

对于序列中第 t 个位置，transformer 给出 hidden state h_t（维度 2560）。

训练时（return_vocab_rewards=False），head 只计算实际下一个 token 的分数：

    h_r[t] = score_A(h_t)                          # 维度 64
    V(t, x_{t+1}) = h_r[t] · score_B.weight[x_{t+1}]  # 标量

这是 ARM shift（自回归移位）：用第 t 步的 hidden state，预测第 t+1 步实际生成 token 的 value。

关键事实：每个位置只取 score_B.weight 的 1 行（对应实际生成的 token），
其余 151935 行完全不参与计算，梯度为 0。

整条序列的 value（用于 BT loss）：

    V(response) = weighted_mean over t of V(t, x_{t+1})

---

### 1.2 Loss 一：BT loss（Bradley-Terry，对比偏好）

输入：一对 (win response w, lose response l)

    V_win  = weighted_mean_t [ V(t, x^w_{t+1}) ]   # win 序列的平均 value
    V_lose = weighted_mean_t [ V(t, x^l_{t+1}) ]   # lose 序列的平均 value

    L_BT = -log( sigma(V_win - V_lose) )

梯度方向：

    d L_BT / d V_win  < 0  →  推 V_win 上升
    d L_BT / d V_lose > 0  →  推 V_lose 下降

结果：win 和 lose 序列里实际出现的所有 token 都会影响梯度更新。
win token 的 score_B 行被推高，lose token 的 score_B 行被推低。

---

### 1.3 Loss 二：TD loss（per-position MSE，正则化）

输入：win 序列 + 数据集里存的标量 reward R_win（来自 Skywork 打分，训练时视为常量）

    L_TD = mean over masked positions t of ( V(t, x^w_{t+1}) - R_win )^2

R_win 是数据集常量，不是模型输出，不产生梯度。
只有 V(t, x^w_{t+1}) 有梯度，被推向 R_win。

注意：L_TD 只作用于 win 序列，lose 序列不参与。

---

### 1.4 总 Loss

    L = L_BT + lambda * L_TD

当前实验使用 lambda = 0.5（训练命令 --td_weight 0.5）：

    L = L_BT + 0.5 * L_TD

---

### 1.5 梯度覆盖范围总结

| 参数 | 来自 L_BT | 来自 L_TD |
|------|----------|----------|
| score_A (64×2560) | ✅ win + lose 双方 | ✅ 仅 win |
| score_B[win tokens 的行] | ✅ 推高 | ✅ 推向 R_win |
| score_B[lose tokens 的行] | ✅ 推低 | ❌ 无 |
| score_B[从未出现的 token 行] | ❌ 无 | ❌ 无 |
| backbone（本实验已 freeze） | frozen | frozen |

**核心不对称：每个位置 151936 个输出中，只有 1 个参与了梯度计算。**
其余 151935 个 value 在训练时从未被算出，梯度永远为零。

---

## 二、与 SIA 官方 MSE loss 的对比

SIA 官方代码（--loss_type mse）用的是标量 head，loss 是：

    L_MSE_official = ( weighted_mean_t[ V(t) ] - R )^2

即先对所有 token 取均值，再和 R 算一次 MSE。

本项目的 L_TD 是：

    L_TD = mean_t [ ( V(t, x_{t+1}) - R_win )^2 ]

两者的数学关系：

    L_TD = Var_t( V(t, x_{t+1}) ) + L_MSE_official

L_TD 比 L_MSE_official 多了一个方差惩罚项：

- L_MSE_official 允许各 token 位置的 value 高低不一，只要平均值等于 R 就满足 loss = 0
- L_TD 要求每个位置的 value 都等于 R，不允许内部有散布

所以 L_TD 是 L_MSE_official 的加强版，约束更严格。

另一个差异：本项目同时使用了 L_BT，官方 MSE 模式没有 BT loss。
BT loss 的对比学习信号比 MSE 更直接地区分好坏 response。

---

## 三、与 FaRMA 论文的对比

FaRMA 论文（arXiv:2502.04517，ICML 2025）提出的 TD loss（Loss_b）：

    L_b(i) = 0.5 * ( V(i, x_i) - max over all t' of V(i+1, t') )^2

这是 Bellman optimality constraint：位置 i 的 value，应该等于下一步所有可能 token 中最高的那个 value。

### 3.1 和本项目 L_TD 的本质区别

| | FaRMA L_b（Bellman TD） | 本项目 L_TD（MSE to R_win） |
|--|------------------------|-----------------------------|
| target 是什么 | max over 151936 tokens of V(i+1, t') | 数据集标量 R_win（常量） |
| target 是否来自模型 | 是（同一个网络的输出） | 否（外部 Skywork 打分） |
| 训练稳定性 | 需要 target network 才能稳定 | target 是常量，天然稳定 |
| 内存需求 | (B, L, 151936) ≈ 5GB per batch | (B, L) + 一个标量，可忽略 |
| 覆盖所有 token | ✅ max 操作隐式约束全词表 | ❌ 只约束实际出现的 token |

### 3.2 为什么没有实现 FaRMA 的 Bellman TD

**原因一：内存**

训练时 (model.py line 319-323) 使用了节省内存的路径：

    W_B = score_B.weight                           # (151936, 64)
    W_B[input_ids[:, 1:]]                          # 只取实际 token 对应的行，(B, L-1, 64)
    scores = (h_r[:, :-1, :] * W_B[...]).sum(-1)   # (B, L-1) 标量

完整的 Bellman target 需要 max over 151936 tokens，即展开 (B, L, 151936) 矩阵：

    batch=8, seq_len=1024, vocab=151936, fp32:
    8 × 1024 × 151936 × 4 bytes ≈ 5 GB（仅此一个中间张量）

**原因二：训练不稳定**

FaRMA 的 Bellman target 是同一个网络的输出 max V(i+1, t')，每步梯度更新后 target 就变了，
两侧互相追赶，容易发散。DQN 等 RL 方法需要独立的 target network（延迟更新的副本）才能稳定。
本项目没有 target network，直接用 Bellman loss 会导致训练不稳定。

**原因三：任务目标不同**

FaRMA 的目标是训练一个严格的 Q-function，能在每一步从 151936 个 candidate 中选最优 action。
本项目的目标是 preference learning（BT 对比），L_TD 只是正则化，用固定的 R_win 作 target 已经足够。

### 3.3 代价：训练-推理不一致

FaRMA 的 Bellman TD 通过 max 操作，让每个位置的 score_B 行都间接收到梯度。
本项目每个位置只有 1 行被更新，推理时 SIA 对从未出现过的 candidate token 打分，
依赖 score_A 的泛化能力；对应的 score_B 行则是随机初始化未更新状态。

这是用内存和稳定性换取的代价，也是后续可以优化的方向（见第四节）。

---

## 四、后续优化方向（不改变现有代码的前提下）

1. **Bellman TD（完整实现）**：需要在 forward 里返回 max(vocab_scores) per position，
   同时引入 target network（或 EMA 更新的副本）。内存代价约 5GB，工程量较大。

2. **Top-K TD**：不对全量 151936 token 取 max，而是只对 SIA 推理时实际用的 top-k（如 top-10）
   candidate token 取 max。内存从 5GB 降到可接受范围，同时缩小训练-推理不一致。

3. **候选 token 采样**：每个位置随机采样 K 个 token（包括实际生成的和随机的），
   对采样的 K 个 token 都计算 V 并参与 loss。这是 REINFORCE / offline RL 的常见做法。

---

## 五、为什么 scalar head 不需要冻结 backbone，vocab_lowrank 需要

### 5.1 实验证据

| 实验 | head | freeze | td_weight | val R² | AlpacaEval Δ |
|------|------|--------|-----------|--------|-------------|
| SIA 官方（scalar, MSE） | scalar | ❌ 不冻结 | — | 高 | 正向 ✅ |
| bt-20260704（vocab_lowrank, BT） | vocab_lowrank | ❌ 不冻结 | 0.01 | 0.6934 | +0.48% ✗ |
| frozen-20260706（vocab_lowrank, BT） | vocab_lowrank | ✅ 冻结 | 0.5 | 0.7491 | +2.98% ✅ |
| sia-backbone-frozen（vocab_lowrank, BT） | vocab_lowrank | ✅ 冻结 | 0.5 | 0.8159 | +4.82% ✅ |

bt-20260704 中 vocab_lowrank 不冻结训练后，val R² 仅 0.6934，报告注明 "backbone damaged"。

### 5.2 根本原因：两种 head 传给 backbone 的梯度性质完全不同

**scalar head → backbone 的梯度：密集、方向一致**

scalar head 的计算：

    token_rewards[t] = w · h_t        （w 只有 2560 个参数）

MSE loss 对 h_t 的梯度：

    d L / d h_t = w^T × 2(V - R) / T

每个位置的梯度方向都是 w^T，乘以同一个误差标量。
LoRA 收到的是方向高度一致的密集信号，非常容易收敛。

**vocab_lowrank → backbone 的梯度：稀疏、方向随机**

vocab_lowrank 使用 ARM shift，位置 t 的计算：

    h_r[t]         = score_A · h_t             # (64,)
    token_rewards[t] = h_r[t] · W_B[x_{t+1}]  # 取 score_B 第 x_{t+1} 行

梯度传回 h_t：

    d L / d h_t = score_A^T · (W_B[x_{t+1}]^T × scalar)

x_{t+1} 每个位置不同，W_B[x_{t+1}] 是 score_B 里不同的行。
LoRA 收到的是各位置"不同 token 对应不同方向"的梯度平均值，大量相互抵消，净信号接近噪声。
LoRA 在噪声驱动下更新 backbone，损坏了预训练表示，泛化能力下降。

### 5.3 头的表达能力决定了谁来做主力

scalar head 只有 2560 个参数，线性映射到 1 个标量，本身几乎没有表达能力。
它无法自己区分 reward，必须依靠 LoRA 调整 backbone，让 h_t 本身编码 reward 信息。
MSE loss 恰好给 LoRA 提供了密集稳定的梯度，所以 LoRA 能够学习，不冻结没问题。

vocab_lowrank head 有 9.9M 参数（score_A + score_B），表达能力充足。
它完全可以从预训练 backbone 的稳定语义表示中自己学出 reward 映射，不需要 LoRA 参与。
LoRA 加入后反而带来噪声梯度，破坏 backbone，所以必须冻结。

### 5.4 总结对比

| | scalar head（不冻结） | vocab_lowrank（不冻结） | vocab_lowrank（冻结） |
|--|---------------------|----------------------|---------------------|
| 每位置传给 backbone 的梯度方向 | 一致（都是 w^T） | 随机（依赖 x_{t+1}） | — |
| LoRA 收到的净信号 | 密集有效 | 大量抵消≈噪声 | — |
| head 自身表达能力 | 极弱（2560 params） | 强（9.9M params） | 强（9.9M params） |
| 谁做主力 | LoRA | 两者竞争互相干扰 | head 独立学习 |
| 结果 | 收敛好 | backbone 被损坏 | 收敛好 |

**一句话**：scalar head 太弱，LoRA 必须做主力，MSE loss 的密集梯度恰好支撑 LoRA 学习。
vocab_lowrank 足够强，不需要 LoRA，但它给 LoRA 的梯度是稀疏随机噪声，冻结才能隔离干扰。

---

## 六、后续优化方向（不改变现有代码的前提下）

1. **Bellman TD（完整实现）**：需要在 forward 里返回 max(vocab_scores) per position，
   同时引入 target network（或 EMA 更新的副本）。内存代价约 5GB，工程量较大。

2. **Top-K TD**：不对全量 151936 token 取 max，而是只对 SIA 推理时实际用的 top-k（如 top-10）
   candidate token 取 max。内存从 5GB 降到可接受范围，同时缩小训练-推理不一致。

3. **候选 token 采样**：每个位置随机采样 K 个 token（包括实际生成的和随机的），
   对采样的 K 个 token 都计算 V 并参与 loss。这是 REINFORCE / offline RL 的常见做法。

---

## 七、相关文档

- `doc/vocab-reward-head-training-loss-survey-20260706.md`：文献综述，94 篇论文，BT+TD 是业界标准
- `doc/farma-paper-summary-zh-20260701.md`：FaRMA 论文中文总结
- `doc/farma-vocab-lowrank-20260703.md`：vocab_lowrank head 架构设计
- `SIA/src/value_model/train.py`：loss 实现（train_epoch_bt 函数，line 333）
- `SIA/src/value_model/model.py`：head forward 实现，ARM shift（line 309-324）
