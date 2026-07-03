# vocab_lowrank Cross-Tokenizer Mapping: Design & Analysis (2026-07-03)

**TL;DR**: 0GM-35B 与 VM-Qwen3-4B 词表不兼容（248k vs 151936），导致 vocab_lowrank head 的 FaRMA 加速路径被 fallback 到 N×K scalar 模式。本文记录两种解决思路的分析：①预计算 token ID 映射表（推荐，可行）；②改用 Qwen3.5 系列作为 VM backbone（词表兼容，但 vLLM 暂不支持 text-only 推理）。

---

## 1. 问题背景

### 1.1 vocab_lowrank 加速原理

vocab_lowrank head 将 VM 的 reward head 分解为：

```
W ∈ R^(V×H)  →  A: (H→rank) + B: (rank→V)
```

推理时只需 **1 次 prefix forward**，得到 `(N, vocab_size)` 的 reward 矩阵，再按 candidate token ID 索引即可：

```python
vocab_rewards = score_B(score_A(hidden_states))  # (N, vocab_size)
scores = vocab_rewards[row_idx, candidate_token_ids]  # (K,)
```

相比 scalar head 的 N×K 次 forward，理论上有约 K≈10× 的 RM 调用加速。

### 1.2 跨 tokenizer 问题

| 模型 | tokenizer | vocab_size |
|------|-----------|------------|
| 0GM-1.0-35B-A3B (LLM) | Qwen2Tokenizer (扩展版) | 248,320 |
| VM-Qwen3-4B-Base (RM) | Qwen2Tokenizer (标准版) | 151,936 |
| Qwen3-VL-30B-A3B-Instruct (LLM) | Qwen2Tokenizer (标准版) | 151,936 |

0GM-35B 与 VM-Qwen3-4B 词表**不兼容**：LLM 的 token ID（最大 248,319）无法直接索引 RM 的 vocab 空间（最大 151,935）。

`RMClient` 检测到 `_cross_tokenizer=True` 后，`score_with_vocab_head_batch` 直接 fallback 到 `score_candidates_batch`（N×K forward），vocab_lowrank 的加速完全失效。

### 1.3 已知 code bug（cross-tokenizer + vocab_lowrank 组合）

`RMClient.__init__` 里 `set_vocab_head_mode(True)` 在 tokenizer 兼容性检测**之前**调用。导致：

- `compute_logits` 只写 `_VOCAB_REWARD_BUFFERS`（vocab path）
- `score_candidates_batch` 读 `_REWARD_BUFFERS`（scalar path）→ 返回 `None`
- **RuntimeError: reward channel returned 0 values, expected N**

即：对 0GM-35B 使用 `--vm_head_type vocab_lowrank` 时会直接 crash，不是静默降级。

---

## 2. 解决方案一：预计算 token 文本映射表（推荐）

### 2.1 核心思路

在**推理前离线**建立映射：

```
ogm_token_id → decode text → encode with Qwen3 tokenizer → qwen3_token_id
```

若结果恰好是 1 个 Qwen3 token，则建立 1-to-1 映射；否则标记为 unmapped（-1）。

映射表存为 numpy int32 数组，shape `(248320,)`，约 1 MB，加载一次常驻内存。

### 2.2 推理时的完整流程

```
1. [startup] 加载 ogm_to_qwen3: np.ndarray(248320,) — int32, -1=unmapped

2. [每次 INTERVENE]
   a. 0GM prefix token IDs → decode text → re-encode with Qwen3 tok
      （复用现有 _get_stable_rm_prefix，映射命中时可用 dict lookup 加速）
   b. 1 次 VM forward(qwen3_prefix_ids) → vocab_rewards (N, 151936)
   c. for each candidate (ogm_cand_id):
          qwen3_id = ogm_to_qwen3[ogm_cand_id]
          score = vocab_rewards[row_idx, qwen3_id] if qwen3_id >= 0 else 0.0
```

### 2.3 两处加速

**加速 1（次要）：消除 candidate token 翻译的 tokenizer 调用**

| 路径 | candidate 翻译方式 |
|------|-------------------|
| 现有 cross-tokenizer scalar | K × (ogm_tok.decode + qwen3_tok.encode) |
| 新方案 | K × dict O(1) lookup |

每次 INTERVENE 约消除 K=10 次 tokenizer Python 调用，节省 ~1–3ms/step。prefix 侧映射命中的 token 同样受益。

**加速 2（主要）：N×K → N 次 VM forward**

| 路径 | VM forward 次数/INTERVENE |
|------|--------------------------|
| 现有 cross-tokenizer scalar | N × K（e.g. 8×10=80） |
| 新方案 vocab_lowrank | N（e.g. 8） |

约 K≈10× 的 RM 调用减少，是主要瓶颈所在。

### 2.4 覆盖率估计

| token 类型 | 映射情况 |
|------------|---------|
| 单字节 ASCII、常用汉字、常见英文词 | ✅ 大概率 1-to-1 |
| 0GM 新增长 token（多字节拼接） | ❌ Qwen3 tokenize 成多个 token |
| 0GM 特殊 domain token | ❌ Qwen3 不认识 |

对中英文日常推理任务，topk candidate 中高频 token 占多数，**估计覆盖率 80–90%**。unmapped candidate 给 0 分（= 不施加 SIA 干预），保守但不引入错误偏差。

### 2.5 prefix 的细节

prefix 仍需走 Qwen3 token 空间（VM embedding 层只有 151936 槽位）。两种处理方式：

- **纯 dict lookup**：对每个 prefix token 查映射表；unmapped token 跳过（prefix 略有缺失，轻微质量损失）
- **混合方式（推荐）**：映射命中 → dict lookup；未命中 → 复用现有 `_get_stable_rm_prefix` 文本路径（保持 prefix 完整性）

现有 `_get_stable_rm_prefix` 已经是逐 token 增量 encode，混合方式只是把其中的"decode+encode"替换成 dict lookup，接口不变。

### 2.6 实现位置

- `scripts/build_token_mapping.py`：离线建表工具（已写，见该文件）
- `src/sia_rm/client.py`：`RMClient.__init__` 加载映射表；修复 `set_vocab_head_mode` 调用时序；`score_with_vocab_head_batch` 添加映射翻译逻辑

---

## 3. 解决方案二：改用 Qwen3.5 作为 VM backbone

### 3.1 动机

Qwen3.5 系列与 0GM-35B 使用完全相同的 tokenizer（Qwen2Tokenizer，vocab_size=248,320）。若以 Qwen3.5-4B-Base 训练 VM，则 `_cross_tokenizer=False`，vocab_lowrank 路径完整可用，无需映射表。

### 3.2 Qwen3.5 可用的小模型

| 模型 | hidden_size | 状态 |
|------|-------------|------|
| Qwen/Qwen3.5-0.8B-Base | 1024 | 可下载 |
| Qwen/Qwen3.5-2B-Base | — | 可下载 |
| Qwen/Qwen3.5-4B-Base | 2560 | 可下载 |

Qwen3.5-4B-Base 的 hidden_size=2560 与当前 Qwen3-4B-Base 一致，vocab_lowrank head 参数量相同（rank=64 的情况下 A: 64×2560, B: 248320×64 ≈ 18M params）。

### 3.3 阻碍：vLLM 对 Qwen3.5 的支持不完整

Qwen3.5 使用**全新混合架构**（`model_type="qwen3_5"`，DeltaNet 线性注意力 + Mamba SSM，非标准 Transformer），与 Qwen3 的架构不兼容：

| 项目 | Qwen3 | Qwen3.5 |
|------|-------|---------|
| model_type | qwen3 | qwen3_5 |
| vLLM 架构类 | Qwen3ForCausalLM | Qwen3_5ForConditionalGeneration（多模态） |
| text-only 类 | ✅ 有 | ❌ 无（issue #39993 closed as not planned） |
| SIA VM 子类 | Qwen3WithScoreForCausalLM | 需重新实现 |

当前 `Qwen3WithScoreForCausalLM` 继承 vLLM 的 `Qwen3ForCausalLM`，无法直接用于 Qwen3.5。需要：

1. 等待 vLLM 正式支持 Qwen3.5 text-only 推理
2. 编写 `Qwen3_5WithScoreForCausalLM`，继承 vLLM 的 Qwen3.5 类
3. 处理权重命名 mismatch（text-only checkpoint 与多模态 wrapper 的 prefix 不同）

**结论**：Qwen3.5 路径在 vLLM 支持完善之前工程量较大，暂不推荐。

---

## 4. 当前状态与推荐路径

| 场景 | 当前状态 | 推荐操作 |
|------|---------|---------|
| Qwen3-VL-30B + vocab_lowrank | ✅ 可用，N→1 forward | 直接上线验证 |
| 0GM-35B + scalar head | ✅ 可用 | 维持现状 |
| 0GM-35B + vocab_lowrank（映射表方案） | 🔧 待实现 | 优先级次于 VL-30B 验证 |
| 0GM-35B + Qwen3.5 VM | ❌ vLLM 不支持 | 等 vLLM 支持后再评估 |

**近期推荐路径**：先在 Qwen3-VL-30B 上完整验证 vocab_lowrank 的质量与速度；验证通过后，再按方案一实现映射表，把加速推广到 0GM-35B。
