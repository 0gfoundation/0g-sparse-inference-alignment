# 0GM-1.0-35B-A3B + Qwen3-4B-RM SIA vs noSIA MMLU-Redux 评测结果

**日期**: 2026-06-02
**主推理模型**: 0GM-1.0-35B-A3B-0427 (Qwen3.5/3.6 MoE, vocab=248044, thinking 模式)
**Value Model (RM)**: VM-Qwen3-4B-merged-for-vllm (训练于 Qwen3-4B base)
**评测**: MMLU-Redux, 30 subjects × 20Q (SIA, 600 题) vs 30 subjects × 30Q (noSIA, 900 题)
**vllm 版本**: 0.19.0 (venv4)

---

## 1. TL;DR

| 指标 | noSIA (matched 30 subj × 30Q = 900Q) | SIA (30 subj × 20Q = 600Q) | Δ |
|------|--------------------------------------|----------------------------|---|
| **准确率** | 75.6% (680/900) | 74.3% (446/600) | **-1.2 pp** |
| **吞吐** | 108.9 tok/s | 51.4 tok/s | **慢 52.8%** (1.97×) |
| per-Q latency (avg) | 11.7 s | 19.6 s | +67% |
| avg tokens/Q | 1271 | 1006 | -21% (SIA 把 thinking 缩短了) |
| 总耗时 | 165 min | 196 min | — |

**核心结论**:
- **准确率几乎持平** (-1.2 pp, mean ≈ 0, median = 0)。13 个 subject 提升 / 5 持平 / 12 下降, 跟 "no-effect + noise" 难以区分
- **速度损失 53%**, 主要来自跨进程 RM HTTP 调用 (~95ms/INTERVENE × 8.3% intervene rate, 详见 [`0gm-35b-sia-rm-inproc-path-20260602.md`](0gm-35b-sia-rm-inproc-path-20260602.md))
- **当前 RM (VM-Qwen3-4B) 没有为 0GM 训练**, alignment 信号本质是噪声, 跟 14B + 同 RM 的 +SIA 收益对比不成立

---

## 2. Per-subject Δaccuracy 完整结果

```
subject                             noSIA(30Q)      SIA(20Q)        Δacc
---------------------------------------------------------------------------
anatomy                            25/30 (83.3%)   15/20 (75.0%)   -8.3%
astronomy                          28/30 (93.3%)   15/20 (75.0%)  -18.3% ⚠️
business_ethics                    23/30 (76.7%)   16/20 (80.0%)   +3.3%
clinical_knowledge                 27/30 (90.0%)   18/20 (90.0%)   +0.0%
college_chemistry                  10/30 (33.3%)    7/20 (35.0%)   +1.7%
college_computer_science           21/30 (70.0%)   14/20 (70.0%)   +0.0%
college_mathematics                20/30 (66.7%)   14/20 (70.0%)   +3.3%
college_medicine                   24/30 (80.0%)   14/20 (70.0%)  -10.0%
college_physics                    21/30 (70.0%)   13/20 (65.0%)   -5.0%
conceptual_physics                 29/30 (96.7%)   19/20 (95.0%)   -1.7%
econometrics                       25/30 (83.3%)   17/20 (85.0%)   +1.7%
electrical_engineering             21/30 (70.0%)   10/20 (50.0%)  -20.0% ⚠️
formal_logic                       20/30 (66.7%)   15/20 (75.0%)   +8.3%
global_facts                       15/30 (50.0%)    9/20 (45.0%)   -5.0%
high_school_chemistry              22/30 (73.3%)   15/20 (75.0%)   +1.7%
high_school_geography              25/30 (83.3%)   19/20 (95.0%)  +11.7% ✅
high_school_macroeconomics         26/30 (86.7%)   18/20 (90.0%)   +3.3%
high_school_mathematics            16/30 (53.3%)    8/20 (40.0%)  -13.3% ⚠️
high_school_physics                17/30 (56.7%)   11/20 (55.0%)   -1.7%
high_school_statistics             25/30 (83.3%)   16/20 (80.0%)   -3.3%
high_school_us_history             25/30 (83.3%)   19/20 (95.0%)  +11.7% ✅
human_aging                        21/30 (70.0%)   12/20 (60.0%)  -10.0%
logical_fallacies                  26/30 (86.7%)   18/20 (90.0%)   +3.3%
machine_learning                   25/30 (83.3%)   17/20 (85.0%)   +1.7%
miscellaneous                      30/30 (100.0%)  20/20 (100.0%)  +0.0%
philosophy                         24/30 (80.0%)   16/20 (80.0%)   +0.0%
professional_accounting            26/30 (86.7%)   19/20 (95.0%)   +8.3%
professional_law                   20/30 (66.7%)   14/20 (70.0%)   +3.3%
public_relations                   22/30 (73.3%)   14/20 (70.0%)   -3.3%
virology                           21/30 (70.0%)   14/20 (70.0%)   +0.0%
---------------------------------------------------------------------------
TOTAL (matched)                   680/900 (75.6%)  446/600 (74.3%)  -1.2%
```

- **最差 3 个**: electrical_engineering -20pp, astronomy -18.3pp, high_school_math -13.3pp
- **最好 3 个**: high_school_geography +11.7pp, high_school_us_history +11.7pp, professional_accounting +8.3pp
- **subject 计数**: 13 提升 / 5 持平 / 12 下降
- mean Δ = -1.22%, median Δ = 0.00%

---

## 3. 速度详细分解

### 3.1 总体

| | noSIA | SIA |
|---|-------|-----|
| total tokens | 1,097,497 | 603,668 |
| total latency | 10,078 s | 11,747 s |
| **throughput** | **108.9 tok/s** | **51.4 tok/s** |
| per-token | 9.18 ms | 18.21 ms (**+9.0 ms/tok**) |

### 3.2 SIA 内部时延 (server pf-summary 累计统计)

| phase | p50 | p95 | max | 含义 |
|-------|-----|-----|-----|------|
| `http_post` (RM /classify) | **94.7 ms** | 124 ms | 140 ms | 单次跨进程 RM 调用 (主瓶颈) |
| `apply_total` (per apply, 含 SKIP+INTERVENE) | 4.0 ms | 77 ms | 122 ms | apply() 总耗时, p50 主要是 SKIP |
| `skip_step` (per SKIP) | 4.0 ms | 4.1 ms | 4.1 ms | 每次 SKIP 的 SIA cleanup |
| `apply_cpu_sync` | 3.6 ms | 3.6 ms | 3.7 ms | entropy + topk + cpu().tolist() |

### 3.3 每 token SIA overhead 拆分

```
每 token 平均 SIA overhead 
  ≈ 0.083 × 96.5 ms (INTERVENE: http_post + intv 内 ops)
  +  0.917 × 4.01 ms (SKIP path 的 SIA cleanup)
  ≈ 8.0 ms (INTERVENE-driven)  +  3.7 ms (SKIP-driven)
  ≈ 11.7 ms/tok

实测: SIA 18.2 ms/tok  vs  noSIA 9.18 ms/tok  →  Δ = +9.0 ms/tok
```

**主瓶颈**: 跨进程 HTTP `/classify` 95 ms/call, 占了 8 ms/tok 中的绝大部分。详见 [`0gm-35b-sia-rm-inproc-path-20260602.md`](0gm-35b-sia-rm-inproc-path-20260602.md) 中"为什么必须跨进程"和"是否能重新走 inproc"的分析。

---

## 4. SIA 干预指标 (从 server DONE 累计统计 380 个请求)

| 指标 | 数值 |
|------|------|
| 干预步数 / 总 step | 36,350 / 436,030 = **8.34%** |
| top-1 flip / 干预步数 | 1,253 / 36,350 = **3.45%** |
| top-1 flip / 总 step | **0.287%** |
| 每请求干预率分布 | p50=7.7%, mean=8.0%, max=20.6%, min=1.0% |
| 每请求 top-1 flip 率 | p50=3.2%, mean=4.1% (相对该请求干预次数) |

**对比 Qwen14B baseline (历史)**:
- 同样 `entropy_threshold=1.0, topk=5, weight=1.0, RM=VM-Qwen3-4B`
- 14B 干预率 **30%+** vs 0GM-35B **8.34%**
- 14B flip 率历史 ~10%+ vs 0GM-35B **3.45%**

---

## 5. 为什么 0GM-35B 上干预率这么低 (深度分析)

### 5.1 entropy 计算回顾

SIA 的 entropy 不是全 vocab entropy, 是 **top-5 logits 重归一化后的 entropy** (`src/sia_vllm_RM.py:698-702`):

```python
topk_result = torch.topk(logits, self._TOPK, dim=-1)        # top-5 raw logits
log_probs = F.log_softmax(topk_result.values.float(), dim=-1)   # softmax 仅在 top-5
probs = log_probs.exp()
entropies = -(probs * log_probs).sum(dim=-1)                # 范围 [0, log(5) ≈ 1.609]
```

threshold=1.0 = top-5 至少**有 62% 接近均匀分布**才触发 INTERVENE。

具象算例:
- **logit 平坦** [10, 9.5, 9, 8.5, 8] → 重归一化 probs ≈ [0.43, 0.26, 0.16, 0.10, 0.06], **entropy ≈ 1.40** → INTERVENE
- **logit 尖锐** [12, 9, 8, 7, 6] → 重归一化 probs ≈ [0.93, 0.05, 0.02, 0.006, 0.002], **entropy ≈ 0.32** → SKIP

### 5.2 候选原因 (按概率从高到低)

#### ① **35B 比 14B 更"自信"** — 主因

更大的预训练模型 → 每 token 决策更确定 → top-5 内 top-1 占绝大多数概率 → 重归一化后 entropy 接近 0。

- 14B 在 chat 场景 top-1 vs top-2 logit 差常 ~1-2 → entropy ~1.0-1.2 (经常 ≥ threshold)
- 35B top-1 vs top-2 logit 差常 ~2-3 → entropy ~0.3-0.5 (远低于 threshold)

这是预训练量更大模型的自然属性, 无 bug。

#### ② **0GM 是 thinking 模型, 大量 token 在低熵推理 prose** — 高概率次因

0GM chat template 自动加 `<think>\n`, 强制模型先输出 chain-of-thought。大段内心独白如:

```
The user wants to identify the disorder characterized by uncontrollable...
Option A: Dyslexia - A learning disorder affecting reading.
Option B: Epilepsy - A neurological disorder characterized by seizures.
...
```

这类 prose 里每个 token (空格、单词后续、连接词、option-list 结构) 都几乎**确定**, top-1 占压倒性概率, entropy 远 < 1.0。

只有真正"关键决策点"才出现高 entropy:
- 选哪个 option (`Therefore the answer is __`)
- 选 `</think>` 收尾还是继续推理
- 自我修正 (`Wait, no, actually __`)

数据印证: SIA mean 1106 tokens/Q, INTERVENE ~91 个/Q (8.3%) — 大致对应"thinking 中真正 explore 的关键节点", 跟论文 sparse junction intervention 理论一致。

Qwen14B 不是 thinking 模型, 每个回答 token 都是"答案 token", 不确定性集中, 30%+ 干预率合理。

#### ③ **MoE 架构输出比 dense 更尖锐** — 中等概率次因

MoE 每 token 激活专门 expert, expert 通常是更**确信**的 specialist → 输出 logit 比同等大小 dense 模型更 peaked。这跟 ① 累加。

#### ④ **vocab 248K vs 152K, 影响间接** — 低概率因素

entropy 是在 top-5 内算的, vocab 大小**不直接影响** top-5 entropy。但 vocab 大 → 全分布概率质量分布得更广 → top-5 内的 top-1 相对更突出 (top-5 之外更多 token 拿走 mass)。属于二阶效应。

#### ⑤ **sampling params 差异 — 已排除**

SIA 的 entropy 在 vllm sampler 的 temperature/top_k/top_p/repetition_penalty **之前**计算 (sampler.py 顺序: float32 cast → SIA processor → apply_penalties → temperature)。两个评测的 sampling params 不同 (0GM: `top_k=20, top_p=0.95, rep_pen=1.0` vs 14B: `top_p=1.0, rep_pen=1.3`), 但**不影响 SIA entropy 判断**。

#### ⑥ **代码 bug — 已排除**

逐行 review 过, 代码 path 两边一致, 无 0GM-specific 分支。差异来自 logits 本身。

### 5.3 为什么 flip 率离谱低 (3.45% of intervened)

flip = SIA 真改了 top-1 选择。条件:
1. top-5 entropy ≥ 1.0 (INTERVENE 触发)
2. **AND** RM reward delta 足够大, 超过 top-1 vs top-2 在 top-5 内的 logit gap

RM 给的是 sigmoid 后概率 (range [0, 1]), 经 `(rm_scores - mean) * weight=1.0` 后 deltas 在 ±0.7 量级 (实测)。

**关键算例** (top-5 entropy=1.0 时):
- 14B: 典型 logit 差 ~0.3-1.0 → RM delta 0.7 经常能 flip → flip 率历史 10%+
- 35B: 典型 logit 差 ~0.5-2.0 → RM delta 0.7 较难 flip → flip 率 3%

即便 35B "entropy 不那么低" 时, 它的 top-5 内部 logit 差距仍比 14B 大 (类似温度更低的情况), RM delta 不够推翻 top-1。

### 5.4 结论

**8% 干预率 + 3.45% flip 主要是"35B + thinking 模型"的客观属性**, 不是 bug:
- **主因 ①**: 35B 模型 top-5 分布比 14B 更尖锐, entropy ≥ 1.0 的 step 占比天然低
- **主因 ②**: thinking 模型大段低熵推理 prose 拖低整体干预率
- **次因 ③**: MoE 输出比 dense 更尖锐
- ④/⑤/⑥ 都不是主因

---

## 6. 准确率 -1.2pp 的可能原因

不能简单说"SIA 不工作", 因为有 13 个 subject 提升 (最大 +11.7pp)。可能的解释:

### 6.1 RM 训练分布 mismatch (主因)

VM-Qwen3-4B-RM **训练于 Qwen3-4B base 的 chat 完成场景**, 不是 thinking 模式。0GM 的 `<think>...</think>` 内心独白是 RM 训练时**没见过的分布**, RM 给出的 reward 在 thinking 阶段本质是 OOD 信号 — 不能保证"奖励正确推理"而是"奖励看起来 helpful 的 phrasing"。

具体表现 (见 astronomy -18.3pp 分析):
- noSIA 在 Q1/Q13/Q17 上 thinking 较长 (1644-1929 tok), 充分对比选项后正确选 B
- SIA 在同样问题上**自信地早收敛** (962-1364 tok), 选错 A
- 共同模式: SIA 的 thinking 里出现 "definitively chose", "physically reasonable", "this matches" 等**自信措辞**, 显示 RM 在奖励"看起来 helpful / 自信" 的 phrasing token

### 6.2 subject 表现分裂的可能解释

- **SIA 表现好的 subject** (geography, us_history, professional_accounting, formal_logic): 答案通常**事实清晰、推理短**, RM 奖励"清晰 phrasing" 跟正确性一致
- **SIA 表现差的 subject** (electrical_engineering, astronomy, math, medicine): 答案需要**多 candidate 仔细比较**, RM 奖励"早收敛 / 自信 phrasing" 反而打断了充分推理

### 6.3 干预率 8% 可能太低

如果 0GM 上真正的"junction"被 threshold=1.0 漏掉了, SIA 实际干预的可能不是关键决策点, 而是一些 marginal 的高熵噪声 step。这会让干预**没有 alignment 收益**, 同时**仍然增加成本**。

---

## 7. 行动建议

按优先级:

1. **Threshold tuning experiment (低成本, ~30 分钟)**:
   - 跑 `entropy_threshold=0.5` 一个 small subset (60Q × 3 subjects)
   - 看干预率能否升到 ~25-30%, accuracy 怎么变
   - 如果干预率升上来 accuracy **改善** → threshold 是问题, 调成 0.5-0.7 跑完整 600Q
   - 如果干预率升上来 accuracy **更差** → RM 信号本身不适配 0GM, 该考虑训新 RM

2. **训一个适配 0GM 的 RM (大工作量)**:
   - 用 0GM 自己生成的 chat (含 `<think>` 段落) 加偏好标签
   - 训一个新的 VM-0GM-7B 或类似
   - 让 RM 训练分布跟 0GM inference 分布对齐

3. **加速 RM 调用 (条件性)**:
   - 只在 #1 或 #2 验证了"SIA 真有 alignment 收益"之后再做
   - 若效果确认, 走 [`0gm-35b-sia-rm-inproc-path-20260602.md`](0gm-35b-sia-rm-inproc-path-20260602.md) 中的方案 A (transformers DynamicCache inproc), 预期 SIA 速度从 51 → ~78-85 tok/s

**优先级理由**: 当前数据下 SIA 在 0GM 上看不到 alignment 收益, 优化速度是 premature optimization。先解决"是否有收益"再解决"如何更快"。
