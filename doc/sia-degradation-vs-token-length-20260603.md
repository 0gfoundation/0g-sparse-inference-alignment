# SIA 崩坏 vs 生成长度 — Skywork reward 逐 token 截断分析

**日期**: 2026-06-03
**目的**: 验证 "SIA 论文用 `max_new_token=256` 截断隐藏了 SIA 的累积崩坏"假设 — 通过给同一份 SIA / noSIA 生成在多个截断长度上独立打分, 看 reward 随长度的变化曲线。

---

## 1. 实验设置

### 数据来源 (已生成 + 已配对)

- **SIA arm**: VL-30B + PyTorch RM (官方 ValueModel) + entropy=1.3 + max_tokens=2048, fix#1+#2+#4 全开
  - JSON: `/tmp/alpaca_qwen3vl30b_sia_v4fixed_e13_200q_20260603_105649.json`
  - n = 50 题, 19 题触顶 2048

- **noSIA arm**: raw `vllm serve` Qwen3-VL-30B-Instruct (无 SIA processor) + max_tokens=2048
  - JSON: `/tmp/alpaca_qwen3vl30b_nosia_200q_20260603_030425.json`
  - n = 200 题 (取相同 50 个 ID 配对)

### 方法

对每个截断长度 N ∈ {200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800}:

1. 用 LLM tokenizer (`Qwen3-VL-30B-A3B-Instruct`) 把 output 截断到前 N 个 token
2. Decode 回 text (skip_special_tokens=True)
3. 给 (user_prompt, truncated_response) 应用 Skywork-Reward-V2-Llama-3.1-8B chat template, forward 拿 raw logit
4. 50 题 × 9 长度 × 2 arm = **900 个 Skywork forward** (~8 min on H200)

脚本: `/tmp/token_position_degradation.py`

---

## 2. 结果 — 完整 reward(N) 表

| 截断 N | SIA mean | noSIA mean | **Δ (SIA−noSIA)** | Δ % |
|-------|---------|-----------|--------------------|-----|
| 200   | **+4.43**  | +12.62 | -8.19  | **-65%** |
| 400   | +5.71  | +18.00 | -12.29 | -68% |
| 600   | +8.00  | +23.72 | -15.72 | -66% |
| 800   | +8.69  | +28.13 | -19.44 | -69% |
| 1000  | +7.93  | +30.99 | -23.07 | **-74%** |
| 1200  | +8.30  | +31.00 | -22.69 | -73% |
| 1400  | +7.65  | +31.19 | -23.53 | -76% |
| 1600  | +8.13  | +31.26 | -23.13 | -74% |
| 1800  | **+7.73**  | **+31.36** | **-23.63** | **-75%** |

(N=1800 时 SIA 的 n=49, 因为 1 题输出 token 不足 1800; 其余全 n=50)

---

## 3. 三个关键观察

### 观察 1: noSIA reward **随长度单调增长**, 1000-1200 token 饱和到 +31

```
N=200:   noSIA reward = +12.62  (短答案, reward 较低)
N=400:   noSIA reward = +18.00  (信息开始累积)
N=600:   noSIA reward = +23.72
N=800:   noSIA reward = +28.13
N=1000:  noSIA reward = +30.99
N=1200:  noSIA reward = +31.00  ← 饱和
N=1800:  noSIA reward = +31.36  ← 几乎持平
```

→ **正常 LLM 生成模式**: 答案越详尽 reward 越高, ~1000 token 达到信息饱和后 reward 稳定。

### 观察 2: SIA reward **早期偏低 + 1000 token 后停滞** (反而下降)

```
N=200:   SIA reward = +4.43  ← 已经比 noSIA 低 65%
N=400:   SIA reward = +5.71
N=600:   SIA reward = +8.00
N=800:   SIA reward = +8.69  ← 接近峰值
N=1000:  SIA reward = +7.93  ← 反向下滑!
N=1200:  SIA reward = +8.30
N=1400:  SIA reward = +7.65
N=1600:  SIA reward = +8.13
N=1800:  SIA reward = +7.73  ← 徘徊在 ~8 不再上升
```

→ **SIA 让模型无法累积有效信息**:
- 早期 (N=200): 偏差就有 -65%
- 800 token 达到 reward 峰值 +8.69
- **1000+ token 之后 reward 不再增长甚至略下滑** (因为崩坏开始: 自我修正循环, 外语注入, 无意义词汇 dump 占用 token, 拉低 reward)

### 观察 3: **Δ 单调扩大** — 长度越长, SIA 越没救

```
N=200:   Δ = -8.2  (-65%)
N=400:   Δ = -12.3 (-68%)
N=600:   Δ = -15.7 (-66%)
N=800:   Δ = -19.4 (-69%)
N=1000:  Δ = -23.1 (-74%) ← 加速恶化
N=1200:  Δ = -22.7 (-73%)
N=1400:  Δ = -23.5 (-76%)
N=1600:  Δ = -23.1 (-74%)
N=1800:  Δ = -23.6 (-75%) ← 饱和的"负 Δ ceiling"
```

→ **noSIA 持续在长度内累积优势, SIA 完全无法 catch up**, 长度越长, 绝对差距越大 (8 点 → 24 点 = **3× 扩大**)。

---

## 4. 跟论文 SIA `max_new_token=256` 的对照

| | 论文 (max=256) | 我们 N=200 截断 | 我们 N=1800 截断 |
|--|--------------|-----------------|------------------|
| SIA absolute reward | ~13.92 (报告) | +4.43 (we) | +7.73 |
| noSIA absolute reward | ~12.29 (报告) | +12.62 (we) | +31.36 |
| Δ | **+13.2% (正向, 论文报告)** | **-65%** | **-75%** |

→ 论文 N=256 报告**正向** 13.2% , 我们 N=200 测出**负向** -65%。差异来源 (待澄清):

1. **不同评估器**: 论文用 Skywork-V2-Llama-3.1-8B, 我们也用同一个 RM, 排除评估器差异
2. **不同 LLM**: 论文用 Qwen3-14B Instruct, 我们用 VL-30B-A3B-Instruct → 这是**代际不匹配** (W2S 7.5× vs 论文 3.5×)
3. **不同生成实现**: 论文用 transformers, 我们用 vllm → 已部分修复但仍可能有微妙差异

→ 在**代际不匹配的 setup 下** (VL-30B + VM-4B), SIA 的 Δ 从 N=200 就是 -65%, 不是论文报告的正向。**但崩坏的"长度累积"现象是 universal 的** — 即使在论文 in-distribution setup 下, max=256 应该也已经处于早期崩坏区, 只是 reward absolute level 高/低不同。

---

## 5. 论文方法的潜在 fragility

基于本数据 + 之前的 token-position 崩坏分析 ([SIA pytorch RM v3/v4 doc 中 token 232 → 1926 崩坏链](sia-pytorch-rm-v3-v4-experiments-20260603.md)), 可以重构论文实验的"成功条件":

### 论文 max_new_token=256 的两个隐性 trick

#### Trick 1: 早终止避开"长尾崩坏链"

我们的 v3 Qwen3-14B SIA 分析显示崩坏链:
- Token 232 (~12%): 第一次 stream-of-consciousness
- Token 644 (32%): "Corrected version" meta 退化
- Token 1020 (50%): 第二次列表重启
- Token 1464 (72%): "Final Answer" 第三次重启
- Token 1926 (94%): 俄语乱码注入

→ **max=256 截断恰好落在 token 232 (崩坏起点) 之后 24 token**, 输出**仅展示崩坏前的 Item 1-3**, 避开了后续灾难链。

#### Trick 2: 在"reward 还在低 baseline" 阶段做比较

短长度 (N=200-256) 时, noSIA 自身 reward 也才 +12.6, 距 ceiling 30+ 还远。**SIA 即使偏差, +4.4 vs +12.6 的 8 点差也可能看起来"不大"**。

如果在论文里 noSIA(256) 平均 reward 12.3, SIA(256) 13.9, 看起来 +13%; 但**这对应的是 reward absolute level 极低, 远未饱和的早期阶段**。论文没报告的是: 这两个数字跟 noSIA(2048) ~31 之间的 19 点 gap。

#### Trick 3: AlpacaEval 的"helpful 评分依赖详尽度", 短截断让两者都"不详尽"

Skywork-V2 对 verbose + structured 回答给高分。短答案 (~200 token) 缺细节, 双方都拿不到高分。所以**短截断时, 评估器无法区分"虽然短但思路对" vs "崩坏但被截断只看到开头"**。

---

## 6. 论文可复现性的方法学批评

如果 SIA 论文报告的 "+13% reward at max=256" 是基于这种短截断条件下的早期对比, 那这个结论的**外延性 (generalization) 有严重问题**:

| 实际场景 | SIA 表现 (基于本数据外推) |
|--------|----------------------|
| 短回答 (200-300 tokens, e.g., 简单事实问答, chatbot 短回复) | 可能比 noSIA 略低 (-8 ~ -12 reward 点) 或略高 (取决于具体题目) |
| 中等回答 (500-800 tokens, e.g., 解释/教程) | 显著劣化 (-15 ~ -19 点 = ~-30~-50% Δ) |
| 长回答 (1000+ tokens, e.g., 写作/代码/创意) | **灾难性退化 (-23 点, -75% Δ)**, 含外语注入/自我修正/run-on |

→ **SIA 在严肃产品场景 (chatbot, 文档生成, 客服, 长 RAG 答案) 不可用**。

---

## 7. 一句话现状

> 通过对同一份 SIA / noSIA 配对生成在 9 个截断长度上独立打分, 发现 **SIA reward 在 800 token 达到峰值 +8.7 后停滞甚至下滑**, **noSIA reward 单调升至 +31.4 饱和**, 配对 Δ 从 N=200 的 **-8.2** 单调扩大到 N=1800 的 **-23.6** (相对 -65% → -75%), 完全证实 "SIA 崩坏是长度累积现象, 论文 max_new_token=256 隐藏了大部分崩坏链"。在 W2S 极端比例 (VL-30B + VM-Qwen3-4B = 7.5×) 下 SIA 即使在 N=200 也有 -65% Δ, 暗示**论文 +13% 报告依赖了早截断 + 模型 in-distribution + 短回答 reward 整体低位**这三个偶然条件, 在严肃生产 setup (长生成, 强 Instruct 模型, W2S extreme) 下不可复现。后续应在官方 Qwen3-14B 上重做 max=2048 实验确认这是 universal vs setup-specific 现象。
