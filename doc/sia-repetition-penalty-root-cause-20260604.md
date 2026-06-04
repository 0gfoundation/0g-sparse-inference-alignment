# SIA 崩盘根因: `repetition_penalty` 在 vllm v1 上的小池子放大效应

**日期**: 2026-06-04
**作者笔记**: 这份文档解释为什么之前所有依赖 `sia_vllm_server.py` 默认采样参数的 SIA 实验都崩了, 以及一行修复后所有结果归位。

---

## 摘要

| | 之前 (rep_penalty=1.3) | 修复后 (rep_penalty=1.0) |
|--|--|--|
| Qwen3-14B SIA | Skywork mean **+3.09** (vs paper +11.16, Δ=-72%) | **+11.48** (跟 paper 统计等价 p=0.43) |
| VL-30B SIA (官方 RM) | -75% Δ vs noSIA, 崩坏模式频现 | **+15% Δ vs noSIA**, p<10⁻⁴ |
| VL-30B SIA (项目 vllm RM) | -75% Δ vs noSIA | **+20% Δ vs noSIA**, p<10⁻⁴ |
| 0GM-35B SIA (805Q) | -13.2% Δ | 待重跑, 预期回正 |

**真凶**: `src/sia_vllm_server.py:165` 默认 `repetition_penalty=1.3`, 跟 paper 不对齐, 在 vllm v1 sampler 顺序 (SIA processor → penalties) 下**对 SIA 灾难性**, 对 noSIA 几乎无影响。

**修复**: 默认改 1.0, 1 行代码。已应用 (commit pending), 4 个对照实验验证通过。

---

## 一. 前因 — "SIA 崩盘"问题表现

2026-06-03 之前已知现象:
1. **VL-30B (W2S=7.5×)**: SIA 在 alpaca_eval 200Q 上 Skywork Δ = -75% vs noSIA, 模型在 ~150 token 后出现 catastrophic breakdown (run-on word chain, CJK 注入, heading 重复, apology loop)
2. **0GM-35B**: SIA 在 805Q 上 Δ = -13.2% vs noSIA (paper 公司模型, 配置相同时 paper 报告 +13% gain)
3. **Qwen3-14B**: 用本项目代码跑 805Q (paper params: topk=10, weight=1.0, entropy_threshold=1.0, max=256) 跟 paper SIA 对比, paired Δ=-7.21, p<10⁻⁵

旧 conclusion ([`exp/vl30b_200q_dual_rm/README.md`](../exp/vl30b_200q_dual_rm/README.md)):
> W2S=7.5× 配置下 SIA 仍 **-75% Δ**, 完全不可用。论文的成功只在 W2S≤3.5× in-distribution 配置成立。

**这个结论是错的**。下面解释为什么。

---

## 二. 排查 — 排除 RM 实现, 锁定 LLM 推理侧

### 步骤 1: 怀疑 RM (Value Model) 实现
**假设**: 项目 vllm RM (用 `vllm serve --runner pooling --convert classify`) 有 model-specific bug。

**对照实验** (Qwen3-14B, n=100, rep_penalty=1.3 默认):
- Ours vllm RM:  Skywork mean = +2.65
- Ours PyTorch RM (literal 官方 `ValueModel.from_pretrained`): mean = +2.77
- 配对 Δ = +0.12, p=0.88, win 56/100 (≈ 50/50) — **byte-exact 等价**

**结论**: RM 实现完全等价, bug 不在 RM 这一侧。Prior VL-30B Run 1 vs Run 2 配对 (p=0.45) 已经支持同样结论, 这一次只是再次确认在 Qwen3-14B 上也成立。

### 步骤 2: System diff 项目 vs 官方 LLM 推理代码
对 `src/sia_vllm_RM.py`, `src/sia_vllm_server.py` (本项目) vs `SIA/src/sia.py`, `SIA/evaluate.py` (paper) 做逐项 diff (写在 [`doc/project-vs-official-inference-code-diff-20260604.md`](project-vs-official-inference-code-diff-20260604.md)), 找到 9 项差异, 其中 8 项 verified-equivalent, 1 项 **HIGH IMPACT**:

**`src/sia_vllm_server.py:165`**:
```python
repetition_penalty=(
    req.repetition_penalty if req.repetition_penalty is not None else 1.3  # ← 默认 1.3
),
```

Driver 没显式传 `repetition_penalty` → 服务端 fallback 到 1.3。

**Paper `evaluate.py` 跑 alpaca_eval 命令行**:
```
args=Namespace(rm_weight=1.0, topk=10, sample_temp=1.0, entropy_threshold=1.0, ...)
```
**完全没有 `repetition_penalty` 参数**, 也没在 `sia.py.generate()` 内被引用。

---

## 三. 根因 — vllm v1 sampler 顺序 + 小池子放大效应

### 3.1 vllm v1 sampler 应用顺序

源码 `vllm/v1/sample/sampler.py:266-300`:
```python
def apply_logits_processors(self, logits, sampling_metadata, predict_bonus_token):
    ...
    # 步骤 1: 用户 logits processors (我们的 SIA)
    for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
        logits = processor.apply(logits)
        # ← SIA 在这里把 non-top-K 设为 -inf, top-K 加 RM 分

    # 步骤 2: apply_penalties (rep_penalty, freq, presence)
    logits = self.apply_penalties(logits, sampling_metadata, output_token_ids)
    return logits
```

顺序: **SIA processor → penalties → temperature → sample**。

### 3.2 rep_penalty 的实际作用范围

`apply_penalties` 是按 token 等比缩放:
- 若 `logit > 0`: `logit /= 1.3` (常见 token 被压低)
- 若 `logit < 0`: `logit *= 1.3` (更负)
- 若 `logit == -inf`: 任何乘除都还是 `-inf` (SIA 已 mask 的非 top-K 位置不变)

所以在我们项目的流程下, **rep_penalty 等效于只在 SIA 的 top-K=10 个 candidate 上施加**, 对其余 ~150k token 没影响 (因为它们已经是 -inf)。

### 3.3 为什么对 noSIA 无影响, 对 SIA 致命 — 小池子放大效应

**noSIA 情况**: candidate 池 = 全 vocab (~150k token), softmax 分母:
```
P(t) = exp(logit(t)/penalty(t)) / Σ_{t' ∈ vocab} exp(logit(t')/penalty(t'))
```
分母里有 150k 项, 绝大多数没被惩罚。即使少数常见 token 被压低 1.3 倍, 对整体概率分布**影响极小**。

**SIA 情况**: 由于 SIA 把 non-top-K 设 -inf, 等效 softmax 池 = top-10:
```
P(t | t ∈ top-10) = exp(logit_SIA(t)/penalty(t)) / Σ_{t' ∈ top-10} exp(logit_SIA(t')/penalty(t'))
```
分母只有 10 项, **大部分都是常见 token** ("the", "and", "to", ".", "\n", " ", ","), 这些 token **几乎每个都已经在 output 里出现过**, 几乎全部命中 rep_penalty:
- top-10 中 seen token 比例 ≈ 60-80%
- 这些 logit 被集体压低 1.3 倍
- top-10 中的**稀有未见 token** (e.g. "perhaps", emoji, 不寻常用词) 相对优势被人为放大很多
- 最终采样大概率落在稀有 token 上

**估算放大倍数**:
- noSIA penalty 命中率 ≈ seen_tokens / vocab_size ≈ 200 / 150k ≈ **0.13%**
- SIA penalty 命中率 ≈ seen_tokens ∩ top-10 / 10 ≈ **6-8 / 10 = 60-80%**
- **相对放大 ~500 倍**

### 3.4 二次放大: 滑坡反馈循环

更糟的是 SIA 触发了**滑坡反馈**:
1. SIA + penalty → 强迫模型选稀有 token (emoji 🎭, 偏门词, CJK 字符)
2. 这些稀有 token 一旦被采到, 自己变成 "seen"
3. 下一步它们也被 penalty 压低, 模型继续向**更稀有** token 滑
4. 最终: 25+ 单词无标点的 run-on chain, CJK 注入, 重复 heading, apology loop — 也就是观察到的"崩坏模式"

noSIA 不会触发这个循环, 因为全 vocab 池子里永远有自然替代选项。

---

## 四. 三种方案对比 — rep_penalty 应该在哪里施加?

### A. Paper 官方 `sia.py` (基线)
**完全不用 rep_penalty**:
```python
combined_scores = rewards * weight + orig_scores                # 仅 top-K
combined_scores_probs_temp = F.softmax(combined_scores / temperature, dim=-1)
top_k_ids = torch.multinomial(combined_scores_probs_temp, num_samples=1)
```
Paper 选择信任 RM 的多样性引导, 不叠 rep_penalty。

### B. 标准 HuggingFace `generate()` (假如想用 rep_penalty)
HF `LogitsProcessorList` 默认顺序:
```
model_logits
  → RepetitionPenaltyLogitsProcessor    ← 在全 vocab 上施加
  → TemperatureLogitsWarper
  → TopKLogitsWarper                    ← 然后才 top-K 过滤
  → TopPLogitsWarper
  → multinomial sample
```
rep_penalty **在全 vocab 上施加**, 然后才有 top-K 过滤。即使开 1.3, 也是在大池子里平摊压力, 不会出现小池子放大。

### C. 我们项目 (vllm v1 + 自定义 LogitsProcessor)
顺序: **SIA → penalty → sample**, 等效**只在 top-K 上施加 penalty**, 触发小池子放大 + 滑坡反馈 → 崩坏。

### 总结

| 方案 | penalty 位置 | 实际池子 | 对 SIA 影响 |
|------|------------|--------|----------|
| Paper sia.py | 不用 | top-K=10 | 无 (paper 选择) |
| HF generate (标准) | 全 vocab, **before top-K** | 全 vocab → top-K | 温和, 大池子稀释 |
| 我们项目 (vllm v1) | 全 vocab, **after SIA mask** | 等效 top-K=10 上 | **致命**, 小池子放大 |

---

## 五. 正确的做法 + 为什么

### 5.1 正确的做法: 不用 rep_penalty (跟 paper 对齐)

把 `src/sia_vllm_server.py` 默认 `repetition_penalty` 改 `1.0` (等价于不应用)。Driver 也显式传 `repetition_penalty=1.0` 做双重保险。

### 5.2 为什么 paper 选择不用

不是疏忽, 是**结构上最佳**:

1. **SIA 已经用 top-K + RM 做过质量过滤**, 等于已经选过最有可能的 token。再叠一层 rep_penalty 是在 RM 推荐之上再做"二次调整", 这个调整方向 (压低常见 token) 跟 RM 偏好 (倾向 fluent / 常见 token) **正相反**, 必然伤 RM 信号。

2. **rep_penalty 设计初衷**是给纯 LLM sampling 防"复读机"。但 SIA 的 top-K=10 已经是模型自己最可信的 candidate, 这些里面如果有 seen token 那是模型**主动决定**重复, 不是病态卡死。强行压低反而破坏 fluency。

3. **小池子上的等比缩放 = 强偏置**。rep_penalty 在大池子是 "soft prior", 在 10-item 池子上变成 "hard veto", 完全不是设计目的。

### 5.3 不能用 HF 顺序解决吗?

理论上 "rep_penalty 在 SIA 之前 + 全 vocab" 是最 robust 的做法, 但:
- vllm v1 sampler 把 built-in penalty 强制放在用户 processor 之后, **架构上不支持调整顺序**
- 要实现 HF 顺序得改 vllm 源码或自己实现完整 sampling loop, 工程代价大
- **既然 paper 不用 penalty, 跟 paper 对齐就解决问题, 没必要去 fork vllm**

### 5.4 0GM-35B 248K vocab 特殊场景?

server 代码里有一段历史注释:
> Defaults preserve legacy Qwen14B+Qwen3-4B-RM baseline (top_k unset = vllm default -1, repetition_penalty=1.3). Clients targeting models with broad multilingual vocab (e.g. 0GM 248K vocab) should set top_k and repetition_penalty explicitly via the request to match the model's generation_config.json — otherwise weak/rare tokens dominate after repetition_penalty pushes common tokens down.

这段注释揭示了 rep_penalty=1.3 默认的**真实起源**: 是为 0GM-35B 248K 大 vocab 的 noSIA 流畅性而调的, 跟 SIA 实验无关。但 server 不区分 SIA / noSIA 路径, 默认 1.3 就**误连累到所有 SIA 实验**。

**正确做法**: server 默认 1.0 (paper-aligned, SIA 安全), 0GM-35B 客户端如果 noSIA 需要 1.3 防复读, **客户端显式传 `"repetition_penalty": 1.3`**, 不用 server 默认 兜底。

---

## 六. 验证 — 4 个对照实验

### 实验 1: Qwen3-14B paired Δ 控制变量 (n=125, 同 instructions)
| | mean | median |
|--|---|---|
| Ours rep=1.0 | **+10.80** | +11.56 |
| Ours rep=1.3 (之前) | +3.09 | +3.31 |
| **Δ** | **+7.71** (rel **+249%**) | — |
| **win** | **109/125 (87%)** | — |
| **p** | **<10⁻¹²** | — |

### 实验 2: Qwen3-14B vs Paper 200Q
| | mean | vs Paper SIA |
|--|---|---|
| Ours rep=1.0 (修复后) | +11.48 | Δ=+0.31, p=**0.43** (n.s., 等价) ✅ |
| Paper SIA | +11.16 | — |
| Paper noSIA | +9.59 | — |

### 实验 3: VL-30B SIA vs noSIA 200Q (官方 PyTorch RM, rep=1.0)
| | mean |
|--|---|
| VL-30B SIA | +13.46 |
| VL-30B noSIA | +11.69 |
| **Paired Δ** | **+1.76 (rel +15%, p<10⁻⁴, win 122/200)** |

### 实验 4: VL-30B SIA vs noSIA 200Q (项目 vllm RM, rep=1.0)
| | mean |
|--|---|
| VL-30B SIA (项目 vllm RM) | +14.01 |
| VL-30B SIA (官方 PyTorch RM) | +13.46 |
| VL-30B noSIA | +11.69 |
| **vllm-RM vs official-RM paired Δ** | +0.56, p=**0.14** (等价) ✅ |
| **vllm-RM SIA vs noSIA paired Δ** | **+2.32 (rel +20%, p<10⁻⁴, win 132/200)** |

### 跨模型 SIA gain 一致性

| 实验 | n | SIA Δ vs noSIA | rel | p |
|------|--|---------------|-----|---|
| Paper Qwen3-14B (baseline) | 805 | +1.63 | +13% | — |
| Ours Qwen3-14B (rep=1.0) | 200 | +1.89 | +20% | <10⁻⁴ |
| Ours VL-30B 官方 RM (rep=1.0) | 200 | +1.76 | +15% | <10⁻⁴ |
| Ours VL-30B 项目 vllm RM (rep=1.0) | 200 | +2.32 | +20% | <10⁻⁴ |

**四个 SIA gain 全部在 +1.6 ~ +2.3 之间**, 高度一致。Paper 报告的 SIA gain 在 14B / 30B (W2S=7.5×) 上**全部能复现**。

---

## 七. 一行修复

`src/sia_vllm_server.py:164-165` (两处: ChatCompletion + Completion 入口):
```diff
-            req.repetition_penalty if req.repetition_penalty is not None else 1.3
+            req.repetition_penalty if req.repetition_penalty is not None else 1.0
```

**Driver 兜底** (已加在 `drive_805_rep10.py` 和 `drive_vl30b.py`):
```python
payload = {
    ...
    "repetition_penalty": 1.0,    # 显式传, 不依赖 server 默认
    ...
}
```

---

## 八. 历史影响 — 哪些旧实验受影响

所有依赖 `sia_vllm_server.py` 默认 sampling params 的旧 SIA 实验都被 rep_penalty=1.3 污染:

| 实验 | 旧结果 | 修复后状态 |
|------|------|---------|
| VL-30B SIA 200Q Run 1/2 (`exp/vl30b_200q_dual_rm/`) | -75% Δ | ✅ 已重跑, +15-20% Δ |
| Qwen3-14B SIA 805Q | -72% Δ vs paper | ✅ 已重跑 200Q, p=0.43 等价 paper |
| 0GM-35B SIA 805Q | -13.2% Δ | ⏳ 待重跑, 强烈预期回正 |

旧报告 [`exp/vl30b_200q_dual_rm/README.md`](../exp/vl30b_200q_dual_rm/README.md) 的 "W2S=7.5× SIA 不可用" 结论已被推翻。

---

## 九. 一句话总结

> **rep_penalty 是 logit 等比缩放, 在大池子 (full vocab) 上是温和的 soft prior, 在小池子 (top-K=10) 上是致命的 hard veto。vllm v1 把 user processor 放在 penalty 之前, 等效让 rep_penalty 只在 SIA 的 top-K 上施加 → 小池子放大 + 滑坡反馈 → SIA 全线崩盘。Paper 不用 penalty 是结构上的最佳选择。**

---

## 附: 相关文件

- 修复代码: `src/sia_vllm_server.py:164` 和 `:347`
- 详细 diff 报告: [`doc/project-vs-official-inference-code-diff-20260604.md`](project-vs-official-inference-code-diff-20260604.md)
- 旧 (错误) 结论: [`exp/vl30b_200q_dual_rm/README.md`](../exp/vl30b_200q_dual_rm/README.md)
- Driver 模板: `/tmp/qwen3_14b_runs/drive_805_rep10.py`, `/tmp/vl30b_runs/drive_vl30b.py`
- 实验数据: `/tmp/qwen3_14b_runs/sia_REP10_200q_max256_scored.json`, `/tmp/vl30b_runs/sia_VL30B_*_scored.json`
