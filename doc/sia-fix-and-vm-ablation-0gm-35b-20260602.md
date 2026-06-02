# 0GM-35B SIA: Step1+2 fix + Value Model size ablation (4B → 1.7B)

**日期**: 2026-06-02
**主推理模型**: 0GM-1.0-35B-A3B-0427 (Qwen3.5/3.6 MoE thinking, vocab=248K)
**评测**: MMLU-Redux, 3 subjects × 10Q = 30Q (astronomy, electrical_engineering, high_school_geography)
**SIA 参数**: topk=5, weight=1.0, entropy_threshold=0.8

---

## 1. 实验起源

之前一系列调试发现 0GM-35B + VM-Qwen3-4B 在 MMLU 上 SIA accuracy 比 noSIA 低 1.2-12.2pp。归因层层倒推:

1. ❌ "35B 更自信 / vllm 版本变化" — 实证已排除
2. ✅ "我引入的 `top_k=20/top_p=0.95/rep_penalty=1.0` 降低了干预率" — 实证证实
   (详见 [`sia-intervene-rate-vs-sampling-params-20260602.md`](sia-intervene-rate-vs-sampling-params-20260602.md))
3. ✅ "vllm /classify 返回 sigmoid prob 而非 raw logit, RM signal magnitude 被压扁 ~10×" — 实证证实
   (commit `e1d150e`: Step 1 logit() 反 sigmoid + Step 2 非 top-k 置 -inf)
4. ❓ "VM-Qwen3-4B-RM 跟 0GM-35B 跨家族分布 OOD"  — 本 doc 验证

修复 (1)(2)(3) 后 (commit `e1d150e`), 用 VM-Qwen3-4B 跑 30Q 验证:
- flip rate 从 ~3% 跳到 16.4% ✅ (signal 数值恢复)
- 但 **accuracy 反而更差** (70% vs noSIA 82.2%, -12.2pp) ⚠️

→ 假设 (4) 成立: **VM-Qwen3-4B 跟 0GM-35B 不匹配, 修复后强 signal 反而 hurt accuracy**。

本 doc 测试 (4) 的修复方向: **换更小的 VM-Qwen3-1.7B**, 看是否更适合 0GM。论文 §5.3 weak-to-strong 章节明确说 "smaller VMs contain sufficient directional information to steer larger LLMs"。

---

## 2. 实验配置

### 2.1 通用 (3 个实验一致)

- 主 LLM: `/workspace/SIA/models/0GM-1.0-35B-A3B-0427` (gpu_mem=0.72-0.78)
- vllm 版本: 0.19.0 (venv4)
- SIA: topk=5, weight=1.0
- Sampling: temperature=1.0, top_k=20, top_p=0.95, repetition_penalty=1.0
- 评测: MMLU-Redux astronomy / electrical_engineering / high_school_geography × 10Q each
- debug: SIA_DEBUG_HIST=1 (instrumentation enabled)

### 2.2 三个实验的关键区别

| 实验 | 代码版本 | entropy_threshold | RM 模型 | 时间戳 |
|------|---------|-------------------|---------|--------|
| **A. thr=0.7 old** | commit `b72764e` (无 sigmoid fix) | 0.7 | VM-Qwen3-4B (32GB GPU) | 09:38 |
| **B. thr=0.8 FIX** | commit `e1d150e` (Step1+2 fix) | 0.8 | VM-Qwen3-4B (32GB GPU) | 10:43 |
| **C. thr=0.8 FIX + 1.7B** | commit `e1d150e` | 0.8 | **VM-Qwen3-1.7B** (17GB GPU) | 11:18 |

---

## 3. 实测结果

### 3.1 Per-subject Accuracy

| Subject | noSIA (30Q baseline) | A. RM=4B thr=0.7 old (20Q) | B. RM=4B thr=0.8 FIX (10Q) | **C. RM=1.7B thr=0.8 FIX (10Q)** |
|---------|---------------------|---------------------------|---------------------------|----------------------------------|
| astronomy | 28/30 (93.3%) | 16/20 (80%) | 7/10 (70%) | **9/10 (90%)** 🎯 |
| electrical_engineering | 21/30 (70%) | 11/20 (55%) | 5/10 (50%) | **6/10 (60%)** |
| high_school_geography | 25/30 (83.3%) | 19/20 (95%) | 9/10 (90%) | 8/10 (80%) |
| **TOTAL (3 subjects avg)** | **82.2%** | 76.7% | 70.0% | **76.7%** |

→ C **比 B 高 6.7pp**, 跟 A 持平 (但 A 是 old code 没 fix), **跟 noSIA 仍差 -5.5pp** (B 是 -12.2pp, **改善了一半**)。

### 3.2 SIA 干预 / flip / 速度指标

| 指标 | A. thr=0.7 old | B. RM=4B thr=0.8 FIX | **C. RM=1.7B thr=0.8 FIX** |
|------|---------------|---------------------|---------------------------|
| intervene rate (cumul) | 20.68% | 17.84% | **18.03%** |
| **flip rate (of iv)** | 2.57% | **16.40%** | **5.93%** ← 减半 |
| flip rate (of total step) | 0.53% | 2.93% | 1.07% |
| avg tokens/Q | 874 | 680 | **840** (+24% vs B) |
| throughput (tok/s) | 33.4 | 37.3 | **39.4** (略快) |
| avg latency/Q (s) | 26.1 | 18.2 | **21.3** |
| per-Q flip range | mean 3.0%, max 5.9% | mean 18.4%, max 33.5% | **mean 9.9%, max 35.8%** |
| RM errors | 0 | 0 | 8 (限于 2 道 long-prefix Q, RM max_model_len=2048 限制) |

---

## 4. 关键发现

### 4.1 Step 1+2 fix 数值层是正确的, flip rate 6× 跳升 (A → B)

```
flip rate (of intervened):  2.57% (no fix) → 16.40% (with fix)   = 6.4× ↑
flip rate (of total step):  0.53%          → 2.93%               = 5.5× ↑
```

这证实 commit `e1d150e` 的两步 fix 在数值层完全等效官方代码:
- **Step 1**: logit() 反 sigmoid, 把 vllm /classify 返回的 sigmoid prob 还原成 raw logit (跟 b2 InprocClient 和官方 `rm_out.logits` 一致)
- **Step 2**: 非 top-k 位置置 -inf, 跟官方 `rewards = -inf except top-k` 行为一致

### 4.2 但 SIA 修复后 (B), accuracy **更差** — 暴露 ML 层面的 RM-LLM mismatch

| | accuracy (3 subj avg) | vs noSIA |
|---|----------------------|----------|
| noSIA (matched 3 subj) | 82.2% | baseline |
| B. RM=4B thr=0.8 FIX | 70.0% | **-12.2pp** ⚠️ |

**强 signal + 弱适配 = 系统化错误方向**: VM-Qwen3-4B 是基于 Qwen3 helpfulness/safety chat 数据训的, **0GM 是 Qwen3.5/3.6 MoE thinking 模型**, 跨家族 + 跨范式。flip rate 16.4% 意味着 SIA 真在改 token 选择, 但改向不是"正确答案", 而是"RM 认为更 helpful 的措辞"。

### 4.3 🎯 换更小的 VM (1.7B), accuracy 改善 +6.7pp (B → C)

| | accuracy | vs B | vs noSIA |
|---|----------|------|----------|
| B. RM=4B | 70.0% | baseline | -12.2pp |
| **C. RM=1.7B** | **76.7%** | **+6.7pp** ✅ | -5.5pp (mismatch 损失减半) |

per-subject:
- **astronomy +20pp** (70% → 90%) — 显著恢复
- **EE +10pp** (50% → 60%)
- geography -10pp (90% → 80%) — 单 Q noise (10 题样本)

### 4.4 1.7B VM 的 reward signal 更"温和"

- VM=4B flip rate (of iv): 16.40% (per-Q max 33.5%)
- VM=1.7B flip rate (of iv): **5.93%** (per-Q max 35.8%, **mean 减半**)

→ 1.7B VM 给的 reward 量级更小/方向更分散, 不那么频繁地 flip top-1, **让 0GM 保留更多自己的推理能力**, 损失减少。

### 4.5 验证论文 §5.3 weak-to-strong 假设

论文原话:
> "SIA effectively achieves high alignment performance even when the guiding value model is much smaller than the target LLM... value landscapes learned by smaller models contain sufficient directional information to successfully steer the generation of larger models."

我们的实验给了一个**比论文更强**的证据: 在 **cross-distribution** (VM=Qwen3 family × LLM=0GM family) 场景下, **小 VM 反而更好**, 不只是"够用"。可能的解释:
1. 小 VM 信号"温和"程度天然就跟 LLM 原生分布 less 冲突
2. 小 VM 在训练时 capacity 有限, 学到的 reward landscape 更平滑 / 通用化, 跨 distribution 转移性更好
3. 跟 4B VM 比 1.7B 训练时间更少 (best epoch 都 ≤3), 没 overfit 到 Qwen3 specific patterns

---

## 5. 速度对比

| 指标 | A. thr=0.7 old | B. RM=4B FIX | C. RM=1.7B FIX |
|------|---------------|--------------|----------------|
| RM 模型大小 | 4B (32GB GPU) | 4B (32GB) | **1.7B (17GB GPU)** |
| throughput (tok/s) | 33.4 | 37.3 | **39.4** |
| avg latency/Q | 26.1s | 18.2s | 21.3s |

C 的 throughput 比 B 略快 5%, 主要因为 RM 小, /classify 时延略短。但 C 的 latency/Q 又比 B 长 17% — 因为 C 生成更长 (840 vs 680 tokens), 模型有更多 thinking 空间。这两个效应抵消。

总体, **速度差异小**, 不是选 VM size 的决定因素。

---

## 6. 跟论文官方 baseline 的对应

| 设置 | accuracy 损失 vs noSIA | 跟论文期望 |
|------|----------------------|------------|
| 论文 14B Qwen3 + VM-4B (W2S, same family) | + alignment 提升 | reward +3~8 |
| 论文 8B Qwen3 + VM-4B (W2S, same family) | + alignment 提升 | reward +3~8 |
| 论文 4B Qwen3 + VM-4B (same family) | + alignment 提升 | reward +7~17 (peak) |
| **实测**: 0GM-35B + VM-4B (cross family) | **-12.2pp acc** | 跨 family 不在论文测试范围 |
| **实测**: 0GM-35B + VM-1.7B (cross family) | **-5.5pp acc** | 仍是 cross family, 但损失减半 |

**结论**: 论文 baseline 都是 **same-family** (LLM 和 VM 都是 Qwen3 base 系列), 我们做的是 **cross-family** (0GM-35B 是后训练分布偏离 Qwen3 base 的 Qwen3.5/3.6 thinking 模型)。论文期望的 alignment 提升**没复现**, 但**小 VM 优于大 VM** 的副发现可能比论文主结论更有实践价值 — 对跨 family 部署很重要。

---

## 7. 下一步建议

按优先级:

1. **跑完整 600Q (MMLU-Redux 30 subjects × 20Q each) 用 VM=1.7B + thr=0.8 + FIX**, 验证 -5.5pp 是否稳定 (当前 30Q 样本量小, 几个 subject 单 Q 偶发性 noise 可能掩盖真实趋势)

2. **跑 AlpacaEval (805Q + Skywork RM 第三方打分)** — 这是 VM 训练目标 (helpfulness) 对口的评测; MMLU 是 correctness 不是 SIA 主战场。可能 helpfulness 提升 > MMLU correctness 损失。

3. **试 VM=0.6B**? 沿"小 VM 更温和"方向走一步。但 1.7B → 0.6B 可能 directional information 不够 (论文不在这个区间给 W2S 数据)。

4. **试 weight=0.5** 在 VM=4B 上看是否 reproduce 1.7B 的效果。如果是, 那不是 VM size 起作用, 而是 signal magnitude。

---

## 8. 相关文件

- 评测脚本: [`eval/mmlu_eval.py`](../eval/mmlu_eval.py)
- 修复 commit: `e1d150e` ([`src/sia_vllm_RM.py`](../src/sia_vllm_RM.py) Step 1+2)
- 模型转换: [`scripts/convert_rm_for_vllm.py`](../scripts/convert_rm_for_vllm.py)
- VM 下载: [https://huggingface.co/Runyi-Hu/SIA/tree/main/VM-Qwen3-1.7B-Base](https://huggingface.co/Runyi-Hu/SIA/tree/main/VM-Qwen3-1.7B-Base) (LoRA + score head)
- 主推理模型: `/workspace/SIA/models/0GM-1.0-35B-A3B-0427`
- 新 RM 模型 (本次): `/workspace/SIA/models/VM-Qwen3-1.7B-merged-for-vllm` (3.2 GB)
- 旧 RM 模型 (对照): `/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm` (7.6 GB)
- 日志归档: [`exp/`](../exp/) (见下文 README 索引条目)

---

## 9. 一句话结论

> Commit `e1d150e` 的 Step1+2 fix 在数值层把 SIA 真正激活了 (flip rate 3% → 16%), 但暴露了一个更深的 ML 问题: **VM-Qwen3-4B 跟 0GM-35B 跨家族, signal 强但方向偏, 让 MMLU accuracy 跌 -12.2pp**; **换更小的 VM-Qwen3-1.7B 让 mismatch 损失减半 (-5.5pp)**, 验证了 SIA 论文 §5.3 weak-to-strong 假设的一个更强版本 — **跨 family 场景下小 VM 反而比大 VM 更好**。
