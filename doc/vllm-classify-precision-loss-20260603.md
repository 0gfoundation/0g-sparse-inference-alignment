# vLLM `/classify` 服务路径的 RM 信号精度损失分析

**日期**: 2026-06-03
**起源**: 在排查 SIA 修复后 (Fix #1+#2+#4 严格对齐官方) 仍出现灾难性退化 (VL-30B 200Q 人工评分 -32%) 时, 怀疑 `convert_rm_for_vllm.py` 把 Qwen3-4B + LoRA + token_reward_head 转换成 vllm-可加载的 Qwen3ForSequenceClassification 这一步可能引入 bug。直接对比**官方 PyTorch ValueModel** vs **我们 merged checkpoint** 在同一批 (prompt, candidate) 上的打分。

---

## 1. 验证设置

用 5 个 SIA-like 输入 (跟前文 verify 测试相同), 对比三种加载方式给出的 score:

```python
user = "What are 3 colors of fruit?"
response = "Three colors of fruit are:"
candidates = [',', 'A', ' a', 'p', 'tool']
# formatted: prefix + response + candidate (跟 Fix #1 后的格式一致, 无 <|im_end|> 后缀)
```

三种加载方式:
1. **官方 PyTorch ValueModel** — `Qwen3-4B + LoRA + token_reward_head.pt`, 走 `ValueModel.from_pretrained(...)`, forward 输出 raw logits, 取 `hidden[:, -1]`
2. **merged-HF (直接用 HF 加载我们的 merged checkpoint)** — `VM-Qwen3-4B-merged-for-vllm` 用 `AutoModelForSequenceClassification.from_pretrained(...)`, HF 标准 forward
3. **vllm /classify** (我们 SIA 实际跑的路径) — `vllm serve --convert classify`, POST /classify 拿 sigmoid prob, 客户端反 sigmoid 还原 raw logit

---

## 2. 结果 1: merged checkpoint **权重 100% 正确**

| Candidate | 官方 PyTorch ValueModel | merged-HF (直接 HF 加载) | Diff |
|-----------|------------------------|-------------------------|------|
| ',' | **-10.057** | -10.063 | +0.006 |
| 'A' | **+1.211** | +1.219 | -0.007 |
| ' a' | -1.607 | -1.602 | -0.006 |
| 'p' | -4.494 | -4.469 | -0.025 |
| 'tool' | **-22.846** | -22.875 | +0.029 |

**max |diff| = 0.029** (bfloat16 精度噪声), **ranking 完全一致**: `['A', ' a', 'p', ',', 'tool']` ✅

→ **`convert_rm_for_vllm.py` 转换流程正确**, merged checkpoint 直接用 HuggingFace `AutoModelForSequenceClassification` 加载, 跟官方 `ValueModel.from_pretrained` 输出一致 (<0.03 logit 差)。

---

## 3. 结果 2: 🚨 **vllm /classify 服务路径有显著精度损失**

对比官方 raw logit vs 经过 vllm /classify + 我们 sigmoid 反向的 score:

| Candidate | 官方 (raw logit) | vllm /classify + 反 sigmoid | **截断/偏移误差** |
|-----------|-----------------|----------------------------|------------------|
| 'A' | **+1.211** | +0.469 | **-0.74** |
| ' a' | -1.607 | -2.007 | -0.40 |
| 'p' | -4.494 | -5.527 | -1.03 |
| ',' | **-10.057** | -11.239 | -1.18 |
| **'tool'** | **-22.846** | **-13.816** | **+9.03 (clamp 在 -13.82)** |

误差类型分两类:

### 3.1 EPS clamping 截断极端值 (主要问题)

我们 `_score_candidates_vllm` 的反 sigmoid 代码:

```python
EPS = 1e-6
scores = [math.log(max(min(p, 1-EPS), EPS) / max(min(1-p, 1-EPS), EPS)) for p in scores]
```

- `EPS = 1e-6` → 反向后最大幅度 ≈ **±log(1e-6 / (1-1e-6)) ≈ ±13.82**
- 官方 raw logit 可以是 **-22.85** (无任何截断)
- 'tool' 候选: 官方给 -22.85, 我们给 -13.82, **少了 9 点惩罚**

### 3.2 vllm 内部 sigmoid + bfloat16 精度损失 (次要)

vllm /classify 内部 sigmoid 输出可能存 bfloat16, bfloat16 只有 ~3 位 decimal 精度。即使没触发 EPS 截断, 中等值也有 ~1 单位的漂移:

- 官方 raw `-10.057` ↔ vllm 反向 `-11.239` (差 1.2)
- 官方 raw `+1.211` ↔ vllm 反向 `+0.469` (差 0.74)
- 官方 raw `-4.494` ↔ vllm 反向 `-5.527` (差 1.0)

总体表现为 **系统性偏向更负**, 跟具体 vllm 实现版本和 batching 行为相关。

---

## 4. 对 SIA 行为的影响

| 维度 | 官方 (直接 PyTorch) | 我们 (vllm 服务) | 后果 |
|------|----|----|------|
| RM delta 最大幅度 | 可以 -22 ~ +5 (spread 27+) | 只能 -13.8 ~ ~+13.8 (单边封顶) | **极坏候选的惩罚不够强** |
| RM delta 精度 | bfloat16 端到端 (1 次精度损失) | sigmoid + 反向, 多次损失 + 不可逆 | **中等幅度信号噪声大** |
| Ranking 保持 | ✅ | ✅ | 排序对 |
| 绝对 magnitude | 准确 | 系统性压缩 + 噪声 | **flip 行为偏差** |

**具体后果**:

- 在 weight=1.0 下, 官方真实信号能 "强力推开 tool 类极坏候选 (-23)", 我们只能 "中度推开 (-14)"
- 反过来, 对没那么坏的 candidate, 我们的信号也被压缩/扭曲
- → 我们的 SIA 看到的是一个 **magnitude 偏小 + 噪声更多的 RM 信号**, 跟论文 / 官方实验里的"干净 raw logit"不一样

---

## 5. 跟之前发现的 root cause 拼图

到目前为止我们已识别的 SIA 退化原因 (按贡献度):

| # | 原因 | 是否解释 SIA 灾难退化 |
|---|------|---------------------|
| 1 | **代际不匹配** (VM-Qwen3-4B 2025-04 vs VL-30B 2025-秋 vs 0GM-35B 2026-早) | **主因 (W2S 推到 7.5-8.75× 远超论文 3.5× 极限)** |
| 2 | **AlpacaEval 只测 helpfulness pillar** (论文 3H 之一, 不是 SIA 强项) | **次因** (论文 Fig 2 显示 SIA 在 AlpacaEval 提升本就最小) |
| 3 | **vllm /classify 精度损失** (本 doc) | **次因 (信号 magnitude 压缩 + 噪声)** |
| 4 | Instruct 模型本身接近 reward ceiling | 次因 (论文 §5.2 自己承认 Instruct 收益小于 Base) |
| 5 | 词表跨族 (0GM-35B 248K vs VM 151K) | 0GM 主因, VL-30B 无关 |
| 6 | thinking 模式 (vs 论文测的 non-thinking) | 0GM 相关, VL-30B 无关 |

→ vLLM 精度损失只是 **第 3 因素**, 不是主因; 但**修复后能从信号源头消除一类系统偏差, 让代际不匹配的真实影响显现**。

---

## 6. 修复方案

| 方案 | 描述 | 难度 | 推荐 |
|------|------|------|------|
| **A**: 不用 vllm, 直接 PyTorch 加载 merged-HF checkpoint | 自建 RM server (FastAPI 包 transformers forward), 直接返回 raw logit, 无 sigmoid | 中等; 失去 vllm batching/prefix caching | ⭐⭐⭐ 推荐先做 |
| **B**: 用 vllm 的 `task=embed` 拿 hidden states, 客户端做最后 head linear | 客户端拿到 hidden 后 manually 做 `score_head(hidden[:, -1])` | 中等; vllm 加速保留 | ⭐⭐ |
| **C**: 把 EPS 改更小 (e.g. 1e-12) 让反向不截断那么早 | 1 行代码改 | 简单, **但 bfloat16 sigmoid 本身就丢精度**, EPS 改了也没用 | ⭐ 无效 |
| **D**: vllm 加载时强制 fp32 sigmoid 输出 | 需要看 vllm 源码 / patch | 较难, 不一定支持 | ⭐⭐ |
| **E**: 验证假设 — 用方案 A 直接 PyTorch 跑 VL-30B 200Q SIA, 看是否退化幅度变小 | 实测最有说服力 | 中等 | ⭐⭐⭐ 推荐验证 |

---

## 7. 一句话现状

> `convert_rm_for_vllm.py` **转换流程正确** (merged checkpoint 直接用 HF 加载与官方 ValueModel 输出 max diff < 0.03), 但 **vllm `/classify` 端点强制 sigmoid 输出, 经我们客户端反 sigmoid + EPS=1e-6 clamping 还原 raw logit, 引入 ~9 点的极值截断 + 多个单位的系统偏移**。这会让 SIA 看到的 RM 信号 magnitude 被压缩、噪声增加。**这不是 SIA 退化的主因 (主因仍是代际不匹配), 但确实是一个独立的精度问题, 修复 (方案 A: 直接 PyTorch 服务 RM) 可以从信号源头净化, 让代际不匹配的真实影响显现, 也便于后续对比实验**。

---

## 8. 验证步骤 (本 doc 复现)

```bash
# 1. 安装 peft 到 venv4 (transformers 4.57.6, 支持 qwen3)
/workspace/SIA/venv4/bin/pip install peft

# 2. 运行验证脚本 (本 doc 5.1 节用的)
/workspace/SIA/venv4/bin/python /tmp/verify_rm_consistency.py
```

脚本对比官方 `ValueModel.from_pretrained` 和 `AutoModelForSequenceClassification.from_pretrained(merged_checkpoint)` 在 5 个 (prompt, candidate) 上的 raw logit, 看是否一致。
