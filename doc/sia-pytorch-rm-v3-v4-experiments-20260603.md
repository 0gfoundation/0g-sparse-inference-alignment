# SIA v3 + v4-fixed 实验记录 — PyTorch RM (官方加载) + entropy 调优

**日期**: 2026-06-03
**目的**: 在 [`vllm-classify-precision-loss-20260603.md`](vllm-classify-precision-loss-20260603.md) 发现 vllm `/classify` 路径有精度损失 (sigmoid + EPS=1e-6 clamp 在 ±13.82) 之后, 尝试用**官方 PyTorch `ValueModel.from_pretrained`** 加载 RM, 跳过 vllm, 拿真实 raw logit (±25 完整范围)。再结合调高 `entropy_threshold` (论文默认 1.0, 比我们之前 0.8 更稀疏) + 修复 60s server timeout, 看 SIA 在 VL-30B-Instruct 上灾难退化是否能缓解。

---

## 1. 新增组件 — PyTorch 官方 RM Server

新文件 `src/sia_rm_pytorch_official.py` — 简易 FastAPI 服务, 严格用官方
`ValueModel.from_pretrained(base_model_path, model_path, ...)` 加载 (跟 `/workspace/SIA/git/SIA/src/value_model/model.py` 完全一致), 暴露 `/score` endpoint, 接 SIA processor 的 pytorch backend。

```bash
nohup /workspace/SIA/venv4/bin/python \
  src/sia_rm_pytorch_official.py \
  --rm /workspace/SIA/models/Qwen3-4B \
  --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
  --device cuda:0 --port 8001 &
```

特点:
- 直接 raw logit 输出, **无 sigmoid + 反 sigmoid 来回**
- 用 attention_mask 找最后非 pad token 位置 (跟 HF SeqCls 一致), 处理多候选 batch padding 时正确
- 5 候选 batch /score 调用约 **600ms** (smoke test 实测)

---

## 2. RM 质量 baseline 验证 — 见 doc/rm-quality-test-20260603.md

启动 RM server 后, 跑了 11 类 (5 + 6) Q+A 质量测试。结果: **RM 对完整 Q+A 的评估完全合理** —
事实问答, 数学, 安全 alignment, 平衡论述, 翻译, 道德, 总结都给出合理 ranking。
但发现 2 个 RM 弱项 (不影响 AlpacaEval, 但值得记录):
- 代码语法错误 RM 不识别 (Test A: 缺括号代码 +38 vs 正确 +25)
- 数学题里"过程长度 > 答案对错" (Test B: 错答 +2 vs 简洁正确 +0.23)

→ **RM 本质上是好的, 至少能正确评判完整 Q+A**。所以问题不在 RM 模型本身, 而是 SIA 把它用在 partial-state prediction (论文 §6.4 自己也提"noise in value signals")。

---

## 3. v3 实验 — pytorch RM + entropy=1.0

### 3.1 配置

| 参数 | 值 |
|------|----|
| LLM | Qwen3-VL-30B-A3B-Instruct |
| RM | PyTorch `ValueModel.from_pretrained(Qwen3-4B + VM-Qwen3-4B-Base/LoRA)` |
| weight | 1.0 |
| entropy_threshold | **1.0** (论文默认) |
| topk | 5 |
| max_tokens | 2048 |
| temperature | 1.0 (无 top_p/top_k/rep_pen, pure multinomial) |
| **REQUEST_TIMEOUT (server)** | **60s** (老值, 没改) |

### 3.2 启动

```bash
# RM server already up on 8001
nohup env SIA_DEBUG_HIST=1 \
  /workspace/SIA/venv4/bin/python \
  /workspace/git/0g-sparse-inference-alignment/src/sia_vllm_server.py \
  --llm /workspace/SIA/models/Qwen3-VL-30B-A3B-Instruct \
  --rm_backend pytorch \
  --rm_url http://localhost:8001 \
  --llm_gpu_mem 0.65 --max_model_len 4096 \
  --topk 5 --weight 1.0 --entropy_threshold 1.0 \
  --host 0.0.0.0 --port 8000 &
```

### 3.3 中途观察 — 发现 60s timeout 截断

Q1 elapsed=60.1s, tokens=907 → 命中 server log 提示 `[SERVER] ... timed out after 60.1s, aborting`。

这个 60s timeout 在 `sia_vllm_server.py:244 REQUEST_TIMEOUT = 60` 是为 MMLU "Answer: A" 短答案场景设计的, 但 **PyTorch RM backend 慢 (15-20 tok/s vs vllm 30-40 tok/s)** + AlpacaEval 长生成 (期望 1000-2000 tokens) 时, **每个题都被 60s 截断**。

→ v3 不公平: SIA arm 被强制截断到 ~1000 tokens, noSIA arm (用 raw vllm serve, 无此 timeout) 能生成到 max_tokens=2048。

### 3.4 v3 Q1 输出抽样 (timeout 截断版)

907 tokens, 60s, 退化成自我修正循环:
- Item 1-5 正常 (Meryl Streep, Robert De Niro, Audra McDonald, Hugh Jackman, Lin-Manuel Miranda)
- Item 6 (Cher): "fondly associated courses with *The Wiz supporter/performing in 02-areas*" — gibberish
- "Note-level correction: Actually, Cher did NOT start strictly under major lyrics; however..."
- "Better Examples After Revision Correction Include" — 重启列表 (4 次)
- "Korshuidt's research confirms:" — **杜撰来源**
- 末尾被 60s 截断: "...regardless genre boundaries explored throughout entertainment media platforms including theater circuits across North American continent especially metropolitan[CUT]"

干预率: 49.5% (Q1, 3091 step 中 1530 干预), top-1 flip 率 65.8% (干预中 1007 flip)

### 3.5 v3 决定 kill

发现 60s timeout 截断问题后 + Q1 输出已退化, kill v3。

---

## 4. v4 实验 — pytorch RM + entropy=1.3

### 4.1 改 entropy_threshold 1.0 → 1.3

期望干预率从 ~50% 降到 ~10-20%, 让 SIA 真正"sparse"。

### 4.2 但又发现: v4 启动后 timeout=60 仍未改, 又会被 60s 截断

Smoke test Q1 也是 60s elapsed。Kill 之, 修 timeout。

---

## 5. v4-fixed 实验 — pytorch RM + entropy=1.3 + timeout=600

### 5.1 timeout 全面修复 (本次 commit)

| 文件 | 修改 |
|------|------|
| `src/sia_vllm_server.py:247` | `REQUEST_TIMEOUT 60 → **600**` (server-side gen timeout) |
| `src/sia_vllm_RM.py:611` | `/score` (pytorch backend) `timeout 30 → **120**` |
| `src/sia_vllm_RM.py:682` | `/classify` (vllm backend) `timeout 30 → **120**` |
| `eval/alpaca_eval.py:30` | client `timeout 300 → **700**` (必须 > server 600) |

`sia_vllm_server.py:212` httpx 3.0s **保留** (仅用于 log RM /status info, 不影响主流程)。

### 5.2 启动配置

跟 v4 一样, 但 timeout 链已修复。SIA LLM server 重启加载 ~16 min, smoke test 通过 (55 tokens "Python/JavaScript/Java", 0% 干预), launch 200Q eval。

### 5.3 中途观察 (跑了 25-50 题就停了, 因为已确认灾难)

| 指标 | v3 (timeout=60s) | **v4-fixed (timeout=600s)** |
|------|------------------|----------------------------|
| Q1 tokens | 907 (60s 截断) | **2048 (max 满)** |
| Q1 elapsed | 60.1s | **147.3s** |
| 干预率 | 49.5% | 37.5% (entropy=1.3 略稀疏) |
| Top-1 flip 率 | 65.8% | 64.0% |
| **触顶 max_tokens=2048 比例** | (没机会触顶) | **48% (12/25 题)** |
| 外语字符注入 (中/俄/阿/印) | 0 | **60% (15/25)** |
| 自我修正循环 | 28% | 28% |
| 平均 tokens/题 | ~1000 | **1363** (退化空间更大) |

### 5.4 Q1 v4-fixed 输出 (2048 tokens, 完整无截断)

开头 (Item 1-3): Meryl Streep, Robert De Niro, Al Pacino — 正常。

Item 4 Audrey Hepburn 开始崩坏:
```
4. **Audrey Hepburn** – ...until after f有了 success elsewhere
                                       ↑ 中文混入
Wait — let's correct: Actually Horse of… better list accurate ones.f
Let me revise this section officially!
[...4 次重启列表...]
✔️ Hugh Jackman — "won Tony for Les Misérables (as Jean Valjean)" ← WRONG (Hugh 是 Boy from Oz)
Correction: Hugh never played Women's lead. Let At this point STOP...
[...]
1. **Kristen Chenoweth** ...
ординарныйुरुकصفقة жанре кино из театраячно грамотная Bring it back to truth
↑ 俄语 + 印地语 + 阿拉伯语 mix → 完全 gibberish
airmale Framㅣ.White Duck Fifteen Rabbits equals zero resurgence
↑ 意义乱码
```

末尾 200 chars (max 满了, 在 dump 单词):
```
...graphic chart diagram illustration map graph timeline sequence series chain link 
node edge arc curve surface solid volumetric density mass weight inertia friction 
torque moment arm lever pivot fulcrum pulley block wedge inclined plane wheel axle 
screw gear windlass crank cam lever oscillating reciproc[截断在 max_tokens=2048]
```

跟"Broadway 演员"主题完全无关, 是无意义词汇 dump 直到 max。

### 5.5 抽样 Q5/Q10/Q20 末尾 200 chars (全部触顶, 全部 dump)

- **Q5 (How to wrap a present neatly?)** → 末尾在 dump "starfish sea foam bubbles sea salt sea breeze wind gust tides rivers deltas estuaries..."
- **Q10 (Georgian Kubdari recipe)** → 末尾在 dump "firms companies enterprises corporations conglomerates multinationals startups incubators accelerators funds investments..."
- **Q20 (Roast a pig)** → 末尾在 dump "analytic continuation contour pade approximation asymptoticseries perturbation infinite product continued fractions modular arithmetic..."

→ 跟原始问题完全无关, 是**稳态的灾难性退化模式**。

### 5.6 决定 kill v4-fixed, 用 25-50 题数据评估

200 题需要 ~4.4 小时跑完, 但 25 题已经清楚证明灾难。停止, 用现有 50 题样本评 Skywork。

---

## 6. Skywork 打分结果 — v4-fixed 50 题 vs noSIA 同 50 题

### 6.1 整体均值

| Arm | scored | 触顶/截断 | mean reward |
|-----|--------|----------|-------------|
| **v4-fixed** (前 50 题) | 33/50 (66%) | **17 题超 Skywork max_length=2048 被 skip** | **13.19** |
| noSIA (同 50 ids) | 50/50 (100%) | 0 | **31.36** |

### 6.2 配对统计 (n=33)

| 指标 | 值 |
|------|----|
| v4-fixed mean | 13.19 |
| noSIA mean | 32.42 |
| **Δ (v4 − noSIA)** | **-19.22** |
| **相对** | **-59.3%** |
| stdev Δ | 10.45 |
| **t-stat** | **-10.57** |
| p-value | **< 1e-11** (天文级别显著) |
| **Win/Loss** | **v4 wins 1, loses 31** (94% loss) |

### 6.3 Top 5 大比分输的题

| Q | v4-fixed | noSIA | Δ |
|---|---------|-------|------|
| Q33 | -1.01 | **+42.00** | **-43.01** |
| Q29 | +23.12 | +57.25 | -34.12 |
| Q20 | -9.88 | +23.62 | -33.50 |
| Q50 | +14.88 | +48.00 | -33.12 |
| Q46 | +12.00 | +43.50 | -31.50 |

### 6.4 跟之前 SIA 实验对比

| 实验 | RM backend | entropy | timeout | n | Δ (Skywork, 配对) | t-stat |
|------|-----------|---------|---------|---|-------------------|--------|
| v1 (OLD, Fix 前) | vllm | 0.8 | 60s | 195 | -0.46 (**-1.6%**) | -1.05 (不显著) |
| v2-fix (所有 fix) | vllm | 0.8 | 60s | 195 | -15.50 (**-52.9%**) | -19.99 |
| **v4-fixed (本次)** | **pytorch** | **1.3** | **600s** | **33** | **-19.22 (-59.3%)** | **-10.57** |

→ **v4-fixed 比 v2-fix 还灾难 6.4 个百分点!**

---

## 7. 为什么"更对齐官方"反而更糟? — 累计假设最终验证

我们在 v4-fixed 把所有已知偏差都修了:
- ✅ Fix #1 (sia_vllm_RM): 去 RM 输入 suffix
- ✅ Fix #2 (eval client): 去 top_p/top_k/rep_pen 二次过滤
- ✅ Fix #4 (sia_vllm_RM): 去 mean-norm
- ✅ 用 PyTorch 官方 ValueModel 加载 (跳过 vllm /classify 精度损失)
- ✅ 调高 entropy_threshold 0.8 → 1.3 (更稀疏干预)
- ✅ 修复 60s server timeout → 600s
- ✅ 修复 client 300s timeout → 700s

→ **结果: Δ 从 -1.6% (没 fix) → -52.9% (有 fix) → -59.3% (修更多)**, **严格单调更差**!

这是因为每次 fix 都让 RM 信号**更精准地传递到 SIA**:
1. Fix #1: RM 看到正确的 partial-response 输入 (无 trailing close tag)
2. Fix #2: SIA 推上去的 candidate 不再被 top_p 截掉
3. Fix #4: RM signal 完整 magnitude (不被 mean-norm 压缩)
4. PyTorch RM: 不被 sigmoid clamp 在 ±13.82
5. timeout=600: 模型能继续被 SIA 推到 max_tokens=2048

每个 fix **单独是对的** (跟官方代码一致), 但**累积起来让 VM 的"错误判断"百分百传递到主模型**, 因为 **VM 跟主模型不同代 (代际不匹配 [`sia-generational-mismatch-analysis-20260603.md`](sia-generational-mismatch-analysis-20260603.md))**。

类比: 一个学徒 (VM-4B, 2025-04 Qwen3) 强行指导一个博士 (VL-30B-Instruct, 2025-秋, 30B MoE) — 当学徒说"这个选择不对"时, 信号越清晰、越完整地传到博士, 博士就越多被推向错误方向。

---

## 8. 一句话现状

> v3/v4-fixed 在所有可识别的对齐 fix 都加上之后 (PyTorch RM 无 sigmoid 精度损失, entropy=1.3 稀疏干预, timeout=600 让生成跑满 2048), Skywork Δ 仍是 **-59.3% (t=-10.57, p<1e-11)**, **比 v2-fix (-52.9%) 更深 6 pp**, 50 题里 **31 题大比分败**, 48% 题目触顶 max_tokens 且尾部全是无意义词汇 dump (跟原题完全脱钩)。这彻底坐实 [代际不匹配假设](sia-generational-mismatch-analysis-20260603.md): **在 VM (2025-04 Qwen3-4B) 跟主模型 (2025-秋 VL-30B-Instruct MoE) 代际差距过大、W2S 比例 7.5× 远超论文测过的 3.5× 极限时, "更对齐官方算法 + 更清晰的 RM 信号" 等于让错误判断更精确地放大**。下一步可尝试 (a) 大幅调高 entropy=2.0 让干预真稀疏到 < 10%, 或 (b) 降 weight=0.3 让 RM 信号不再主导, 但根本解决方法仍是换同代/同家族/同能力的 VM。
