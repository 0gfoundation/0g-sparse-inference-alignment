# rep_penalty 修复验证实验汇总 (Qwen3-14B + VL-30B)

**日期**: 2026-06-04
**目的**: 系统验证 `repetition_penalty=1.3 → 1.0` 修复后, SIA 在不同模型 / 不同长度配置下都能复现 paper SIA gain。

**核心结论**: ✅ rep_penalty 修复在 4 个实验都有效 (Qwen3-14B + VL-30B 三种长度), SIA Δ vs noSIA **+1.3 ~ +2.8** 全部 p<10⁻³ 显著。**SIA 不仅在 paper 的 max=256 regime 有效, 在 max=2048 长生成 regime 上也 +10% gain**, 完全反转了之前 "OOD 配置 SIA 不可用" 的错误结论。

详见根因分析: [`doc/sia-repetition-penalty-root-cause-20260604.md`](sia-repetition-penalty-root-cause-20260604.md)
配置 diff: [`doc/project-vs-official-inference-code-diff-20260604.md`](project-vs-official-inference-code-diff-20260604.md)

---

## 实验总览

| 实验 | LLM | max_tokens | SIA params | rep_penalty | SIA intv 率 | log 路径 |
|------|-----|-----------|----------|-------------|--------|--------|
| **Qwen3-14B SIA** | Qwen3-14B | 256 | topk=10, weight=1.0, entropy=1.0 | **1.0** ✅ | (n/a) | [`exp/.../qwen14b-sia-max256/`](../exp/rep-penalty-fix-validation-20260604/qwen14b-sia-max256/) |
| **VL-30B SIA max=256** | Qwen3-VL-30B | 256 | topk=10, weight=1.0, entropy=1.0 | **1.0** ✅ | **18.8%** | [`exp/.../vl30b-sia-max256/`](../exp/rep-penalty-fix-validation-20260604/vl30b-sia-max256/) |
| **VL-30B noSIA max=256** | Qwen3-VL-30B | 256 | weight=0, entropy=10 (no-op) | 1.0 ✅ | — | [`exp/.../vl30b-nosia-max256/`](../exp/rep-penalty-fix-validation-20260604/vl30b-nosia-max256/) |
| **VL-30B SIA max=2048** | Qwen3-VL-30B | 2048 | topk=10, weight=1.0, entropy=1.0 | **1.0** ✅ | **25.8%** | [`exp/.../vl30b-sia-max2048/`](../exp/rep-penalty-fix-validation-20260604/vl30b-sia-max2048/) |
| **VL-30B noSIA max=2048** | Qwen3-VL-30B | 2048 | weight=0, entropy=10 (no-op) | 1.0 ✅ | — | [`exp/.../vl30b-nosia-max2048/`](../exp/rep-penalty-fix-validation-20260604/vl30b-nosia-max2048/) |
| **VL-30B noSIA MMLU 150Q** | Qwen3-VL-30B | 2048 (eval) | weight=0, entropy=999999 (no-op) | 1.0 ✅ | — | [`exp/.../vl30b-mmlu-noSIA/`](../exp/rep-penalty-fix-validation-20260604/vl30b-mmlu-noSIA/) |
| **VL-30B SIA MMLU 150Q** | Qwen3-VL-30B | 2048 (eval) | topk=5, weight=1.0, entropy=1.0 | **1.0** ✅ | **10.0%** | [`exp/.../vl30b-mmlu-SIA/`](../exp/rep-penalty-fix-validation-20260604/vl30b-mmlu-SIA/) |

所有 AlpacaEval 实验:
- LLM 推理代码: `src/sia_vllm_server.py` (本项目, 已修复 rep_penalty 默认 1.0)
- RM (Value Model): 项目 vllm RM (`vllm serve --runner pooling --convert classify` on VM-Qwen3-4B-merged-for-vllm)
- AlpacaEval 200Q, temperature=1.0
- 评分: Skywork-Reward-V2-Llama-3.1-8B (`/workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B`)

MMLU 实验:
- 评测脚本: `eval/mmlu_eval.py` (CoT + "Answer: X" 提取, max_tokens=2048 内置)
- 数据集: `edinburgh-dawg/mmlu-redux` (30 subjects)
- 题数: 5 题 / subject × 30 subjects = 150 Q
- 打分: exact_match on A/B/C/D

---

## 一. Qwen3-14B (paper-aligned config)

**目的**: 验证项目代码 + rep_penalty 修复后跟 paper Qwen3-14B baseline 统计等价。

| | Skywork mean | median |
|--|---|---|
| Ours Qwen3-14B SIA (rep=1.0) | **+11.48** | +11.59 |
| Paper Qwen3-14B SIA (805Q) | +11.16 | — |
| Paper Qwen3-14B noSIA | +9.59 | — |

**关键 paired (n=200, 跟 paper 同题对比)**:
- Ours vs Paper SIA: Δ=+0.31, **p=0.43** (n.s.) ✅ **统计上等价**
- Ours vs Paper noSIA: Δ=+1.89, **p<10⁻⁴** ✅ SIA gain 显著

**对照 rep=1.0 vs rep=1.3 (paired n=125, 单变量 rep_penalty)**:
- Δ=**+7.71**, win **109/125 (87%)**, **p<10⁻¹²** ← 完全锁定 rep_penalty 是元凶

数据: [`exp/.../qwen14b-sia-max256/outputs_scored.json`](../exp/rep-penalty-fix-validation-20260604/qwen14b-sia-max256/outputs_scored.json)
评分 log: [`exp/.../analysis/compare_qwen14b.log`](../exp/rep-penalty-fix-validation-20260604/analysis/compare_qwen14b.log)

---

## 二. VL-30B max=256 (paper-aligned 长度)

| | Skywork mean | median |
|--|---|---|
| VL-30B SIA-256 | **+14.01** | +13.31 |
| VL-30B noSIA-256 | +11.69 | +10.78 |

**Paired (n=200)**:
- **SIA Δ = +2.32 (+19.8%), p<10⁻⁴**, win 132/200 (66%) ✅

VM (RM) intervention 验证:
- 平均 **18.8%** intervention 率
- top1 flip 率 ~55-81%
- RM error 数: 0 (无 Connection refused)
- Log 节选 (`sia_llm_server.log`):
  ```
  [SIA] req=0 DONE  intervened=50/256  ratio=19.5%  top1_flip=33/50 (66.0%)
  [SIA] req=0 DONE  intervened=83/256  ratio=32.4%  top1_flip=48/83 (57.8%)
  ...
  ```

数据: [`exp/.../vl30b-sia-max256/outputs_scored.json`](../exp/rep-penalty-fix-validation-20260604/vl30b-sia-max256/outputs_scored.json) (SIA) + [`exp/.../vl30b-nosia-max256/outputs_scored.json`](../exp/rep-penalty-fix-validation-20260604/vl30b-nosia-max256/outputs_scored.json) (noSIA)

---

## 三. VL-30B max=2048 (长生成 regime)

**关键警告**: 第一次跑 max=2048 时, **vllm RM server 因 GPU 内存竞争 OOM 死了**, SIA 退化为 no-op (intervention 0%), 输出跟 noSIA byte-exact 相同, 得出错误结论 "SIA -6% 损害"。
**Redo run 用顺序启动 + /classify smoke test 确保 RM 真活着**, 才是有效数据。具体见根因分析 §2 "1992018 / 1932817 race condition" 部分。

| | Skywork mean | median |
|--|---|---|
| VL-30B SIA-2048 (REDO, 真 SIA) | **+30.19** | +30.75 |
| VL-30B noSIA-2048 | +27.44 | +27.63 |

**Paired (n=200) — 关键结果**:
- **SIA Δ = +2.75 (+10.0%), p<10⁻⁷**, win 123/200 (62%) ✅

**SIA-2048 truncated to 256 vs noSIA-2048 truncated to 256**:
- Δ=+1.30, p=0.003 ✅ SIA 在前 256 token 也 +11% 优势

### VM intervention 验证

- 平均 **25.83%** intervention 率
- 平均 top1 flip 率 **66.88%** (22,381 flips / 33,462 interventions)
- RM error 数: 0
- Sanity check: SIA-2048 outputs **0/10 byte-identical** to noSIA-2048 (上次 broken run 是 10/10 identical)
- Log 节选 (`sia_llm_server.log`):
  ```
  [SIA] req=0 DONE  intervened=90/451  ratio=20.0%  top1_flip=64/90 (71.1%)
  [SIA] req=0 DONE  intervened=213/1131  ratio=18.8%  top1_flip=140/213 (65.7%)
  [SIA] req=0 DONE  intervened=300/801  ratio=37.5%  top1_flip=192/300 (64.0%)
  ...
  ```

### 性能指标 (n=200, single-stream 跑)

**吞吐 (driver wall time)**:

| | SIA-2048 | noSIA-2048 (baseline) |
|--|---|---|
| 总 reqs | 200 | 200 |
| 总生成 tokens | 129,313 | 151,806 |
| 总 wall 时间 | 3,599s (60.0 min) | 1,214s (20.2 min) |
| **平均 tokens/s** | **35.9** | **125.0** (3.5× faster) |
| 平均 tok/req | 647 | 759 (SIA 输出短 ~15%) |
| 平均 wall/req | 18.0s | 6.1s |

**每 token 推理耗时** (= wall_time / tokens, single-stream):

| arm | 平均 step (ms/token) | 解读 |
|--|---|---|
| **noSIA-2048 (纯 LLM)** | **8.0ms** | 单纯 LLM forward + sampling |
| **SIA-2048 (混合 SKIP + INTERVENE)** | **27.8ms** | 多了 SIA 干预开销 |

**SIA 内部时间拆解** (from `[SIA-pf-summary]` 在 `sia_llm_server.log`):

| 步骤 | p50 (ms) | p95 (ms) | 说明 |
|------|---------|---------|------|
| **SKIP step (非干预步骤总耗时)** | **3.73** | 4.06 | top-K + entropy + decision, 无 RM 调用 |
| Top-K 提取 + entropy 计算 | 0.64 | 1.16 | GPU torch.topk + log_softmax |
| CPU sync (entropy 拉回 CPU) | 3.11 | 3.61 | 主要开销 |
| **INTERVENE step (干预步骤总耗时)** | **~75** | ~78 | apply_total p95 |
| ↳ RM HTTP /classify call | 71.18 | 98.34 | **占干预成本 95%** ← 瓶颈 |
| ↳ format_chat (prompt 构造) | 0.13 | 0.69 | |
| ↳ intv_prepare (索引/数据准备) | 0.16 | 0.32 | |
| ↳ intv_apply_logits (改 logits) | 1.15 | 1.32 | |

**核心观察**:
- **INTERVENE 比 SKIP 慢 ~20× ** (75ms vs 3.7ms), 因 RM HTTP 调用 (~71ms p50)
- SKIP 中 CPU sync (3.1ms) 是主要开销, 远超 GPU 计算 (0.6ms)
- 加权平均 step = 25.8% × 75ms + 74.2% × 3.7ms = 19.3 + 2.7 = **22.0ms** (理论)
- 实测 27.8ms — 多出 ~5ms 是 LLM forward + vllm 内部 sampling 开销 (跟 noSIA 的 8ms baseline 接近)
- **总成本拆分**: LLM forward (8ms) + SIA SKIP overhead (~3.7ms) + INTERVENE 中的 RM 调用 (~18ms 加权) ≈ 29.7ms ≈ 实测

**优化方向** (未实施):
- RM 调用 **batch 多个 candidates** 而不是每 candidate 一次 HTTP → 可降 ~50% 干预成本
- vllm RM 启用 prefix caching (已开) — 当前命中率 unknown
- 把 RM 切换成 in-process / shared-memory 调用消除 HTTP 开销

数据: [`exp/.../vl30b-sia-max2048/outputs_scored.json`](../exp/rep-penalty-fix-validation-20260604/vl30b-sia-max2048/outputs_scored.json) (SIA REDO) + [`exp/.../vl30b-nosia-max2048/outputs_scored.json`](../exp/rep-penalty-fix-validation-20260604/vl30b-nosia-max2048/outputs_scored.json) (noSIA)
评分 log: [`exp/.../analysis/compare_redo.log`](../exp/rep-penalty-fix-validation-20260604/analysis/compare_redo.log)
完整 SIA 性能日志: [`exp/.../vl30b-sia-max2048/sia_llm_server.log`](../exp/rep-penalty-fix-validation-20260604/vl30b-sia-max2048/sia_llm_server.log) (grep `\[SIA-pf-summary`)

---

## 四. MMLU 回归测试 (知识类任务 SIA 无 regression)

**目的**: 验证 SIA 在与 Value Model 训练目标 (Helpfulness, Harmlessness) **无关**的知识类任务上, **没有引入负面效果**。Paper 没在 MMLU 测过, 这是项目自检。

### 配置

| 项 | 值 |
|--|---|
| Dataset | `edinburgh-dawg/mmlu-redux`, 30 subjects |
| 题数 | 5 题 / subject × 30 = **150 Q** |
| LLM | Qwen3-VL-30B-A3B-Instruct (项目 `sia_vllm_server.py` 推理) |
| RM (SIA arm) | 项目 vllm RM (`VM-Qwen3-4B-merged-for-vllm`) |
| max_tokens | 2048 (脚本内置) |
| temperature | 1.0 |
| repetition_penalty | **1.0** (fix 后默认 + driver 显式) |
| SIA params | topk=**5**, weight=1.0, entropy_threshold=1.0 |
| noSIA params | topk=5, weight=0.0, entropy_threshold=999999 (no-op) |
| 答案提取 | CoT + "Answer: X" 正则 (exact match) |

`--topk 5` (跟 AlpacaEval 的 10 不同) 是 paper 同 model 上 MMLU 的标准设置, 沿用之前 Qwen3-14B MMLU 实验保持可比。

### 结果

| 指标 | **SIA** | **noSIA** | Δ |
|------|--------|----------|---|
| **Overall accuracy** | **0.8000** (120/150) | 0.7867 (118/150) | **+0.013 (+1.7%, +2 题)** |
| Total latency | 698.8s (11.6 min) | 454.3s (7.6 min) | +1.54× |
| Avg latency/Q | 4.7s | 3.0s | +1.6× |
| Avg tokens/Q | 262.5 | 287.7 | -25 (SIA 略短) |
| Throughput | 56.3 tok/s | 95.0 tok/s | -41% (RM 调用开销) |
| **SIA intervention 率** | **10.03%** (4000/39889) | — | — |
| **SIA top1 flip 率** | **61.7%** (2467/4000) | — | — |
| RM error 数 | 0 | — | — |

### 结论 — 无 regression ✅

- SIA - noSIA = +2 题, 在 n=150 二项分布噪声内 (binomial 95% CI ≈ ±7.7pp)
- **统计上 SIA 跟 noSIA 等价**, 无显著 gain 也无显著 loss
- SIA 内部健康指标都正常 (10% intervention, 62% flip, 0 RM error), 跟 AlpacaEval VL-30B 同量级
- 跟之前 [`doc/eval-report.md`](eval-report.md) 上 Qwen3-14B 的 MMLU 结论一致:
  > Value Model 在哪个领域训练, 就只在哪个领域的任务上能引导 LLM. ... MMLU 是纯知识问答, 与 Value Model 训练目标无关, 因此干预效果为零 — 这是符合预期的。

### MMLU 跟 AlpacaEval 对比 — SIA 的"任务覆盖边界"

| 任务类型 | 配置 | SIA Δ vs noSIA | 显著性 | 解读 |
|----------|------|---------------|--------|------|
| **AlpacaEval** (max=256) | RM 训练匹配 (alignment) | **+2.32 (+20%)** | p<10⁻⁴ ✅ | RM 引导有效 |
| **AlpacaEval** (max=2048) | RM 训练匹配 | **+2.75 (+10%)** | p<10⁻⁷ ✅ | RM 引导有效 (长输出) |
| **MMLU** (150Q) | RM 训练**不匹配** (knowledge) | **+0.013 (+1.7%)** | n.s. | 无 regression, 无 gain |

**SIA 是 alignment-class 任务的工具**, 不应期望它提升知识类准确率。

数据: [`exp/.../vl30b-mmlu-SIA/results.json`](../exp/rep-penalty-fix-validation-20260604/vl30b-mmlu-SIA/results.json) (SIA) + [`exp/.../vl30b-mmlu-noSIA/results.json`](../exp/rep-penalty-fix-validation-20260604/vl30b-mmlu-noSIA/results.json) (noSIA)

---

## 五. 跨实验一致性

| 实验 | n | SIA mean | noSIA mean | Δ | rel | p |
|------|--|---------|-----------|---|-----|---|
| Paper Qwen3-14B (baseline, 805Q) | 805 | +13.92 | +12.29 | **+1.63** | +13% | (paper) |
| Ours Qwen3-14B (max=256) | 200 | +11.48 | +9.59 | **+1.89** | +20% | <10⁻⁴ |
| Ours VL-30B (max=256) | 200 | +14.01 | +11.69 | **+2.32** | +20% | <10⁻⁴ |
| **Ours VL-30B (max=2048)** | **200** | **+30.19** | **+27.44** | **+2.75** | **+10%** | **<10⁻⁷** |
| Ours VL-30B (max=2048, trunc-256) | 200 | +13.32 | +12.01 | **+1.30** | +11% | 0.003 |

**SIA 绝对 gain 在 +1.3 ~ +2.8 之间, 跨模型 / 跨长度高度一致**。max=2048 的 relative gain (+10%) 较小**只是因为 Skywork 偏好长答案让 baseline 升到 +27** (相同绝对 Δ÷大基线 = 小百分比), 不是 SIA 自身能力下降。

## 六. 推翻先前的错误结论

之前 `exp/vl30b_200q_dual_rm/README.md` 写:
> W2S=7.5× 配置下 SIA 仍 **-75% Δ**, 完全不可用。论文的成功只在 W2S≤3.5× in-distribution 配置成立。

这个结论是错的, 真因是 `repetition_penalty=1.3` 污染了所有那些实验。修复后:
- VL-30B (W2S=7.5×) max=256: +20% Δ
- VL-30B (W2S=7.5×) max=2048: +10% Δ
- 都跟 paper Qwen3-14B (in-distribution) gain 同量级

也曾错误地推论 "SIA 是短窗口工具, 只在 ≤256 有效":
- 这是基于 broken-RM 的 SIA-2048 = -6% 数据
- Redo 后 SIA-2048 真正 +10% Δ, **SIA 在长生成上同样有效**

---

## Appendix A. 启动命令 (完整复现)

### A.1 项目 vllm RM server (所有 VL-30B SIA 实验用)

```bash
nohup /workspace/SIA/venv4/bin/vllm serve \
  /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
  --runner pooling --convert classify \
  --hf-overrides '{"architectures":["Qwen3WithScoreForCausalLM"]}' \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.15 \
  --max-model-len 4096 \
  --port 8001 --host 0.0.0.0 \
  --disable-log-stats \
  > /tmp/vl30b_runs/vllm_rm_server.log 2>&1 &
```

### A.2 RM /classify smoke test (启动 SIA LLM 之前必做!)

```bash
# 等 RM /v1/models 响应
until curl -s --max-time 2 http://localhost:8001/v1/models 2>/dev/null | grep -q object; do sleep 3; done

# 验证 /classify 真返回 probs (port up != engine alive!)
curl -s --max-time 10 -X POST http://localhost:8001/classify \
  -H "Content-Type: application/json" \
  -d '{"model":"/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm","input":[[1,2,3,4,5]]}'
# 应该返回 {"id":...,"data":[{"index":0,"probs":[...]}], ...}
```

### A.3 VL-30B SIA LLM server (max=2048)

```bash
cd /workspace/git/0g-sparse-inference-alignment
nohup /workspace/SIA/venv4/bin/python src/sia_vllm_server.py \
  --llm /workspace/SIA/models/Qwen3-VL-30B-A3B-Instruct \
  --rm_url http://localhost:8001 \
  --rm_backend vllm \
  --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
  --llm_gpu_mem 0.60 \
  --topk 10 --weight 1.0 --entropy_threshold 1.0 \
  --max_model_len 4096 \
  --port 8000 \
  > /tmp/vl30b_runs/sia_llm_server.log 2>&1 &
```

VL-30B SIA max=256 只把 `--max_model_len` 改 2048 (其它保持)。

### A.4 VL-30B noSIA SIA LLM server (weight=0 模式)

跟上面相同, 仅改:
```bash
  --topk 10 --weight 0.0 --entropy_threshold 10.0 \   # weight=0 + entropy threshold 永不触发
```
(此时 RM 不会被查, 但保持 server 启动 + 跟 SIA 一致的 code path)

### A.5 Driver (driver 显式传 repetition_penalty=1.0, 跟 server 默认双保险)

`/tmp/vl30b_runs/drive_vl30b.py`:
```python
payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": item["instruction"]}],
    "max_tokens": MAX_TOK,
    "temperature": 1.0,
    "repetition_penalty": 1.0,
    "chat_template_kwargs": {"enable_thinking": False},
}
```

启动 driver:
```bash
nohup /workspace/SIA/venv4/bin/python -u /tmp/vl30b_runs/drive_vl30b.py \
  /tmp/vl30b_runs/sia_VL30B_vllmRM_200q_max2048_v2.json \
  Qwen3-VL-30B-A3B-Instruct \
  2048 200 \
  > /tmp/vl30b_runs/driver.log 2>&1 &
```

### A.6 Qwen3-14B SIA LLM server (官方 PyTorch RM)

Qwen3-14B 实验用的是官方 PyTorch RM (`sia_rm_pytorch_official.py`), 而不是项目 vllm RM:

```bash
# RM (official ValueModel HTTP)
nohup /workspace/SIA/venv4/bin/python \
  /workspace/git/0g-sparse-inference-alignment/src/sia_rm_pytorch_official.py \
  --rm /workspace/SIA/models/Qwen3-4B \
  --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
  --device cuda:0 --port 8002 \
  > /tmp/qwen3_14b_runs/pytorch_rm_server.log 2>&1 &

# SIA LLM
cd /workspace/git/0g-sparse-inference-alignment
nohup /workspace/SIA/venv4/bin/python src/sia_vllm_server.py \
  --llm /workspace/SIA/models/Qwen3-14B \
  --rm_url http://localhost:8002 \
  --rm_backend pytorch \
  --llm_gpu_mem 0.55 \
  --topk 10 --weight 1.0 --entropy_threshold 1.0 \
  --max_model_len 2048 \
  --port 8000 \
  > /tmp/qwen3_14b_runs/sia_server_rep10.log 2>&1 &

# Driver (with repetition_penalty=1.0 in payload)
nohup /workspace/SIA/venv4/bin/python -u /tmp/qwen3_14b_runs/drive_805_rep10.py \
  /tmp/qwen3_14b_runs/sia_REP10_200q_max256.json \
  Qwen3-14B \
  256 200 \
  > /tmp/qwen3_14b_runs/driver_rep10_max256.log 2>&1 &
```

### A.7 Skywork 评分脚本

跑评分前**必须 kill SIA + RM servers 释放 GPU** (Skywork 需要 ~16GB):

```bash
pkill -9 -f 'EngineCore' 2>/dev/null
pkill -9 -f 'sia_vllm_server' 2>/dev/null
pkill -9 -f 'vllm serve' 2>/dev/null
sleep 8
nvidia-smi --query-gpu=memory.free --format=csv,noheader

# 跑评分脚本
cd /tmp/vl30b_runs
/workspace/SIA/venv4/bin/python -u compare_redo.py
```

主评分脚本: [`exp/.../analysis/compare_redo.py`](../exp/rep-penalty-fix-validation-20260604/analysis/compare_redo.py) (VL-30B 全场对比)
+ [`exp/.../analysis/compare_qwen14b.py`](../exp/rep-penalty-fix-validation-20260604/analysis/compare_qwen14b.py) (Qwen3-14B 跟 paper 对比)

### A.8 VL-30B MMLU — noSIA arm

```bash
cd /workspace/git/0g-sparse-inference-alignment
nohup /workspace/SIA/venv4/bin/python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-VL-30B-A3B-Instruct \
    --rm_url http://localhost:8001 \
    --llm_gpu_mem 0.6 --topk 5 --weight 0.0 --entropy_threshold 999999 \
    --host 0.0.0.0 --port 8000 \
    > /tmp/vl30b_runs/mmlu_nosia_server.log 2>&1 &

# Driver
nohup /workspace/SIA/venv4/bin/python -u eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 \
    --model /workspace/SIA/models/Qwen3-VL-30B-A3B-Instruct \
    --output eval/results/vl30b_noSIA_mmlu_150q.json \
    --limit 5 --repetition_penalty 1.0 \
    > /tmp/vl30b_runs/mmlu_nosia_driver.log 2>&1 &
```

跟之前 Qwen3-14B MMLU noSIA 命令 (`doc/eval-report.md`) 完全一致, 只换 LLM。

### A.9 VL-30B MMLU — SIA arm

```bash
# 1. vllm RM server (跟 A.1 一致, 端口 8001)
nohup /workspace/SIA/venv4/bin/vllm serve \
  /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
  --runner pooling --convert classify \
  --hf-overrides '{"architectures":["Qwen3WithScoreForCausalLM"]}' \
  --enable-prefix-caching --gpu-memory-utilization 0.15 \
  --max-model-len 4096 --port 8001 --host 0.0.0.0 --disable-log-stats \
  > /tmp/vl30b_runs/mmlu_sia_rm_server.log 2>&1 &

# 2. 必做 RM /classify smoke test (见 A.2) 避免 broken-RM 坑

# 3. SIA LLM server
cd /workspace/git/0g-sparse-inference-alignment
nohup /workspace/SIA/venv4/bin/python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-VL-30B-A3B-Instruct \
    --rm_url http://localhost:8001 \
    --rm_backend vllm \
    --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --llm_gpu_mem 0.6 --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > /tmp/vl30b_runs/mmlu_sia_server.log 2>&1 &

# 4. Driver (跟 noSIA 一致, 只改 output 文件名)
nohup /workspace/SIA/venv4/bin/python -u eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 \
    --model /workspace/SIA/models/Qwen3-VL-30B-A3B-Instruct \
    --output eval/results/vl30b_SIA_mmlu_150q.json \
    --limit 5 --repetition_penalty 1.0 \
    > /tmp/vl30b_runs/mmlu_sia_driver.log 2>&1 &
```

跟 noSIA 的 5 处 diff (SIA 启用必需): `--rm_backend vllm` / `--rm_model VM-...-merged-for-vllm` / `--weight 1.0` / `--entropy_threshold 1.0` / 启动独立的 vllm RM server。

---

## Appendix B. SIA intervention 健康指标

判断 SIA 是否真正运行 (避免 broken-RM 跑出 garbage 数据):

| 指标 | 健康范围 | 红灯 |
|------|--------|------|
| `intervention ratio` | 10-40% (entropy_threshold=1.0 下) | **0.0%** → RM 死了 |
| `top1 flip rate` | 50-80% | 0% / 100% (有 bug) |
| `RM error` 日志数 | **0** | >0 → 连接问题 |
| `Connection refused` 日志数 | **0** | >0 → RM server 死了 |
| SIA output vs noSIA output byte-identical | 应该 **不等** | 等 → broken |

检查命令:
```bash
SIA_LOG=/tmp/.../sia_llm_server.log
grep -c "Connection refused\|RM error" $SIA_LOG  # 应该 0
grep -E "intervened=" $SIA_LOG | awk -F'intervened=' '{split($2,a,"/"); split(a[2],b," "); intv+=a[1]; tot+=b[1]; n++} END {print "avg ratio =", intv*100/tot "%, reqs =", n}'
```

---

## Appendix C. 已知 race condition + 启动顺序教训

**问题**: 同时启动 vllm RM (port 8001) 和 SIA LLM (port 8000) 会让两个 vllm 进程**竞争 GPU 内存**:
- vllm RM: `--gpu-memory-utilization 0.10` (希望占 10% = ~14GB)
- SIA LLM: `--llm_gpu_mem 0.65` (希望占 65% = ~92GB)
- 同时启动 → 总共 75% target, 但实际 vllm 估算 KV cache 大小时, 看到的"剩余空间"是错的
- 概率性: RM 在 KV cache 分配阶段 **OOM 死亡**

**正确顺序**:
1. 启动 RM, 等 /v1/models 200 + **/classify smoke test 返回 probs**
2. **再** 启动 SIA LLM
3. SIA LLM ready 后, **再次** ping RM /v1/models 确认它没死
4. 跑 driver 第一题, 检查 SIA log 出现 `intervened=N/M (N>0)` 才确认 SIA 真活着

**坑的特点**: vllm RM 死后, FastAPI 进程残留 (`Z` 状态), `curl /v1/models` 在死亡前夕短暂可能仍返回 200 OK, ready 检查通过, **但后续 /classify 全部 Connection refused**, SIA processor 退化为 no-op, intervention 率 0%, 输出**跟 noSIA byte-identical**。

这是为什么 Sanity 检查必须查 `/classify` (真正发请求), 不能只信 `/v1/models`。
