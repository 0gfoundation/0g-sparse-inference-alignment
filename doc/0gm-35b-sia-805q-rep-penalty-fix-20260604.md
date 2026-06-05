# 0GM-35B SIA vs noSIA AlpacaEval 805Q — rep_penalty fix 翻案 (2026-06-04)

**TL;DR**: 用 `rep_penalty=1.0` 修复 + no-think 三重保险跑 0GM-1.0-35B-A3B-0427 AlpacaEval 805Q SIA 和 noSIA, **真 apples-to-apples 配对 (n=783)**:

- **SIA mean = +16.04, noSIA mean = +10.59**
- **Δ = +5.50 (+51.6%), win 537/783 (69%), t=+12.06, p ≈ 0** ✅ 极显著

彻底翻案之前"0GM SIA 不可用 (Δ=-13.2%)"的错误结论。

---

## 1. 配置

### 1.1 三进程拓扑 (SIA arm)
| 组件 | venv | gpu_mem | 说明 |
|------|------|---------|------|
| RM (Qwen3-4B Score) | venv4 (vllm 0.19.0) | 0.15 | HTTP `vllm serve --runner pooling --convert classify` on port 8001 |
| 主 LLM (0GM-35B SIA) | venv4 (vllm 0.19.0) | 0.72 | `sia_vllm_server.py --rm_backend vllm --weight 1.0 --entropy 1.0` on port 8000 |
| Driver (AlpacaEval) | venv4 | — | `eval/alpaca_eval.py --no_think_prompt --ban_think_token --disable_thinking` |

### 1.2 二进程拓扑 (noSIA arm)
| 组件 | venv | gpu_mem | 说明 |
|------|------|---------|------|
| 主 LLM (0GM-35B noSIA) | venv4 (vllm 0.19.0) | 0.72 | `sia_vllm_server.py --rm_backend vllm --weight 0.0 --entropy 999999` on port 8000 (SIA processor 退化为 no-op, RM 永不查) |
| Driver (AlpacaEval) | venv4 | — | 跟 SIA driver 完全一致, 只改输出文件名 |

**唯一的实质差异**: SIA `--weight 1.0 --entropy_threshold 1.0` (开干预) vs noSIA `--weight 0.0 --entropy_threshold 999999` (永不干预)。Code path 完全一致 (`sia_vllm_server.py`)。

### 1.3 SIA 参数

| 参数 | 值 |
|------|---|
| `--topk` (SIA candidates) | 5 |
| `--weight` (SIA) | 1.0 (SIA arm) / 0.0 (noSIA arm) |
| `--entropy_threshold` | 1.0 (SIA arm) / 999999 (noSIA arm) |
| `--max_tokens` (driver) | 2048 |
| `--temperature` | 1.0 |
| `--top_p` | 0.95 (0GM 248K vocab 必须设, 防 OOV 漂移) |
| `--top_k` | 20 |
| **`--repetition_penalty`** | **1.0** (2026-06-04 修复, server 默认从 1.3 改为 1.0) |
| Thinking 抑制 | **`--no_think_prompt --ban_think_token --disable_thinking`** (三重保险, 0GM 训练含 thinking pattern, 单一 flag 不够) |
| `--limit` | (无, 跑全 805 题) |

### 1.4 Skywork 评分
- Model: `Skywork-Reward-V2-Llama-3.1-8B`
- Method: `apply_chat_template + AutoModelForSequenceClassification`, raw logit (无 sigmoid)
- max_length: 4096 token

---

## 2. 输出质量

| 指标 | SIA (805/805) | noSIA (805/805) |
|------|---------------|------------------|
| 完成数 | 805 / 805 ✅ | 805 / 805 ✅ |
| 错误数 | 0 ✅ | 0 ✅ |
| avg tokens / Q | 242 | 228 |
| min / max tokens | 1 / 2048 | 3 / 604 (实测无 max=2048 截断) |
| 撞 max=2048 cap | 3 / 805 (0.4%) | 0 / 805 |
| 含 `<think>` tag | **0 / 805** ✅ no-think 三重保险成功 | **0 / 805** ✅ |
| CJK 注入 | 0 / 805 | 0 / 805 |

**风格差异 (符合 SIA 设计意图)**:
- noSIA: 信息精简 (Q1 Broadway 61 tokens 1 段列表, Q15 Thanksgiving turkey 117 tokens 简述)
- SIA: 内容更详细+结构化 (Q1 Broadway 310 tokens, 5 个 actor 各一段含具体剧目+年代; Q15 含 1621 年 Pilgrims/Wampanoag 背景)

跟之前 SIA-thinkbrief 805Q (rep=1.3 bug 版本) 对比:
- 旧版: 普遍 "Thinking Process:" 浪费 2048 tokens 无真实答案 → reward 低
- 新版: 直接回答, 长度自适应

---

## 3. 性能指标

### 3.1 Driver 吞吐 (805 reqs 各自)

| 指标 | SIA arm | noSIA arm | 比率 |
|------|---------|-----------|------|
| 总 reqs | 805 | 805 | — |
| 总生成 tokens | 195,182 | 183,567 (推估 805 × 228) | — |
| 总 driver wall (per-Q elapsed 求和) | **101.9 min** | **31.2 min** | SIA 慢 **3.27×** |
| 平均 tokens / Q | 242 | 228 | — |
| 平均 wall / Q | 7.6s | 2.3s | SIA 慢 3.3× |
| **Client throughput** | **31.9 tok/s** | **98.0 tok/s** | noSIA 快 **3.07×** |

**SIA 的 slowdown 完全来自 RM HTTP 调用** (24% 步骤 × 71ms × 805 reqs)。其它 code-path 完全一致, 主 LLM forward 速度本身相同。

### 3.2 SIA Server 健康 (整个 SIA 跑共 214,262 generation steps)

| 指标 | 值 | 健康范围 |
|------|---|---------|
| **总 generation steps** | **214,262** | — |
| **Intervened steps** | **51,927** | — |
| **Intervention ratio** | **24.24%** | 10-30% 健康 ✅ |
| **Top-1 flip ratio** | **60.31%** (31,316 / 51,927) | 50-80% 健康 ✅ |
| **RM error count** | **0** | 必须 0 ✅ |
| **Connection refused** | **0** | 必须 0 ✅ |

### 3.3 SIA Server — Per-step 时间拆解 (`[SIA-pf-summary @51900]` 末尾累计统计)

| 阶段 | p50 (ms) | p95 (ms) | max (ms) | 说明 |
|------|---------|---------|---------|------|
| **SKIP step (非干预步)** | **0.73** | 1.41 | 2.22 | top-K + entropy + decision, 无 RM call |
| ↳ apply_topk_ent | 0.69 | 1.25 | 1.89 | GPU torch.topk + log_softmax |
| ↳ apply_cpu_sync | 0.08 | 0.18 | 0.31 | entropy 拉回 CPU |
| ↳ format_chat | 0.10 | 0.22 | 0.75 | prompt 构造 |
| **INTERVENE step (干预步)** | (apply_total p95 ≈ 95.7) | 95.7 | 118.2 | 含 RM call 的总耗时 |
| ↳ **http_post (RM /classify call)** | **70.78** | **99.14** | 116.05 | **干预成本主要瓶颈** |
| ↳ parse_response (JSON 解析) | 0.18 | 0.30 | 0.38 | 几乎免费 |
| ↳ intv_prepare (索引/数据准备) | 0.13 | 0.22 | 0.37 | 几乎免费 |
| ↳ intv_apply_logits (改 logits) | 1.13 | 1.30 | 1.72 | 几乎免费 |

**核心观察**:
- INTERVENE step 比 SKIP step **慢 ~95×** (71ms vs 0.73ms), 95%+ 成本在 RM HTTP call
- 加权平均 step ≈ 24.24% × 71ms + 75.76% × 0.73ms = **17.8 ms / token**
- 加上主 LLM forward (~14ms), driver 端 31.4 ms / token, 跟实测 31.9 tok/s 吻合

### 3.4 跟同期 VL-30B AlpacaEval (HTTP RM) 性能对比

| | 0GM-35B 805Q (本次) | VL-30B 200Q (max=2048) |
|---|---|---|
| 主 LLM 模型 | 35B MoE (3B active) | 30B MoE (3B active) |
| Intervention 率 | 24.24% | 25.83% |
| Flip 率 | 60.31% | 66.88% |
| RM call p50 (http_post) | **70.78ms** | 71.18ms |
| SKIP step p50 | 0.73ms | 3.73ms (VL-30B SKIP path 较慢) |
| Client throughput | 31.9 tok/s | 35.9 tok/s |

两者 RM 调用成本几乎完全一致 (都 vllm RM HTTP 路径), 主 LLM 速度也接近 (3B active MoE)。

---

## 4. Skywork 评分结果

### 4.1 总体 mean

| arm | n | mean | median | std |
|-----|---|------|--------|-----|
| **SIA 805Q (rep=1.0 fix, no-think)** | 799 | **+16.04** | +16.25 | 12.21 |
| **noSIA 805Q (rep=1.0, no-think, apples-to-apples)** | 787 | **+10.59** | +10.81 | 12.10 |

n < 805 是因为 6 (SIA) / 18 (noSIA) 个 record 输出为空或非常短, 评分跳过。

### 4.2 配对比较 (n=783, 真 apples-to-apples)

| 指标 | 值 |
|------|---|
| **Δmean** | **+5.499** |
| **relative gain** | **+51.6%** |
| **Win** | **537 / 783 (69%)** |
| t-statistic | **+12.06** |
| **p-value** | **≈ 0** (实际 < 10⁻²⁵) |

配对方法: 按 `instruction` 字段精确匹配, 两 arm 都有 reward 才计入。

### 4.3 跟历史 0GM-35B SIA 实验对比

| 实验 | rep_penalty | thinking 抑制 | 结果 |
|------|------------|--------------|------|
| 旧 0GM 805Q SIA (broken) | server **1.3** (bug) | `--brief_instruction` 仅 | mean +11.16, **Δ = -13.2% vs noSIA** ❌ |
| **本次 0GM 805Q SIA (FIXED)** | **1.0** | **`--no_think_prompt --ban_think_token --disable_thinking`** | mean **+16.04**, **Δ = +51.6% vs noSIA (apples-to-apples)** ✅ |

**`rep_penalty` 修复在 0GM-35B 上的效果**: 跟 Qwen3-14B (+20%), VL-30B (+10~20%) 一致, **SIA 在 0GM 上也有显著正向 gain, 甚至是所有模型里相对 gain 最大的**。

---

## 5. 结论

### 5.1 推翻先前结论
之前 `doc/sia-fix-and-vm-ablation-0gm-35b-20260602.md` 写 "0GM-35B SIA Δ = -13.2%, 不可用"。**这个结论是错的**, 真因是两层污染:
1. `repetition_penalty=1.3` server 默认 (今天修复) — 在所有 SIA 路径上系统性破坏 RM 信号
2. `--brief_instruction` 不足以抑制 0GM thinking, 答案被 2048 tok 截断 → Skywork 低分

### 5.2 修复路径
1. **代码侧**: `src/sia_vllm_server.py:164,347` 默认 `repetition_penalty=1.0` (commit f0f4e1c, 2026-06-04)
2. **Driver 侧**: 显式传 `repetition_penalty=1.0` 双重保险
3. **0GM 特殊**: thinking 抑制必须用 **三重保险** `--no_think_prompt --ban_think_token --disable_thinking` (`--brief_instruction` 不够强)

### 5.3 跨模型 SIA gain 一致性 (rep_penalty=1.0 修复后, 全 apples-to-apples paired)

| 实验 | n_paired | Δ vs noSIA | rel | p |
|------|--|----|-----|---|
| Paper Qwen3-14B (baseline) | 805 | +1.63 | +13% | (paper) |
| Ours Qwen3-14B (max=256) | 200 | +1.89 | +20% | <10⁻⁴ |
| Ours VL-30B (max=256) | 200 | +2.32 | +20% | <10⁻⁴ |
| Ours VL-30B (max=2048) | 200 | +2.75 | +10% | <10⁻⁷ |
| **Ours 0GM-35B (805Q)** | **783** | **+5.50** | **+51.6%** | **~0** ⭐ |

0GM-35B 的相对 gain 最大: 因为 0GM noSIA baseline (mean +10.59) 较低 (VL-30B noSIA mean +27.4 多), 同样的绝对 Δ 在小 baseline 上 rel 更大。

### 5.4 速度代价 (RM HTTP 调用主导)

| | SIA | noSIA | 倍数 |
|--|-----|-------|------|
| Throughput (driver tok/s) | 31.9 | 98.0 | noSIA 快 3.07× |

SIA 的成本几乎完全来自 24% 步骤里 71ms 的 RM HTTP call。如果未来能把 RM 调用降到 17ms (b2 inproc 量级), SIA throughput 可恢复到 ~85 tok/s — 但 0GM 上 b2 inproc 受 vllm 0.19 cudagraph 全局 flag 阻断, 暂不可行 (详见 `doc/0gm-35b-sia-rm-inproc-path-20260602.md`)。

---

## Appendix A. 完整启动命令

所有日志已 archive 到 [`exp/0gm-35b-sia-805q-rep-penalty-fix-20260604/`](../exp/0gm-35b-sia-805q-rep-penalty-fix-20260604/)。

### A.1 SIA arm

**Step 1: vllm RM (Qwen3-4B Score) HTTP server, port 8001**
```bash
mkdir -p /tmp/0gm_runs
nohup /workspace/SIA/venv4/bin/vllm serve \
    /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --runner pooling --convert classify \
    --hf-overrides '{"architectures":["Qwen3WithScoreForCausalLM"]}' \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.15 \
    --max-model-len 4096 \
    --port 8001 --host 0.0.0.0 \
    --disable-log-stats \
    > /tmp/0gm_runs/rm_4B_server.log 2>&1 &
```
**日志**: [`exp/.../rm_4B_server.log`](../exp/0gm-35b-sia-805q-rep-penalty-fix-20260604/rm_4B_server.log)

**Step 2 (必做): RM `/classify` smoke test**
```bash
until curl -s --max-time 2 http://localhost:8001/v1/models 2>/dev/null | grep -q object; do sleep 3; done
curl -s --max-time 10 -X POST http://localhost:8001/classify \
  -H "Content-Type: application/json" \
  -d '{"model":"/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm","input":[[1,2,3,4,5]]}'
# 期望返回 {"id":...,"data":[{"index":0,"probs":[...]}], ...}
```

**Step 3: 0GM-35B SIA LLM server, port 8000**
```bash
nohup /workspace/SIA/venv4/bin/python \
    /workspace/git/0g-sparse-inference-alignment/src/sia_vllm_server.py \
    --llm /workspace/SIA/models/0GM-1.0-35B-A3B-0427 \
    --rm_backend vllm \
    --rm_url http://localhost:8001 \
    --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --llm_gpu_mem 0.72 --max_model_len 4096 \
    --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > /tmp/0gm_runs/sia_llm_server.log 2>&1 &
```
**日志**: [`exp/.../sia_llm_server.log`](../exp/0gm-35b-sia-805q-rep-penalty-fix-20260604/sia_llm_server.log) (含全部 `[SIA-pf-summary]` 性能数据 + 全 805 req 的 `intervened=/top1_flip=` 行)

**Step 4: Driver 805Q (no-think 三重保险)**
```bash
nohup /workspace/SIA/venv4/bin/python \
    /workspace/git/0g-sparse-inference-alignment/eval/alpaca_eval.py \
    --base_url http://localhost:8000/v1 \
    --model 0GM-1.0-35B-A3B-0427 \
    --max_tokens 2048 \
    --temperature 1.0 --top_p 0.95 --top_k 20 --repetition_penalty 1.0 \
    --no_think_prompt --ban_think_token --disable_thinking \
    --output /tmp/0gm_runs/sia_0gm_805q_max2048_nothink.json \
    > /tmp/0gm_runs/driver_nothink.log 2>&1 &
```
**输出 (含 Skywork rewards)**: [`exp/.../sia_805q_outputs_scored.json`](../exp/0gm-35b-sia-805q-rep-penalty-fix-20260604/sia_805q_outputs_scored.json)

### A.2 noSIA arm (跟 SIA byte-exact 一致, 仅 2 处差异)

**Step 1: noSIA SIA-server, port 8000 (无 vllm RM 启动)**
```bash
nohup /workspace/SIA/venv4/bin/python \
    /workspace/git/0g-sparse-inference-alignment/src/sia_vllm_server.py \
    --llm /workspace/SIA/models/0GM-1.0-35B-A3B-0427 \
    --rm_backend vllm \
    --rm_url http://localhost:8001 \
    --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --llm_gpu_mem 0.72 --max_model_len 4096 \
    --topk 5 --weight 0.0 --entropy_threshold 999999 \
    --host 0.0.0.0 --port 8000 \
    > /tmp/0gm_runs/nosia_llm_server.log 2>&1 &
```
跟 SIA Step 3 的 diff: `--weight 0.0 --entropy_threshold 999999` (SIA processor 退化为 no-op, 永不查 RM)。`--rm_backend vllm` / `--rm_url` / `--rm_model` 保留, 让 code path 完全一致 (server 启动时一条 "RM status query failed" warning 是预期的, 不影响后续 noSIA 推理)。

**日志**: [`exp/.../nosia_llm_server.log`](../exp/0gm-35b-sia-805q-rep-penalty-fix-20260604/nosia_llm_server.log)

**Step 2: Driver 805Q (跟 SIA driver 完全一致, 只改输出文件名)**
```bash
nohup /workspace/SIA/venv4/bin/python \
    /workspace/git/0g-sparse-inference-alignment/eval/alpaca_eval.py \
    --base_url http://localhost:8000/v1 \
    --model 0GM-1.0-35B-A3B-0427 \
    --max_tokens 2048 \
    --temperature 1.0 --top_p 0.95 --top_k 20 --repetition_penalty 1.0 \
    --no_think_prompt --ban_think_token --disable_thinking \
    --output /tmp/0gm_runs/nosia_0gm_805q_max2048_nothink.json \
    > /tmp/0gm_runs/driver_nosia.log 2>&1 &
```
**输出 (含 Skywork rewards)**: [`exp/.../nosia_805q_outputs_scored.json`](../exp/0gm-35b-sia-805q-rep-penalty-fix-20260604/nosia_805q_outputs_scored.json)

### A.3 Skywork 评分 + 配对对比

跑前 kill 两个 server 释放 GPU (Skywork 需要 ~16GB):
```bash
pkill -9 -f 'EngineCore' 2>/dev/null
pkill -9 -f 'sia_vllm_server' 2>/dev/null
pkill -9 -f 'vllm serve' 2>/dev/null
sleep 8

# 跑评分脚本
cd /tmp/0gm_runs
/workspace/SIA/venv4/bin/python -u score_pair.py
```

**评分脚本**: [`exp/.../score_pair.py`](../exp/0gm-35b-sia-805q-rep-penalty-fix-20260604/score_pair.py)
**评分 log**: [`exp/.../score_pair.log`](../exp/0gm-35b-sia-805q-rep-penalty-fix-20260604/score_pair.log)

---

## Appendix B. SIA intervention 健康检查命令

```bash
SIA_LOG=exp/0gm-35b-sia-805q-rep-penalty-fix-20260604/sia_llm_server.log

# 检查 RM 是否有 error (must be 0)
grep -c "Connection refused\|RM error" $SIA_LOG

# 总 intervention ratio
grep 'intervened=' $SIA_LOG | awk -F'intervened=' \
  '{split($2,a,"/"); split(a[2],b," "); intv+=a[1]; tot+=b[1]; n++} \
   END {printf "Reqs: %d  steps: %d  intv: %d (%.2f%%)\n", n, tot, intv, intv*100/tot}'

# 总 flip ratio
grep 'top1_flip=' $SIA_LOG | awk -F'top1_flip=' \
  '{split($2,a,"/"); split(a[2],b," "); flip+=a[1]; total+=b[1]} \
   END {if (total>0) printf "Flip: %d/%d = %.2f%%\n", flip, total, flip*100/total}'

# 最近 SIA-pf-summary
grep 'SIA-pf-summary' $SIA_LOG | tail -1
```

---

## Appendix C. 模型 + 数据集 paths

- 主 LLM: `/workspace/SIA/models/0GM-1.0-35B-A3B-0427` (Qwen3_5MoeForConditionalGeneration, 35B / 3B active MoE)
- VM (Value Model): `/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm` (Qwen3-4B + score head merged for vllm)
- Skywork RM: `/workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B`
- AlpacaEval dataset: `/workspace/SIA/data/alpaca_eval/alpaca_eval.json` (805 instructions)
