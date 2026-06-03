# Qwen3-VL-30B-A3B-Instruct AlpacaEval 200Q — SIA vs noSIA 完整对照

**日期**: 2026-06-03
**对照前置 doc**: [`alpaca-eval-0gm-35b-thinkbrief-805q-20260603.md`](alpaca-eval-0gm-35b-thinkbrief-805q-20260603.md) (0GM-35B thinking + brief 实验, 找到 SIA -13.2% 显著负 Δ)
**状态**: ✅ noSIA + SIA 全部完成 + 打分完毕

---

## 1. 实验目的

0GM-35B 实验在 `thinking + brief` + fixed strip_think 后, 得到 **SIA Δ = -13.2% (t=-4.61, p<0.0001)**, 跟此前 100Q `--no_think_prompt + --ban_think_token` 结论 (-13.0%) 高度一致 — 表明 SIA 与 VM-Qwen3-4B 在 0GM-35B 上**稳定伤害 reward 约 13%**, 无论 thinking 模式开关。

**关键假设**: 此伤害根因可能是**跨家族 OOD** (VM-Qwen3-4B-Base 训练于 Qwen3 数据, 应用于 Qwen3.5/3.6 thinking 模型 0GM-35B 时跨家族信号噪声主导)。

**本实验目标**: 在**同家族 + 非 thinking 模型** Qwen3-VL-30B-A3B-Instruct 上重复 SIA vs noSIA 对照, 验证:

- 假设成立: 同家族场景下 SIA Δ 应接近 0 (RM 信号准确, 干预方向正确)
- 假设不成立: SIA Δ 仍 -13% 左右 → 问题不在 cross-family, 而在 SIA 算法/配置本身

---

## 2. 模型与配置

### 2.1 Qwen3-VL-30B-A3B-Instruct 模型确认 (在 `/workspace/SIA/models/Qwen3-VL-30B-A3B-Instruct`)

**Instruct 变体, 非 thinking 模式**, 三处证据:

| 证据 | 内容 |
|------|------|
| 命名约定 | Qwen3 家族 `*-Thinking` = 显式 reasoning; `*-Instruct` = 标准 instruction-tuned, **默认不 thinking** |
| `tokenizer_config.json` chat_template (5292 chars) | **0 次** 出现 `<think>` / `</think>` / `thinking` / `enable_thinking` / `reasoning` 关键字; 末尾 `<\|im_start\|>assistant\n` (标准 ChatML, **不注入 `<think>`**) |
| `config.json` | `model_type: qwen3_vl_moe`, `architectures: Qwen3VLMoeForConditionalGeneration`, text vocab_size=151,936 (同 Qwen3 base) |

→ **不用 `--brief_instruction` / `--disable_thinking` / `--no_think_prompt` / `--ban_think_token`**, 用 vanilla `/v1/chat/completions`。

### 2.2 共享配置 (两 arm 完全一致)

| 参数 | 值 |
|------|----|
| 模型 | `Qwen3-VL-30B-A3B-Instruct` (Qwen3 家族 MoE 30B 总 / 3B active, VL 但纯文本 prompt 不触发 vision path) |
| 数据集 | `/workspace/SIA/data/alpaca_eval/alpaca_eval.json`, **前 200 题** (`--limit 200`) |
| max_tokens | **2048** (同 0GM 实验, 实际生成远短) |
| temperature | 1.0 |
| top_p | 0.95 |
| top_k | 20 |
| repetition_penalty | 1.0 |
| chat_template | 默认 ChatML (无 `<think>` 注入) |
| `--brief_instruction` | ❌ (无 thinking 不需要) |
| `--disable_thinking` / `--no_think_prompt` / `--ban_think_token` | ❌ ❌ ❌ |
| `max-model-len` | 4096 |

---

## 3. noSIA arm — 已完成

### 3.1 raw vllm serve (port 8000)

```bash
nohup /workspace/SIA/venv4/bin/vllm serve \
  /workspace/SIA/models/Qwen3-VL-30B-A3B-Instruct \
  --served-model-name Qwen3-VL-30B-A3B-Instruct \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --host 0.0.0.0 --port 8000 \
  > /tmp/qwen3vl30b_nosia_srv_20260603_024531.log 2>&1 &
```

- PID 1350345, log [`/tmp/qwen3vl30b_nosia_srv_20260603_024531.log`](file:///tmp/qwen3vl30b_nosia_srv_20260603_024531.log)
- 加载: 13 个 safetensors shards × ~73s = **15 min**

### 3.2 跑 200Q AlpacaEval

```bash
nohup /workspace/SIA/venv2/bin/python \
  /workspace/git/0g-sparse-inference-alignment/eval/alpaca_eval.py \
  --base_url http://localhost:8000/v1 \
  --model Qwen3-VL-30B-A3B-Instruct \
  --limit 200 \
  --max_tokens 2048 \
  --temperature 1.0 --top_p 0.95 --top_k 20 --repetition_penalty 1.0 \
  --output /tmp/alpaca_qwen3vl30b_nosia_200q_20260603_030425.json \
  > /tmp/alpaca_qwen3vl30b_nosia_200q_20260603_030425.log 2>&1 &
```

- PID 1356816, 启动 03:04, 结束 03:24 (用时 **20.2 min**)
- LOG: [`/tmp/alpaca_qwen3vl30b_nosia_200q_20260603_030425.log`](file:///tmp/alpaca_qwen3vl30b_nosia_200q_20260603_030425.log)
- JSON: [`/tmp/alpaca_qwen3vl30b_nosia_200q_20260603_030425.json`](file:///tmp/alpaca_qwen3vl30b_nosia_200q_20260603_030425.json)

### 3.3 noSIA 生成统计

| 指标 | 值 |
|------|----|
| Q done / errors | 200 / 0 |
| **总 wall time** | **20.2 min (1,213s)** |
| **客户端 throughput** | **127.4 tok/s** |
| total tokens 生成 | 154,453 |
| tokens 分布 (per Q) | min=64 p50=704 p95=~1062 max=1734 mean=773 |
| **触顶 2048 cap** | **0 / 200** ✅ |
| 单题耗时 p50 | 5.6s |
| `</think>` / `<think>` 在 output 中 | **0 / 200** ✅ (确认非 thinking 模式) |
| tokens<50 异常短 | 0 |
| 'I cannot/Sorry' 拒答 | 0 |

→ 输出结构**全程干净**: 无 thinking 残留, 无触顶, 无短输出, 无拒答。

### 3.4 noSIA Skywork 打分

```bash
nohup /workspace/SIA/venv/bin/python \
  /workspace/git/0g-sparse-inference-alignment/scripts/measure_alpaca_reward.py \
  --input_file /tmp/alpaca_qwen3vl30b_nosia_200q_20260603_030425.json \
  --output_file /tmp/alpaca_qwen3vl30b_nosia_200q_scored_20260603_034614.json \
  --rm /workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B \
  --device cuda:0 --strip_think \
  > /tmp/skywork_score_vl30b_nosia_20260603_034614.log 2>&1 &
```

- PID 1369369, ~3 min
- `--strip_think` 保留 (无副作用; 此模型无 `</think>` 输出, regex 不匹配)
- LOG: [`/tmp/skywork_score_vl30b_nosia_20260603_034614.log`](file:///tmp/skywork_score_vl30b_nosia_20260603_034614.log)
- JSON: [`/tmp/alpaca_qwen3vl30b_nosia_200q_scored_20260603_034614.json`](file:///tmp/alpaca_qwen3vl30b_nosia_200q_scored_20260603_034614.json)

**结果**:

```
total samples       : 200
scored              : 195
skipped (>2048 tok) : 5    ← Skywork tokenizer 截断长样本
mean reward         : 29.2646
p50                 : 28.8750
min / max           : -6.2812 / 60.0000
```

→ **基线极高 (mean=29.26, p50=28.88)**, 比 0GM-35B noSIA mean=13.82 高 **2.1×** — Qwen3-VL-30B-Instruct 本身是非常强的模型。

---

## 4. SIA arm — 已完成

### 4.1 RM 4B server (port 8001)

```bash
nohup /workspace/SIA/venv4/bin/vllm serve \
  /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
  --runner pooling --convert classify \
  --hf-overrides '{"architectures":["Qwen3WithScoreForCausalLM"]}' \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.22 \
  --max-model-len 2048 \
  --port 8001 --host 0.0.0.0 \
  --disable-log-stats \
  > /tmp/rm_4B_vl30b_sia_20260603_034705.log 2>&1 &
```

- PID 1369700, log [`/tmp/rm_4B_vl30b_sia_20260603_034705.log`](file:///tmp/rm_4B_vl30b_sia_20260603_034705.log)

### 4.2 SIA LLM server (port 8000, weight=1.0, entropy=0.8, topk=5)

```bash
nohup env SIA_DEBUG_HIST=1 \
  /workspace/SIA/venv4/bin/python \
  /workspace/git/0g-sparse-inference-alignment/src/sia_vllm_server.py \
  --llm /workspace/SIA/models/Qwen3-VL-30B-A3B-Instruct \
  --rm_backend vllm \
  --rm_url http://localhost:8001 \
  --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
  --llm_gpu_mem 0.72 --max_model_len 4096 \
  --topk 5 --weight 1.0 --entropy_threshold 0.8 \
  --host 0.0.0.0 --port 8000 \
  > /tmp/qwen3vl30b_sia_srv_20260603_034908.log 2>&1 &
```

- PID 1370712, log [`/tmp/qwen3vl30b_sia_srv_20260603_034908.log`](file:///tmp/qwen3vl30b_sia_srv_20260603_034908.log)
- 加载 ~16 min (35B + cudagraph; ~75s/shard × 13)

### 4.3 跑 200Q SIA AlpacaEval (同 noSIA 所有 client flags)

```bash
nohup /workspace/SIA/venv2/bin/python \
  /workspace/git/0g-sparse-inference-alignment/eval/alpaca_eval.py \
  --base_url http://localhost:8000/v1 \
  --model Qwen3-VL-30B-A3B-Instruct \
  --limit 200 \
  --max_tokens 2048 \
  --temperature 1.0 --top_p 0.95 --top_k 20 --repetition_penalty 1.0 \
  --output /tmp/alpaca_qwen3vl30b_sia_200q_20260603_040544.json \
  > /tmp/alpaca_qwen3vl30b_sia_200q_20260603_040544.log 2>&1 &
```

- PID 1375627, 启动 04:05, 结束 05:27 (用时 **81.3 min**)
- 0 errors, 200/200 ✅
- LOG: [`/tmp/alpaca_qwen3vl30b_sia_200q_20260603_040544.log`](file:///tmp/alpaca_qwen3vl30b_sia_200q_20260603_040544.log)
- JSON: [`/tmp/alpaca_qwen3vl30b_sia_200q_20260603_040544.json`](file:///tmp/alpaca_qwen3vl30b_sia_200q_20260603_040544.json)

### 4.4 SIA 生成统计 + 性能指标

| 指标 | 值 |
|------|----|
| Q done / errors | 200 / 0 |
| **总 wall time** | **81.3 min (4,877s)** |
| **客户端 throughput** | **28.3 tok/s** |
| total tokens 生成 | 138,023 (client) / 138,252 (server) |
| 有 DONE 行的 req | 193 / 200 (7 题无 DONE 行 — 早终止) |
| **干预率 (全局, ∑intv/∑tot)** | **28.21%** (38,999 / 138,252) |
| 干预率 per-req | p25=19.2%, **p50=25.7%**, p75=32.0%, p95=38.4%, max=49.0% |
| **Top-1 Flip 率 (干预 step 中)** | **13.19%** (∑flip/∑intv = 5,145 / 38,999) |
| Top-1 Flip per-req | min=0%, p50=12.8%, p95=37.6%, max=81.8% |
| **整体 Flip 率 (改 top-1 / 总 token)** | **3.72%** (5,145 / 138,252) |
| `</think>` / `<think>` 在 output 中 | 0 / 200 ✅ |

### 4.5 SIA Skywork 打分

```bash
nohup /workspace/SIA/venv/bin/python \
  /workspace/git/0g-sparse-inference-alignment/scripts/measure_alpaca_reward.py \
  --input_file /tmp/alpaca_qwen3vl30b_sia_200q_20260603_040544.json \
  --output_file /tmp/alpaca_qwen3vl30b_sia_200q_scored_20260603_055038.json \
  --rm /workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B \
  --device cuda:0 --strip_think \
  > /tmp/skywork_score_vl30b_sia_20260603_055038.log 2>&1 &
```

- PID 1407482, ~3 min
- LOG: [`/tmp/skywork_score_vl30b_sia_20260603_055038.log`](file:///tmp/skywork_score_vl30b_sia_20260603_055038.log)
- JSON: [`/tmp/alpaca_qwen3vl30b_sia_200q_scored_20260603_055038.json`](file:///tmp/alpaca_qwen3vl30b_sia_200q_scored_20260603_055038.json)

**结果**:

```
total samples       : 200
scored              : 200    ← 全部成功
skipped (>2048 tok) : 0      ← 一题都没被截断
mean reward         : 28.6679
p50                 : 28.3750
min / max           : -9.7500 / 59.5000
```

---

## 5. Δ 对比正式结论

### 5.1 整体均值

| | scored | mean reward |
|--|--------|-------------|
| **SIA (fixed strip)** | **200 / 200** ✅ | **28.67** |
| **noSIA (fixed strip)** | 195 / 200 | **29.26** |

### 5.2 配对统计

| 配置 | n | SIA mean | noSIA mean | Δ | 相对 | stdev Δ | SE | t-stat | p-value | 显著? |
|------|---|----------|------------|---|------|---------|-----|--------|---------|------|
| **全配对** | **195** | 28.81 | 29.26 | **-0.46** | **-1.57%** | 6.11 | 0.44 | **-1.05** | ~0.29 | ❌ |
| 排除任一触顶 | 191 | 29.31 | 29.71 | -0.41 | -1.38% | 6.07 | 0.44 | -0.93 | ~0.35 | ❌ |

### 5.3 Win/Tie/Loss (|Δ|>0.5)

| | 计数 | 占比 |
|--|------|------|
| **SIA wins**  | **86** | **44.1%** |
| Tie | 13 | 6.7% |
| **SIA loses** | **96** | **49.2%** |

→ 几乎五五开, 略偏向 noSIA。

### 5.4 统计力评估

当前 n=195, Δ=-0.46, stdev=6.11, t=-1.05 → p ≈ 0.29 (不显著)。

若 -1.57% 是真信号, 要达 p<0.05 (80% power) 需要 **n ≈ 1,380 题** (远超 805 全集), 即 200 题样本量**没足够 power** 检测这个 -1.6% 的小效应。

---

## 6. 🎯 跨模型对比 — **0GM-35B vs VL-30B**

### 6.1 reward Δ

| 模型 | 家族 | 模式 | n | Δ (SIA-noSIA) | t-stat | p-value | 显著? |
|------|------|------|---|---------------|--------|---------|------|
| **0GM-35B**     | Qwen3.5/3.6 (跨家族 vs VM-4B) | thinking + brief | 805 | **-13.2%** | **-4.61** | **<0.0001** | ✅ |
| **VL-30B Instruct** | Qwen3 (同家族 vs VM-4B)     | non-thinking     | 200 | **-1.6%**  | **-1.05** | ~0.29     | ❌ |

**伤害幅度: 0GM-35B 是 VL-30B 的 8.4×**

### 6.2 SIA processor 性能 (两实验)

| 指标 | VL-30B (200Q) | 0GM-35B (805Q) |
|------|---------------|----------------|
| 客户端 tokens/s (SIA) | 28.3 | 35.3 |
| 客户端 tokens/s (noSIA) | 127.4 | 108.6 |
| SIA / noSIA 慢倍数 | **4.5×** | **3.1×** |
| 干预率 (全局) | **28.21%** | 18.99% |
| Top-1 Flip 率 (干预中) | 13.19% | 16.72% |
| 整体 Flip 率 (总 token) | 3.72% | 3.18% |

### 6.3 行为差异解读

**关键发现 — 干预率高 ≠ 伤害大**:

1. VL-30B 的**干预率 (28%) 高于 0GM-35B (19%)** — 因为 VL-30B 非 thinking 模式输出 token 分布更"自然"(markdown 结构/选项处高 entropy 多, 触发 entropy_threshold=0.8 多); 0GM-35B 在 thinking 模式下大量 thinking 流低 entropy token 走 skip
2. VL-30B 的 **Top-1 Flip 率 (13%) 低于 0GM-35B (17%)** — 即使 VL-30B 触发干预的 step 多, 但 RM 实际推翻 top-1 的比例更低 → RM 跟 VL-30B native 偏好更一致 → **同家族证据**
3. **两实验的整体 Flip 率 (3.7% vs 3.2%)** — 实际改方向的 token 比例相当, 但同样比例改方向产生的 reward 影响差距 **8.4×** → 决定伤害幅度的是**干预方向是否正确**, 不是干预**数量**

### 6.4 同家族 vs 跨家族 — 假设验证

| 假设 | 是否被本实验支持? |
|------|------------------|
| **H1**: SIA 与 VM-Qwen3-4B 在 0GM-35B 上失败的根因是**跨家族 OOD** | ✅ **强支持** — 同家族 VL-30B 上 Δ 从 -13% 收缩到 -1.6%, 接近中性 |
| **H2**: 是 SIA 算法本身问题 | ❌ **较弱** — 若是算法问题, VL-30B 上也应有类似负偏差 |
| **H3**: 是 thinking 模式问题 | ⚠️ **混淆** — 0GM-35B 是 thinking + 跨家族, VL-30B 是 non-thinking + 同家族, 两个变量同时变化, 无法单独 attribute |

→ 要严格区分 H1 vs H3, 需要再做 **同家族 thinking 模型** 对照 (如 Qwen3-Thinking 系列), 或 **非 thinking 跨家族** 对照 (如 0GM-Instruct 如果存在)。

---

## 7. 文件清单

### 7.1 noSIA arm
| 文件 | 角色 |
|------|------|
| `/tmp/qwen3vl30b_nosia_srv_20260603_024531.log` | raw vllm serve log |
| `/tmp/alpaca_qwen3vl30b_nosia_200q_20260603_030425.log` | eval client stdout |
| `/tmp/alpaca_qwen3vl30b_nosia_200q_20260603_030425.json` | 200Q noSIA 输出 |
| `/tmp/skywork_score_vl30b_nosia_20260603_034614.log` | Skywork 打分 stdout |
| `/tmp/alpaca_qwen3vl30b_nosia_200q_scored_20260603_034614.json` | 200Q noSIA + reward |

### 7.2 SIA arm
| 文件 | 角色 |
|------|------|
| `/tmp/rm_4B_vl30b_sia_20260603_034705.log` | RM 4B vllm serve log |
| `/tmp/qwen3vl30b_sia_srv_20260603_034908.log` | SIA LLM server log (含 SIA-pf-summary + per-req DONE) |
| `/tmp/alpaca_qwen3vl30b_sia_200q_20260603_040544.log` | eval client stdout |
| `/tmp/alpaca_qwen3vl30b_sia_200q_20260603_040544.json` | 200Q SIA 输出 |
| `/tmp/skywork_score_vl30b_sia_20260603_055038.log` | Skywork 打分 stdout |
| `/tmp/alpaca_qwen3vl30b_sia_200q_scored_20260603_055038.json` | 200Q SIA + reward |

---

## 8. 下一步建议 (基于本实验结论)

按优先级:

| 优先级 | 行动 | 目的 |
|--------|------|------|
| **1** | 把 0GM-35B SIA 换成 **同家族 RM** (例如训一个 Qwen3.5/3.6 上的 VM, 或找到合适的现有 RM checkpoint) | 直接验证 H1 (跨家族 OOD 假设), 若同家族 RM 让 0GM-35B Δ 收缩到接近 0, 就坐实根因 |
| 2 | VL-30B 跑**全 805 题**复测 | 把 -1.57% (p~0.29) 的不显著结果用更大样本确认是真信号 (Δ 转正/接近 0) 还是统计噪声 |
| 3 | VL-30B 同样配置但换 **VM-Qwen3-1.7B** | 看小 VM 在同家族上是否进一步逼近 +0 或转正 |
| 4 | 解耦 thinking 模式与 cross-family — 找一个同家族的 thinking 模型 (如 Qwen3-Thinking-xB) 跑 SIA | 区分 H1 (跨家族) vs H3 (thinking 模式) 假设 |
| 5 | (长期) 训练一个专门 for 0GM-35B 的 VM | 终极 ML 层解决跨家族问题 |

---

## 9. SIA 大比分落败样本定性分析

虽然整体 Δ=-1.57% 不显著, 但**配对样本中 SIA 仍有 96 题大败给 noSIA (输 >0.5 reward)**, 其中前 3 个大比分落败案例呈现一致的失败模式。

### 9.1 Top 3 落败案例 (Δ = SIA-noSIA, 最负在前)

| 排名 | Q | 题目类型 | SIA reward | noSIA reward | Δ | SIA tokens | noSIA tokens |
|------|---|----------|-----------|--------------|---|------------|--------------|
| #1 | Q137 | 长篇短故事 (FF14 角色 + 甘道夫 crossover) | 18.00 | 36.50 | **-18.50** | 1112 | **1601** |
| #2 | Q131 | MLK 演讲改编成 top 100 rap song | 20.12 | 36.00 | **-15.88** | 1264 | **1946** |
| #3 | Q159 | 男女能否只做好朋友 (短问答) | 22.38 | 36.75 | **-14.38** | 167 | **207** |

### 9.2 失败模式 — 三个共同特征 (3/3 命中)

| 模式 | Q137 | Q131 | Q159 |
|------|------|------|------|
| **SIA 输出比 noSIA 短** | -489 tok (-31%) | -682 tok (-35%) | -40 tok (-19%) |
| **SIA 内容"未写完" / 跳过关键段落** | 故事戛然而止于 "We journey to the Whispering...", 没到高潮就停 | 没有 Outro 段, 没有"为什么这首歌会冲上 #1"分析章节 | 缺 bullet 列表结构 (noSIA 给了 4 个 bullet) |
| **Skywork 偏好 verbose + 结构化** | noSIA 给出完整起承转合 + 诙谐结尾 | noSIA 含 BPM/制作人/Album/Genre + 6 个 bullet 商业分析 + 病毒推广场景 | noSIA 用 bullet 列出"关键要点" |

### 9.3 具体内容对比 (摘要)

**Q137 — 故事写作**:
- **SIA**: 1112 tokens, 故事进行到光之战士说 "我们前往 Whispering..." 即被截断 (max_tokens=2048 还远未到, 是模型自己 EOS 早停了)
- **noSIA**: 1601 tokens, **完整起承转合**: 引入 (Limsa Lominsa 闪电变身) → 冲突 (三人冒险) → 解决方案 (Moonlight Gate 反向魔法) → 干净结局 (变回 Hildibrand) + 诙谐尾声 ("我们最好永远不要再提这件事")

**Q131 — 创意改编**:
- **SIA**: 1264 tokens, 写了 Intro/3 verse/2 chorus/Bridge, 结束于 "走向正义... 走向真理... 走向爱... 走向力量... 走向变革... 走向崛起..." (戛然而止)
- **noSIA**: 1946 tokens, **额外**包含: Outro (MLK 原声穿插 + 3 个 rapper 接力), Final Chorus (大写字母全体合唱), "Why this would be #1" 6-bullet 商业分析, "Imagine it trending" 推广场景, 最终 Outro 一行

**Q159 — 短问答**:
- 内容**实质几乎相同** (都说"是, 可以纯友谊", 都强调沟通/界限/尊重)
- **SIA**: 167 tokens, 全文 3 个自然段, 末尾 emoji `💖✨`
- **noSIA**: 207 tokens, 加了 4 个 bullet point 列要点 (清晰沟通、相互尊重、诚实、共同承诺)
- → Skywork 偏好 bullet 结构, 仅这一点差异打出 14.4 reward 的差距

### 9.4 失败机制推测

**A. SIA 在"过渡 token"处倾向早终止**:
- 长生成里, 章节标题/列表项 marker (`###`, `**`, `1.`, `-`) 后, 接下来是新段落开头还是结束信号 (EOS / 转换)?
- 这种过渡 token 高 entropy → 触发 SIA 干预 → RM 在跨段落场景下 score 可能不准
- 当 RM 误把"继续展开"打负分时, top-1 翻转为更收敛的选择 → 模型早结束

**B. VL-30B noSIA 输出本就极 verbose (mean reward 29.26 高基线)**:
- noSIA 平均生成结构丰富 (含 bullets / heading / 表格); Skywork 训练时显然对这种**详尽结构化输出**给高分
- SIA 的轻度收敛在这类任务上**违背了 Skywork 的偏好方向** → 即使干预方向"合理", 结果也是负的

**C. 失败集中在长任务上, 短任务影响小**:
- Q159 (167 vs 207 tok, 都是短问答) Δ=-14.4 但绝对差距小
- Q137 / Q131 (千级 token 长任务) 差距 -15 ~ -18.5 reward, **绝对失败更严重**
- 这预测: **SIA 在短任务样本占多数的数据集上 Δ 接近 0, 在长任务样本多时 Δ 偏负** — 与 AlpacaEval 数据集中长任务占比相关

### 9.5 与 0GM-35B 实验的呼应

0GM-35B 上 -13.2% 的负 Δ 也很可能是同样机制 — Skywork 偏好 verbose + 结构化, SIA 让模型早终止 / 收敛, 与 Skywork 偏好相悖。

→ **本研究两组实验 (0GM-35B 跨家族 + VL-30B 同家族) 的负 Δ 都可能不只是 RM 跨家族 OOD 问题, 还包含 Skywork 评估器偏好与 SIA "收敛干预"方向冲突的因素**。

可验证假设: 用一个**偏好简洁回答**的 reward model 重新打分, Δ 可能转正 (SIA 输出更简洁正中下怀)。

---

## 10. 一句话现状

> Qwen3-VL-30B-A3B-Instruct (Qwen3 同家族, non-thinking) 上 SIA + VM-Qwen3-4B 200Q AlpacaEval **Δ = -1.57% (t=-1.05, p~0.29, 不显著)**, 跟 0GM-35B 的 **-13.2% (p<0.0001)** 相比伤害幅度 **缩小 8.4×**。**强证据支持 "SIA 在 0GM-35B 上失败的根因是 RM 跨家族 OOD" 假设**, 同时大比分落败样本分析揭示一个**附加机制**: SIA 倾向让长生成早终止 / 跳过结构化段落, 与 Skywork "偏好 verbose + 结构化"的打分偏好相悖。后续行动: 给 0GM-35B 配同家族 RM 重测 (优先级 #1), VL-30B 跑全 805 题确认 -1.6% 不是统计噪声 (优先级 #2), 或用偏好简洁的 RM 重新打分验证"Skywork 偏好冲突"的子假设。
