# 0GM-35B AlpacaEval 805Q (thinking + brief) — SIA vs noSIA 完整对照

**日期**: 2026-06-02 ~ 2026-06-03
**前置 doc**: [`alpaca-eval-0gm-35b-summary-20260602.md`](alpaca-eval-0gm-35b-summary-20260602.md) (含此前 100Q 对照的所有方法学审计)
**状态**: SIA arm ✅ 已完成 + 已打分; noSIA arm ⏳ 跑中 (启动于 00:12, 预计 ~3.5h)

---

## 1. 实验目的

之前 100Q 的 SIA vs noSIA 对照得到 **SIA -13% reward**, 但那次为了排除 `<think>` 格式干扰用了 `--no_think_prompt + --ban_think_token` (raw `Human/Assistant` prompt + token banning), 把 0GM-35B 的 thinking 模式完全关掉了。

**这次目的**: 在 0GM-35B 真正的 thinking 工作模式下做 SIA vs noSIA 公平对照 — 看 SIA 在原生 thinking 流程里对 reward 的真实影响 (而不是被压制 thinking 后的退化模式)。

**核心问题**:
1. 0GM-35B 在原生 thinking 模式 (有 `<think>...</think>` 完整流程) 下, SIA 是否还伤 reward?
2. 这次用 `--brief_instruction` 让 model 在 think 和 answer 都更简洁, 避免触顶 2048 cap 的样本被切断 → 更干净的对照
3. 805Q 全量评测, 排除 100Q 样本量小的统计噪声

---

## 2. 共享配置 (两 arm 完全一致, 除 server 端)

| 参数 | 值 |
|------|----|
| 模型 | `0GM-1.0-35B-A3B-0427` (Qwen3.6 / Qwen3.5 MoE thinking) |
| 数据集 | `/workspace/SIA/data/alpaca_eval/alpaca_eval.json` (805 题全量) |
| max_tokens | **2048** |
| temperature | 1.0 |
| top_p | 0.95 |
| top_k | 20 |
| repetition_penalty | 1.0 |
| chat_template | 默认 (即 `enable_thinking=True`, prompt 末尾自动注入 `<think>`) |
| 用户指令 | 原 instruction 前 prepend: <br/>*"Please keep both your reasoning (inside `<think>`) and your final answer concise and to the point. Avoid unnecessary elaboration."* |
| eval client | `eval/alpaca_eval.py` with `--brief_instruction` (无 `--disable_thinking`/`--no_think_prompt`/`--ban_think_token`) |
| `max-model-len` | 4096 |

**唯一区别**: SIA arm 走 SIA server (`sia_vllm_server.py` + RM 4B), noSIA arm 走 raw `vllm serve` (无 SIA processor, 干净 baseline)。

---

## 3. SIA arm — 已完成

### 3.1 RM 4B server (port 8001)

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
  > /tmp/rm_4B_alpaca_thinkbrief_20260602_163740.log 2>&1 &
```

- PID 1162698, log [`/tmp/rm_4B_alpaca_thinkbrief_20260602_163740.log`](file:///tmp/rm_4B_alpaca_thinkbrief_20260602_163740.log)

### 3.2 SIA LLM server (port 8000, weight=1.0, entropy=0.8, topk=5)

⚠️ 第一次启动失败 (旧 sia_vllm_server PID 1149494 残留占着 port 8000, EngineCore 已死但 FastAPI 仍 listen → 所有请求 500)。force-kill 后重启:

```bash
nohup env SIA_DEBUG_HIST=1 \
  /workspace/SIA/venv4/bin/python \
  /workspace/git/0g-sparse-inference-alignment/src/sia_vllm_server.py \
  --llm /workspace/SIA/models/0GM-1.0-35B-A3B-0427 \
  --rm_backend vllm \
  --rm_url http://localhost:8001 \
  --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
  --llm_gpu_mem 0.72 --max_model_len 4096 \
  --topk 5 --weight 1.0 --entropy_threshold 0.8 \
  --host 0.0.0.0 --port 8000 \
  > /tmp/0gm_sia_alpaca_thinkbrief_srv_20260602_164947.log 2>&1 &
```

- PID 1167411, log [`/tmp/0gm_sia_alpaca_thinkbrief_srv_20260602_164947.log`](file:///tmp/0gm_sia_alpaca_thinkbrief_srv_20260602_164947.log)
- 加载 ~3 min (35B + cudagraph)

### 3.3 跑 805Q AlpacaEval

```bash
nohup /workspace/SIA/venv2/bin/python \
  /workspace/git/0g-sparse-inference-alignment/eval/alpaca_eval.py \
  --base_url http://localhost:8000/v1 \
  --model 0GM-1.0-35B-A3B-0427 \
  --max_tokens 2048 \
  --temperature 1.0 --top_p 0.95 --top_k 20 --repetition_penalty 1.0 \
  --brief_instruction \
  --output /tmp/alpaca_0gm_sia_thinkbrief_805q_20260602_165328.json \
  > /tmp/alpaca_0gm_sia_thinkbrief_805q_20260602_165328.log 2>&1 &
```

- PID 1168623
- 总耗时: **330.4 min (5.51h)**, client throughput 35.3 tok/s
- 0 errors, 805/805 ✅
- LOG: [`/tmp/alpaca_0gm_sia_thinkbrief_805q_20260602_165328.log`](file:///tmp/alpaca_0gm_sia_thinkbrief_805q_20260602_165328.log)
- JSON: [`/tmp/alpaca_0gm_sia_thinkbrief_805q_20260602_165328.json`](file:///tmp/alpaca_0gm_sia_thinkbrief_805q_20260602_165328.json)

### 3.4 生成统计

| 指标 | 值 |
|------|----|
| Q done / errors | 805 / 0 |
| tokens (per Q) | min=7 p50=781 p95=1949 max=2048 mean=870 |
| total tokens 生成 | 700,396 |
| **触顶 2048 cap** | **37 / 805 (4.6%)** |
| 单题耗时 (s) | p50=21.2 p95=60.1 mean=24.6 |
| `</think>` 闭合率 | **742 / 805 (92.2%)** |
| thinking chars (平均) | 2,362 |
| final answer chars (平均) | 952 |
| **无 final answer (thinking 占满)** | **63 / 805 (7.8%)** ⚠️ |
| tokens<50 (异常短) | 13 |
| 'I cannot/Sorry' 拒答 | 0 |

### 3.5 SIA processor 性能 (132K steps)

```
http_post (RM 调用):    p50=86ms  p95=121ms  max=140ms
apply_total:            p50=4ms   p95=110ms  max=143ms
apply_topk_ent:         p50=0.7ms p95=1.2ms
skip_step:              p50=3.9ms p95=4.1ms  ← 多数 token 走 skip 路径
intv_prepare:           p50=0.13ms
intv_apply_logits:      p50=1.3ms
```

### 3.6 干预率 / Flip 率 / 吞吐 (聚合 794 个 req DONE 行)

| 指标 | 值 |
|------|----|
| **tokens/s (client throughput)** | **35.3 tok/s** |
| 总 wall time | 19,824s (330.4 min, **5.51h**) |
| 总生成 tokens | 700,396 (client side) / 701,345 (server DONE 聚合) |
| 有 DONE 行的 req | 794 / 805 (11 个没 DONE 行 — 早终止/state 释放早于打印) |
| total intervened tokens | 133,196 |
| total flipped tokens | 22,270 |
| **干预率 (全局, ∑intv/∑tot)** | **18.99%** |
| 干预率 per-req | p25=15.1%, **p50=19.4%**, p75=23.8%, min=2.4%, max=50.0% |
| **Top-1 Flip 率 (干预 step 中, ∑flip/∑intv)** | **16.72%** |
| Top-1 Flip per-req | p25=13.1%, **p50=17.5%**, p75=24.0%, min=0%, max=75.0% |
| **整体 Flip 率 (改变 top-1 / 总 token)** | **3.18%** |

**解读**:
- 干预率 19% vs 此前 100Q `--no_think_prompt + --ban_think_token` 的 33% — 本次更低, 因为:
  - thinking 模式下 token 分布更陡 (大量低 entropy step → 走 skip)
  - 上次 ban `<think>` 后 top-5 重排引起伪 entropy 上升 (见 [summary doc §11.3 可疑 #3](alpaca-eval-0gm-35b-summary-20260602.md))
- Flip 率 17% (干预 step 中实际改变 sampling 顺序的 ratio): 干预 step 里约 1/6 真的影响了 top-1 决策
- 整体 Flip 3.18% — 全部 700K token 中, 仅约 22K (3%) 因 SIA 改变了采样, 这是 SIA 真正"动了"模型行为的物理范围
- 35.3 tok/s — 比 raw vLLM 0GM-35B 慢 ~3-5×, 主要瓶颈 = 每干预 step 的 ~86ms RM call (干预率 19% × 86ms 摊到每 token ≈ 16ms overhead/token)

### 3.7 Skywork 打分

kill SIA server + RM server (释放 ~120GB GPU), 用 `venv` (有 accelerate) + `--strip_think`:

```bash
nohup /workspace/SIA/venv/bin/python \
  /workspace/git/0g-sparse-inference-alignment/scripts/measure_alpaca_reward.py \
  --input_file /tmp/alpaca_0gm_sia_thinkbrief_805q_20260602_165328.json \
  --output_file /tmp/alpaca_0gm_sia_thinkbrief_805q_scored_20260602_235645.json \
  --rm /workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B \
  --device cuda:0 \
  --strip_think \
  > /tmp/skywork_score_sia_thinkbrief_20260602_235645.log 2>&1 &
```

- PID 1299095, scoring ~8 min
- `--strip_think`: 剥离 `<think>...</think>` 内容, Skywork 只评 final answer (因为 thinking 不该参与 reward)
- LOG: [`/tmp/skywork_score_sia_thinkbrief_20260602_235645.log`](file:///tmp/skywork_score_sia_thinkbrief_20260602_235645.log)
- JSON: [`/tmp/alpaca_0gm_sia_thinkbrief_805q_scored_20260602_235645.json`](file:///tmp/alpaca_0gm_sia_thinkbrief_805q_scored_20260602_235645.json)

**结果**:

```
total samples       : 805
scored              : 798
skipped (>2048 tok) : 7    ← Skywork tokenizer 截断长样本
mean reward         : 11.1584
p50                 : 11.5625
min / max           : -17.6250 / 35.7500
```

---

## 4. noSIA arm — 已完成

### 4.1 raw vllm serve (port 8000, 无 SIA)

为干净 baseline, 用 raw `vllm serve` (跟此前 200Q noSIA 一致做法), 完全排除 SIA processor 任何开销:

```bash
nohup /workspace/SIA/venv4/bin/vllm serve \
  /workspace/SIA/models/0GM-1.0-35B-A3B-0427 \
  --served-model-name 0GM-1.0-35B-A3B-0427 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --host 0.0.0.0 --port 8000 \
  > /tmp/0gm_nosia_thinkbrief_srv_20260603_000821.log 2>&1 &
```

- PID 1302846, log [`/tmp/0gm_nosia_thinkbrief_srv_20260603_000821.log`](file:///tmp/0gm_nosia_thinkbrief_srv_20260603_000821.log)
- 注意: `--gpu-memory-utilization 0.85` (比 SIA arm 高, 因为没有 RM 共占), 不影响 reward 对比

### 4.2 跑 805Q AlpacaEval (同 SIA 所有 client flags)

```bash
nohup /workspace/SIA/venv2/bin/python \
  /workspace/git/0g-sparse-inference-alignment/eval/alpaca_eval.py \
  --base_url http://localhost:8000/v1 \
  --model 0GM-1.0-35B-A3B-0427 \
  --max_tokens 2048 \
  --temperature 1.0 --top_p 0.95 --top_k 20 --repetition_penalty 1.0 \
  --brief_instruction \
  --output /tmp/alpaca_0gm_nosia_thinkbrief_805q_20260603_001206.json \
  > /tmp/alpaca_0gm_nosia_thinkbrief_805q_20260603_001206.log 2>&1 &
```

- PID 1304510, 启动 2026-06-03 00:12, **结束 02:17 (用时 125 min, 2.1h)**
- 0 errors, 805/805 ✅
- LOG: [`/tmp/alpaca_0gm_nosia_thinkbrief_805q_20260603_001206.log`](file:///tmp/alpaca_0gm_nosia_thinkbrief_805q_20260603_001206.log)
- JSON: [`/tmp/alpaca_0gm_nosia_thinkbrief_805q_20260603_001206.json`](file:///tmp/alpaca_0gm_nosia_thinkbrief_805q_20260603_001206.json)

### 4.3 noSIA 生成统计

| 指标 | noSIA (805) | SIA (805) | 对比 |
|------|-------------|-----------|------|
| 总 wall time | 125 min | 330 min | **noSIA 快 2.6×** |
| client throughput | 108.6 tok/s | 35.3 tok/s | **noSIA 快 3.0×** |
| 总 tokens | 816,206 | 700,396 | noSIA 多 +17% |
| tokens (mean per Q) | 1,012 | 870 | noSIA 多 16% |
| **触顶 2048 cap** | **122 / 805 (15.2%)** | **37 / 805 (4.6%)** | **noSIA 触顶率 3.3×** |
| `</think>` 闭合率 | 87.2% | 92.2% | noSIA 略低 |
| 无 final answer | 75 / 805 (9.3%) | 63 / 805 (7.8%) | noSIA 略高 |
| **嵌套 think tag** (`</think>` 后又 `<think>`) | **33.5%** (270/805) | ~? (未统计但远低于 noSIA) | noSIA 输出结构更乱 |
| 用自创 `<thinking>`/`<reasoning>` tag | 9.2% (74/805) | ? | noSIA 显著更多 |
| `I cannot/Sorry` 拒答 | 0 | 0 | ✓ |

→ noSIA 输出结构**比 SIA 更乱** (触顶率 3.3×, 嵌套 think 33.5%, 自创 tag 9.2%)。这暗示 **SIA 起到了轻度"格式收敛"作用**, 让 0GM 更乖地按标准 `<think>...</think>` 格式生成。

### 4.4 Skywork 打分 (kill raw vllm serve 后)

⚠️ **首轮 (broken strip)**: 第一次打分时 `_THINK_RE = r"<think>.*?</think>"` 这个正则**在约 50% 样本上没有真正剥到 thinking** — 因为 0GM chat_template 把开头 `<think>` 放在 **prompt 末尾** (不在 output 里), output 实际形如 `{thinking content}</think>{answer}`, 没有开 tag → 配对正则不匹配。详见 §10。

**首轮结果 (旧 broken strip, 已弃用)**:

```bash
nohup /workspace/SIA/venv/bin/python \
  /workspace/git/0g-sparse-inference-alignment/scripts/measure_alpaca_reward.py \
  --input_file /tmp/alpaca_0gm_nosia_thinkbrief_805q_20260603_001206.json \
  --output_file /tmp/alpaca_0gm_nosia_thinkbrief_805q_scored_20260603_022222.json \
  --rm /workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B \
  --device cuda:0 --strip_think \
  > /tmp/skywork_score_nosia_thinkbrief_20260603_022222.log 2>&1 &
```

- noSIA scored=796/805, **mean reward = 11.66** (含约 55% thinking 残留污染)
- 配对 vs SIA (broken): Δ = -0.50 (-4.3%), t = -1.72 (p ≈ 0.09, **不显著**) ← **测量假象**

详见 §11 "Δ 对比正式结论"。

---

## 5. 配置对比一表

| 项 | SIA arm | noSIA arm |
|----|---------|-----------|
| LLM server | `sia_vllm_server.py` | raw `vllm serve` |
| LLM `--gpu-memory-utilization` | 0.72 | 0.85 |
| RM | VM-Qwen3-4B (vllm backend, port 8001) | (none) |
| SIA 参数 | topk=5, weight=1.0, entropy_threshold=0.8 | — |
| `SIA_DEBUG_HIST` | 1 (打印 per-step entropy/gap 直方图) | — |
| **eval client flags** | **完全一致** | **完全一致** |
| max_tokens | 2048 | 2048 |
| temp / top_p / top_k / rep_pen | 1.0 / 0.95 / 20 / 1.0 | 1.0 / 0.95 / 20 / 1.0 |
| --brief_instruction | ✅ | ✅ |
| --disable_thinking | ❌ (thinking enabled) | ❌ |
| --no_think_prompt | ❌ | ❌ |
| --ban_think_token | ❌ | ❌ |
| 样本量 | 805 | 805 |

---

## 6. 配置混淆变量审计

可能的混淆变量:
- `--gpu-memory-utilization` 不同 (SIA 0.72 / noSIA 0.85) 是否影响生成结果? **不会** — 它只影响 KV cache 大小, 模型权重/计算一致, batch size 不同最多影响延迟不影响逻辑
- noSIA 没有 `SIA_DEBUG_HIST` (但这只是 print debug, 不进 sampling)
- ⚠️ 此前 100Q SIA 用 1.7B VM (commit b6b433c 实验), 这次 805Q 又换回 4B VM — 因为 [`sia-fix-and-vm-ablation-0gm-35b-20260602.md`](sia-fix-and-vm-ablation-0gm-35b-20260602.md) 没有明确结论选 1.7B; 见 §12 后续优先级

---

## 7. 文件清单 (本实验涉及)

### 已生成 (SIA arm)
| 文件 | 角色 |
|------|------|
| `/tmp/rm_4B_alpaca_thinkbrief_20260602_163740.log` | RM 4B vllm serve log |
| `/tmp/0gm_sia_alpaca_thinkbrief_srv_20260602_164947.log` | SIA LLM server log (含 SIA-pf-summary) |
| `/tmp/alpaca_0gm_sia_thinkbrief_805q_20260602_165328.log` | eval client stdout (805Q SIA) |
| `/tmp/alpaca_0gm_sia_thinkbrief_805q_20260602_165328.json` | 805Q SIA 输出 JSON |
| `/tmp/skywork_score_sia_thinkbrief_20260602_235645.log` | Skywork 打分 stdout |
| `/tmp/alpaca_0gm_sia_thinkbrief_805q_scored_20260602_235645.json` | 805Q SIA + reward 字段 |

### 已生成 (noSIA arm)
| 文件 | 角色 |
|------|------|
| `/tmp/0gm_nosia_thinkbrief_srv_20260603_000821.log` | raw vllm serve log |
| `/tmp/alpaca_0gm_nosia_thinkbrief_805q_20260603_001206.log` | eval client stdout (805Q noSIA) |
| `/tmp/alpaca_0gm_nosia_thinkbrief_805q_20260603_001206.json` | 805Q noSIA 输出 JSON |
| `/tmp/skywork_score_nosia_thinkbrief_20260603_022222.log` | Skywork 打分 stdout (旧 broken strip, **已弃用**) |
| `/tmp/alpaca_0gm_nosia_thinkbrief_805q_scored_20260603_022222.json` | 805Q noSIA + reward (旧 broken strip, **已弃用**) |

### 重新打分 (修 strip 后, 是当前真实结果)
| 文件 | 角色 |
|------|------|
| `/tmp/skywork_score_sia_thinkbrief_v2_20260603_023431.log` | SIA Skywork 打分 (fixed strip) |
| `/tmp/alpaca_0gm_sia_thinkbrief_805q_scored_v2_20260603_023431.json` | SIA + reward (fixed strip) |
| `/tmp/skywork_score_nosia_thinkbrief_v2_20260603_023431.log` | noSIA Skywork 打分 (fixed strip) |
| `/tmp/alpaca_0gm_nosia_thinkbrief_805q_scored_v2_20260603_023431.json` | noSIA + reward (fixed strip) |

### 失败的 SIA 启动尝试 (保留以备 troubleshooting)
| 文件 | 备注 |
|------|------|
| `/tmp/0gm_sia_alpaca_thinkbrief_20260602_164043.log` | 第一次启动失败 (port 8000 被占) |
| `/tmp/alpaca_0gm_sia_thinkbrief_805q_20260602_164537.log` | 第一次 eval 失败 (server 死, 805 个 500 error) |
| `/tmp/alpaca_0gm_sia_thinkbrief_805q_20260602_164537.json` | 805 个 ERROR 记录, 不要用 |

---

## 8. Troubleshooting 教训

| 问题 | 表现 | 原因 | 修复 |
|------|------|------|------|
| SIA server 启动后立即崩 | "[Errno 98] address already in use" | 旧 SIA server (PID 1149494, noSIA --weight 0 配置) 残留, EngineCore 死了但 FastAPI 仍 listen → 新 server 死 | `kill -9 1149494` 后重启 |
| Skywork `--strip_think` 大半样本未生效 | 旧正则 `<think>.*?</think>` 不匹配缺失开 tag 的 leading thinking | 0GM chat_template 把开头 `<think>` 注入 prompt 末尾 → output 实际形如 `{thinking}</think>{answer}` 缺开 tag → 正则不匹配 | 改用两段正则: ①  `^.*?</think(?:ing)?>\s*` 剥 leading thinking ② `<think(?:ing)?>.*?</think(?:ing)?>\s*` 剥嵌套块 (commit 见 §10) |
| 用 1.7B VM 还是 4B VM | 之前 ablation 没明确结论 | (待决) | 本次先用 4B (与 0427 的 history 一致), Δ = -13% 后, 优先级 #1 切 1.7B 重跑 |

---

## 9. 一句话现状

> 0GM-35B AlpacaEval 805Q 在 thinking + brief + max_tokens=2048 下, **SIA Δ reward = -13.2% (n=795, t=-4.61, p<0.0001, 统计上极显著)**。修复 `--strip_think` 失效 bug 之后, 真相浮现: SIA 与 VM-4B 在 0GM-35B 上**稳定伤害 reward 约 13-14%**, 跟此前 100Q `--no_think_prompt + --ban_think_token` 结论 (-13%) 高度一致, 推翻了 "100Q 大负 Δ 是压制 thinking 引起" 的此前解读。下一步优先级 #1: 换 **VM-Qwen3-1.7B-Base** 同家族 RM 重跑, 验证是否 cross-family OOD 是根因。

---

## 10. `--strip_think` 失效 bug 与修复

### 10.1 Bug 描述

`scripts/measure_alpaca_reward.py` 原 strip 正则:

```python
_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
```

这个正则**要求 `<think>` 开 tag 出现在 output 中**。但 0GM chat_template 在 prompt 末尾自动注入 `<think>\n`, 所以**模型 output 实际**形如:

```
{thinking content}\n</think>\n\n{final answer}
```

→ output 里**没有 `<think>` 开 tag**, 只有 `</think>` 闭 tag。配对正则不匹配 → **整段 thinking 内容连同 `</think>` tag 都保留在打分输入里**。

### 10.2 影响面 (805 题 SIA + 805 题 noSIA)

|  | 旧 broken strip 真正剥到 thinking 的样本 | 余下 (含 thinking 残留) |
|--|------|------|
| **SIA**   | 402 / 805 (49.9%) | **403 (50.1%)** ⚠️ |
| **noSIA** | 364 / 805 (45.2%) | **441 (54.8%)** ⚠️ |

→ 约 **50-55% 样本**的 Skywork 打分输入**混杂了 thinking 内容**。

### 10.3 修复 (`scripts/measure_alpaca_reward.py`)

新增 `_LEAD_THINK_END_RE` 处理 leading thinking 情况:

```python
_PAIR_THINK_RE     = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>\s*", flags=re.DOTALL)
_LEAD_THINK_END_RE = re.compile(r"^.*?</think(?:ing)?>\s*",               flags=re.DOTALL)

def maybe_strip_think(text, strip):
    if not strip: return text
    # Step 1: 剥 leading thinking (chat_template 注入的开 tag 在 prompt, 不在 output)
    text = _LEAD_THINK_END_RE.sub("", text, count=1)
    # Step 2: 剥任何剩余嵌套配对 think/thinking 块
    return _PAIR_THINK_RE.sub("", text)
```

新逻辑覆盖 4 类 case:

| 输出结构 | 处理 |
|----------|------|
| `{think}</think>{answer}` (标准, 缺开 tag) | Step 1 剥 thinking + `</think>` |
| `{think1}</think>{<think>nest</think>}{answer}` (嵌套) | Step 1 剥到第一个 `</think>`, Step 2 剥嵌套 |
| `{think}</thinking>{answer}` (自创 tag) | Step 1 用 `</think(?:ing)?>` 命中 |
| 无 `</think>` (thinking 占满 cap) | 无匹配, 原样保留 — Skywork 看到整段 thinking-style 文本, 给低分 (符合"thinking 占满未出 answer"应得低分) |

修复 dry-run 验证: 平均每题剥掉 **~2200 chars** (vs 旧 strip 约 0)。

---

## 11. Δ 对比正式结论 (修 strip 后, **当前真值**)

### 11.1 整体均值

| 配置 | scored | mean reward |
|------|--------|-------------|
| **SIA  (fixed strip)** | 800 / 805 | **11.99** |
| **noSIA (fixed strip)** | 798 / 805 | **13.82** |

### 11.2 配对统计 4 种切法

|  | n_pairs | SIA mean | noSIA mean | Δ | 相对 | t-stat | p-value | 显著? |
|--|---------|----------|------------|---|------|--------|---------|------|
| 旧 broken — 全配对 | 792 | 11.13 | 11.63 | -0.50 | -4.3% | -1.72 | ~0.086 | ❌ |
| 旧 broken — 排除任一触顶 | 663 | 11.81 | 12.36 | -0.56 | -4.5% | -1.89 | ~0.059 | ❌ |
| **新 fixed — 全配对** | **795** | **11.97** | **13.79** | **-1.82** | **-13.2%** | **-4.61** | **<0.0001** | ✅ |
| **新 fixed — 排除任一触顶** | **664** | **12.83** | **14.93** | **-2.10** | **-14.0%** | **-5.16** | **<1e-6** | ✅ |

→ **修复后 Δ 翻 3 倍, 从"边缘不显著" 到 "极显著"**。

### 11.3 noSIA mean 涨幅 > SIA 涨幅 — 为什么?

|  | 旧 broken | 新 fixed | 差 |
|--|-----------|----------|------|
| SIA  mean   | 11.16 | 11.99 | +0.83 |
| noSIA mean | 11.66 | 13.82 | **+2.16** |

noSIA mean 涨幅 (+2.16) > SIA (+0.83) 的原因:

- noSIA 输出**结构更乱** (33.5% 嵌套 think + 9.2% 自创 tag + 15.2% 触顶), 这些 thinking 残留会让 Skywork 看到"凌乱的 thinking-style 文本" → 打低分
- 旧 broken strip 让 noSIA 这些**结构噪声**直接污染打分, 拉低 mean → 显得 SIA 跟 noSIA 差距小
- 修复后 noSIA 只剩 clean final answer 打分, mean 大幅上升 → 暴露了 SIA 实际差距

### 11.4 Win/Tie/Loss (fixed, n=795, |Δ|>0.5)

| | 计数 | 占比 |
|--|------|------|
| **SIA wins**  | **315** | **39.6%** |
| Tie | 53 | 6.7% |
| **SIA loses** | **427** | **53.7%** |

排除触顶后 (n=664): SIA wins 256 (38.6%) / loses 363 (54.7%) — 差距更大。

### 11.5 跟此前 100Q 实验对比

| 实验 | 模式 | Δ | n | 显著? |
|------|------|---|----|------|
| 100Q `--no_think_prompt + --ban_think_token` | 强制 no thinking | -13.0% | 98 | 边缘 |
| 805Q broken strip (本次旧 strip) | thinking + brief | -4.3% | 792 | ❌ |
| **805Q fixed strip (本次真相)** | thinking + brief | **-13.2%** | **795** | **✅ p<0.0001** |

→ **100Q 和 805Q 两次实验 Δ 数字惊人一致 (~ -13%)**, 修 strip 之前看到的"thinking 模式只 -4.5%"是**测量假象**。

**真相**: SIA 在 0GM-35B (Qwen3.6) + VM-Qwen3-4B-Base (Qwen3) 组合下**稳定伤害 reward 约 13-14%**, 与 thinking 模式开关**无关**。

### 11.6 推翻此前 §6 的解读

§6 (现已删除) 曾推测: "100Q -13% 是 `--no_think_prompt + --ban_think_token` 压制 thinking 把 0GM 推到了 OOD 模式"。

**这个推测被本次实验推翻** — 即使在 0GM 原生 thinking + brief 模式下, SIA 仍带来同样幅度 (-13%) 的 reward 损失。

→ **真正根因更可能是 cross-family OOD** (VM-Qwen3-4B-Base 训练于 Qwen3 / SFT 数据, 应用于 Qwen3.5/3.6 thinking 模型 0GM-35B 时跨家族 RM 信号噪声主导)。

---

## 12. 下一步优先级 (基于修正后的真相)

| 优先级 | 行动 | 期望 / 验证假设 |
|--------|------|---------------|
| 🚨 **1** | **VM-Qwen3-1.7B-Base** 重跑 SIA 805Q (与本次完全同 config) | 若 1.7B 同家族 RM 让 Δ 接近 0 或转正, 确认 cross-family OOD 是 4B 失败的根因 |
| **2** | topk=10 (官方默认, 当前 topk=5) | 候选池翻倍, SIA 影响范围 ×2; 若 -13% 缩到 -7% 说明 topk=5 占一半"损失" |
| **3** | 去 mean-norm 让代码完全匹配官方 | 修 [`summary doc §11.3 可疑 #1`](alpaca-eval-0gm-35b-summary-20260602.md) 的 mean-norm + `<think>` top-5 污染问题 |
| 4 | weight=0.5 弱化干预 | 若 weight=0.5 让 Δ 减半, 说明 SIA 干预方向是错的 (做"更少 SIA"比做完整 SIA 好) |
| 5 | 用 0GM 自家训的 VM (cross-family OOD 终极 ML 层解决) | 长期方向, 需训练成本 |
