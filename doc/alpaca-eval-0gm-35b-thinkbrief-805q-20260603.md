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

## 4. noSIA arm — 跑中

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
- 注意: `--gpu-memory-utilization 0.85` (比 SIA arm 高, 因为没有 RM 共占), 不影响吞吐对比 (此实验目标是 reward 不是吞吐)

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

- PID 1304510 (启动于 2026-06-03 00:12)
- 预计 ~3.5h (无 RM 调用开销, 比 SIA 快约 2h)
- LOG (预): [`/tmp/alpaca_0gm_nosia_thinkbrief_805q_20260603_001206.log`](file:///tmp/alpaca_0gm_nosia_thinkbrief_805q_20260603_001206.log)
- JSON (预): [`/tmp/alpaca_0gm_nosia_thinkbrief_805q_20260603_001206.json`](file:///tmp/alpaca_0gm_nosia_thinkbrief_805q_20260603_001206.json)

### 4.3 待跑: Skywork 打分 (与 SIA 同方式)

```bash
# kill raw vllm serve (释放 GPU) 后:
nohup /workspace/SIA/venv/bin/python \
  /workspace/git/0g-sparse-inference-alignment/scripts/measure_alpaca_reward.py \
  --input_file /tmp/alpaca_0gm_nosia_thinkbrief_805q_20260603_001206.json \
  --output_file /tmp/alpaca_0gm_nosia_thinkbrief_805q_scored_<TS>.json \
  --rm /workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B \
  --device cuda:0 \
  --strip_think \
  > /tmp/skywork_score_nosia_thinkbrief_<TS>.log 2>&1 &
```

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

## 6. 预期判读

跑完 noSIA + 打分后, 得到 SIA mean (11.16) vs noSIA mean (待出) 的 Δ:

| 情景 | 含义 |
|------|------|
| **Δ > 0** (SIA > noSIA) | SIA 在原生 thinking 模式下成功提升 reward — 推翻此前 100Q -13% 结论, 说明问题不是 SIA 本身, 而是 100Q 跑的 `--no_think_prompt + --ban_think_token` 把 0GM 推到了 OOD 模式 |
| **Δ ≈ 0** | SIA 在 thinking 模式下不显著 — 用 4B VM 在 0GM-35B (Qwen3.6) 上 cross-family OOD 太大, RM 信号噪声主导 |
| **Δ < 0** (SIA < noSIA) | SIA 在 thinking 模式仍伤 reward, 与 100Q 结论一致 — 这才是 0GM-35B SIA 的真实问题, 建议下一步: 换 1.7B 同家族 VM 或 weight=0.5 |

可能的混淆变量:
- `--gpu-memory-utilization` 不同 (SIA 0.72 / noSIA 0.85) 是否影响生成结果? **不会** — 它只影响 KV cache 大小, 模型权重/计算一致, batch size 不同最多影响延迟不影响逻辑
- noSIA 没有 `SIA_DEBUG_HIST` (但这只是 print debug, 不进 sampling)
- ⚠️ 此前 100Q SIA 用 1.7B VM (commit b6b433c 实验), 这次 805Q 又换回 4B VM — 因为 [`sia-fix-and-vm-ablation-0gm-35b-20260602.md`](sia-fix-and-vm-ablation-0gm-35b-20260602.md) 没有结论选 1.7B; 后续 Δ 不理想可考虑再用 1.7B 跑

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

### 跑中 (noSIA arm)
| 文件 | 角色 |
|------|------|
| `/tmp/0gm_nosia_thinkbrief_srv_20260603_000821.log` | raw vllm serve log |
| `/tmp/alpaca_0gm_nosia_thinkbrief_805q_20260603_001206.log` | eval client stdout (805Q noSIA) |
| `/tmp/alpaca_0gm_nosia_thinkbrief_805q_20260603_001206.json` | 805Q noSIA 输出 JSON |

### 待生成 (noSIA Skywork 打分阶段)
| 文件 | 角色 |
|------|------|
| `/tmp/skywork_score_nosia_thinkbrief_<TS>.log` | Skywork 打分 stdout |
| `/tmp/alpaca_0gm_nosia_thinkbrief_805q_scored_<TS>.json` | 805Q noSIA + reward 字段 |

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
| 用 1.7B VM 还是 4B VM | 之前 ablation 没明确结论 | (待决) | 本次先用 4B (与 0427 的 history 一致), 若 Δ 不理想再换 1.7B 跑 |

---

## 9. 一句话现状

> 0GM-35B AlpacaEval 805Q 在 thinking + brief instruction + max_tokens=2048 下, **SIA mean reward = 11.16 (798/805 scored)**; noSIA arm 跑中, 完成后做 Δ 对比, 验证 SIA 在原生 thinking 模式下到底是 +/-/平。
