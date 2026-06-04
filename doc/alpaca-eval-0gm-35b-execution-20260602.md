# 0GM-35B AlpacaEval 200Q + Skywork 第三方打分 — 执行记录

**日期**: 2026-06-02
**目标**: 在 0GM-1.0-35B-A3B-0427 上跑 AlpacaEval, 用 Skywork-Reward-V2-Llama-3.1-8B 打第三方 reward 分, 用于 SIA vs noSIA 对照实验。
**对照前置 doc**: [`alpaca-eval-plan-20260602.md`](alpaca-eval-plan-20260602.md) (规划), 本 doc 是执行记录。
**当前状态**: ✅ noSIA arm 已完成, ⏳ SIA arm 未开始

---

## 1. 跟规划 doc 的关键调整

执行时跟规划 doc 有 4 处调整 (实际跑过才发现需要):

| 调整 | 原因 |
|------|------|
| **加 `--disable_thinking` 参数** | 0GM 默认 chat template 强制 `<think>\n` prefix → 800+ tok thinking 占满 256 cap → 输出全是不完整 thinking 而非答案。用 `chat_template_kwargs={"enable_thinking": false}` 跳过 thinking (vllm 0.19 OpenAI server 支持透传) |
| max_tokens: 256 → **512** | 256 即便不 thinking 也常被切断 (0GM 输出较 verbose), 512 折中 |
| --limit: 805 → **200** | 用户决定先用 200 题确认 pipeline + 节省时间, 不跑全 805 |
| noSIA 用 **raw `vllm serve`** (而非 sia_vllm_server 设 `--weight 0`) | 干净排除 SIA processor 开销, 更接近 vanilla baseline |

---

## 2. 完整命令记录 (noSIA arm)

### 2.1 启 noSIA server (raw vllm serve, port 8000)

```bash
nohup /workspace/SIA/venv4/bin/vllm serve \
    /workspace/SIA/models/0GM-1.0-35B-A3B-0427 \
    --served-model-name 0GM-1.0-35B-A3B-0427 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --enable-prefix-caching \
    --host 0.0.0.0 --port 8000 \
    > 0gm_nosia_alpaca_20260602_115018.log 2>&1 &
```

- SRV_PID 之前两次试启动失败 (旧 EngineCore 还占着 GPU 显存), 第三次 PID 1076287 成功
- 加载时长: ~5 min (35B + cudagraph)

### 2.2 跑 200Q AlpacaEval generation

```bash
nohup /workspace/SIA/venv2/bin/python eval/alpaca_eval.py \
    --base_url http://localhost:8000/v1 \
    --model 0GM-1.0-35B-A3B-0427 \
    --limit 200 \
    --max_tokens 512 \
    --temperature 1.0 --top_p 0.95 --top_k 20 --repetition_penalty 1.0 \
    --disable_thinking \
    --output /tmp/alpaca_0gm_nosia_200q_20260602_122922.json \
    > /tmp/alpaca_0gm_nosia_200q_20260602_122922.log 2>&1 &
```

- 实际跑了 **14.2 分钟**, throughput **109.3 tok/s**
- 输出: 200/200, 0 errors, total tokens 92,818
- avg 480 tok/Q, max 512 (cap), min 248
- **92% 仍 hit 512 cap** — 0GM 是 verbose 输出风格 (list/markdown), 但内容**是真实答案** (无 `<think>`)

### 2.3 kill server + 用 Skywork RM 打分

需要 kill 主 LLM 释放 GPU 给 Skywork (单卡 H200 装不下 35B + Skywork 8B + cudagraph)。

```bash
# kill noSIA server (释放 ~120GB)
kill <PID>
pgrep -f EngineCore | xargs -r kill -9   # cleanup zombies
```

注意 Skywork 用 venv (不是 venv4) 跑, 因为 `accelerate` 只装在 venv:

```bash
nohup /workspace/SIA/venv/bin/python scripts/measure_alpaca_reward.py \
    --input_file /tmp/alpaca_0gm_nosia_200q_20260602_122922.json \
    --output_file /tmp/alpaca_0gm_nosia_200q_scored.json \
    --rm /workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B \
    --device cuda:0 \
    > /tmp/skywork_score_nosia_20260602_125550.log 2>&1 &
```

- Skywork RM 加载 ~29s (4 个 shards, BF16, ~16GB)
- 打分 200 题用了约 **80s** (吞吐 ~2.5 题/s)

---

## 3. noSIA 打分结果

```
=== summary ===
  total samples : 200
  scored        : 200
  skipped (error/empty)     : 0
  skipped (too long > 2048) : 0
  mean reward   : 18.0551
  p50           : 17.25
  min / max     : -6.22 / 44.25
```

### 跟历史 14B A1 baseline 对比

| Setup | model | mean reward (noSIA) |
|-------|-------|---------------------|
| 历史 A1 (官方 evaluate.py, 805Q) | Qwen3-14B | 12.29 |
| **本次 (vllm-based, 200Q, disable_think)** | **0GM-35B** | **18.06** |

→ 0GM noSIA reward 比 14B noSIA 高 **+47%**, 这是合理的:
- 0GM-35B 模型更大/更新, 自然生成质量本来就更高
- 不能直接跟 14B A1 数字对比 (不同模型 + 不同 prompt 路径 + 不同 max_tokens)
- 真正有意义的是 **0GM SIA - 0GM noSIA 的 Δ** (等 SIA arm 跑完)

---

## 4. 归档文件 (在 `exp/` 下)

5 个文件全部归档完:

| 文件 | 角色 | 路径 |
|------|------|------|
| noSIA vllm serve log | 主 LLM server (raw vllm) | [`exp/alpaca_0gm_nosia_server_20260602_122922.log`](../exp/alpaca_0gm_nosia_server_20260602_122922.log) |
| AlpacaEval generation log | eval client 跑 200Q 的 stdout | [`exp/alpaca_0gm_nosia_200q_eval_20260602_122922.log`](../exp/alpaca_0gm_nosia_200q_eval_20260602_122922.log) |
| AlpacaEval generation JSON | 200 条 `{id, instruction, output, tokens, elapsed, prompt, result}` | [`exp/alpaca_0gm_nosia_200q_20260602_122922.json`](../exp/alpaca_0gm_nosia_200q_20260602_122922.json) |
| Skywork 打分 log | 打分 stdout (含 cur_mean_reward 进度) | [`exp/alpaca_0gm_nosia_200q_scored_20260602_122922.log`](../exp/alpaca_0gm_nosia_200q_scored_20260602_122922.log) |
| Skywork 打分 JSON | 200 条 (原数据 + `reward` 字段) | [`exp/alpaca_0gm_nosia_200q_scored_20260602_122922.json`](../exp/alpaca_0gm_nosia_200q_scored_20260602_122922.json) |

---

## 5. 涉及的代码

- 新增 [`eval/alpaca_eval.py`](../eval/alpaca_eval.py): HTTP client, 跟 `mmlu_eval.py` 同风格, 加 `--disable_thinking` 支持 (透传 `chat_template_kwargs={"enable_thinking": false}`)
- 新增 [`scripts/measure_alpaca_reward.py`](../scripts/measure_alpaca_reward.py): Skywork RM 打分脚本, 支持 `--strip_think` (本次因 disable_thinking 已生效, 不需要剥离)

---

## 6. 下一步 — SIA arm

待执行 (用户决定后启动):

```bash
# kill Skywork process (释放 ~16GB)
# 启 RM 1.7B server (port 8001)
nohup /workspace/SIA/venv4/bin/vllm serve \
    /workspace/SIA/models/VM-Qwen3-1.7B-merged-for-vllm \
    --runner pooling --convert classify \
    --hf-overrides '{"architectures":["Qwen3WithScoreForCausalLM"]}' \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.12 \
    --max-model-len 2048 \
    --port 8001 --host 0.0.0.0 \
    --disable-log-stats &

# 启 0GM SIA server (port 8000), 接 1.7B RM, thr=0.8 + Step1+2 fix
nohup env SIA_DEBUG_HIST=1 /workspace/SIA/venv4/bin/python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/0GM-1.0-35B-A3B-0427 \
    --rm_backend vllm \
    --rm_url http://localhost:8001 \
    --rm_model /workspace/SIA/models/VM-Qwen3-1.7B-merged-for-vllm \
    --llm_gpu_mem 0.78 --max_model_len 4096 \
    --topk 5 --weight 1.0 --entropy_threshold 0.8 \
    --host 0.0.0.0 --port 8000 &

# 跑同样 200Q AlpacaEval (相同参数, 唯一区别是 base_url 现在指 SIA server)
/workspace/SIA/venv2/bin/python eval/alpaca_eval.py \
    --base_url http://localhost:8000/v1 \
    --model 0GM-1.0-35B-A3B-0427 \
    --limit 200 --max_tokens 512 \
    --temperature 1.0 --top_p 0.95 --top_k 20 --repetition_penalty 1.0 \
    --disable_thinking \
    --output alpaca_0gm_sia_200q_<TS>.json

# kill 两个 server, 用 Skywork 打 SIA 的 JSON
/workspace/SIA/venv/bin/python scripts/measure_alpaca_reward.py \
    --input_file alpaca_0gm_sia_200q_<TS>.json \
    --output_file alpaca_0gm_sia_200q_scored_<TS>.json \
    --rm /workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B \
    --device cuda:0

# 最终对比: Δ reward = SIA_mean - noSIA_mean = ? vs 18.06
```

预期时长:
- 启 RM + 0GM SIA: ~5 min
- 200Q SIA generation (intervene 增加 RM 调用开销): ~20-25 min
- 打分: ~3 min

总 **~30 分钟**。

---

## 7. 一句话现状

> 0GM-35B noSIA 在 200Q AlpacaEval 上, mean reward = **18.06** (使用 disable_thinking + max_tokens=512), 高于历史 14B A1 noSIA (12.29) 约 +47%, 符合更大模型/更新模型基线更高的预期。等 SIA arm 跑完后做 Δ 对比。
