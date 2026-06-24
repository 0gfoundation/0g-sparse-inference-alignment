# 0GM-35B AlpacaEval 200Q + Skywork Third-Party Scoring — Execution Record

**Date**: 2026-06-02
**Goal**: Run AlpacaEval on 0GM-1.0-35B-A3B-0427, score with Skywork-Reward-V2-Llama-3.1-8B as a third-party reward signal, for use in the SIA vs noSIA comparison experiment.
**Reference planning doc**: [`alpaca-eval-plan-20260602.md`](alpaca-eval-plan-20260602.md) (plan); this doc is the execution record.
**Current status**: noSIA arm complete; SIA arm not yet started

---

## 1. Key Adjustments from the Planning Doc

4 adjustments from the planning doc were made during execution (discovered only after actually running):

| Adjustment | Reason |
|------|------|
| **Added `--disable_thinking` parameter** | 0GM's default chat template forces a `<think>\n` prefix → 800+ token thinking fills the 256 token cap → output is entirely incomplete thinking rather than an answer. Use `chat_template_kwargs={"enable_thinking": false}` to skip thinking (vllm 0.19 OpenAI server supports pass-through) |
| max_tokens: 256 → **512** | 256 frequently truncates even without thinking (0GM outputs are relatively verbose); 512 is a compromise |
| --limit: 805 → **200** | User decided to use 200 questions first to confirm the pipeline and save time; not running all 805 |
| noSIA uses **raw `vllm serve`** (not sia_vllm_server with `--weight 0`) | Cleanly excludes SIA processor overhead; closer to a vanilla baseline |

---

## 2. Complete Command Record (noSIA arm)

### 2.1 Start noSIA server (raw vllm serve, port 8000)

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

- Two previous startup attempts for SRV_PID failed (old EngineCore was still holding GPU memory); the third attempt PID 1076287 succeeded
- Load time: ~5 min (35B + cudagraph)

### 2.2 Run 200Q AlpacaEval generation

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

- Actual run time: **14.2 minutes**, throughput **109.3 tok/s**
- Output: 200/200, 0 errors, total tokens 92,818
- avg 480 tok/Q, max 512 (cap), min 248
- **92% still hit 512 cap** — 0GM uses a verbose output style (lists/markdown), but content **is real answers** (no `<think>`)

### 2.3 Kill server + score with Skywork RM

The main LLM must be killed to free GPU for Skywork (a single H200 cannot fit 35B + Skywork 8B + cudagraph).

```bash
# kill noSIA server (free ~120GB)
kill <PID>
pgrep -f EngineCore | xargs -r kill -9   # cleanup zombies
```

Note: Skywork runs in venv (not venv4) because `accelerate` is only installed in venv:

```bash
nohup /workspace/SIA/venv/bin/python scripts/measure_alpaca_reward.py \
    --input_file /tmp/alpaca_0gm_nosia_200q_20260602_122922.json \
    --output_file /tmp/alpaca_0gm_nosia_200q_scored.json \
    --rm /workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B \
    --device cuda:0 \
    > /tmp/skywork_score_nosia_20260602_125550.log 2>&1 &
```

- Skywork RM load: ~29s (4 shards, BF16, ~16GB)
- Scoring 200 questions took approximately **80s** (throughput ~2.5 Q/s)

---

## 3. noSIA Scoring Results

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

### Comparison with Historical 14B A1 Baseline

| Setup | model | mean reward (noSIA) |
|-------|-------|---------------------|
| Historical A1 (official evaluate.py, 805Q) | Qwen3-14B | 12.29 |
| **This run (vllm-based, 200Q, disable_think)** | **0GM-35B** | **18.06** |

→ 0GM noSIA reward is **+47%** higher than 14B noSIA; this is expected:
- 0GM-35B is a larger/newer model with inherently higher baseline generation quality
- Direct comparison with 14B A1 numbers is not valid (different model + different prompt path + different max_tokens)
- The meaningful metric is **0GM SIA - 0GM noSIA delta** (pending SIA arm completion)

---

## 4. Archived Files (under `exp/`)

All 5 files archived:

| File | Role | Path |
|------|------|------|
| noSIA vllm serve log | Main LLM server (raw vllm) | [`exp/alpaca_0gm_nosia_server_20260602_122922.log`](../exp/alpaca_0gm_nosia_server_20260602_122922.log) |
| AlpacaEval generation log | eval client stdout for 200Q run | [`exp/alpaca_0gm_nosia_200q_eval_20260602_122922.log`](../exp/alpaca_0gm_nosia_200q_eval_20260602_122922.log) |
| AlpacaEval generation JSON | 200 records `{id, instruction, output, tokens, elapsed, prompt, result}` | [`exp/alpaca_0gm_nosia_200q_20260602_122922.json`](../exp/alpaca_0gm_nosia_200q_20260602_122922.json) |
| Skywork scoring log | Scoring stdout (includes cur_mean_reward progress) | [`exp/alpaca_0gm_nosia_200q_scored_20260602_122922.log`](../exp/alpaca_0gm_nosia_200q_scored_20260602_122922.log) |
| Skywork scoring JSON | 200 records (original data + `reward` field) | [`exp/alpaca_0gm_nosia_200q_scored_20260602_122922.json`](../exp/alpaca_0gm_nosia_200q_scored_20260602_122922.json) |

---

## 5. Code Involved

- Added [`eval/alpaca_eval.py`](../eval/alpaca_eval.py): HTTP client, same style as `mmlu_eval.py`, with `--disable_thinking` support (passes through `chat_template_kwargs={"enable_thinking": false}`)
- Added [`scripts/measure_alpaca_reward.py`](../scripts/measure_alpaca_reward.py): Skywork RM scoring script, supports `--strip_think` (not needed this time because disable_thinking was already in effect)

---

## 6. Next Steps — SIA arm

Pending execution (starts once user decides):

```bash
# kill Skywork process (free ~16GB)
# Start RM 1.7B server (port 8001)
nohup /workspace/SIA/venv4/bin/vllm serve \
    /workspace/SIA/models/VM-Qwen3-1.7B-merged-for-vllm \
    --runner pooling --convert classify \
    --hf-overrides '{"architectures":["Qwen3WithScoreForCausalLM"]}' \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.12 \
    --max-model-len 2048 \
    --port 8001 --host 0.0.0.0 \
    --disable-log-stats &

# Start 0GM SIA server (port 8000), connected to 1.7B RM, thr=0.8 + Step1+2 fix
nohup env SIA_DEBUG_HIST=1 /workspace/SIA/venv4/bin/python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/0GM-1.0-35B-A3B-0427 \
    --rm_backend vllm \
    --rm_url http://localhost:8001 \
    --rm_model /workspace/SIA/models/VM-Qwen3-1.7B-merged-for-vllm \
    --llm_gpu_mem 0.78 --max_model_len 4096 \
    --topk 5 --weight 1.0 --entropy_threshold 0.8 \
    --host 0.0.0.0 --port 8000 &

# Run same 200Q AlpacaEval (same parameters, only difference is base_url now points to SIA server)
/workspace/SIA/venv2/bin/python eval/alpaca_eval.py \
    --base_url http://localhost:8000/v1 \
    --model 0GM-1.0-35B-A3B-0427 \
    --limit 200 --max_tokens 512 \
    --temperature 1.0 --top_p 0.95 --top_k 20 --repetition_penalty 1.0 \
    --disable_thinking \
    --output alpaca_0gm_sia_200q_<TS>.json

# Kill both servers, score the SIA JSON with Skywork
/workspace/SIA/venv/bin/python scripts/measure_alpaca_reward.py \
    --input_file alpaca_0gm_sia_200q_<TS>.json \
    --output_file alpaca_0gm_sia_200q_scored_<TS>.json \
    --rm /workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B \
    --device cuda:0

# Final comparison: delta reward = SIA_mean - noSIA_mean = ? vs 18.06
```

Expected duration:
- Start RM + 0GM SIA: ~5 min
- 200Q SIA generation (intervention adds RM call overhead): ~20-25 min
- Scoring: ~3 min

Total: **~30 minutes**.

---

## 7. One-Line Status

> 0GM-35B noSIA on 200Q AlpacaEval: mean reward = **18.06** (using disable_thinking + max_tokens=512), approximately +47% higher than the historical 14B A1 noSIA (12.29), consistent with the expectation that a larger/newer model has a higher baseline. Waiting for the SIA arm to complete before computing the delta.
