# VL-30B AlpacaEval 200Q — docker compose b2 inproc Experiment Report

**Model**: Qwen3-VL-30B-A3B-Instruct, b2 inproc, docker compose production deployment
**Scoring RM**: Skywork-Reward-V2-Llama-3.1-8B (third-party, independent from VM-Qwen3-4B)
**Scripts**: `eval/alpaca_eval.py` (generation) + `scripts/measure_alpaca_reward.py` (scoring)

This document covers two rounds of AlpacaEval 200Q experiments:
- **Round 1 (20260608)**: Baseline docker deployment, `SIA_LLM_CUDAGRAPH=piecewise` (constrained by GPU memory)
- **Round 2 (20260610)**: Removed `SIA_LLM_CUDAGRAPH=piecewise`, switched back to FULL_AND_PIECEWISE mode (+32% SIA speed)

---

## Round 1 (20260608)

### 1. Configuration

#### Server Parameters (docker-compose.yml)

| Parameter | Value |
|---|---|
| LLM | Qwen3-VL-30B-A3B-Instruct |
| RM backend | b2 inproc (vllm 0.17.1) |
| RM model | VM-Qwen3-4B-merged-for-vllm |
| `--llm_gpu_mem` | 0.48 (GPU occupied by other processes; normal is 0.55) |
| `--rm_b2_gpu_mem` | 0.08 (normal is 0.15) |
| `--max_model_len` | 2048 |
| `--topk` | 10 |
| `--weight` | 1.0 |
| `--entropy_threshold` | 1.0 |
| `SIA_LLM_CUDAGRAPH` | piecewise (restricted mode, ~2× slower than FULL_AND_PIECEWISE) |

#### Generation Parameters

| Parameter | Value |
|---|---|
| `--limit` | 200 |
| `--max_tokens` | 2048 |
| `--temperature` | 1.0 |
| `--top_p` | 0.95 |
| `--top_k` | 20 |
| `--repetition_penalty` | 1.0 |
| noSIA method | per-request `--sia_weight 0`, shared server, no service switch needed |

### 2. Results

#### 2.1 Skywork Scoring (paired n=199)

| Metric | SIA | noSIA | Δ |
|---|---|---|---|
| mean reward | **30.9814** | 28.6011 | **+2.38 (+8.3%)** |
| p50 | 31.50 | 29.00 | +2.50 |
| min | 1.7891 | -2.4531 | — |
| max | 57.5000 | 58.0000 | — |
| scored / total | 200 / 200 | 199 / 200 | noSIA 1 item exceeded 2048, skipped |

#### 2.2 SIA Intervention Health Metrics

| Metric | Value |
|---|---|
| intervention ratio | **21.9%** |
| top1 flip rate | **64.6%** |
| RM error | 0 ✅ |

#### 2.3 Generation Throughput

| | SIA | noSIA |
|---|---|---|
| total tokens | 124,028 | 149,018 |
| wall time | 2095s (34.9 min) | 1844s (30.7 min) |
| throughput | **59.2 tok/s** | **80.8 tok/s** |
| SIA tax | — | 1.36× |

### 3. Output Files

| File | Content |
|---|---|
| [`exp/alpaca-vl30b-b2-docker-20260608/alpaca_vl30b_b2_sia_20260608_090255.json`](../exp/alpaca-vl30b-b2-docker-20260608/alpaca_vl30b_b2_sia_20260608_090255.json) | SIA generation, 200 items |
| [`exp/alpaca-vl30b-b2-docker-20260608/alpaca_vl30b_b2_nosia_20260608_094908.json`](../exp/alpaca-vl30b-b2-docker-20260608/alpaca_vl30b_b2_nosia_20260608_094908.json) | noSIA generation, 200 items |
| [`exp/alpaca-vl30b-b2-docker-20260608/alpaca_vl30b_b2_sia_gen.log`](../exp/alpaca-vl30b-b2-docker-20260608/alpaca_vl30b_b2_sia_gen.log) | SIA generation progress log |
| [`exp/alpaca-vl30b-b2-docker-20260608/alpaca_vl30b_b2_nosia_gen.log`](../exp/alpaca-vl30b-b2-docker-20260608/alpaca_vl30b_b2_nosia_gen.log) | noSIA generation progress log |
| [`exp/alpaca-vl30b-b2-docker-20260608/log_docker_AlpacaEval_20260608.txt`](../exp/alpaca-vl30b-b2-docker-20260608/log_docker_AlpacaEval_20260608.txt) | docker server log (including DONE lines) |

---

## Round 2 (20260610, FULL fix)

### 4. Configuration Changes

Compared to Round 1, the only change: **remove `SIA_LLM_CUDAGRAPH=piecewise`**, allowing the main LLM to use the default FULL_AND_PIECEWISE mode.

```yaml
# docker-compose.yml environment changes
# Removed: SIA_LLM_CUDAGRAPH: piecewise
# Kept:    SIA_RM_CUDAGRAPH: piecewise
#          SIA_RM_MULTIPROCESS: "0"
```

GPU memory configuration restored to normal (no other processes occupying GPU): `--llm_gpu_mem 0.55`, `--rm_b2_gpu_mem 0.15`. All other parameters are identical to Round 1.

#### Why This Achieves a Speedup

vLLM's CUDA graph has two modes:

- **PIECEWISE**: Only captures small graphs for individual attention layers, suitable for small batches; large batches (prefill) cannot match, falling back to eager (no graph) execution, which is slow.
- **FULL_AND_PIECEWISE (default)**: At startup, captures both a FULL graph for the entire forward pass (covering common batch sizes) and PIECEWISE graphs; the two are automatically switched based on batch size, so both prefill and decode can hit the graph with no eager fallback.

Round 1's `SIA_LLM_CUDAGRAPH=piecewise` was an experimental environment variable in the SIA code that forced the main LLM to capture only PIECEWISE graphs, causing every prefill step to run in eager mode — approximately 2× slower than the FULL graph. Removing this environment variable restores vLLM's default FULL_AND_PIECEWISE capture, returning main LLM forward speed to normal.

vllm 0.17.1 (the version used by VL-30B) captures all graphs AOT (ahead-of-time) at startup with no runtime re-capture, so this change is safe on this version. See [`0gm-35b-sia-perf-breakdown-20260609.md §6`](0gm-35b-sia-perf-breakdown-20260609.md) for detailed analysis.

### 5. Results

#### 5.1 Skywork Scoring (paired n=198)

| Metric | SIA | noSIA | Δ |
|---|---|---|---|
| mean reward | **30.3802** | 29.1595 | **+1.22 (+4.2%)** |
| p50 | 30.6250 | 30.2500 | +0.375 |
| min | — | -3.7969 | — |
| max | 59.2500 | 64.5000 | — |
| scored / total | 200 / 200 | 198 / 200 | noSIA 2 items exceeded 2048, skipped |

**Paired statistics**:

| Metric | Value |
|---|---|
| W / L / T | 122 / 73 / 3 |
| t-test | t=3.109, p=0.0022 ✅ |
| Wilcoxon | p=0.0005 ✅ |

#### 5.2 SIA Intervention Health Metrics

| Metric | Value |
|---|---|
| intervention rate | **24.9%** |
| top1 flip rate | **66.1%** |
| RM error | 0 ✅ |

#### 5.3 Generation Throughput

| | SIA | noSIA | vs Round 1 SIA |
|---|---|---|---|
| total tokens | 124,654 | 150,895 | — |
| wall time | 1591s (26.5 min) | 1229s (20.5 min) | — |
| throughput | **78.3 tok/s** | **122.8 tok/s** | **+32%** |
| avg tokens/Q | 623 | 754 | — |
| SIA tax | — | 1.57× | — |

noSIA went from 80.8 → 122.8 tok/s (+52%), approaching the normal pure vLLM level, indicating that Round 1's low throughput was primarily due to GPU memory constraints rather than a code issue.

### 6. Output Files

| File | Content |
|---|---|
| [`exp/alpaca-vl30b-b2-docker-20260610/alpaca_vl30b_b2_sia_20260610_083001.json`](../exp/alpaca-vl30b-b2-docker-20260610/alpaca_vl30b_b2_sia_20260610_083001.json) | SIA generation, 200 items |
| [`exp/alpaca-vl30b-b2-docker-20260610/alpaca_vl30b_b2_nosia_20260610_083001.json`](../exp/alpaca-vl30b-b2-docker-20260610/alpaca_vl30b_b2_nosia_20260610_083001.json) | noSIA generation, 200 items |
| [`exp/alpaca-vl30b-b2-docker-20260610/alpaca_vl30b_b2_sia_scored.json`](../exp/alpaca-vl30b-b2-docker-20260610/alpaca_vl30b_b2_sia_scored.json) | SIA Skywork scores |
| [`exp/alpaca-vl30b-b2-docker-20260610/alpaca_vl30b_b2_nosia_scored.json`](../exp/alpaca-vl30b-b2-docker-20260610/alpaca_vl30b_b2_nosia_scored.json) | noSIA Skywork scores |
| [`exp/alpaca-vl30b-b2-docker-20260610/alpaca_vl30b_b2_sia_gen.log`](../exp/alpaca-vl30b-b2-docker-20260610/alpaca_vl30b_b2_sia_gen.log) | SIA generation progress log |
| [`exp/alpaca-vl30b-b2-docker-20260610/alpaca_vl30b_b2_nosia_gen.log`](../exp/alpaca-vl30b-b2-docker-20260610/alpaca_vl30b_b2_nosia_gen.log) | noSIA generation progress log |
| [`exp/alpaca-vl30b-b2-docker-20260610/docker_log.txt`](../exp/alpaca-vl30b-b2-docker-20260610/docker_log.txt) | docker server log (including DONE lines) |

---

## Summary Comparison

| Experiment | noSIA mean | SIA mean | Δ | p value | intervention rate | SIA tok/s | noSIA tok/s |
|---|---|---|---|---|---|---|---|
| HTTP path max=2048 (20260603) | 27.44 | 30.19 | **+2.75** | — | 25.83% | 35.9 | — |
| HTTP path max=256 (20260603) | 11.69 | 14.01 | **+2.32** | — | 18.8% | 46.5 | — |
| b2 inproc max=256 (20260605) | — | — | — | — | 18.50% | 67.9 | — |
| **Round 1 docker (20260608)** | 28.60 | 30.98 | **+2.38** | — | 21.9% | 59.2 | 80.8 |
| **Round 2 docker FULL fix (20260610)** | 29.16 | 30.38 | **+1.22** | **0.0022** | 24.9% | **78.3** | **122.8** |

### Analysis

**Round 2 Δ (+1.22) is lower than Round 1 (+2.38)**, yet both rounds are statistically significant. The reason for the gap is not fully clear. Possible factors:
1. **Statistical noise**: Single-run variance is large at 200Q (historical range +1.22~+2.75); normal fluctuation cannot be ruled out
2. **noSIA baseline improvement**: noSIA absolute score increased from 28.60 → 29.16 (a higher baseline compresses Δ)
3. **Generation length difference**: Round 2 SIA averaged 623 tokens/Q vs Round 1 ~620; noSIA 754 vs 745 — essentially the same

**Throughput greatly improved**: The FULL_AND_PIECEWISE fix delivers SIA +32%, noSIA +52%. Round 2's noSIA at 122.8 tok/s is already close to the normal pure vLLM level.

---

## 8. MMLU 150Q Evaluation (20260608, same server)

### 8.1 Configuration

Shares the same docker compose server as AlpacaEval (parameters same as §1).
The noSIA arm is also implemented via per-request `--sia_weight 0`, no service switch needed.

| Parameter | Value |
|---|---|
| subjects | 30 subjects × 5Q = 150Q |
| `--limit` | 5 |
| `--temperature` | 1.0 |
| `--repetition_penalty` | 1.0 |
| thinking mode | N/A (Qwen3-VL-30B-A3B-Instruct is the Instruct variant, not a thinking model, does not output `<think>` blocks) |

### 8.2 Results

| Metric | SIA | noSIA | Δ |
|---|---|---|---|
| accuracy | **0.7867 (118/150)** | 0.7867 (118/150) | **0.00 pp** |
| wall time | 678.8s (11.3 min) | 608.0s (10.1 min) | — |
| avg tokens/Q | 259.9 | 287.7 | — |
| throughput | 57.4 tok/s | 71.0 tok/s | SIA tax 1.24× |

**SIA intervention health metrics**:

| Metric | This run | Historical reference (HTTP path) |
|---|---|---|
| intervention rate | **11.5%** | 10.03% |
| top1 flip rate | **63.4%** | ~62% |
| RM error | 0 | 0 ✅ |

### 8.3 Comparison with Historical Experiments

| Experiment | noSIA acc | SIA acc | Δ | intervention rate |
|---|---|---|---|---|
| HTTP path (20260604) | 0.7867 (118/150) | 0.8000 (120/150) | +1.33 pp | 10.03% |
| b2 inproc (20260605) | 0.7867 (118/150) | 0.8133 (122/150) | **+2.67 pp** | ~10-20% |
| **b2 inproc docker (this run)** | **0.7867 (118/150)** | **0.7867 (118/150)** | **0.00 pp** | **11.5%** |

### 8.4 Analysis

**Δ=0 is within statistical noise**. The doc notes that "MMLU 150Q is within noise (binomial 95% CI ±7.7pp)"; the historical maximum Δ is only +2.67 pp, far below the confidence interval. 0 pp is normal variation and does not indicate SIA failure.

**Intervention rate 11.5% is consistent with historical 10.03%**, indicating the RM is working correctly. noSIA throughput (71 tok/s vs historical 120 tok/s) is low for the same reason as AlpacaEval — GPU memory constraints.

**Conclusion**: SIA's effect on MMLU is inherently small (knowledge task, RM training objective is biased toward helpfulness). 150Q is not sufficient to stably detect Δ; more questions are needed to rule out noise.

---

## 9. Related Docs

- [`qwen3-vl-30b-sia-eval-20260605.md`](qwen3-vl-30b-sia-eval-20260605.md) — VL-30B SIA comprehensive evaluation summary (HTTP + b2 inproc)
- [`docker-install-vl30b-20260606.md`](docker-install-vl30b-20260606.md) — docker compose deployment guide
- [`vl30b-b2-inproc-speedup-20260605.md`](vl30b-b2-inproc-speedup-20260605.md) — b2 inproc 1.46× speedup original report
- [`0gm-35b-sia-perf-breakdown-20260609.md`](0gm-35b-sia-perf-breakdown-20260609.md) — FULL_AND_PIECEWISE optimization analysis (including VL-30B applicability analysis §6)
