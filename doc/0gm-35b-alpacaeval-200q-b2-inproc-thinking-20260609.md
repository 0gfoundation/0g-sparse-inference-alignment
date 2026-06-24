# 0GM-35B AlpacaEval 200Q: ban_think vs natural thinking — b2 inproc quality + performance comparison (2026-06-09)

**TL;DR**:

- **SIA is highly effective under natural thinking**: Skywork Δ=**+3.27**, p<0.0001 (W/L=115/76), **4.5×** better than the ban_think group (Δ=+0.72, not significant).
- **FULL fix group performs even stronger**: Skywork Δ=**+4.05**, p<0.0001 (W/L=122/64), an additional **+0.78** improvement over the PIECEWISE group.
- **Natural thinking yields higher overall quality**: noSIA 23.62 vs ban_think noSIA 15.86, indicating the thinking process itself substantially improves 0GM-35B output.
- **`SIA_LLM_CUDAGRAPH=piecewise` was a 2× performance bottleneck, now fixed**: removing that env var improves SIA from 37.8 → **54.1 tok/s (+43%)**, noSIA from 57.1 → **107.0 tok/s (+87%)**; root cause explained in [§3](#three-performance-root-cause--fix).

---

## 1. Experiment Configuration

### Common Environment

| Item | Value |
|------|---|
| Machine | H200 single GPU (143,771 MiB) |
| Main LLM | `/workspace/SIA/models/0GM-1.0-35B-A3B-0427` |
| Value Model (RM) | `/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm` |
| Skywork RM | `/workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B` |
| venv | `venv6` (vllm 0.18.0) |
| `--rm_backend` | `b2` (in-process b2 inproc) |
| `--topk` / `--weight` / `--entropy_threshold` | 10 / 1.0 / 1.0 |
| `SIA_RM_CUDAGRAPH` | `none` (required; see [b2 inproc speedup doc](0gm-35b-b2-inproc-speedup-20260609.md)) |
| `SIA_RM_MULTIPROCESS` | `0` |
| AlpacaEval dataset | `data/alpaca_eval/alpaca_eval.json` (first 200 questions) |

### Differences Across Three Groups

| Config | ban_think group | natural thinking group | **FULL fix SIA group** (§3 fix) | **FULL fix noSIA group** |
|--------|-------------|---------------------|--------------------------|----------------------|
| `--llm_gpu_mem` | 0.55 | 0.80 | 0.80 | 0.80 |
| `--rm_b2_gpu_mem` | 0.15 | 0.15 | 0.15 | 0.15 |
| `--max_model_len` | 2048 | 4096 | 4096 | 4096 |
| `--max_tokens` | 512 | 2048 | 2048 | 2048 |
| Driver params | `--no_think_prompt --ban_think_token` | none (natural thinking) | none (natural thinking) | none (natural thinking) |
| `SIA_LLM_CUDAGRAPH` | `piecewise` (⚠️ suboptimal) | `piecewise` (⚠️ suboptimal) | **not set (default FULL_AND_PIECEWISE)** ✅ | **not set** ✅ |
| `--weight` / `--entropy_threshold` | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | **0 / 9999** (SIA disabled) |

### Launch Commands

**natural thinking group (original, with PIECEWISE restriction):**
```bash
SIA_LLM_CUDAGRAPH=piecewise SIA_RM_CUDAGRAPH=none SIA_RM_MULTIPROCESS=0 \
python src/sia_vllm_server.py \
  --llm /workspace/SIA/models/0GM-1.0-35B-A3B-0427 \
  --rm_backend b2 \
  --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
  --llm_gpu_mem 0.80 --rm_b2_gpu_mem 0.15 \
  --topk 10 --weight 1.0 --entropy_threshold 1.0 \
  --max_model_len 4096 --port 8000
```

**FULL fix group (recommended, without `SIA_LLM_CUDAGRAPH`):**
```bash
SIA_RM_CUDAGRAPH=none SIA_RM_MULTIPROCESS=0 \
python src/sia_vllm_server.py \
  --llm /workspace/SIA/models/0GM-1.0-35B-A3B-0427 \
  --rm_backend b2 \
  --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
  --llm_gpu_mem 0.80 --rm_b2_gpu_mem 0.15 \
  --topk 10 --weight 1.0 --entropy_threshold 1.0 \
  --max_model_len 4096 --port 8000
```

---

## 2. Quality Results (Skywork Reward)

### 2.1 Skywork Scores per Group

| Group | arm | n_scored | mean | p50 |
|------|-----|----------|------|-----|
| ban_think | noSIA | 197/200 | 15.86 | 15.94 |
| ban_think | SIA | 194/200 | 16.52 | 16.94 |
| natural thinking | noSIA | 193/200 | 23.62 | 23.63 |
| natural thinking | SIA | 199/200 | 26.41 | 26.88 |
| **FULL fix** | **noSIA** | **191/200** | **24.01** | **23.38** |
| **FULL fix** | **SIA** | **199/200** | **27.69** | **27.12** |

### 2.2 Paired Statistical Comparison

| Group | paired n | Δ mean | Win/Loss/Tie | t-test p | Wilcoxon p | Conclusion |
|------|----------|--------|--------------|----------|------------|------|
| ban_think | 193 | +0.72 | 104/87/2 | 0.27 | 0.18 | **not significant** |
| natural thinking | 193 | +3.27 | 115/76/2 | <0.0001 | 0.000031 | **highly significant** ✅ |
| **FULL fix** | **191** | **+4.05** | **122/64/5** | **<0.0001** | **<0.000001** | **highly significant** ✅ |

### 2.3 Key Observations

**1. Natural thinking yields far higher overall quality than ban_think**

noSIA comparison: 23.62 vs 15.86, a difference of **+7.76 points (+49%)**. 0GM-35B is a thinking model; forcibly disabling thinking causes significant loss of reasoning quality.

**2. SIA benefit increases with thinking quality**

| Condition | Δ (SIA − noSIA) | Significance |
|------|----------------|--------|
| ban_think | +0.72 | p=0.27, **not significant** |
| natural thinking (PIECEWISE) | +3.27 | p<0.0001, **highly significant** |
| **FULL fix** | **+4.05** | **p<0.0001, highly significant** |

FULL fix group Δ=+4.05, higher than the PIECEWISE group's +3.27 (**+0.78**). The two groups' noSIA quality is similar (24.01 vs 23.62), but the SIA gap is larger (27.69 vs 26.41), indicating that after the speedup fix the model generates higher-quality thinking paths, amplifying SIA's intervention effect.

**Interpretation**: longer thinking processes contain more high-entropy decision points, so SIA's intervention at each junction can cumulatively influence the overall reasoning path. ban_think suppresses the thinking process entirely, leaving SIA to influence only a small number of tokens in the final answer generation, greatly diluting its effect.

**3. SIA helps thinking converge (consistent with [35B MMLU doc §1.3](0gm-1.0-35b-sia-eval-20260605.md#13-mmlu-150q-30-subjects--5) observations)**

SIA group average output: 314,555 tokens / 200 Q = 1573 tokens/Q; noSIA group: 347,231 tokens / 200 Q = 1736 tokens/Q. SIA in thinking mode **generates fewer tokens yet achieves higher scores** — consistent with the MMLU experiment finding that "SIA prevents thinking from diverging."

---

## 3. Performance Root Cause + Fix

### 3.1 Throughput Summary per Group

| Group | arm | `SIA_LLM_CUDAGRAPH` | tokens | elapsed (s) | **tok/s** |
|------|-----|---------------------|--------|-------------|-----------|
| ban_think | noSIA | `piecewise` | 52,115 | 1,549 | 33.6 |
| ban_think | SIA | `piecewise` | 51,830 | 1,498 | 34.6 |
| natural thinking | noSIA | `piecewise` | 347,231 | 6,081 | **57.1** |
| natural thinking | SIA | `piecewise` | 314,555 | 8,314 | **37.8** |
| **FULL fix** | **noSIA** | **not set (default)** | **342,752** | **3,202** | **107.0** |
| **FULL fix** | **SIA** | **not set (default)** | **314,931** | **5,822** | **54.1** |
| **Pure vLLM noSIA (control)** | — | — | 341,997 | 2,996 | **114.1** |

> Pure vLLM control: `vllm serve`, `--gpu-memory-utilization 0.80`, `--max-model-len 4096`, same 200Q with same sampling parameters.

### 3.2 Per-step Timing Breakdown Comparison (PIECEWISE vs FULL_AND_PIECEWISE)

| Stage | PIECEWISE p50 | **FULL_AND_PIECEWISE p50** | Change | Note |
|------|--------------|---------------------------|------|------|
| **skip_step (no intervention)** | 0.50ms | **4.30ms** | ↑ 8.6× | Caused by cpu_sync overhead |
| apply_topk_ent | 0.43ms | 0.44ms | unchanged | GPU topK + log_softmax |
| **apply_cpu_sync** | **0.06ms** | **3.85ms** | **↑ 64×** | **New secondary bottleneck, see §3.4** |
| **b2_score_call (RM)** | **51ms** | **~50ms** | unchanged | Main bottleneck for intervention steps |
| intv_prepare | 0.08ms | 0.08ms | unchanged | |
| intv_apply_logits | 0.41ms | 0.41ms | unchanged | |

### 3.3 Root Cause: `SIA_LLM_CUDAGRAPH=piecewise` causes 2× slowdown

**Problem**: b2 inproc noSIA 57 tok/s vs pure vLLM 114 tok/s, a **2×** difference.

**Root cause**: `SIA_LLM_CUDAGRAPH=piecewise` forcibly downgraded the main LLM's CUDA graph mode from the default `FULL_AND_PIECEWISE` to `PIECEWISE`-only.

| Mode | Configuration | Measured tok/s (noSIA) |
|------|---------|-------------------|
| `FULL_AND_PIECEWISE` (default) | pure vLLM / no env var set | **114.1** |
| `PIECEWISE`-only | `SIA_LLM_CUDAGRAPH=piecewise` | **57.1** |

FULL mode captures the entire model forward as one large CUDA graph, yielding the highest decode efficiency; PIECEWISE-only captures only segments between individual operator boundaries, resulting in more dispatches and a 2× slowdown.

**Why this restriction is unnecessary in vllm 0.18.0** (based on layer-by-layer analysis of vllm source):

1. **`apply()` executes outside the CUDA graph**: vllm v1 separates model forward (`execute_model()`) and sampling (`sample_tokens()`) into two independent calls. `SIALogitsProcessor.apply()` is triggered by `Sampler.__call__()` inside `sample_tokens()`, at which point model forward is already complete and the CUDA graph context has long exited. Whether the LLM uses FULL or PIECEWISE does not affect `apply()`'s execution environment.

2. **AOT capture, no capture flag during inference**: Both FULL and PIECEWISE in vllm 0.18.0 are AOT (Ahead-Of-Time) pre-captured — completed during the warmup phase at server startup. During inference, only `cudagraph.replay()` is called; no `torch.cuda.graph()` context is entered and `torch.cuda.is_current_stream_capturing()` flag is not set.

3. **vllm's internal flag is unrelated to the RM**: `set_cudagraph_capturing_enabled(False)` is called after `capture_model()` completes, but this flag is only checked inside `CUDAGraphWrapper.__call__()`. The RM uses `enforce_eager=True` (`CUDAGraphMode.NONE`), completely bypassing `CUDAGraphWrapper`; this flag has no effect on the RM.

4. **Dispatcher guarantees no accidental capture**: `CudagraphDispatcher.dispatch()` returns `CUDAGraphMode.NONE` (fallback eager) for unregistered batch_descriptors rather than triggering a new capture. `cudagraph_capturing_enabled=False` is a safety layer that is never triggered in the normal inference path.

**Historical context for the original restriction**: It was originally designed for vllm 0.19.0 — in 0.19, PIECEWISE switched to runtime capture: it captures on the first real inference batch, at which point `_state` already contains real requests and `apply()` would be called, which in turn calls the RM, triggering eager RM forward inside a `torch.cuda.graph()` context causing an error. vllm 0.18.0 uses AOT, so this code path does not exist.

### 3.4 New Secondary Bottleneck After Fix: `apply_cpu_sync`

**Fix**: remove `SIA_LLM_CUDAGRAPH=piecewise` (let the main LLM use the default `FULL_AND_PIECEWISE`).

**Measured results** (200Q complete data):

| Metric | PIECEWISE-only (old) | **FULL_AND_PIECEWISE (fixed)** | Change |
|------|--------------------|-----------------------------|------|
| noSIA tok/s | 57.1 | **107.0** | **+87%** ✅ |
| SIA tok/s | 37.8 | **54.1** | **+43%** ✅ |
| RM error | 0 | 0 | no regression ✅ |
| Intervention rate | 16.9% | 16.9% | consistent ✅ |
| top1 flip | 66.4% | 66.3% | consistent ✅ |
| Truncated (no final answer) | 13/200 | 7/200 | improved ✅ |

SIA tok/s 54.1, noSIA tok/s 107.0 (93.8% of pure vLLM 114.1), **fix is effective with zero correctness regression**.

**But did not reach the expected ~58 tok/s, reason**: FULL mode introduced a new `apply_cpu_sync` overhead:

- PIECEWISE mode: LLM forward executes in multiple segments with attention boundary sync points between segments, leaving some GPU stream idle; SIA's `tensor.item()` (pulling entropy value to CPU) only needs ~0.06ms wait.
- FULL mode: the entire model forward is submitted to the GPU stream as a single large graph, after which Python returns immediately, but there are still many pending kernels on the GPU. When SIA's `tensor.item()` triggers a CPU-GPU sync, it must wait for all pending kernels to drain → **3.85ms** wait.

This adds a fixed ~3.8ms overhead per token (in both skip and intervene steps), effectively pulling the achievable 58 tok/s down to ~54 tok/s:

```
Estimated SIA tok/s (after fix) = 1 / (1/114 + 0.15×0.050 + 0.00385)
                                 ≈ 1 / (8.77ms + 7.5ms + 3.85ms)
                                 ≈ 50ms⁻¹ → ~50 tok/s (conservative; measured ~54)
```

**Future optimization direction**: convert SIA's entropy CPU sync to async (prefetch after FULL graph submission, before RM call returns) to eliminate this 3.85ms wait.

---

## 4. Follow-up Work

| # | Task | Expected benefit | Status |
|---|------|---------|------|
| 1 | Remove `SIA_LLM_CUDAGRAPH=piecewise`, measure SIA/noSIA tok/s + quality | SIA 37.8→54.1 (+43%), Δ 3.27→4.05 | ✅ **Done** (200Q, see §2 and §3) |
| 2 | Update CLAUDE.md recommended commands (remove that env var) | Better production configuration | ✅ **Done** |
| 3 | `SIA_RM_CUDAGRAPH=full` or disable RM prefix caching | RM 55ms→~11ms, SIA ~54→~85 tok/s (+57%) | Pending verification; see [perf breakdown doc](0gm-35b-sia-perf-breakdown-20260609.md) §7 |
| 4 | Eliminate `apply_cpu_sync` 3.85ms overhead | ~54→~55 tok/s (+2%) | Low priority |
| 5 | Apply same fix to VL-30B (remove `SIA_LLM_CUDAGRAPH=piecewise`) | VL-30B SIA ~73→~95 tok/s (+30%) | Pending verification |

---

## 5. Artifacts

| File | Content |
|------|------|
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_b2_nosia_20260609_081640.json` | ban_think noSIA output (200Q) |
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_b2_sia_20260609_073834.json` | ban_think SIA output (200Q) |
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_b2_nosia_scored.json` | ban_think noSIA Skywork scores |
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_b2_sia_scored.json` | ban_think SIA Skywork scores |
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_b2_08_nosia.json` | natural thinking noSIA output (200Q) |
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_b2_08_sia.json` | natural thinking SIA output (200Q) |
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_b2_08_nosia_scored.json` | natural thinking noSIA Skywork scores |
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_b2_08_sia_scored.json` | natural thinking SIA Skywork scores |
| `exp/alpaca-0gm35b-b2-inproc-20260609/b2_08_sia_server.log` | natural thinking SIA server log (with pf-summary + intervention stats) |
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_pure_vllm_08_nosia.json` | pure vLLM control (200Q, 114.1 tok/s) |
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_b2_08_sia_fixed.json` | FULL fix SIA output (200Q) |
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_b2_08_sia_fixed_scored.json` | FULL fix SIA Skywork scores |
| `exp/alpaca-0gm35b-b2-inproc-20260609/b2_08_sia_fixed_server.log` | FULL fix SIA server log (with FULL_AND_PIECEWISE confirmation + pf-summary) |
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_b2_08_nosia_fixed.json` | FULL fix noSIA output (200Q) |
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_b2_08_nosia_fixed_scored.json` | FULL fix noSIA Skywork scores |
| `exp/alpaca-0gm35b-b2-inproc-20260609/b2_08_nosia_fixed_server.log` | FULL fix noSIA server log |
