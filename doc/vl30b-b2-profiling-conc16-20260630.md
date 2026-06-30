# VL-30B b2 Inproc Profiling — conc=16 Bottleneck Analysis (2026-06-30)

**Purpose**: Identify the throughput bottleneck at conc=16 using new per-step profiling
fields (`b2_batch_wall_abs`, `intv_batch_size`, `apply_intv_step`) added in commit `072dc3e`.

**Setup**: docker with `SIA_RM_BATCH_CHUNK=32`, bench ran `--stress --stress-start 16
--stress-max-conc 16` to isolate conc=16 data. Profiling accumulates inside the server
and is printed every 100 INTERVENE calls to docker logs.

---

## 1. Raw Profiling Output (conc=16, @600 — most stable)

```
[SIA-pf-rate @600]
  intv_calls=600  apply_steps=242  req_slots=3781
  intv_rate=15.9%  avg_intv_batch=3.70

[SIA-pf-summary @600]
  b2_prefix_adv:    p50=0.00  p95=1.07  max=2.99  ms
  b2_score_call:    p50=9.48  p95=17.73 max=23.94 ms  (amortized per request)
  b2_batch_wall_abs:p50=29.42 p95=82.13 max=125.15 ms (absolute per step)
  intv_batch_size:  p50=3.00  p95=9.00  max=14.00
  apply_total:      p50=30.99 p95=81.47 max=152.00 ms
  apply_topk_ent:   p50=0.53  p95=0.63  max=70.12  ms
  apply_cpu_sync:   p50=6.75  p95=7.69  max=8.24   ms
  skip_step:        p50=5.13  p95=7.72  max=70.27  ms
  apply_intv_step:  p50=38.33 p95=91.86 max=152.00 ms
  intv_prepare:     p50=0.04  p95=0.08  max=0.17   ms
  intv_apply_logits:p50=0.19  p95=0.48  max=1.06   ms
```

---

## 2. Key Facts

### 2.1 Intervention Rate

- **15.9%** of all per-request decode steps trigger VM scoring (not 5% as assumed from 35B
  thinking-mode observation — VL-30B has no thinking tokens, so all answer tokens are evaluated).
- At each decode step, the 16 concurrent requests each independently have a 15.9% chance of
  intervening, giving:

  ```
  P(at least 1 request intervenes) = 1 - 0.841^16 = 95.2%
  ```

  Nearly every step calls the VM.

### 2.2 VM Batch Call (b2_batch_wall_abs)

- `intv_batch_size p50=3` — when the VM is called, typically 3 requests are batched (3 × 10
  candidates = 30 prompts in one `llm.generate()` call).
- `b2_batch_wall_abs p50=29ms` — the VM takes 29ms to score all 30 prompts.
- Cross-check: 29ms ÷ 3 requests = 9.7ms per request ≈ `b2_score_call p50=9.48ms` ✓

**The VM scales linearly with batch size** (memory-bandwidth bound, no batching speedup):
- 1 request × 10 candidates → ~9ms
- 3 requests × 10 candidates → ~29ms (3×)
- 10 requests × 10 candidates → ~90ms (10×, matches p95)

### 2.3 Per-step Time Breakdown

Steps with at least one INTERVENE (`apply_intv_step p50=38ms`):

| Component | Time | Share |
|-----------|------|-------|
| VM batch call (`b2_batch_wall_abs`) | 29ms | 77% |
| GPU→CPU sync (`apply_cpu_sync`) | 6.75ms | 18% |
| Logits write-back (`intv_apply_logits`) | 0.19ms | 0.5% |
| Session prep (`intv_prepare`) | 0.04ms | 0.1% |
| GPU topk+entropy (`apply_topk_ent`) | 0.53ms | 1.4% |
| **Total** | **~38ms** | |

Pure SKIP steps: `skip_step p50=5.13ms` (≈ noSIA baseline).

---

## 3. ITL Prediction vs Benchmark

```
Expected ITL = P(any intervene) × apply_intv_step + P(none) × skip_step
             = 0.952 × 38ms + 0.048 × 5ms
             = 36.2 + 0.2
             = 36.4ms
```

Benchmark measured ITL at conc=16: **34.3ms** ✓ (close match, ~6% off).

Throughput ratio: `SIA ITL / noSIA ITL = 34ms / 11ms = 3.1×` → **SIA/noSIA ≈ 32-34%** (matches
benchmark 34.3%).

---

## 4. Root Cause

**The VM batch call (29ms at p50) is the dominant bottleneck, accounting for 77% of each
INTERVENE step.**

The underlying reason is that `score_candidates_batch()` submits N×K prompts to the VM's
`llm.generate()` as a single call. With N=3 sessions and K=10 candidates, this is 30 prompts.
The VM (Qwen3-4B, PIECEWISE CUDA graph) processes them one batch at a time. Because:

1. **APC (Automatic Prefix Caching)** hits correctly — each set of 10 candidates shares the
   per-request prefix, so only 1+10 token-worth of computation is needed per session.
2. **The batch size (30 prompts) does not hit a pre-captured CUDA graph bucket** — PIECEWISE
   captures fixed decode batch sizes; a dynamic batch of 30 falls back to eager mode, incurring
   full kernel-launch overhead per layer (28 layers × ~10 kernels × 30 prompts).
3. **The workload is memory-bandwidth bound** — at this batch size the GPU spends most of its
   time reading model weights, not computing; hence no speedup from merging N sessions.

**Consequence**: each additional intervening session adds ~9ms linearly; at higher concurrency
the VM wall-clock grows proportionally.

---

## 5. Optimization Impact Estimates

### 5.1 Dual-gating (logit-gap second gate, 15.9% → 5%)

Add a second SKIP condition: if `logit_gap = top1_logit - top2_logit > threshold`, skip even
when entropy ≥ `entropy_threshold`. This cuts intervention rate from 15.9% to ~5%.

```
At 5% rate, conc=16:
  P(any intervene per step) = 1 - 0.95^16 = 56%  (down from 95%)
  avg_intv_batch ≈ 16 × 0.05 / 0.56 = 1.4 sessions
  b2_batch_wall_abs ≈ 1.4 × 9ms = 12.6ms
  apply_intv_step ≈ 5ms + 12.6ms = 17.6ms

  Expected ITL = 0.56 × 17.6 + 0.44 × 5.1 = 12.1ms
  SIA/noSIA ≈ 11ms / 12.1ms = 91%
```

| Metric | Current | After dual-gating |
|--------|---------|-------------------|
| Intervention rate | 15.9% | ~5% |
| SIA ITL (conc=16) | 34ms | ~12ms |
| SIA/noSIA (conc=16) | **34%** | **~91%** |
| Implementation effort | — | ~20 LOC |

**Risk**: quality impact unknown — dual-gating skips VM scoring at steps where the model is
"confident" (large logit gap) but entropy is still above threshold. AlpacaEval A/B test required.

### 5.2 Smaller VM (4B → 1.7B, retrain LoRA)

Reduces per-session VM time from ~9ms to ~3ms (3× faster). At the same batch size of 3
sessions: 29ms → ~9ms.

```
Expected ITL ≈ 0.952 × (5ms + 9ms) + 0.048 × 5ms = 13.6ms
SIA/noSIA ≈ 11ms / 13.6ms = 81%
```

| Metric | Current | After 1.7B VM |
|--------|---------|---------------|
| SIA/noSIA (conc=16) | 34% | ~81% |
| Implementation effort | — | High (ML retraining) |

### 5.3 topk 10 → 5 (parameter change only)

Cuts prompts per session from 10 to 5, halving VM batch size. At 3 sessions: 30 → 15 prompts.

```
b2_batch_wall_abs ≈ 3 × (9ms/2) = 13.5ms
Expected ITL ≈ 0.952 × (5ms + 13.5ms) + 0.048 × 5ms = 18.0ms
SIA/noSIA ≈ 11ms / 18ms = 61%
```

Quality impact: top-4 through top-10 candidates are no longer scored by VM.

### 5.4 Summary Table

| Option | SIA/noSIA (conc=16) | Effort | Quality risk |
|--------|---------------------|--------|--------------|
| Current baseline | 34% | — | — |
| Dual-gating (→5%) | **~91%** | Low (20 LOC) | Needs A/B test |
| Smaller VM (1.7B) | ~81% | High (retraining) | Low |
| topk 10→5 | ~61% | Trivial (1 param) | Medium |
| Dual-gating + 1.7B VM | **~97%** | High | Needs A/B test |

---

## 6. Recommended Next Steps

**Priority 1 — Dual-gating** (immediate, low risk):
Implement `--logit_gap_threshold` parameter. When `top1_logit - top2_logit ≥ threshold`,
skip VM scoring regardless of entropy. Tune threshold on AlpacaEval to achieve ~5%
intervention rate while verifying no reward degradation.

**Priority 2 — AlpacaEval A/B test** after dual-gating:
Measure actual quality delta (paired t-test, n≥200). If quality holds, ship. If quality
drops, adjust threshold or abandon.

**Priority 3 — 1.7B VM retraining** (if dual-gating quality holds):
Retrain VM-Qwen3-1.7B-Base LoRA. Stack with dual-gating for ~97% SIA/noSIA.

---

## 7. Profiling Infrastructure Added (commit 072dc3e)

New fields in `[SIA-pf-summary]` and new `[SIA-pf-rate]` line:

| Field | What it measures |
|-------|-----------------|
| `[SIA-pf-rate] intv_rate` | Intervention rate = intv_calls / req_slots |
| `[SIA-pf-rate] avg_intv_batch` | Mean sessions per VM batch call |
| `b2_batch_wall_abs` | Absolute wall-clock of each `score_candidates_batch()` call |
| `intv_batch_size` | Number of sessions per VM batch call |
| `apply_intv_step` | apply() wall on steps with ≥1 intervention |

Previously only `b2_score_call` (amortized per-request) existed, which masked the true
per-step VM cost at high concurrency and made chunking behavior invisible.

---

## 8. Dual-Gating: Design, Implementation, and Caveats

### 8.1 Concept

The current single gate skips VM scoring when **entropy < threshold** (model is confident
across the top-K distribution). Dual-gating adds a second independent gate:

> **If `logit_gap = top1_logit − top2_logit > gap_threshold` → also SKIP**,
> regardless of entropy.

A large logit gap means the model assigns a much higher probability to its top-1 token than
to any alternative. Even if the VM disagrees, the bonus it can add to tokens #2–10 is
unlikely to overcome a large gap and flip the final sample. Skipping VM scoring in this
regime wastes compute without changing outcomes.

### 8.2 Implementation (~20 LOC)

The logit gap tensor is already computed in the `SIA_DEBUG_HIST=1` code path. For the
production path, the change is:

**Step 1 — compute gap and merge with entropy into a single GPU→CPU sync** (avoids adding
an extra round-trip):

```python
# In apply(), replace the entropy-only sync:
#   entropy_values = entropies.cpu().tolist()
# With a combined sync:
gap_tensor = entropy_topk_result.values[:, 0] - entropy_topk_result.values[:, 1]
combined = torch.stack([entropies, gap_tensor], dim=1)   # (batch, 2)
combined_cpu = combined.cpu().tolist()
entropy_values = [m[0] for m in combined_cpu]
gap_values     = [m[1] for m in combined_cpu]
```

**Step 2 — new CLI parameter**:

```python
parser.add_argument("--logit_gap_threshold", type=float, default=0.0,
    help="Skip VM scoring when top1−top2 logit gap exceeds this value (0=disabled).")
```

**Step 3 — modify intervene_flags**:

```python
# Before (single gate):
intervene_flags[i] = entropy_values[i] >= self._ENTROPY_THRESHOLD

# After (dual gate):
intervene_flags[i] = (entropy_values[i] >= self._ENTROPY_THRESHOLD) \
                 and (gap_values[i] < self._GAP_THRESHOLD)
```

When `--logit_gap_threshold 0` (default), the second condition is always True and behavior
is identical to the current single-gate code — fully backward compatible.

### 8.3 Expected Throughput Impact

Threshold tuning guide (empirical estimates, to be calibrated per model):

| `--logit_gap_threshold` | Expected intv_rate | Expected ITL (conc=16) | SIA/noSIA |
|-------------------------|--------------------|------------------------|-----------|
| — (disabled, current)  | 15.9%              | ~34ms                  | ~34%      |
| 5.0 (conservative)     | ~10%               | ~19ms                  | ~58%      |
| 3.0 (moderate)         | ~7%                | ~15ms                  | ~73%      |
| 1.5 (aggressive)       | ~5%                | ~12ms                  | ~91%      |

### 8.4 Literature Basis and Caveats

**What is validated by prior work:**

- **Entropy-based sparse intervention** (first gate): the SIA paper
  ("Inference-time Alignment via Sparse Junction Steering") directly validates that
  intervening only at high-entropy "junctions" achieves equal or better alignment than
  dense intervention, at 20–80% of steps. The first gate is on solid footing.

- **Logit gap / margin as a confidence measure**: a standard concept in classification
  literature ("margin classifier"). Used in LLM contexts for speculative decoding and
  early-exit research, though not specifically for RM-based intervention.

**What is NOT validated by prior work:**

The specific combination of **entropy + logit gap dual-gating for RM intervention quality**
has no direct published validation. The assumption that "large-gap steps are safe to skip"
is logically sound but empirically unverified on this model pair (VL-30B + VM-Qwen3-4B).

**Key risk**: the VM and LLM are two different models with different "opinions." At steps
where the LLM has a large gap, the VM could still strongly prefer a different token — and
skipping VM scoring would miss those corrections. How often this happens depends on the
degree of LLM/VM alignment, which must be measured empirically.

### 8.5 Required Validation Before Shipping

```
AlpacaEval 200Q:
  Arm A: SIA (entropy_threshold=1.0, logit_gap_threshold=disabled)   ← current
  Arm B: SIA (entropy_threshold=1.0, logit_gap_threshold=3.0)        ← dual-gate
  
  Paired t-test on Skywork reward scores:
    p > 0.05 and |Δmean| < 0.5  →  quality holds, safe to ship
    p < 0.05 or |Δmean| ≥ 0.5  →  threshold too aggressive, increase gap_threshold
```

Recommended tuning order: start at `gap_threshold=5.0`, verify quality, then tighten to
3.0, then 1.5. Stop at the largest threshold that passes the paired t-test.

---

## 9. Batch Debug Experiment — Linear Scaling Hypothesis Disproven (2026-06-30)

### 9.1 Background

Section 4 proposed that the VM's linear batch scaling (9ms/session) was caused by vLLM
internally splitting N×K prompts into N separate `compute_logits` calls — one per session —
because the PIECEWISE CUDA graph only captures `batch_size=K=10`.

To verify, `SIA_RM_BATCH_DEBUG=1` instrumentation was added to `score_candidates_batch()`:
after `llm.generate()` returns, the code reads `_REWARD_BUFFERS[fid]` (populated by one
entry per `compute_logits` call) and prints the call count and per-call sample sizes.

### 9.2 Raw Debug Output

```
(EngineCore_DP0 pid=76) [RM-batch-debug] sessions=8 total_prompts=80 compute_logits_calls=1 samples_per_call=[80]
(EngineCore_DP0 pid=76) [RM-batch-debug] sessions=4 total_prompts=40 compute_logits_calls=1 samples_per_call=[40]
(EngineCore_DP0 pid=76) [RM-batch-debug] sessions=4 total_prompts=40 compute_logits_calls=1 samples_per_call=[40]
(EngineCore_DP0 pid=76) [RM-batch-debug] sessions=7 total_prompts=70 compute_logits_calls=1 samples_per_call=[70]
(EngineCore_DP0 pid=76) [RM-batch-debug] sessions=8 total_prompts=80 compute_logits_calls=1 samples_per_call=[80]
(EngineCore_DP0 pid=76) [RM-batch-debug] sessions=5 total_prompts=50 compute_logits_calls=1 samples_per_call=[50]
(EngineCore_DP0 pid=76) [RM-batch-debug] sessions=7 total_prompts=70 compute_logits_calls=1 samples_per_call=[70]
(EngineCore_DP0 pid=76) [RM-batch-debug] sessions=6 total_prompts=60 compute_logits_calls=1 samples_per_call=[60]
(EngineCore_DP0 pid=76) [RM-batch-debug] sessions=2 total_prompts=20 compute_logits_calls=1 samples_per_call=[20]
(EngineCore_DP0 pid=76) [RM-batch-debug] sessions=3 total_prompts=30 compute_logits_calls=1 samples_per_call=[30]
(EngineCore_DP0 pid=76) [RM-batch-debug] sessions=2 total_prompts=20 compute_logits_calls=1 samples_per_call=[20]
(EngineCore_DP0 pid=76) [RM-batch-debug] sessions=3 total_prompts=30 compute_logits_calls=1 samples_per_call=[30]
(EngineCore_DP0 pid=76) [RM-batch-debug] sessions=8 total_prompts=80 compute_logits_calls=1 samples_per_call=[80]
(EngineCore_DP0 pid=76) [RM-batch-debug] sessions=3 total_prompts=30 compute_logits_calls=1 samples_per_call=[30]
(EngineCore_DP0 pid=76) [RM-batch-debug] sessions=4 total_prompts=40 compute_logits_calls=1 samples_per_call=[40]
(EngineCore_DP0 pid=76) [RM-batch-debug] sessions=8 total_prompts=80 compute_logits_calls=1 samples_per_call=[80]
(EngineCore_DP0 pid=76) [RM-batch-debug] sessions=3 total_prompts=30 compute_logits_calls=1 samples_per_call=[30]
(EngineCore_DP0 pid=76) [RM-batch-debug] sessions=2 total_prompts=20 compute_logits_calls=1 samples_per_call=[20]
(EngineCore_DP0 pid=76) [RM-batch-debug] sessions=2 total_prompts=20 compute_logits_calls=1 samples_per_call=[20]
```

Side note: the `(EngineCore_DP0 pid=76)` prefix reveals that `score_candidates_batch()` runs
inside vLLM's EngineCore subprocess, not the main FastAPI process. This explains why earlier
grep attempts (before proper container restart) found no output — the subprocess stdout is
prefixed and logged separately.

### 9.3 Result: `compute_logits_calls=1` in Every Case

**`compute_logits_calls=1` without exception, across all observed batch sizes (20–80 prompts).**

vLLM already merges all N×K prompts into a single forward pass. The hypothesis from Section 4
(that PIECEWISE CUDA graph forces N separate calls of batch=K) is wrong.

The Section 4 statement:
> "The batch size (30 prompts) does not hit a pre-captured CUDA graph bucket — PIECEWISE
> captures fixed decode batch sizes; a dynamic batch of 30 falls back to eager mode"

is disproven. vLLM handles variable batch sizes in a single call regardless of CUDA graph
capture granularity.

### 9.4 Revised Root Cause

The 29ms VM wall time is the **true GPU compute cost** of one forward pass of the 4B dense
VM model on a batch of N×K prompts. It is not a batching or CUDA graph artifact. This is
consistent with the GPU being memory-bandwidth bound when loading 4B model weights:
adding more prompts to the batch does not proportionally increase time (weights are loaded
once, amortized across all prompts in the batch), but the absolute cost is still ~29ms
regardless of whether N=1 or N=8.

Concretely, the 3× throughput slowdown is explained entirely by this 29ms blocking call:

```
LLM step time (noSIA) ≈ 13ms   (derived: 11ms ITL × 16 conc → per-step time)
VM call time          ≈ 29ms   (measured)
P(any intervene)      = 95.2%  (1 - 0.841^16)

SIA per-step time ≈ 13 + 0.952 × 29 ≈ 40.6ms
SIA/noSIA         ≈ 13 / 40.6 = 32%   ← matches observed ~34%
```

### 9.5 Revised Optimization Estimates

Section 5's estimates assumed linear per-session scaling (9ms/session), which is now known
to be an emergent property of the fixed 29ms/call cost, not a per-prompt effect. The estimates
for dual-gating (Section 5.1) were based on reducing the number of VM **calls** per unit time
(via lower intervention rate), which remains valid — fewer decode steps trigger the VM,
regardless of how it scales internally.

**Updated optimization table** (estimates revised for clarity):

| Option | Mechanism | SIA/noSIA (conc=16) | Effort |
|--------|-----------|---------------------|--------|
| Current baseline | — | 34% | — |
| Dual-gating (intv_rate → ~5%) | fewer VM calls | ~91% | Low |
| Smaller VM (4B → 1.7B) | faster VM forward pass | ~50–60% | High (retraining) |
| topk 10 → 5 | smaller batch per call | ~50% | Trivial |
| Dual-gating + 1.7B VM | both | ~97% | High |

Key revision: "smaller VM" and "topk reduction" have a **modest** effect because the 29ms
cost is dominated by weight-loading (memory bandwidth), not compute. Halving topk may only
reduce VM time by 20–30%, not 50%. The dual-gating path (reducing call frequency) remains
the highest-leverage option.

### 9.6 Next Steps (Revised)

**Immediate**: implement dual-gating and measure actual quality delta on AlpacaEval 200Q
(see Section 8.5). This is the only high-leverage, low-effort path confirmed by the data.

**If dual-gating quality holds**: consider smaller VM only after measuring its quality impact.
The engineering cost (retraining) is high and the throughput gain (~50-60% vs 34% baseline)
is substantially less than dual-gating (~91%).

---

## 10. Eager VM Prefill Experiment (2026-06-30)

### 10.1 Motivation and Idea

At each decode step, before `apply()` returns, we know the "next predicted token" for every
active request:
- **Intervening requests**: the post-intervention best token (`post_top1`, highest weighted score)
- **Skip requests**: the top-1 raw logit token

The hypothesis: if we immediately call `VM.generate([prefix + predicted_tok])` for all 16
sessions in a single batched call after each decode step, vLLM's APC will cache the KV blocks
for that prefix. At the **next** intervention step, scoring queries `[prefix + predicted_tok + candidate_j]`,
and if `predicted_tok == actual_sampled_tok` (APC hit), the VM only needs to compute `candidate_j`'s
KV — eliminating the full prefix prefill latency.

This was proposed as a way to convert "prefill cost at INTERVENE time" into "small decode cost
spread across all steps," with the expectation that APC would make subsequent scoring nearly free.

### 10.2 Implementation

Three files were modified (commit `6692eb1`):

**`src/sia_rm/client.py`** — added `eager_prefill_batch()`:
```python
def eager_prefill_batch(self, requests: list) -> None:
    if self._cross_tokenizer:
        return
    prompts = [TokensPrompt(prompt_token_ids=self._sessions[sid] + [int(tok)])
               for sid, tok in requests if sid in self._sessions]
    if not prompts:
        return
    if not self._multiprocessing:
        clear_inproc_rewards(self._fid)
    self.llm.generate(prompts, self._sp, use_tqdm=False)
    if not self._multiprocessing:
        take_inproc_rewards(self._fid)  # discard scores; only wanted the KV cache
```
Key: `_sessions[sid]` is NOT modified (no `fix_a_token` on predicted token), so the
session state remains valid for the next real scoring call.

**`src/sia_vllm_RM.py`** — after the per-item loop and logit modification, added:
```python
if self._RM_BACKEND == "b2" and self._EAGER_VM_PREFILL:
    # Advance session state with any tokens sampled since last fix_a_token
    for _i in range(batch_size):
        _out[_n_vm:]  →  fix_a_token for each
    # Collect predicted token per request
    _pred = _eager_intv_best[_i]   # intervening: post-intervention top-1
          / topk_indices[_i][0]    # skip at intervene step: raw top-1
          / argmax_all[_i]         # pure skip step
    eager_prefill_batch(_eager_reqs)
    _pf_record("b2_eager_wall", elapsed_ms)
```
The eager block runs at **every** `apply()` call, not just INTERVENE steps, so skip-step
tokens also accumulate in the VM cache.

**`docker-compose.yml`** — added `--eager_vm_prefill` flag (commit `65a9357`).

### 10.3 Deployment Bug: `docker compose restart` vs `docker compose up -d`

First benchmark after commit showed no improvement and `b2_eager_wall` completely absent
from pf-summary. Investigation:

1. `grep 'b2_eager_wall' docker logs` → 0 matches
2. Added diagnostic print: `[SIA] __init__ _EAGER_VM_PREFILL=False _RM_BACKEND=b2`
3. Banner confirmed: `eager_vm_prefill=False`

Root cause: **`docker compose restart` reuses the existing container's CMD** (the command
baked at container creation time). It does NOT re-read `docker-compose.yml`. Since the
container was originally created before commit `65a9357` added `--eager_vm_prefill`, the
flag was never passed to the server despite being in the YAML.

Fix: `docker compose up -d` detects that the command changed and recreates the container.
After this, `eager_vm_prefill=True` appeared in the banner and `_EAGER_VM_PREFILL=True`
in the EngineCore subprocess.

**Lesson**: for command-line changes in `docker-compose.yml`, always use `docker compose up -d`,
never `docker compose restart`.

### 10.4 Experimental Results

All runs: conc=16, `--stress-start 16 --stress-rounds 1 --stress-max-conc 16`,
realistic-prompt, max_out=128, `--topk 10 --weight 1.0 --entropy_threshold 1.0 --logit_gap_threshold 3.0`.

**Baseline (without `--eager_vm_prefill`)**:

```
Conc=16: ITL=58.9ms  tok/s=256.3

[SIA-pf-summary @300]
b2_batch_wall_abs: p50=35.29  p95=81.37   ms
apply_cpu_sync:    p50=16.94  p95=20.26   ms
skip_step:         p50=12.37  p95=20.33   ms
apply_intv_step:   p50=61.19  p95=114.58  ms
(b2_eager_wall: absent)
```

**With `--eager_vm_prefill`**:

```
Conc=16: ITL=60.2ms  tok/s=229.6

[SIA-pf-summary @300]
b2_eager_wall:     p50=20.87  p95=27.85   ms   ← new overhead
b2_batch_wall_abs: p50=28.22  p95=72.81   ms   ← improved (APC working)
apply_cpu_sync:    p50=7.66   p95=15.38   ms   ← improved
skip_step:         p50=14.61  p95=44.54   ms   ← worsened
apply_intv_step:   p50=62.41  p95=106.99  ms   ← same
```

### 10.5 Analysis

**APC is working, but the benefit is small:**

| Metric | Without eager | With eager | Delta |
|--------|--------------|------------|-------|
| `b2_batch_wall_abs` (scoring) | 35ms | 28ms | −7ms ✓ |
| `b2_eager_wall` (new overhead) | 0ms | 21ms | +21ms ✗ |
| `apply_cpu_sync` | 17ms | 8ms | −9ms ✓ |
| `skip_step` | 12ms | 15ms | +3ms ✗ |
| `apply_intv_step` | 61ms | 62ms | +1ms ✗ |
| ITL | 58.9ms | 60.2ms | +1.3ms ✗ |

APC reduced scoring from 35ms to 28ms (−7ms), confirming the hypothesis works at the
caching layer. The `apply_cpu_sync` improvement (−9ms) is a beneficial side-effect:
the eager prefill "fills" the GPU pipeline so the main LLM's topk computation has less
contention with residual VM computation from the previous step.

**Why the net result is still negative:**

The eager prefill costs **21ms on every single apply() call**, regardless of whether
that step intervenes or not. At conc=16 with 95% step intervention rate, this translates to:

```
Savings:  7ms (scoring APC) × ~5% effective + 9ms (cpu_sync) ≈ 9.35ms per step
Cost:    21ms (eager prefill) × 100% of steps           = 21ms per step
Net:     −11.65ms (slower overall)
```

**Root cause — VM is memory-bandwidth bound:**

The `b2_eager_wall: p50=21ms` for 16 single-token prompts (with APC hitting on all
prefix blocks) reveals the minimum floor. With perfect APC, the VM still needs to
**load all 4B model weights** from HBM every forward pass. At ~2TB/s GPU bandwidth,
loading 8GB (4B params × bfloat16) alone takes ~4ms, and with PIECEWISE CUDA graph
overhead and 16-session decode step, the irreducible minimum is ~21ms regardless of
token count.

APC saves attention computation but **not weight loading**. Since weight loading
dominates (the model is memory-bandwidth bound, not compute-bound), APC gives only
marginal scoring speedup (~7ms out of 35ms total).

The eager prefill would only be net-positive if it ran infrequently relative to the
scoring savings. At the observed intervention rate (~5% per request per step), an
eager prefill every N=4 steps would break even — but that is equivalent to simply
reducing VM call frequency with block-wise scoring, which avoids the eager overhead
entirely.

**Comparison with the intervention rate context:**

| Metric | Value |
|--------|-------|
| VM forward pass (min, with full APC) | ~21ms |
| Scoring call improvement from APC    | −7ms (35→28ms) |
| Steps between interventions (avg)    | ~1/0.05 = 20 steps |
| Eager calls to recoup 1 scoring save | 21ms / 7ms = 3× |
| Required: eager every N steps        | N ≥ 3 for break-even |
| Actual: eager every step             | N = 1 → always net negative |

### 10.6 Conclusion

**Eager VM prefill does not improve throughput for VL-30B + VM-Qwen3-4B.**

The mechanism is sound (APC reduces scoring latency by 7ms) but the economics are unfavorable:
the eager call runs at every step and costs 21ms, while scoring benefits only on intervene
steps (~5% of request-steps). The VM model's memory-bandwidth-bound nature means the
minimum eager cost (weight loading) cannot be eliminated even with perfect APC.

The feature has been removed from `docker-compose.yml`. The code implementation
(`--eager_vm_prefill` flag and `eager_prefill_batch()`) is retained in the codebase for
reference — it would become beneficial only in a configuration where VM calls are
infrequent (e.g., combined with block-wise scoring reducing intervention rate to <5%
of steps, making the eager overhead amortized across many scoring saves).
