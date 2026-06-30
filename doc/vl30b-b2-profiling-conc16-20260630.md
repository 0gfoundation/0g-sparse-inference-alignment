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
