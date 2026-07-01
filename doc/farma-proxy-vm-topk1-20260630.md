# FaRMA Proxy Experiment: vm_topk=1 (2026-06-30)

## 1. Background

The vocabulary-wide scoring head (FaRMA, arxiv 2502.04517) replaces K separate VM forward
passes with a single forward that outputs reward scores for the entire vocabulary, then reads
off scores for each candidate by index. The theoretical reduction in VM computation is K× (K=topk=10).

Before committing to retraining the VM with this architecture, we ran a proxy experiment to
estimate the actual throughput gain in our batched b2 inproc setup.

**Prior theoretical estimate**: only ~1.1–1.2× improvement, based on the assumption that the
VM is memory-bandwidth bound and VM batch size is always 16×10=160 per intervention step.

## 2. Experimental Design

**Key insight**: To isolate the effect of reducing VM candidates from K=10 to K=1 per request
without changing the intervention rate, we introduced a `--vm_topk` parameter (separate from
`--topk`):

- `--topk 10`: entropy gate still uses 10 candidates → **intervention rate unchanged**
- `--vm_topk 1`: only 1 candidate (top-1 from LLM logits) is sent to VM per request
- VM scores the 1 candidate, modifies that token's logit; the other 9 candidates fill to `-inf`
- This preserves apple-to-apple comparison on **when** interventions happen, changing only
  **how many candidates** the VM processes per call

This is the closest approximation of FaRMA's throughput characteristics achievable without
retraining the VM:
- FaRMA: 1 VM forward → vocab-size reward vector → index 10 scores → full quality preserved
- This proxy: 1 VM forward → 1 score → 1 candidate modified → **quality degraded** (acceptable for throughput test)

## 3. Configuration

| Parameter | Value |
|-----------|-------|
| Main LLM | Qwen3-VL-30B-A3B-Instruct |
| VM | VM-Qwen3-4B-merged-for-vllm |
| `--topk` | 10 (entropy gate, unchanged) |
| `--vm_topk` | 1 (candidates sent to VM) |
| `--entropy_threshold` | 1.0 (production value) |
| `--weight` | 1.0 |
| `--rm_backend` | b2 (inproc) |
| `SIA_RM_CUDAGRAPH` | piecewise |
| `SIA_RM_MULTIPROCESS` | 0 |
| Concurrency sweep | 1 / 2 / 4 / 8 / 16 |
| Input tokens | ~422 |
| Max output tokens | 128 |

## 4. Results

### 4.1 Benchmark (bench_30b.py --only-concurrency)

| Conc | ITL mean | Out tok/s | Req Lat |
|------|----------|-----------|---------|
| 1    | 9.0ms    | 80.6      | 1588ms  |
| 2    | 10.5ms   | 183.5     | 1394ms  |
| 4    | 12.8ms   | 305.4     | 1671ms  |
| 8    | 16.2ms   | 475.0     | 2143ms  |
| 16   | 20.2ms   | 759.8     | 2680ms  |

### 4.2 pf-summary (@ step 1800)

```
b2_prefix_adv:      p50=0.00  p95=1.02  max=2.90
b2_score_call:      p50=3.50  p95=9.56  max=32.81
b2_batch_wall_abs:  p50=9.74  p95=13.91 max=32.81
intv_batch_size:    p50=1.00  p95=6.00  max=15.00
apply_total:        p50=5.79  p95=20.35 max=73.60
apply_topk_ent:     p50=0.49  p95=0.59  max=73.44
apply_cpu_sync:     p50=4.55  p95=7.67  max=8.77
apply_intv_step:    p50=17.12 p95=22.91 max=51.76
intv_prepare:       p50=0.04  p95=0.08  max=0.15
intv_apply_logits:  p50=0.22  p95=0.53  max=1.03
```

### 4.3 Clean topk=10 baseline (bench_30b.py, same date)

Run after vm_topk=1, with `--vm_topk` removed and all other parameters identical.
This is the apple-to-apple control group.

| Conc | ITL mean | Out tok/s | Req Lat |
|------|----------|-----------|---------|
| 1    | 9.6ms    | 78.0      | 1640ms  |
| 2    | 12.7ms   | 155.1     | 1649ms  |
| 4    | 18.1ms   | 217.6     | 2347ms  |
| 8    | 23.8ms   | 327.2     | 3111ms  |
| 16   | 48.9ms   | 322.3     | 6333ms  |

pf-summary for topk=10 (@ step 1800):

```
b2_score_call:      p50=11.01 p95=18.63 max=68.04
b2_batch_wall_abs:  p50=22.04 p95=61.51 max=137.31
intv_batch_size:    p50=2.00  p95=6.00  max=15.00
apply_cpu_sync:     p50=4.60  p95=17.14 max=20.46
apply_intv_step:    p50=29.15 p95=83.77 max=148.15
intv_apply_logits:  p50=0.40  p95=2.41  max=2.75
```

### 4.4 Head-to-head comparison

| Metric | topk=10 (clean) | vm_topk=1 | Speedup |
|--------|----------------|-----------|---------|
| conc=16 ITL | 48.9ms | **20.2ms** | **2.4×** |
| conc=16 tok/s | 322.3 | **759.8** | **2.36×** |
| conc=16 / noSIA ratio | 31% | **73%** | +42pp |
| b2_batch_wall_abs p50 | 22.04ms | **9.74ms** | **2.26×** |
| apply_intv_step p50 | 29.15ms | **17.12ms** | 1.70× |
| apply_intv_step p95 | 83.77ms | **22.91ms** | **3.65×** |
| intv_batch_size p50 | 2.00 | 1.00 | — |

noSIA reference (conc=16): ~1040 tok/s, ITL ~13.6ms.

## 5. Analysis

### 5.1 Why the gain was larger than the pre-experiment prediction

Prior theoretical analysis assumed VM batch size at each intervention step was
`num_sessions × topk = 16 × 10 = 160`, predicting only ~1.1–1.2× improvement
(VM is memory-bandwidth bound; weight loading cost is roughly fixed regardless of batch size).

The actual picture, from the clean topk=10 baseline pf-summary:

**`intv_batch_size p50=2.00`** for topk=10; **`intv_batch_size p50=1.00`** for vm_topk=1.

At the median intervention step, only **1–2 sessions** are at a high-entropy token simultaneously
(not all 16). With a per-session intervention rate of ~5% and 16 sessions, the expected number
of co-intervening sessions per step is 0.8, so the median is 1–2, not 16.

The actual candidate counts per VM call:
- topk=10: **2 sessions × 10 candidates = ~20 sequences** at median intervention step
- vm_topk=1: **1 session × 1 candidate = ~1 sequence** at median intervention step

A ~20× reduction in per-call VM workload (not 10×), which maps to the observed
`b2_batch_wall_abs` drop: 22ms → 9.74ms (2.26×). The sub-linear ratio (20× workload
reduction → 2.26× time) confirms the VM has a fixed per-call overhead (CUDA graph dispatch,
kernel launch) that dominates at very small batch sizes.

### 5.2 The intv_batch_size difference is self-reinforcing

The `intv_batch_size` difference between the two runs (p50=2 vs p50=1) is not a configuration
artifact — it is a **consequence of VM call duration**:

- topk=10: VM call takes ~22ms. During these 22ms, more sessions accumulate pending
  interventions → next call receives a larger batch → stays slow (vicious cycle).
- vm_topk=1: VM call takes ~9.74ms. Fewer sessions accumulate → next batch stays small
  → stays fast (virtuous cycle).

This self-reinforcing dynamic explains why the tail latency improvement is even larger than
the median: `apply_intv_step p95` went from 83.77ms to 22.91ms (3.65×), because large
accumulated batches (p95 = 6 sessions × 10 candidates = 60 sequences) are the worst case
for topk=10, but only 6 sequences for vm_topk=1.

### 5.3 Throughput saturation at conc=16 for topk=10

```
topk=10:  conc=8 → 327 tok/s,  conc=16 → 322 tok/s  (plateau — VM is the bottleneck)
vm_topk=1: conc=8 → 475 tok/s, conc=16 → 760 tok/s  (continues scaling linearly)
```

With topk=10, adding concurrency beyond 8 does not increase throughput because the VM
cannot keep up. With vm_topk=1, the system scales correctly up to conc=16.

### 5.4 Remaining gap from noSIA

At conc=16: vm_topk=1 achieves ITL 20.2ms vs noSIA 13.6ms — **1.5× overhead**, down from
3.6× (topk=10 clean: 48.9ms / 13.6ms). The remaining 6.6ms gap is the irreducible cost of:

- `apply_cpu_sync` p50=4.55ms: GPU→CPU sync to read topk candidate indices. Now the
  **dominant overhead**, overtaking VM call time. Unchanged between vm_topk=1 and topk=10
  (4.55ms vs 4.60ms), so it is independent of VM workload.
- VM call amortized: 9.74ms × ~5% intervention rate ≈ 0.5ms average per step.

The new bottleneck is `apply_cpu_sync`, not the VM itself.

## 6. Conclusion

### 6.1 FaRMA is strongly motivated

The clean A/B experiment (same date, same docker config, only `--vm_topk` toggled) gives
a definitive result:

| | topk=10 | vm_topk=1 | FaRMA (expected) |
|---|---|---|---|
| conc=16 tok/s | 322 | 760 | **~760** |
| conc=16 / noSIA | 31% | 73% | **~73%** |
| Quality | full | degraded (1 candidate) | **full (10 candidates)** |

FaRMA achieves the same throughput as vm_topk=1 by running 1 VM forward on the prefix and
reading all K=10 scores from the vocab-wide output vector. It gets the throughput of vm_topk=1
with the quality of topk=10.

### 6.2 Updated priority

P-1 (vocabulary-wide scoring head) is **confirmed as the highest-priority item** with strong
empirical backing. The prior estimate of 1.1–1.2× was based on a wrong assumption (VM batch
always 160); actual measured gain with the clean baseline is **2.36× throughput, 2.4× ITL**,
raising conc=16 throughput from 31% to 73% of noSIA.

### 6.3 New bottleneck: apply_cpu_sync

With VM call time no longer dominant, `apply_cpu_sync` at 4.55ms is now the largest overhead.
This is the GPU→CPU sync to read topk candidate indices. Worth investigating whether this can
be reduced further (e.g., keeping topk indices on GPU longer, or restructuring the sync timing).

### 6.4 Quality note

This proxy experiment modifies only 1 candidate's logit per intervention (the LLM top-1 token).
This degrades alignment quality and should **not** be used in production. The experiment was
purely for throughput measurement. Real FaRMA restores full 10-candidate quality.
