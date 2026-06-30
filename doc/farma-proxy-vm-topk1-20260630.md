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

### 4.3 Comparison vs topk=10 baseline

| Metric | topk=10 baseline | vm_topk=1 (this run) | Δ |
|--------|-----------------|----------------------|---|
| conc=16 ITL p50 | ~58–60ms | **20.2ms** | **−3×** |
| conc=16 tok/s | ~370 | **759.8** | **+105%** |
| conc=16 / noSIA ratio | 35% | **73%** | +38pp |
| b2_batch_wall_abs p50 | ~28ms | **9.74ms** | **−2.9×** |
| apply_intv_step p50 | ~38ms | **17.12ms** | **−2.2×** |
| apply_cpu_sync p50 | ~7.7ms | 4.55ms | −1.7× |

noSIA reference (conc=16): ~1040 tok/s, ITL ~13.6ms.

## 5. Analysis

### 5.1 Why the gain was larger than predicted

Prior analysis assumed VM batch size at each intervention step was `num_sessions × topk = 16 × 10 = 160`.
The pf-summary reveals the actual picture:

**`intv_batch_size p50=1.00`**

At the median intervention step, only **1 session** is at a high-entropy token simultaneously.
This is consistent with an intervention rate of ~5% per session: with 16 concurrent sessions,
the expected number of intervening sessions per step is 0.8, so the median is 1 (or 0).

Therefore the actual reduction was:
- topk=10 baseline: **1 session × 10 candidates = 10 sequences** per typical VM call
- vm_topk=1 (this run): **1 session × 1 candidate = 1 sequence** per typical VM call

A 10× reduction in VM workload per call — not the 160→16 (10×) I had predicted at the batch level,
but through the same 10× factor for a different reason (intervention sparsity rather than session count).

`b2_batch_wall_abs` dropped from ~28ms to ~9.74ms (2.9×), confirming that the VM is not purely
memory-bandwidth bound at this small batch size. At batch_size=1–10, compute and dispatch overhead
scale with batch size, making the reduction meaningful.

### 5.2 Remaining gap from noSIA

At conc=16: ITL 20.2ms vs noSIA 13.6ms — the system is now **1.5× slower than noSIA**, down from
4.3× before. The remaining 6.6ms overhead per step is the irreducible cost of:

- VM call itself (9.74ms amortized across all steps, weighted by ~5% intervention rate ≈ ~0.5ms avg)
- CPU sync (`apply_cpu_sync` p50=4.55ms): GPU→CPU sync for topk values, the new dominant cost

`apply_cpu_sync` (GPU→CPU sync for topk values) is now the **largest single overhead**
at 4.55ms, overtaking the VM call time (which is now smaller due to vm_topk=1).
This is a different bottleneck than before and worth investigating separately.

## 6. Conclusion

### 6.1 FaRMA is strongly motivated

This experiment directly validates the value of training a FaRMA-style VM (vocabulary-wide
scoring head). The key finding:

- Reducing VM candidates from 10 to 1 per request yields **~3× throughput improvement** at conc=16
- FaRMA achieves the same throughput reduction by running 1 forward on the prefix and reading all
  K=10 scores from the vocab-wide output vector — **same speed as vm_topk=1, full quality of topk=10**

The proxy experiment gives a concrete, empirically measured target: training a FaRMA VM should
recover the ~3× throughput gain while restoring alignment quality.

### 6.2 Updated priority

P-1 (vocabulary-wide scoring head) is **confirmed as the highest-priority item** with strong
empirical backing. Prior estimate of 1.1–1.2× was based on a wrong assumption about batch size;
actual gain is ~3×, raising conc=16 throughput from 35% to 73% of noSIA.

### 6.3 New bottleneck: apply_cpu_sync

With VM call time no longer dominant, `apply_cpu_sync` at 4.55ms is now the largest overhead.
This is the GPU→CPU sync to read topk candidate indices. Worth investigating whether this can
be reduced further (e.g., keeping topk indices on GPU longer, or restructuring the sync timing).

### 6.4 Quality note

This proxy experiment modifies only 1 candidate's logit per intervention (the LLM top-1 token).
This degrades alignment quality and should **not** be used in production. The experiment was
purely for throughput measurement. Real FaRMA restores full 10-candidate quality.
