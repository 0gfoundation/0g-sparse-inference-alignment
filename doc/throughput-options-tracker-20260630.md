# SIA Throughput Options Tracker — VL-30B b2 Inproc (2026-06-30)

This document catalogs every throughput-related optimization that has been considered for SIA.
Two sections: **Promising** (top, ordered by priority) and **Ruled Out** (bottom, with evidence).

Priority ordering for the promising section weighs three factors together:
`confidence of gain × ease of implementation × risk level`.

**Current baseline** (VL-30B b2 inproc, conc=16, 2026-06-29):
- ITL p50 ≈ 58–62 ms (vs noSIA ≈ 13.6 ms, 4–5×)
- Throughput ≈ 350–370 tok/s (vs noSIA ≈ 1040 tok/s, 35%)
- VM per-call latency p50 ≈ 28 ms (with APC, scoring 16 sessions × topk=10)
- Intervention rate ≈ 5% per step (entropy gate, VL-30B no-thinking)

---

## Promising Methods (highest priority first)

---

### P-1 · Vocabulary-wide Scoring Head (FaRMA)

**Source**: FaRMA, arxiv 2502.04517, ICML 2025  
**Type**: VM architecture change (requires retraining)  
**Estimated speedup**: **2.36× system throughput (conc=16), confirmed by proxy experiment**

Current approach scores K=10 candidates with K separate forward passes, even when batched.
FaRMA changes the reward head from `[hidden → scalar]` to `[hidden → vocab_size]`:
the VM does one forward on the current prefix and reads off the reward for every candidate
from the output vector in one shot — K forwards → 1 forward.

**Proxy experiment result (2026-06-30)**: Using `--vm_topk 1` to send 1 candidate per request
to the VM (same entropy gate / intervention rate as `topk=10`) measured directly on VL-30B:

| Metric | topk=10 baseline | vm_topk=1 proxy | FaRMA (expected) |
|--------|-----------------|-----------------|-----------------|
| conc=16 tok/s | 322 | 760 | **~760** |
| % of noSIA (1339 tok/s) | 24% | 57% | **~57%** |
| b2_batch_wall_abs p50 | 22.04ms | 9.74ms | ~9.74ms |
| ITL overhead vs noSIA | 4.41× | 1.82× | **~1.82×** |

FaRMA achieves the throughput of `vm_topk=1` while preserving full alignment quality
(10 candidates scored, not 1). The proxy underestimates FaRMA quality but matches throughput.
See `doc/farma-proxy-vm-topk1-20260630.md` for full analysis.

**When to implement**: during the Month 2 vocab-aligned VM retraining (see P-2).
The architecture decision must be locked before training starts; the code change is minimal
(add the vocabulary projection, change the scoring call to a single forward + index lookup).
**Can be combined with** P-5 (low-rank head factorization) to limit the added parameter cost.

---

### P-2 · Vocabulary-aligned VM Retraining (Qwen3.5-4B base)

**Source**: Month 2 plan (sia-6month-roadmap-20260622.md §Month 2)  
**Type**: Model replacement (requires retraining)  
**Estimated speedup**: −7ms per VM call (cross-tokenizer overhead eliminated)

Current VM-Qwen3-4B has 151k-token vocabulary vs VL-30B/0GM-35B's 248k-token vocabulary.
This mismatch forces:
- BPE boundary re-tokenization (CPU, ~2ms per call for 0GM-35B)
- Prevents shared KV cache (APC cannot work across different tokenizations)
- Blocks vocabulary-wide scoring head (P-1) entirely

Training Qwen3.5-4B (same vocabulary as 0GM-35B) as VM eliminates all three.
For VL-30B specifically, the cross-tokenizer forced overhead is ~7ms of the ~28ms call cost.
Combined with P-1, the total VM call cost drops from 28ms → ≈5ms.

Also unlocks the ARM training objective (P-3), which requires per-token reward prediction
and works best when the VM vocabulary matches the main LLM.

---

### P-3 · ARM Training Objective (GenARM)

**Source**: GenARM, ICLR 2025, arxiv 2410.08193  
**Type**: Training method (requires retraining)  
**Estimated gain**: quality improvement per VM call (not directly latency)

ORM (Outcome Reward Model) assigns a single reward at the end of a complete sequence.
SIA needs per-token rewards at intermediate decode steps — using an ORM forces a full forward
at each step with the partial sequence, which is semantically misaligned.

ARM (Autoregressive Reward Model) trains the VM to predict token-level rewards directly,
matching the per-step intervention structure of SIA. GenARM shows ARM training targets
outperform ORM in quality even at the same model scale.

This is the natural training objective for SIA and should be the default target for
any new VM training run. It primarily improves output quality, which indirectly allows
keeping `--weight` lower (less aggressive intervention) without quality loss.

---

### P-4 · Two-stage Cascade: 0.6B Pre-filter

**Source**: Month 2 task 2.3 (sia-6month-roadmap-20260622.md); VM-Qwen3-0.6B-Base on HuggingFace  
**Type**: Inference strategy change  
**Estimated speedup**: −50–80% of 4B VM calls

Use a tiny 0.6B VM to pre-rank K=10 candidates and discard the bottom 8–9, then
call the 4B VM only on the 1–2 survivors. The 0.6B VM is ≈6.7× smaller → calls complete
in ≈4ms; the expected 4B call skip rate is ~75%, making the cascade net 3–4× faster
than calling 4B every time.

Key assumption: the 0.6B VM's ranking is sufficiently correlated with the 4B VM's ranking
that always scoring the 0.6B top-1 with the 4B VM captures the same alignment effect.
This is a PoC question — the cascade pair must be empirically validated.

Pre-trained VM-Qwen3-0.6B-Base is already available on HuggingFace (SIA repo).
Prototype effort: ~1–2 weeks for the cascade inference path + A/B quality validation.

---

### P-5 · Low-rank Reward Head

**Source**: arxiv 2407.04615, TMLR 2025  
**Type**: VM architecture (companion to P-1)  
**Estimated gain**: reduces parameter count of vocab-wide head

The vocabulary-wide head matrix is `[hidden_dim × vocab_size]`; for Qwen3.5-4B with
248k vocabulary this is ≈4B×248k = enormous. Factorize it as `[d × r] × [r × V]`
with r ≪ V — a low-rank bottleneck that keeps expressiveness while drastically cutting
parameter count and output bandwidth.

This is a natural companion to P-1; implement them together during Month 2 VM training.
Effort: minimal, just changes the reward head architecture before training starts.

---

### P-6 · Multiplicative Distribution Fusion (LLMdoctor)

**Source**: LLMdoctor, arxiv 2601.10416, Jan 2026  
**Type**: Logits intervention change (no retraining)  
**Estimated gain**: quality (+3–5% AlpacaEval win-rate); minimal throughput impact  
**Effort**: 1–2 days

Replace additive logit biasing `logits += weight × vm_score` with multiplicative fusion:
`π_decode ∝ π_base^α · π_r^β` (product of distributions, then renormalize).
LLMdoctor shows 62.10% win rate vs GenARM and 76.00% vs ARGS on GPT-4o head-to-head.

This is primarily a **quality** improvement with near-zero throughput cost
(no extra GPU compute — just a different math operation on already-computed scores).
Its priority here is high because of the extreme ease and low risk: purely a formula
change in `SIALogitsProcessor.apply()`, fully reversible.

Also directly addresses reward hacking risk (Inference-Time Reward Hacking, arxiv 2506.19248):
the product-of-distributions formulation is less prone to the proxy-reward over-optimization
that causes the inverted-U true-reward curve.

---

### P-7 · Adaptive Soft Mixing / Learned Intervention Router (TARo)

**Source**: "To Intervene or Not" (arxiv 2606.11201, ACL 2026); TARo (arxiv 2603.18411)  
**Type**: Gate replacement (one is rule-based, one requires training)  
**Estimated gain**: quality per intervention; potentially lower effective intervention rate

Two related directions:

**Rule-based soft mixing** ("To Intervene or Not"): when the base model's max-token
probability < 0.4, instead of binary intervene/skip, blend the two distributions with
a soft weight proportional to the VM confidence ratio. Reduces reward hacking at
low-uncertainty positions. No training required. Effort: ~1 week.

**Learned router (TARo)**: trains a token-level binary router (end-to-end with VM)
that learns which positions benefit from intervention. Achieves up to +22.4% quality
improvement by skipping ineffective interventions. Requires joint router training.
Indirectly helps throughput by reducing intervention frequency beyond what entropy
thresholding alone achieves.

---

### P-8 · Hydragen Shared-prefix Attention for VM Scoring

**Source**: Hydragen, arxiv 2402.05099  
**Type**: VM serving kernel change  
**Estimated speedup**: reduces per-call compute for K candidates (attention FLOPs, not weight loading)

In the current SIA scoring batch, all K=10 candidates share an identical prefix
(prompt + generated tokens so far). Hydragen computes the shared-prefix attention
once and applies it to all K candidates, then runs only the K short suffix
attention passes independently. This reduces attention FLOPs from K×L² to L² + K×1.

**Caveat**: VM-Qwen3-4B is memory-bandwidth bound (HBM bandwidth limits weight loading),
so Hydragen's reduction in attention FLOPs may not translate directly to wall-clock
savings unless the weight loading is the dominant bottleneck. Worth benchmarking after
P-1+P-2 (with vocab-wide head, the weight loading still dominates).
Effort: ~2 weeks to modify VM serving kernel.

---

### P-9 · Accept/Reject Sampling Intervention (RSD)

**Source**: RSD, arxiv 2501.19324, ICML 2025  
**Type**: Intervention strategy change (algorithmic)

Instead of always biasing logits by VM scores, adopt an accept/reject loop:
sample a token from the base distribution; if VM scores it above a threshold, accept;
otherwise resample from a VM-weighted distribution. This provides a more principled
sampling guarantee (same stationary distribution as target policy) and may reduce
effective VM calls when the base model's first sample is already high-quality.

**Caveat**: RSD's published results are at sequence/step level, not token level.
Token-level accept/reject lacks direct top-conference validation for the SIA setting.
Treat as exploratory (Month 4 plan). Prioritize after P-6 (multiplicative fusion).

---

### P-10 · Tiny Linear Judge on Main LLM Hidden States (Judge Decoding)

**Source**: Judge Decoding, arxiv 2501.19309  
**Type**: VM replacement (requires training)  
**Estimated speedup**: VM latency 28ms → <0.1ms (if quality holds)  
**Risk**: HIGH — quality extremely uncertain

Train a 16.4k-parameter linear layer that reads the main LLM's internal hidden states
to score token candidates. The main LLM's forward pass already runs at every step;
this adds negligible compute. Training: ~500 triples (question + correct candidate +
wrong candidate), ~1.5 hours. If it works, per-call latency drops 300×.

**Why it is not ranked higher**: the approach is highly speculative for SIA's use case.
Judge Decoding validates on a different task (speculative decoding acceptance criterion).
Whether the main LLM's internal representations carry sufficient token-level reward signal
for open-ended instruction following is an open empirical question. No paper validates this
for SIA-style inference-time alignment. Attempt only if P-1 through P-4 have been validated.

---

### P-11 · In-process RM Executor (Bypass vLLM IPC)

**Source**: b2-pure-engineering-optimizations.md, item B-1  
**Type**: Engineering change to RM client  
**Estimated speedup**: +3.5–5 tok/s (eliminates 5–7ms IPC overhead per intervention)  
**Applies to**: Qwen3-14B / original b2 path; partially relevant to VL-30B b2 inproc

Current b2 inproc still routes through nested EngineCore + /dev/shm pipe.
Replacing this with a direct `GPUModelRunner.execute_model()` call eliminates the
zmq + pickle + shared-memory handshake (~5–7ms overhead per intervention).

High engineering risk (vLLM internal API, must be re-adapted on each vLLM upgrade).
The VL-30B b2 setup already gets much of this benefit from inproc mode; the remaining
IPC overhead is smaller than in the original HTTP path.

---

### P-12 · Pure Engineering Microoptimizations (Group A)

**Source**: b2-pure-engineering-optimizations.md, Group A  
**Type**: Code-only changes (no model changes, no risk to output quality)  
**Estimated speedup**: +1.7–3 tok/s combined

Three independent changes, safe to bundle:

- **Disable INTERVENE debug logs + 2 GPU syncs**: `rm_scores.min()/.max().item()` + print
  call force GPU→CPU synchronization. Guard behind `SIA_LOG_LEVEL` env var. (+1–1.5 tok/s)
- **Explicit CUDA graph capture for batch=5**: `--cuda-graph-sizes [1,2,4,5,8,...]` ensures
  a batch of 5 candidates doesn't fall back to eager mode. Must pass as list, not scalar.
  (+0.5–1 tok/s)
- **Tensor copy optimization in SIALogitsProcessor**: eliminate redundant list concatenations
  and per-candidate tensor allocations inside the intervention hot path. (+0.2–0.5 tok/s)

These are the safest items on this list: accuracy is mathematically unaffected.

---

### P-13 · Async Two-GPU Pipeline

**Source**: parallel-decoding-design.md; sia-6month-roadmap-20260622.md  
**Type**: Infrastructure change  
**Estimated gain**: VM completely off critical path for single-request throughput

Move the VM to a dedicated second GPU. The main LLM decodes on GPU-0; after each step,
GPU-1 scores candidates asynchronously while GPU-0 immediately starts the next decode step.
At step N+1, the scores from step N are consumed to bias logits.

**Caveat**: introduces a one-step lag in the reward signal (scores from step N modify
logits at step N+1). Whether this lag degrades alignment quality needs empirical
validation. Also requires 2-GPU deployment, which is a hardware constraint.
For conc=16 throughput improvement, the gain is most pronounced for long generations.

---

## Ruled-Out Methods (tried or theoretically dismissed)

---

### R-1 · Eager VM Prefill ❌ Tried — Net Negative

**Experiment**: implemented and benchmarked 2026-06-29 (Section 10 of
`vl30b-b2-profiling-conc16-20260630.md`)

**Idea**: after each decode step, speculatively prefill the VM's KV cache with the
predicted next token, so the next intervention step hits APC and skips the tail prefill.

**Result**: ITL +1.3ms worse (60.2ms vs 58.9ms baseline).
- APC DID work: VM scoring dropped 35ms → 28ms (−7ms). ✓
- But eager prefill cost: b2_eager_wall p50=21ms, runs on EVERY step (not just interventions). ✗
- Net: −7ms × 5% intervention rate benefit vs +21ms × 100% step cost → net negative.

**Root cause**: VM-Qwen3-4B is memory-bandwidth bound. Its ~8GB weights must be read from HBM
at every forward pass regardless of APC (APC only saves the attention FLOPs, not the weight
bandwidth). 8GB ÷ HBM bandwidth ≈ 21ms irreducible minimum per forward pass.
Any eager-prefill strategy hits this floor on every step, not just on intervention steps.

---

### R-2 · Dual-gating with Logit-Gap Threshold ❌ Tried — Ineffective

**Experiment**: `--logit_gap_threshold 3.0` deployed in production docker-compose

**Idea**: skip intervention when both entropy < threshold AND logit gap between top-1
and top-2 tokens > gap threshold (dual confirmation of model confidence).

**Result**: no measurable throughput improvement for VL-30B.
Root cause: for VL-30B (no thinking tokens), entropy and logit gap are **negatively
correlated** — when entropy is low (model is confident), the logit gap is also usually
large. The dual gate fires on the same positions as the entropy gate alone, adding no
further skip rate beyond what `--entropy_threshold` already provides.

This parameter should be removed from docker-compose (it's currently still present
as a dead-weight configuration).

---

### R-3 · Block-wise Scoring (B=4) ❌ Theoretically Dismissed

**Idea**: score the VM only every B=4 tokens instead of every token;
reuse the previous step's score for the B−1 steps in between.

**Why rejected**: no paper basis for fixed-interval skipping in SIA. More critically,
this randomly misses the high-entropy positions that are precisely the tokens where
intervention matters most. The entropy gate (--entropy_threshold) provides the principled
version of "don't always score" — it skips low-entropy (high-confidence) steps, which are
the safe positions to skip. Block-wise scoring would skip both safe and critical positions
with equal probability, directly undermining the core SIA mechanism.
Rejected by team consensus. Not worth experimenting.

---

### R-4 · FP8 RM Quantization ❌ Tried — Slight Regression

**Experiment**: vllm-rm-followup-optimizations.md §G4 (HTTP vLLM-RM path, 2026-05-26)

**Result**: −5% throughput (28.08 vs 29.65 tok/s before FP8).

Three failure layers:
1. **Memory-bandwidth bound**: VM is bottlenecked by weight loading (8GB from HBM), not
   arithmetic throughput. FP8 GEMM accelerates compute but not bandwidth.
2. **Uncalibrated FP8 KV cache**: without per-layer q/k/v scale calibration, vLLM falls
   back to a safety path that adds FP8 ↔ BF16 format conversion overhead.
3. **Small input matrices**: each scoring call processes ≈8 rows (5 candidates × 1–2 tokens
   after prefix cache hit). FP8's compute advantage is proportional to matrix size;
   at M≈8 the kernel launch overhead dominates.

Conclusion: FP8 quantization cannot help a memory-bandwidth-bound VM on small scoring batches.

---

### R-5 · Linear Scaling Hypothesis (PIECEWISE CUDA Graph) ❌ Disproven

**Hypothesis**: VL-30B's ITL scales linearly with concurrent load because PIECEWISE CUDA
graph mode causes one `compute_logits` call per concurrent request (vs FULL_AND_PIECEWISE
which batches them).

**Disproof**: added `compute_logits_calls` profiling counter. Result: always 1, regardless
of concurrency level. The linear ITL scaling is due to GPU resource contention (SM/memory
bandwidth sharing between main LLM and VM), not from extra forward passes.

---

### R-6 · vLLM HTTP-RM Scheduler Tuning (G2) ❌ No Benefit on HTTP Path

**Experiment**: vllm-rm-followup-optimizations.md §G2  
Tuned `--max-num-seqs`, `--max-num-batched-tokens`, etc. on the HTTP vLLM-RM server.

**Result**: near-zero improvement (~0.07 tok/s). The bottleneck was the shared GPU
alternation model (main LLM and RM interleave on the same device), not the vLLM scheduler.
Scheduling parameter tuning attacks the wrong bottleneck.

---

### R-7 · Static CUDA Graph Bucketing for HTTP-RM ❌ Slightly Negative

**Experiment**: vllm-rm-followup-optimizations.md §5.1 (#9, 2026-05-21)  
Added static bucket sizes to the HTTP vLLM-RM (`--cuda-graph-sizes`).

**Result**: −2% throughput (24.96 vs 25.41 tok/s). Implementation complexity with bucket
degradation paths and mismatch between bucket granularity and actual batch size distribution
caused more overhead than it saved. Contrast with P-12's item X-7 (explicit batch=5 capture
for b2 inproc), which is more targeted and lower risk.

---

### R-8 · Async Dual CUDA Stream on Single GPU ❌ Architecture Limit

**Idea**: run main LLM and VM on the same GPU using two CUDA streams to overlap compute.

**Why it doesn't work**: both models saturate HBM bandwidth independently. On a single GPU,
two bandwidth-bound workloads sharing the memory bus cannot truly run in parallel — they
compete for the same bottleneck resource. SM compute time-slicing is possible but the
memory wall is serialized. This is the fundamental reason why P-13 (two-GPU pipeline)
requires a second physical GPU and cannot be approximated on a single card.

---

---

### R-9 · K=10候选共享Prefix KV Cache ❌ Already Implemented via APC

**Idea**: K=10个候选在同一干预步中共享相同的前缀（prompt + 已生成response），如果b2 inproc没有复用这段prefix的KV cache，则每个候选都会重新encode全量前缀，浪费大量计算。

**Result**: APC已经自动处理了这个问题。`vl30b-b2-profiling-conc16-20260630.md` §4明确记录：

> "APC hits correctly — each set of 10 candidates shares the per-request prefix, so only **1+10 token-worth of computation is needed per session**."

即每次干预步，10个候选只需encode 1个新token（共享prefix KV通过APC命中），不是10个完整序列。这个优化不是新的优化空间，vLLM APC已经在做了。

**Why this appeared in the survey**: 文献调研中RAD（EMNLP 2023）提出用causal RM实现prefix KV复用，当时未确认b2 inproc是否已实现。确认后：该方案已被APC覆盖，无需额外工程投入。

---

*Last updated: 2026-07-01*
*VL-30B configuration: b2 inproc, vllm 0.17.1, VM-Qwen3-4B, PIECEWISE CUDA graph, SIA_RM_MULTIPROCESS=0*
