# Concurrent throughput — what's actually left to try, prioritized (2026-07-17)

## Purpose

Full re-review of every plan/roadmap/optimization doc under `doc/`, cross-checked against
external literature (deep-research workflow, 102 sub-agents, 20 primary sources fetched,
25 claims adversarially 3-vote-verified). Goal: a single priority list of throughput
improvements that do **not** sacrifice generation quality, excluding anything already
proven not to work or reasoned not to work. Two items that looked like live internal
contradictions turned out, on closer reading, to already be resolved — see §0.

## §0. Correcting the record: two "contradictions" are already resolved, not open

Before prioritizing, two items flagged as unresolved internal disagreements are not
actually live disagreements once the docs are read in date order:

**Block-wise scoring.** `sia-6month-roadmap-20260622.md` (2026-06-22) proposed it as the
Month-1 headline task. `throughput-options-tracker-20260630.md` (2026-06-30, 8 days
later) rejects it: "Block-wise scoring would skip both safe and critical positions with
equal probability, directly undermining the core SIA mechanism... Rejected by team
consensus." The later doc supersedes the earlier plan — the roadmap is simply stale on
this one task, not evidence of an ongoing dispute. **Verdict: stays rejected.**

**Dual-gating (logit-gap second gate).** `vl30b-b2-profiling-conc16-20260630.md` proposed
the idea with a theoretical estimate (intervention rate 15.9% → ~5%, SIA/noSIA ratio
34% → 91%), explicitly flagged as "empirically unverified." The team then actually shipped
`--logit_gap_threshold 3.0` to production and measured it: `throughput-options-tracker-
20260630.md` R-2 reports **zero measurable improvement**, because on VL-30B entropy and
logit-gap are *negatively correlated* — low entropy already implies a large gap, so the
second gate almost never fires beyond what the entropy gate alone already provides.
This is a real A/B result, not a theory-vs-theory standoff. **Verdict: the raw
logit-gap-as-second-gate idea is empirically dead for this model. Still open** (see
priority 3 below): whether a *different, less-correlated* secondary signal, properly
calibrated or learned rather than hand-picked, could do what the raw logit gap couldn't.

## §1. What external literature actually adds

Full report: 102 agents, 20 sources, 9 confirmed findings, 16 refuted after adversarial
verification. Headline: **the literature does not hand the team a ready fix for any of
its three open problems** — it mostly narrows what's plausible and rules out some
tempting-looking analogs.

1. **Disaggregated serving (DistServe/Mooncake-style) does not validate a specific
   throughput number for splitting main-LLM/RM onto 2 GPUs.** Published disaggregation
   gains come overwhelmingly from resolving *queuing delay* and *large cross-node
   KV-cache transfer* (prefill is only 2–23% of P95 TTFT in one measured cluster) — a
   cost profile that doesn't match moving a small RM's tiny candidate-token payload.
   Neither confirms nor refutes the team's estimate; the mechanism papers describe
   isn't the mechanism this system would rely on.
   - **Caution surfaced, not confirmed**: one paper (Nexus) claims single-GPU dynamic SM
     partitioning beats 2-GPU disaggregation by 1.4×. This specific claim didn't survive
     adversarial verification in this pass (evidentiary reasons), but it's exactly the
     kind of result that would undercut a 2-GPU plan — worth reading directly before
     committing real engineering time to a 2-GPU build-out.

2. **The core "batch coupling" problem (one request's synchronous RM call blocks its
   whole co-batched group inside a live vLLM step) has no published solution.**
   AugServe and Clairvoyant — the two closest scheduling papers — both operate at the
   request-admission/queueing layer; Clairvoyant explicitly states continuous-batching
   engines like vLLM "gain nothing" from its approach and disclaims multi-GPU/distributed
   relevance. This is a genuine, currently-unaddressed gap in the public literature, not
   something the team overlooked.

3. **No verified precedent exists for async/pipelined per-step auxiliary scoring either.**
   The two closest-looking analogs — PicoSpec (speculative pipelining without waiting for
   verifier feedback) and SSD (draft model prepares for the likely verification outcome
   while verification is in flight) — both failed adversarial verification (0-3 each).
   A separate, confirmed finding: batched speculative decoding's "ragged tensor problem"
   (sequences accepting different numbers of draft tokens desynchronize position IDs /
   attention masks / KV state) is real and independently corroborated by a production
   vLLM GitHub PR (#14645) fixing the identical bug. **Anyone attempting request-class
   segregation inside vLLM's batching should expect to hit this exact failure class.**

4. **On gating: the deciding factor is calibration rigor, not "which idea is right."**
   Confidence-style secondary gates are empirically prone to "confident miscalibration"
   under distribution shift — 72.2% of cases needing escalation get a confidently wrong
   "skip" instead of an uncertain one (this is a distinct, more general mechanism than
   VL-30B's specific entropy/logit-gap correlation, but points the same direction: don't
   trust a hand-picked raw threshold). Two positive counter-examples: (a) applying
   Conformal Risk Control / UCB-style calibration to an analogous rollback threshold in
   soft speculative decoding (BiLD) gives real, statistically-guaranteed speedups over an
   uncalibrated threshold; (b) Mixture-of-Depths shows a small, separately-trained
   predictor can approximate an expensive routing decision at ~97% fidelity with minimal
   downstream quality loss. **Net: if dual-gating is revisited, don't hand-pick another
   raw threshold — calibrate one properly or train a cheap predictor.**

Nothing found, confirmed or refuted, on block-wise vs. entropy-gated scoring specifically
— that one is settled by the team's own A/B (§0), not by external literature.

## §2. Prioritized list — highest reliability first

Excludes everything already tagged tried-and-failed or quality-sacrificing in the
existing docs (all quantization variants, static CUDA-graph batch bucketing, same-GPU
dual-CUDA-stream overlap, transformers-KV-cache RM backend, block-wise scoring, raw
logit-gap dual-gating, reduced K / raised entropy_threshold / smaller unvalidated VM).

### Priority 1 — Micro-optimizations already identified, zero quality risk, cheapest to ship
`throughput-options-tracker-20260630.md`'s "safest items" list (removing dead
`--logit_gap_threshold` config, cutting debug-log GPU syncs, per-candidate tensor
allocation cleanup in the hot path) — small (+0.2–0.5 tok/s each) but mathematically
risk-free and not gated on anything else. Do these first simply because they're free.

### Priority 2 — Resolve the still-live gating question properly, per the calibration finding
Not "retry logit-gap" (already empirically dead on VL-30B, §0) — instead: either (a)
apply a formal calibration procedure (conformal risk control / UCB-style, per BiLD) to
whatever secondary signal is tried, with a real held-out calibration set and a stated
tolerance, or (b) train a tiny learned predictor (a few-parameter classifier on cheap
per-step features — entropy, top-2 identity, maybe hidden-state norm) that predicts
"would the RM's top-1 differ from the LLM's top-1," mirroring Mixture-of-Depths'
97%-accurate router. The roadmap already cites the right precedent for (b) (TARo,
"Learning Adaptive LLM Decoding") but never built it. This is the one place external
literature gives an actionable, non-obvious correction to the internal approach: stop
hand-picking thresholds, calibrate or learn one.

### Priority 3 — 2-GPU physical separation, but test the *synchronous* variant first, not the async one
Two distinct proposals exist in the docs and they are not the same risk profile:
- **Synchronous split** (move LLM and RM to separate physical GPUs, identical semantics
  to today, no algorithmic change): removes the same-GPU HBM bandwidth contention that
  `R-8` in the tracker documents as the reason dual-CUDA-stream overlap fails on one
  card. This is the genuinely zero-quality-risk version — worth provisioning a second
  GPU to test empirically, since neither the team's own estimate nor the external
  literature settles the actual number (disaggregation literature's gain mechanism
  doesn't transfer cleanly either way, per §1.1). Test before further theorizing.
- **Async pipelined split** (`P-13` in the tracker: GPU-1 scores step N while GPU-0
  starts decoding step N+1, consuming scores one step late): this is **not**
  zero-quality-risk as previously assumed — the tracker's own text flags "introduces a
  one-step lag in the reward signal... whether this lag degrades alignment quality needs
  empirical validation," and external literature found no verified precedent for this
  exact async pattern (§1.3). Treat as a distinct, higher-risk follow-on to test only
  after the synchronous split's ceiling is known, with an explicit AlpacaEval quality
  check on the lagged variant before considering it for production.

### Priority 4 — Do not build custom request-class-segregated batching inside vLLM
Both the internal batch-coupling problem and the external literature search converge on
the same conclusion: no one has published a working pattern for segregating
"needs-extra-compute-this-step" requests from others inside a live continuous-batching
engine's synchronous forward pass, and the closest attempted analog (batched speculative
decoding) has a documented, production-confirmed failure mode (ragged tensor problem)
when heterogeneous per-step compute is naively batched. This would be genuine, unscoped
research, not an optimization task — deprioritize relative to 1–3 given the time/risk
ratio, unless the team is prepared to treat it as an open engineering research project.

### Priority 5 — Direct GPU-tensor reward sharing (skip `/dev/shm` IPC)
Carried over unchanged from the internal inventory as a legitimate, no-quality-cost
micro-optimization; external research didn't specifically bear on this (it's a pure
intra-process engineering detail), so its priority rests entirely on the internal
estimate. Fine to pursue opportunistically alongside Priority 1, but not blocking.

## §3. What NOT to re-attempt (already exhausted, no new evidence changes this)

All quantization (FP8/INT8/INT4/AWQ) — proven memory-bandwidth-bound. VM CUDA graph on
0GM-35B — structurally impossible (prefill-heavy workload vs. PIECEWISE's fixed-shape
assumption), three rounds tried. vLLM classify/pooling and transformers+DynamicCache RM
backends — 5-8× slower, already measured. Same-GPU dual-CUDA-stream overlap — both
models are independently bandwidth-bound, R-8 confirms this is the same reason a 2-GPU
split (not stream tricks) is the only lever left on that specific bottleneck. Block-wise
scoring and raw logit-gap dual-gating — both empirically dead per §0, not theory.
`vm_topk=1`, reduced `--topk`, raised `--entropy_threshold`, unvalidated smaller-VM
swap — all explicitly quality-sacrificing, excluded per this request's constraint.
