# SIA Roadmap（May 2026 Retrospective + July–December 2026 Plan）

---

## Overview

### What Is SIA

**SIA (Sparse Inference-time Alignment)** is a system that intervenes in model outputs in real time during inference.

In plain terms: every time the model generates a token, SIA evaluates candidate tokens in parallel and biases the probability distribution toward higher-quality options. Model weights are never modified — SIA operates purely at inference time, acting as a real-time "quality gatekeeper" that filters each token before it is sampled.

### Current Results (as of 2026-06-22)

The core system was completed within two months. Both primary models — 0GM-VL-35B and Qwen3-VL-30B — are near production-ready:

- **Effectiveness validated**: With SIA enabled on 0GM-VL-35B, the overall dialogue quality (AlpacaEval win rate) reached **65.4%** (125 out of 191 pairwise comparisons judged the SIA version as superior); MMLU accuracy shows no significant degradation
- **Performance cost**: Single-request throughput with SIA is approximately **59–64%** of the no-SIA baseline (0GM-35B: 59–61%, VL-30B: 64%); under high concurrency (16 concurrent requests), throughput drops to approximately **35%** of no-SIA, meaning the same hardware can serve about 65% fewer requests

### Three Core Pain Points (Current State)

| Pain Point | Current State | Root Cause |
|------------|--------------|------------|
| **Effectiveness highly dependent on configuration** | 0GM-35B natural thinking: **win rate 65.4% (191 pairs), Δ=+4.05**; 0GM-35B ban_think: **win rate 54%, Δ=+0.72 (not significant)**; VL-30B: **win rate 63% (122/195), Δ=+4.2%~+8.3%, significant in both rounds**; Qwen3-14B same-family: **+13.2% (paper reproduction)**. MMLU: same-family (VL-30B) shows no significant degradation; 0GM-35B cross-family drops **−5.5 to −12.2pp** | **① VM generational/family mismatch (primary cause)**: VM-4B is based on Qwen3 base (training cutoff April 2025); 0GM-35B is a 2026 new model. Capability gap W2S=8.75×, time gap 8–12 months, vocabulary mismatch 248K vs 151K; three comparison results are strictly monotonic (larger W2S → worse performance). **② Out-of-distribution VM usage**: VM training task is complete-answer preference scoring; SIA's actual task is step-level value prediction on partial response + 1 token — distribution mismatch causes high scoring signal noise. **③ Cross-tokenizer BPE boundary noise**: 0GM-35B vocabulary 248K vs VM vocabulary 151K; before stable prefix optimization, APC misses caused unstable scoring (stable prefix has partially resolved this). ④ VM training data volume/quality not yet fully validated |
| **High VM latency** | **0GM-35B ~30ms/call** (after stable prefix optimization; previously as high as 34–50ms and grew linearly with sequence length); **VL-30B 11ms/call** (with CUDA graph enabled) | **① No CUDA graph (largest contributor)**: vllm 0.18.0 has a WeakSet bug where `clear_all_graphs()` during the main LLM profiling phase clears the RM's CUDA graph, causing a runtime crash; 0GM-35B is forced into eager mode. On VL-30B, CUDA graph reduces VM latency from 71ms→11ms (6.4×). **② Cross-tokenizer BPE boundary overhead (0GM-35B specific)**: each step requires decode+re-encode; combined with eager dispatch overhead, this accounts for ~7ms gap — the second-largest contributor to the latency difference between 0GM-35B and VL-30B. **③ Small-batch memory-bound**: each VM forward has a very small batch size (topk×1–2 tokens), but must read the full ~8GB of weights, leaving the GPU in a memory-bound state |
| **High-concurrency throughput loss** | After batch scoring optimization: conc=16 SIA tok/s 369, **35% of noSIA**. Full concurrency sweep (35B): conc=4→48%, 8→42%, 16→35%. **30B comparison**: 66% at conc=16 (vs 35B's 35%); the gap stems from cross-tokenizer overhead making each 0GM-35B VM call ~3ms vs 30B's ~1.5ms, accumulating to 2× when repeated 16 times serially | Three bottlenecks at conc=16 (ITL 3.0×, 41.3ms vs noSIA 13.6ms): ① **APC partial tail compute ~15ms/batch (largest contributor, structurally irreducible)**: each VM batch forward's candidate prefix is only ~15 tokens, which does not fill one APC block (16 tokens); the tail fragment must be fully prefilled every step. ② **No CUDA graph ~5ms/call (fixable)**: vllm 0.18.0 WeakSet bug forces VM into eager mode, adding kernel dispatch overhead per batch step. ③ **Cross-tokenizer CPU encoding ~0.5ms** (slightly improved after P-3 optimization). The ~15ms is a structural lower bound; ~5ms can be fixed via CUDA graph |

<sub>Abbreviations used below: **E** = effectiveness highly dependent on configuration; **L** = high VM latency; **T** = high-concurrency throughput loss.</sub>

### Two Phases of Work Focus

The past two months were an **"engineering alignment" phase**: using the NTU SIA paper (arxiv 2602.21215) as a reference, we deliberately made no changes to any algorithm or model — the intervention logic, VM architecture, and training weights were kept exactly as-is. The goal was first to reproduce the paper's claimed results in a production environment, then eliminate deployment overhead through purely engineering means (b2 inproc architecture, replacing low-speed PyTorch with vLLM, batch scoring merging). This ensured that any changes in effectiveness could be attributed to the engineering implementation, not algorithmic changes. High-concurrency throughput improved 54% from baseline, with very low optimization risk, deterministic gains, and results directly measurable in throughput numbers.

The next 6 months enter the **"going beyond the original paper" phase**: with engineering alignment complete, we will now **actively push past the boundaries of the original SIA paper**, boldly experimenting with algorithmic changes (product-of-distributions fusion, block-wise scoring, accept/reject sampling), model improvements (same-vocabulary VM retraining, stronger base models, larger training datasets), and training data enhancements (step-level annotation, multimodal preference pairs). **Each of these directions will affect effectiveness**, with results that may be positive or negative — there are no guarantees of improvement over the current state. Therefore, the success criteria for the plan going forward are more often "A/B produces a conclusion" rather than "a specific number must be hit," and milestones accordingly retain both conservative and optimistic scenarios.

### 6-Month Plan

The plan is divided into three phases. The core objective is to reduce SIA's performance cost from "significant" to "negligible" while continuously improving alignment effectiveness.

**Phase 1 (Month 1): Reduce scoring frequency; establish GPT-4 evaluation baseline**  
By intelligently skipping "unimportant tokens," reduce Value Model call frequency to 1/4 of its current rate (approximately 75% reduction, from ~20% intervention rate to ~5%), drastically reducing overhead with minimal loss in alignment effectiveness. Simultaneously, build a GPT-4 standard evaluation baseline on top of the existing Skywork judge baseline to provide industry-comparable quantitative benchmarks for all future improvements.

**Phase 2 (Months 2–3): Switch to a faster scoring model**  
The current scoring model uses a different vocabulary from the main model, requiring extra vocabulary re-encoding at each scoring step (CPU encoding ~2ms + eager dispatch ~5ms, ~7ms total overhead). Training a scoring model with an identical vocabulary is expected to reduce single-call scoring latency from ~30ms to ~23ms (eliminating the ~7ms cross-tokenizer overhead); with a simultaneous CUDA graph fix, this could further drop to ~11ms. We will also validate whether "product-of-distributions fusion" (a more precise intervention method than the current additive approach) can further improve alignment effectiveness.

**Phase 3 (Months 4–6): Stronger scoring capability + multimodal extension**  
Evaluate the feasibility of step-level intervention (PRM) — first validate the incremental effect using a training-free step-level pause-and-rerank approach (having the model explore 3 directions at each reasoning step end and choosing the best one to continue, with parameter design drawing from AdaDec), then decide based on conclusions whether to train an external PRM (the approach validated by RSD on MATH500/AIME). Simultaneously extend SIA to multimodal inputs supporting image inputs, covering a broader range of business scenarios.

### 6-Month Goals

| Dimension | Current (2026-06) | Conservative Goal (CUDA graph still unavailable) | Optimistic Goal (after CUDA graph is restored) |
|-----------|-------------------|---------------------------------------------------|------------------------------------------------|
| High-concurrency throughput (SIA vs no-SIA) | **35%** | **≥ 53%** | **≥ 72%** |
| Single-call scoring latency | ~30ms | ~23ms (eliminate cross-tokenizer ~7ms) | ~11ms (with additional CUDA graph) |
| Multimodal support | Text only | Text + images | Text + images |
| Standard evaluation baseline | Skywork judge baseline established; GPT-4 judge TBD | Month 1: establish GPT-4 judge, track continuously | Month 1: establish GPT-4 judge, track continuously |

---

## Execution Order Overview

```
Month -2 ██ Project kickoff ██ Batch forward ██ First effectiveness evaluation   [Done]
Month -1 ██ b2 inproc ██ Bug fixes ██ Production readiness ██ High-concurrency opt [Done]
─────────────────────── Current (2026-06-22) ──────────────────────────────────────
Month 1  ██ block-wise scoring ██ Dual entropy gating ██ GPT-4 evaluation baseline
Month 2  ██ Two-stage coarse-filter PoC ██ Data collection ██ ARM VM training
Month 3  ██ Same-vocabulary VM launch (incl. MoE selection) ▶ Product-of-distributions fusion A/B
Month 4  ██ Stronger VM ██ accept/reject ▶ Tiny judge PoC ▶ Multimodal data prep
Month 5  ██ PRM feasibility evaluation (full-month primary track)
Month 6  ██ Multimodal VM training ██ Multimodal VM launch
```

The completed phases represent actual output from the first two months. The execution plan phases strictly depend on prior phases: reduce VM call frequency early → upgrade to a stronger model mid-term → carry out fundamental improvements late.

---

## Month -2 (2026-04-21 ~ 2026-05-21): Project Cold Start

> The project launched on 2026-04-21. Within one month, we completed preliminary research, foundational framework construction, and the first effectiveness validation.

**Key Accomplishments:**

- **Paper study and conceptual understanding**: Read the SIA paper (arxiv 2602.21215), understood the core idea of sparse intervention at critical decision junctions; ran the paper's inference intervention pipeline using the original experiment code (github.com/hurunyi/SIA) to confirm method viability
- **SIA core implementation**: Token-level intervention framework based on vLLM `LogitsProcessor`, OpenAI-compatible HTTP API, per-request dynamic `sia_weight` adjustment
- **First effectiveness evaluation**: 0GM-VL-35B AlpacaEval + MMLU initial runs confirm alignment signal is present; measured intervention rate ~20% (approximately 80% of tokens skip the VM call because entropy is below the threshold)
- **VM performance baseline**: SIA single-request ~40 tok/s vs noSIA ~114 tok/s; confirmed VM serial calls (one GPU forward per candidate) as the primary bottleneck
- **Batch forward optimization**: K candidates merged from K serial GPU forwards into 1 batch forward, reducing VM compute by ~60%
- **KV prefix caching exploration**: Experimented with PyTorch DynamicCache prefix reuse; unsatisfactory results due to PyTorch having no APC mechanism
- **vLLM RM backend established**: Implemented `--rm_backend vllm`, calling the vLLM classify interface via HTTP for scoring (groundwork for b2 inproc)

**End-of-month status**: SIA is operational with preliminary effectiveness validation; single-request SIA/noSIA ratio ~35%; clear performance improvement headroom identified.

---

## Month -1 (2026-05-22 ~ 2026-06-22): Architectural Breakthrough + Production Readiness Complete

> Completed core architecture upgrade, fixed a major effectiveness regression bug, and launched both models to the marketplace.

**Key Accomplishments:**

**1. b2 inproc VM architecture deployed** (late May)  
Converted VM from an independent HTTP process to an embedded nested vLLM instance (in-process function call), completely eliminating network round-trip overhead (~20–30ms per call). Combined with VM CUDA Graph (Qwen3-VL-30B) and main LLM CUDA Graph fix (removing PIECEWISE-only restriction):
- Qwen3-VL-30B: HTTP ~36 tok/s → b2 inproc **78.3 tok/s** (**+117%**)
- 0GM-VL-35B: HTTP ~32 tok/s → b2 inproc + PIECEWISE fix **54.1 tok/s** (**+69%**); further to **66–68 tok/s** (**+106%**) after combining with §3 stable prefix

**2. Critical `repetition_penalty` bug fix** (June 4)  
Discovered that the default `repetition_penalty` value of 1.3, when stacked with SIA logit intervention, caused evaluation data to show SIA Δ = −13% to −75% (incorrect conclusion). After fixing to 1.0, SIA effectiveness fully recovered to positive results:
- 0GM-VL-35B MMLU thinking mode controlled experiment: no significant accuracy drop between SIA vs noSIA

**3. 0GM-VL-35B cross-tokenizer optimization** (early June)  
The stable prefix solution eliminates APC invalidation caused by cross-tokenizer BPE boundary merging: VM call latency reduced from linearly growing with sequence length (34ms→~50ms) to a **fixed ~30ms**, with end-to-end throughput improving +22%.

**4. Dual-model marketplace deployment complete** (mid-June)  
Both 0GM-VL-35B and Qwen3-VL-30B have completed Docker deployment, OpenAI-compatible API, full integration tests (run_all.sh), max_model_len extended to 32768 tokens, and multimodal image input support (image requests bypass SIA).

**5. High-concurrency batch scoring optimization** (June 18)  
Merged N concurrent request VM calls from N serial calls into 1 GPU batch forward, combined with incremental cross-tokenizer prefix caching (P-3):
- conc=16 throughput: 239 → **369 tok/s** (**+54%**)
- conc=16 ITL multiplier: 4.8× → **3.0×**

**6. Explored and ruled out failed paths** (June 22)  
Both vllm classify runner (7–8× slower, APC doesn't work) and transformers + DynamicCache (5–8× slower, TP=4 advantage not replicable) were experimentally ruled out, avoiding repeated exploration. Additionally, RM CUDA graph (piecewise mode) after three rounds of systematic fixes remained 2.7–3.5× slower than eager mode; the root cause is that RM is a prefill-heavy workload, fundamentally incompatible with PIECEWISE optimization targets, and has been fully ruled out — see [Appendix: 0GM-35B CUDA Graph Debugging History](#appendix-cuda-graph).

---

## Current State (End of 2026-06-22)

### Performance Data

| Scenario | Model | SIA | noSIA | Ratio |
|----------|-------|-----|-------|-------|
| Single-request throughput | 0GM-VL-35B | 66–68 tok/s | ~112 tok/s | **59–61%** |
| Single-request throughput | Qwen3-VL-30B | 78.3 tok/s | ~122.8 tok/s | **64%** |
| conc=16 throughput | 0GM-VL-35B | 369 tok/s | 1040 tok/s | **35%** |
| Single-request ITL | 0GM-VL-35B | ~11ms (~480 tokens) / 34ms (32K tokens) | ~9ms | **~1.2–1.4×** (typical scenario) |
| conc=16 ITL | 0GM-VL-35B | 41.3ms | 13.6ms | **3.0×** |

| Component | Data |
|-----------|------|
| VM per-call latency p50 | 0GM-VL-35B ~30ms, Qwen3-VL-30B ~11ms (p95=17ms) |
| VM intervention rate (entropy_threshold=1.0) | ~20% (0GM-VL-35B), ~25% (Qwen3-VL-30B) |
| Multimodal support | ❌ text-only VM; image requests bypass SIA |

### Alignment Effectiveness

| Evaluation | Result | Notes |
|------------|--------|-------|
| MMLU thinking mode (0GM-VL-35B, 150Q controlled) | No significant change SIA vs noSIA | rep_penalty bug eliminated |
| AlpacaEval win-rate (0GM-VL-35B) | **65.4%** (Skywork judge, **191 pairs**) | Skywork judge baseline established (9 of 200Q excluded due to exceeding RM limit); GPT-4 judge not yet run, to be established in Month 1 |
| AlpacaEval Skywork Δ (0GM-VL-35B) | **+5.45 reward (+22.7%)** | SIA mean 29.46 vs noSIA 24.01 (191-pair basis, experiment stable-prefix-20260610) |
| Qwen3-VL-30B AlpacaEval (b2 inproc, 200Q) | SIA Skywork mean **+1.22~+2.38 (+4.2%~+8.3%)**, significant in both rounds | GPT-4 judge not yet run; two-round Δ variation is normal statistical noise (see doc/alpaca-eval-vl30b-b2-docker-20260608.md) |

<sub>MMLU note: Multiple experiments (VL-30B ±1–3pp, 14B ±1pp) are all statistically insignificant; 0GM-35B thinking mode showed a +12pp exception, mainly because SIA reduces thinking truncation (noSIA cap-hit 25.3% → SIA 8.0%), not a genuine knowledge improvement — awaiting repeated confirmation.</sub>

---

## Month 1 (2026-07): Engineering Quick Win — Reduce VM Call Frequency

**Primary track**: No model changes, pure code modifications — reduce the effective VM call rate from ~20% to ~5%.

### Task 1.1: block-wise scoring (B=4)

Modify `SIALogitsProcessor`: on top of the existing entropy gating (~20% trigger rate), add a block-wise strategy — only every 4th token is eligible to trigger VM scoring, reducing the actual VM call rate from ~20% to ~5%.

> **Expected gains upon completion**: Effective VM call rate drops from the current ~20% (entropy gating) to ~5% (block-wise stacked), reducing amortized VM overhead per token by ~75% (per-call p50 latency ~30ms unchanged; calls reduced to 1/4 of current), with conc=16 throughput expected to rise from 369 tok/s to ~500+ tok/s.

- **Theoretical basis**: Reducing VM call count to 1/B directly lowers cumulative VM overhead — an obvious engineering corollary that needs no separate citation. Actual speedup will be measured via SIA A/B testing. **On whether skipping 3/4 of high-entropy tokens causes effectiveness loss**: The original SIA paper (arxiv 2602.21215) provides direct evidence — intervening at only ~20% of high-entropy junctions (rather than 100% of all tokens) achieves or exceeds the alignment effectiveness of full intervention, indicating that most token positions contribute near-zero intervention value, and intervention value is highly non-uniform across a sequence. Block-wise applies a further 3/4 skip on top of entropy gating (which already filtered to ~20%): immediately after an intervention, the model has been guided onto a high-reward trajectory; the next few tokens naturally follow that trajectory, so the marginal value of immediately re-intervening is low (the cooldown window design is based exactly on this). The actual upper bound on effectiveness loss depends on how many of the skipped tokens are truly independent high-value junctions — this is the core quantitative question that Month 1 A/B (B=2/4/8) will answer.
- **Implementation options**:
  - **Fixed block (simple)**: Only check entropy at positions 4/8/12/…; simple to implement, but critical tokens that fall within the first three positions of a block will be mechanically skipped.
  - **Cooldown window (recommended)**: After each intervention trigger, suppress checks for the next N tokens (`cooldown_remaining` counter decrements); when no intervention occurs, normal entropy checks resume. Advantage: positions near a recently corrected location are low-risk and can be skipped; critical junctions will not be missed due to a fixed stride. Equally simple to implement (add a per-request counter in `apply()`); expected to show less effectiveness loss than fixed block. Both options included in Month 1 A/B comparison.
- **Effort**: ~1 week (code changes concentrated in the `apply()` method, with token buffer logic adjustments)
- **Risk**: Effectiveness may slightly drop; A/B comparison needed to find the optimal B value; if B=4 causes unexpectedly large score drop, fall back to B=2

**Additional alternative — Vocabulary-wide Reward Head** ([arxiv 2502.04517](https://arxiv.org/abs/2502.04517), ICML 2025): The current SIA performs a separate forward pass for each of the top-K candidates, even with batch merging — still K sequences. This paper redesigns the RM head to output **reward scores for the entire vocabulary** in a single forward pass, going from K passes → 1 pass, theoretically improving VM latency by topK×. Effort: requires modifying the VM inference interface; better suited for implementation together with the same-vocabulary VM training in Month 2, rather than modifying the existing model in Month 1. This month, simply review the architectural approach.

### Task 1.2: Dual-Gate Replacing Single Entropy Gate

Current gating: intervene when main model entropy > θ (single entropy).  
Change to: intervene when main model entropy > θ₁ **AND** the VM's previous step score shows low divergence between candidates (dual-gate).  
Reduces low-quality ineffective interventions while also reducing VM call count by ~20-30%.

**Specific definition of "VM divergence"**: In the VM's last call, the reward score difference between the top-1 candidate and the second-highest candidate is < δ (small gap = VM is uncertain which is better = current position is semantically sensitive, worth continued intervention); large gap (VM strongly prefers one candidate) = VM signal is stable in this region, short-term intervention value is low, can skip. Boundary handling when combined with block-wise: if the last VM call was more than B steps ago (i.e., during a block-wise skip period), recent divergence data is missing — **default to triggering intervention** (conservative strategy, avoiding missed critical positions due to lack of historical data).

> **Expected gains upon completion**: VM effective call rate (latency/throughput) reduced by an additional ~20-30% on top of block-wise, providing an extra ~10-15% end-to-end throughput improvement; reducing ineffective interventions also slightly improves alignment effectiveness (lower proportion of noisy interventions).

- **Theoretical basis**: **"To Intervene or Not"** ([arxiv 2606.11201](https://arxiv.org/abs/2606.11201), ACL 2026) directly studies the decision of "when to intervene," proposing a **soft mixture weight** computed from the confidence ratio of two models when the base model's confidence is low (max token probability < 0.4), replacing binary intervention; experiments show non-uniform intervention outperforms always-intervening. This supports the dual-gate direction of this task, though the paper's mechanism is soft mixing rather than hard thresholding. **TARo** (arxiv 2603.18411) proposes an end-to-end **learned** token-level router that automatically determines intervention strength at each position, showing up to +22.4% improvement in experiments; the conclusion supports "adaptive strategies outperform fixed strategies," but TARo's router requires training, unlike the rule-based score-difference gating in this task.
- **Note**: EASD (arxiv 2512.23765)'s "dual entropy" targets two generative models (draft + target), both of which have token probability distributions for computing entropy — this is **not directly applicable** to SIA's reward model scenario, and is not used as theoretical basis for this task.
- **Supplementary reference**: Learning Adaptive LLM Decoding ([arxiv 2603.09065](https://arxiv.org/abs/2603.09065), 2026 preprint) suggests introducing a learned routing policy (small classifier head, deciding whether to intervene based on context features) — can be combined with dual-gate as a Month 1 advanced exploration.
- **Effort**: ~1 week (add a per-request state cache to `apply()` recording last VM call's score difference and steps since last call)

### Task 1.3: Establish GPT-4 Standard Evaluation Baseline

The existing Skywork judge has produced the 65.4% AlpacaEval baseline, which serves as a useful quick-iteration reference but has two limitations: evaluation circularity and non-comparable scope (see below). This task uses the same 200-question AlpacaEval fixed test set, switches to GPT-4 as judge, and establishes a reproducible evaluation pipeline aligned with industry standards.

**Why the existing Skywork judge results are insufficient:**

1. **Circularity problem (Goodhart's Law)**: SIA's working mechanism is to call the VM (a reward model) at each decoding step to shift the logit distribution — essentially optimizing "to get higher RM scores." If we use Skywork (also a reward model) to judge whether SIA's outputs are better, the evaluator and the optimization target are the same class of system, creating an inherent circular dependency: SIA's score improvement may only reflect "better alignment with RM preferences," not actual quality improvement as perceived by humans. GPT-4 as an evaluator is not a reward model; its judgments are orthogonal to SIA's optimization objective, providing a more independent validation of whether the effect generalizes.

2. **Non-comparability**: When the academic and industry communities publish AlpacaEval numbers, GPT-4 Turbo is the default judge (as defined by the AlpacaEval 2.0 standard). The current Skywork judge's 65.4% is an internal reference number that cannot be directly compared with any published paper's numbers, creating a scope problem for external claims.

> **Expected gains upon completion**: Not a direct performance optimization, but provides a quantifiable basis for all subsequent alignment effectiveness improvements — every optimization has a win-rate number for comparison, avoiding the "we changed something but can't measure the effect" problem; the VM quality baseline also sets a reference point for Month 2 training objectives.

- **Evaluation framework**: AlpacaEval 2.0 (arxiv 2404.04475) uses GPT-4 as judge to compute win-rate, with a fixed 200-question set ensuring cross-experiment comparability, at approximately 100× lower cost than human evaluation.
- **VM quality baseline**: Simultaneously use RMB ([arxiv 2410.09893](https://arxiv.org/abs/2410.09893), ICLR 2025) and RewardBench 2 ([arxiv 2506.01937](https://arxiv.org/abs/2506.01937), 2026) to evaluate the raw scoring quality of VM-Qwen3-4B, establishing a "VM capability → SIA effectiveness" mapping to guide Month 2 VM training objectives.
- **⚠️ Reward Hacking Risk**: Inference-Time Reward Hacking ([arxiv 2506.19248](https://arxiv.org/abs/2506.19248), NeurIPS 2025 Spotlight) shows that RM intervention at inference time exhibits an **inverted-U curve for true reward** — the proxy reward monotonically increases with intervention strength, but the true reward begins declining after the optimal point; this is a universal pattern, not only triggered at high `--weight` values. Task 1.3's evaluation pipeline should simultaneously design reward anomaly detection metrics (e.g., output length distribution, repetition rate, perplexity) to prevent SIA from "inflating scores through degradation" on certain questions.
- **Effort**: ~1 week

### Month 1 Success Criteria

| Metric | Month 1 Goal | Current Baseline |
|--------|-------------|-----------------|
| conc=16 SIA tok/s | **≥ 550** | 369 |
| conc=16 ITL (ms) | **≤ 28** | 41.3 |
| AlpacaEval evaluation | **Runnable, with results** | None |

---

## Month 2 (2026-08): Same-Vocabulary VM Training

**Primary track**: Full focus on same-vocabulary VM data collection and training, in preparation for Month 3 deployment. No new throughput improvements this month; Month 1 engineering gains (block-wise + dual entropy gating) continue to apply.

**Why same-vocabulary is the highest priority**: The current VM (Qwen3-4B) and the main LLM (0GM-VL-35B) use different tokenizers (Qwen3 151K vocabulary vs Qwen3.5 248K vocabulary). This means the prefix token IDs seen by the VM do not correspond one-to-one with the tokens generated by the main LLM, causing systematic noise in the scoring signal. Same-vocabulary training resolves this in one stroke:

- Cross-tokenizer noise eliminated → improved scoring precision (better alignment effectiveness)
- Stable prefix workaround can be retired → eliminates cross-tokenizer CPU encoding overhead (~2ms) and eager mode kernel dispatch overhead caused by cross-tokenizer issues (~5ms); b2_score_call drops from ~30ms to **~23ms** (VM latency reduction; CUDA graph still unavailable, so cannot reach ~11ms)
- HBM data read per call unchanged, but quality is better (indirect improvement to concurrent throughput)

> **Note (from prior experiments)**: RM CUDA graph on 0GM-35B failed after three rounds of fixes and has been fully ruled out. The root cause is that RM is a prefill-heavy workload, fundamentally incompatible with vllm PIECEWISE's optimization targets — same-vocabulary VM training cannot change this fundamental nature. The latency improvement from same-vocabulary training **comes solely from eliminating cross-tokenizer overhead (~7ms)**, not from CUDA graph — see [Appendix: 0GM-35B CUDA Graph Debugging History](#appendix-cuda-graph).

### Task 2.1: Preference Data Collection

Training data will be built through three main paths:

1. **Public datasets**: Directly reuse high-quality open-source preference datasets (e.g., UltraFeedback, HelpSteer2, Skywork-Reward-Preference-80K, etc.) covering general instruction-following scenarios; filter subsets matching the style of 0GM-VL-35B as needed.
2. **LLM synthetic data**: Using public prompt sets (AlpacaEval 200Q, Open-Hermes, etc.) as input, have 0GM-VL-35B generate N candidates per prompt (temperature sampling), use the existing VM to score candidates, and construct chosen/rejected preference pairs. This approach can be scaled as needed and ensures prompt distribution matches real inference scenarios.
3. **NTU collaborator data**: The original SIA paper is from NTU; directly request the preference datasets used to train the SIA Value Model from the collaborators, which can serve as the highest-quality cold-start data, greatly reducing data construction overhead. Prioritize confirming availability before Month 2 begins.

> **Expected gains upon completion**: No direct performance improvement, but it is a prerequisite for Task 2.2; data quality and scale directly determine the alignment effectiveness ceiling after the same-vocabulary VM launches in Month 3.

- **Effort**: ~1-2 weeks
- **Risk**: If public datasets don't match the target scenario style, increase the proportion of synthetic data

### Task 2.2: VM-Qwen3.5-4B Training — Using ARM Training Objective

Train a reward head using Qwen3.5-4B (248K vocabulary, same as 0GM-VL-35B) as the base.

> **Expected gains upon completion** (realized upon Month 3 launch): Alignment effectiveness: cross-tokenizer noise eliminated, VM scoring quality significantly improved; VM latency: b2_score_call ~30ms → ~23ms (eliminating combined cross-tokenizer overhead ~7ms: CPU encoding ~2ms + eager mode dispatch ~5ms; CUDA graph still unavailable, ~23ms is the practical lower bound); if vocabulary-wide head is implemented, VM latency additionally improves by up to topK× (theoretically up to 10× for topK=10).

**Training objective**: Follow the original SIA VM's ARM (Autoregressive Reward Model) training approach — VM outputs a reward at each token position; at inference time, the value at the end of the prefix is used as the intervention score. This is a necessary requirement for SIA's token-level intervention (ORM is only meaningful at the end of a complete answer; it cannot score mid-decoding). Theoretical basis: GenARM (ICLR 2025, arxiv 2410.08193). The core change in this round is **switching the base to Qwen3.5-4B (same vocabulary as 0GM-VL-35B)**; the training objective itself is unchanged.

Simultaneously evaluate whether a suitable small MoE base is available (when VRAM permits, MoE has lower per-call read bandwidth).

**🎯 Architecture Design Goal — Vocabulary-wide Scoring Head** (from [arxiv 2502.04517](https://arxiv.org/abs/2502.04517), ICML 2025):  
When training the new VM, **do not continue using the old "input one candidate sequence → output one scalar" interface**; instead, design the reward head as "input current prefix → output reward vector for entire vocabulary." This reduces SIA's scoring of top-K candidates from **K forwards → 1 forward**, with a theoretical upper bound of topK× speedup (theoretically up to 10× for topK=10); the paper (FaRMA) measured actual speedup of ~**6–6.5×** — below the theoretical upper bound, as the increased output scale of the RM head partially offsets the reduction in call count; actual gains to be measured via SIA A/B. Implementation: change RM head from `[hidden → scalar]` to `[hidden → vocab_size]`; labels remain preference pairs. This is the **highest-priority architectural decision for this round of VM training** and does not require separate effort — just confirm the interface design before training begins.

**Additional training techniques**:
- **Low-rank Reward Head** ([arxiv 2407.04615](https://arxiv.org/abs/2407.04615), TMLR 2025): Factorize the vocabulary-wide head's matrix from `[d × V]` into `[d × r] × [r × V]` (r ≪ V); particularly effective for Qwen3.5's vocab=248K, and naturally compatible with the architecture goal above.
- **From r to Q\* theoretical inspiration** ([arxiv 2404.12358](https://arxiv.org/abs/2404.12358), COLM 2024): This paper proves that DPO is equivalent to token-level implicit Q-learning, where the LLM's log-prob difference β(log π_θ − log π_ref) encodes an implicit reward value. **SIA engineering derivation**: When annotation data is insufficient, the log-prob difference between 0GM-VL-35B and a reference model can serve as weak-supervised pseudo-labels to assist VM cold-start — this application is an extension of the theory to SIA, not a direct contribution from the original paper.
- **RED** (reward redistribution, [arxiv 2411.08302](https://arxiv.org/abs/2411.08302), venue TBD): Uses differential scores of sequence prefixes from an existing RM to recover token-level rewards — r̃ᵗ = R(x,y≤t) − R(x,y≤t−1) — no training required, different from distillation. Can serve as a fallback path when token-level annotations are unavailable.

- **Effort**: ~2-3 weeks (training + preliminary offline validation)

### Task 2.3: Two-Stage Coarse-Filter PoC — 0.6B VM as Gatekeeper for 4B VM (Experimental, 1–2 weeks)

**Principle**: RSD (ICML 2025, arxiv 2501.19324), SSS (EMNLP 2025, arxiv 2508.15044), and GSI (ICLR 2026, arxiv 2506.04118) have validated the "small model fast pre-filter, large model fine-ranking" two-stage architecture. SIA adaptation: first score K candidates with VM-Qwen3-0.6B; if candidate score variance < threshold (differences are not significant, intervention value is low), skip the 4B VM; otherwise call the 4B VM for fine ranking.

> **Note**: The above three papers all involve a small LLM gating a large LLM, not "small RM gating large RM" — this transfer to the VM scenario has no direct top-conference validation.

**Speed bottleneck**: VM calls have a ~12ms fixed overhead (vLLM scheduler + IPC, does not scale with model size). The 0.6B GPU compute is only ~2ms, totaling approximately **14ms** per call. Therefore, the skip rate must exceed **47%** before any net gain is realized; at a 50% skip rate, average overhead drops from only 30ms to approximately **29ms**.

> **Expected gains upon completion (conservative)**: If measured skip rate ≥ 60%, average VM overhead approximately **23–26ms** (vs current 30ms, ~15–23% reduction); effectiveness loss pending A/B measurement.

- **Effort**: 1–2 weeks; runs in parallel with 2.1 data collection, does not block Month 2 primary track
- **Success criteria**: Stable dual-LLM-in-process architecture (prerequisite — vllm v1 co-process behavior not yet validated) + 4B VM calls reduced ≥ 30% + AlpacaEval win-rate loss ≤ 1%

### Month 2 Success Criteria

| Metric | Month 2 Goal | Month 1 Baseline |
|--------|-------------|-----------------|
| Same-vocabulary ARM VM training | **Training complete, offline validation passed** | — |
| MoE selection conclusion | **Dense 4B or MoE selection finalized** | — |
| Two-stage coarse-filter PoC | **Architecture stability validated, preliminary effectiveness data available** | — |
| conc=16 SIA tok/s | **Maintain ≥ 550** | ≥ 550 |

> 📌 This is a training month; performance numbers remain unchanged. The latency and throughput gains from the same-vocabulary VM (b2_score_call ~30ms → ~23ms, conc=16 throughput ≥ 550 (conservative)) are realized after Month 3 deployment.

---

## Month 3 (2026-09): Same-Vocabulary VM Launch — Quality and Efficiency Both Improved

**Primary track**: Complete same-vocabulary VM training, finalize selection (dense 4B vs MoE), deploy to production, replace the existing VM.

### Task 3.1: Same-Vocabulary VM Training and Model Selection

**Dense 4B** (default option): Comparable scale to the current VM; same vocabulary eliminates cross-tokenizer encoding noise. Note: CUDA graph remains unavailable for 0GM-35B (root cause is RM prefill-heavy workload, unrelated to tokenizer — see Month 2 notes); latency improvement comes from eliminating cross-tokenizer overhead, not CUDA graph.

> **Expected gains upon completion**: Finalizes dense 4B vs MoE selection, providing technical decision basis for Task 3.2 deployment; ARM training objective is theoretically superior to ORM; same-vocabulary + ARM combination significantly improves alignment scoring quality.

**MoE selection evaluation** (if a suitable pretrained MoE base is available): MoE in memory-bandwidth-bound scenarios reads only active parameters per forward (~3B), theoretically faster than dense models of comparable quality. However, **note that vllm loads all expert weights** (not just the active portion). Using 30B-A3B as an example:

| Comparison | Current VM (Qwen3-4B dense) | MoE 30B-A3B |
|------------|------------------------------|-------------|
| Per-GPU forward read bandwidth (BF16) | ~2 GB | ~1.5 GB (active 3B × 2B ÷ 4 GPUs) |
| Model capability | 4B | Approaches 30B level |
| VRAM usage (TP=4) | ~2 GB/GPU | **~15 GB/GPU** (30B × 2 bytes ÷ 4 GPUs, full expert weights) |

> **⚠️ VRAM infeasible**: In the current 4×A100 80GB configuration, the main LLM occupies 75% (60 GB/GPU), the 4B VM occupies 13% (10.4 GB/GPU), leaving approximately 10 GB/GPU. The 30B-A3B MoE VM requires **~15 GB/GPU** for weights alone (exceeding available space). **30B-A3B MoE is infeasible as a VM on current hardware** unless the main LLM's gpu_mem_utilization is reduced (which would shorten the maximum context length and concurrency).
>
> **Viable MoE options**: If a small MoE in the Qwen3.5 family exists (e.g., 7B-A3B, VRAM ~3.5 GB/GPU), it could be considered. However, no publicly available Qwen3.5-7B-A3B checkpoint exists for VM training; this option requires waiting for a suitable base model to be released.

**Default: dense 4B** (Qwen3.5-4B): Comparable scale to the current VM; same vocabulary eliminates noise; VRAM confirmed as feasible.

### Task 3.2: Same-Vocabulary VM Deployment and Validation

Replace the production VM, re-run AlpacaEval, validate effectiveness improvement.

> **Expected gains upon completion**: Alignment effectiveness: cross-tokenizer noise completely eliminated; AlpacaEval win-rate improvement quantifiable for the first time. VM latency: b2_score_call ~30ms → **~23ms** (eliminating cross-tokenizer encoding ~7ms; CUDA graph still unavailable for 0GM-35B; conservative gain as stated; further reduction to ~11ms possible if CUDA graph is later fixed). Concurrent throughput: conc=16 throughput conservative expectation **≥550 (maintaining baseline)**; improvement headroom after eliminating cross-tokenizer overhead; if CUDA graph is later fixed, can rise to ≥750 tok/s (see deliverable criteria table below).

### Task 3.3: Product-of-Distributions Fusion A/B Experiment (1-2 days, from LLMdoctor)

**Source**: LLMdoctor (arxiv 2601.10416, January 2026) changes additive logit biasing to product-of-distributions fusion (π_decode ∝ π_base^α · π_r^β), achieving 62.10% win vs. GenARM (ICLR 2025) and 76.00% win vs. ARGS in head-to-head GPT-4o judge evaluation.

> **Expected gains upon completion (if successful)**: AlpacaEval win-rate +3-5% additional gain (alignment effectiveness improvement); the change only modifies the `SIALogitsProcessor.apply()` fusion logic — extremely low engineering cost, can be rolled back at any time; results directly determine the priority of Month 4 accept/reject experiments.

Core formula:

> Current SIA: `logits[token] += weight * RM_score`  
> Product-of-distributions change: `α * log_prob_base[token] + β * log_prob_rm[token]` (log-space weighted average)

**This change requires no VM retraining** — only modifies the score fusion logic in `SIALogitsProcessor.apply()`, with extremely low engineering cost. Combined with the new same-vocabulary VM deployed in Month 3, three A/B configurations can be simultaneously validated:
1. Old VM + additive (current production, control group)
2. New ARM VM + additive
3. New ARM VM + product-of-distributions

**⚡ Fast early validation**: If Month 1 has bandwidth remaining (block-wise and dual entropy gating complete early), run a quick "product-of-distributions vs additive" comparison using the existing old VM at the end of Month 1 (1-2 days, 50 AlpacaEval questions) to get an early signal without waiting until Month 3.

**Priority rationale**: Extremely low cost (1-2 days), strong literature evidence, and results directly determine the necessity of Month 4 accept/reject experiments — if product-of-distributions fusion already yields significant effectiveness improvement, accept/reject experiment priority can be lowered.

- **Effort**: 1-2 days of code + ~1 week of A/B validation
- **Risk**: Very low (change can be rolled back at any time)
- **Success criteria**: Product-of-distributions win-rate higher than additive by ≥ 1% (statistically significant)

### Month 3 Success Criteria

| Metric | Month 3 Conservative Goal (CUDA graph still unavailable) | Month 3 Optimistic Goal (CUDA graph restored) | Month 2 Baseline |
|--------|----------------------------------------------------------|-----------------------------------------------|-----------------|
| conc=16 SIA tok/s | **≥ 550** | **≥ 750** | ≥ 550 |
| b2_score_call p50 | **~23ms** (eliminate cross-tokenizer ~7ms, eager mode) | **~11ms** (with additional CUDA graph) | ~30ms |
| AlpacaEval win-rate vs noSIA | **Quantifiable, positive improvement** | — | 65.4% (Skywork judge) |
| Product-of-distributions vs additive fusion comparison | **Conclusion: whether product-of-distributions is superior** | — | — |

### Month 3 System Backup Directions (Exploratory, 1-2 weeks if bandwidth allows)

The following directions have literature support but are not yet on the primary track; Month 3 can launch exploratory parallel experiments:

1. **Hydragen Shared-Prefix Attention** ([arxiv 2402.05099](https://arxiv.org/abs/2402.05099), arxiv preprint (ICLR 2025 not confirmed))  
   In SIA's VM scoring, all topK candidates share exactly the same prefix (prompt + already-generated tokens). Hydragen extracts this shared portion for a single forward pass, theoretically reducing VM candidate scoring from K independent computations → 1 shared-prefix + K very short suffix attentions, with significant concurrent throughput improvement. Requires modifying the VM serving kernel; ~2 weeks of effort.

---

## Month 4 (2026-10): VM Capability Upgrade + Intervention Strategy Experiments

**Primary track**: Based on the ARM 4B validation conclusions from Month 3, decide whether to scale up the model; simultaneously experiment with accept/reject intervention mode (as an exploratory experiment, without strong commitments).

### Task 4.1: Scale Decision — Train ARM 8B if ARM 4B Underperforms

**Decision logic** (from literature research): Literature (GenARM, ICLR 2025, arxiv 2410.08193) shows that the key to VM is the **training objective**, not model scale — the ARM token-level training objective outperforms ORM in experiments. Based on this (not a direct paper comparison): under sufficient training, ARM 4B should theoretically outperform ORM 8B. After Month 3 deploys ARM 4B, first validate effectiveness with AlpacaEval.

> **Expected gains upon completion**: AlpacaEval win-rate (alignment effectiveness) ≥ +5% vs noSIA; if ARM 4B is sufficient, saves training resources and directs the budget to Month 5 PRM feasibility evaluation or multimodal; if scaling to 8B is needed, block-wise already reduces call frequency, making the higher per-call latency of 8B acceptable under b2.

- If ARM 4B win-rate already reaches ≥ +5%: **skip 8B training**, redirect resources to Month 5 PRM feasibility evaluation or multimodal
- If ARM 4B effectiveness is still insufficient: scale up to 8B under the ARM training objective (by this point block-wise has already reduced call frequency, making 8B's higher per-call latency manageable)

- **Effort**: 2-3 weeks (contingent on Month 3 results)

### Task 4.2: accept/reject Intervention Mode Experiment (Exploratory)

Current mode: VM score added to logits (logit biasing).  
New mode: high VM score → accept top-1; low VM score → force resample from top-K.

> **Expected gains upon completion (if successful)**: Alignment effectiveness: more precise token selection, theoretically reduces "bias injection"; determines the final production intervention strategy, laying the groundwork for Month 5 long-term direction; if less effective than logit biasing, quickly rules it out to avoid repeated future experiments.

**Note**: RSD (ICML 2025, arxiv 2501.19324)'s accept/reject operates at the **reasoning step/sequence level**, not the token level, and uses an architecture of two independent large-and-small models, which differs from SIA's single model + external VM approach. **Token-level accept/reject lacks direct top-conference validation**; this task is an exploratory experiment with uncertain results requiring A/B measurement.

**Prerequisite (from 2026 research)**: Month 3's Task 3.3 product-of-distributions fusion experiment has very low cost (1-2 days) and strong literature support; it should be validated before accept/reject. If Month 3's product-of-distributions fusion already yields ≥ +3% win-rate improvement, this task's priority can be reduced and does not need to be forced in Month 4; if Month 3's product-of-distributions fusion is not significant, proceed with accept/reject in Month 4.

- **Effort**: 1-2 weeks of experimentation (contingent on Month 3 product-of-distributions fusion conclusions)
- **Success criteria**: accept/reject mode AlpacaEval win-rate higher than the current best fusion mode

### Task 4.3: Tiny Judge PoC (Execute if bandwidth allows)

**Source**: Judge Decoding (arXiv 2025 preprint, arxiv 2501.19309) replaces the acceptance criterion of speculative decoding with a 16,384-parameter linear layer, achieving 3.9–9.7× acceleration on Llama-405B inference with 500 triplets and 1.5 hours of training.

**SIA analogy**: Inside the main LLM's (0GM-VL-35B) LogitsProcessor, train an **extremely small linear scoring head** (<10M parameters) based on top-K logit distribution features, replacing the external 4B VM. **The 35B main model is completely frozen; only the scoring head is trained** — first run 35B inference to collect top-K logit features (saved to disk, only a few MB), then train the scoring head offline separately; the two steps can be executed independently and easily fit on a single 144GB card. If feasible, per-call latency drops from ~30ms to <0.1ms, fundamentally resolving L/T.

> **Expected gains upon completion (if successful)**: VM latency: VM per-call latency ~30ms → <0.1ms (300×); concurrent throughput: VM completely off the critical path, throughput approaches noSIA levels; even partial success can significantly reduce VM call frequency.

- **Effort**: ~1 week of code + 1 week of effectiveness comparison
- **Risk**: Accuracy may be insufficient to guide token selection (VM signal is more fine-grained than speculative decoding's accept/reject)
- **Success criteria**: Tiny judge vs 4B VM AlpacaEval win-rate difference ≤ 3%
- **Failure handling**: If accuracy is insufficient, the experimental conclusion still has value (proves the boundary of this direction)

### Task 4.4: Multimodal VM Data Collection Launch (1-2 weeks, preparation for Month 6 training)

**Purpose**: The prerequisite for multimodal VM training (Month 6 primary track) is multimodal preference data construction; data scale determines training quality. Use potential bandwidth this month to start early, avoiding the situation where Month 6 begins without data.

> **Expected gains upon completion**: No direct performance improvement, but eliminates data wait bottleneck for Month 6 multimodal VM training, effectively compressing Month 6's 4-week workload to 2-3 weeks.

- **Content**: Build image-text mixed preference pairs (Qwen3-VL-30B as reference LLM); data sources: public image-text preference datasets (e.g., RLHF-V, MMInstruct, etc.) + synthetic VQA/image-text conversation data (using public prompt sets as input, multi-sampling from LLM, then VM labeling)
- **Effort**: 1-2 weeks (lightweight, can run in parallel while 4.1/4.2 are running)
- **Prerequisite**: Execute only if bandwidth is available this month; if 4.1 needs 8B training + 4.2 needs experimentation, defer to Month 5

### Month 4 Success Criteria

| Metric | Month 4 Goal | Month 3 Baseline |
|--------|-------------|-----------------|
| AlpacaEval win-rate vs noSIA | **≥ +5% (ARM 4B or ARM 8B)** | Positive improvement |
| Intervention mode comparison | **Logit bias vs accept/reject with actual measurements** | — |
| Tiny judge PoC | **Experiment complete, effectiveness comparison data available** (if bandwidth allows) | — |
| Multimodal VM data | **Collection started** (if bandwidth allows) | — |
| conc=16 SIA tok/s | **Maintain ≥ 550** (conservative); if Month 3 CUDA graph already fixed, maintain ≥ 750 | ≥ 550 |

---

## Month 5 (2026-11): PRM Feasibility Evaluation

**Primary track**: Evaluate the feasibility of Process Reward Model in SIA's inference scenario — clarify the incremental effectiveness ceiling of PRM vs ARM VM, and the engineering implementation path for step-level intervention in vLLM streaming serving. This month **does not commit to training and launching PRM**; the deliverable is "A/B produces a conclusion." Multimodal VM data collection has been prepared in parallel during Month 4; training and launch are concentrated in Month 6.

### Task 5.1: PRM Feasibility Evaluation — Step-Level Intervention for Reasoning Tasks

#### Limitations of the Existing ARM VM

SIA's current intervention approach is **per-token scoring**: at each high-entropy position, using the Value Model to evaluate candidate token quality and bias the logit distribution. This works well for general conversation scenarios.

However, in math/reasoning tasks under think mode, the model generates a complete reasoning process, where each "step" (e.g., "Step 1: simplify the left side") typically spans tens to hundreds of tokens. **The blind spot of per-token intervention is: it can correct the wording of individual tokens, but cannot judge "whether the direction of this entire reasoning step is correct"** — by the time the step ends and the direction is found to be wrong, the preceding tokens have already been generated.

The PRM (Process Reward Model) approach is: **at the end of each reasoning step, evaluate "did this step go correctly," and if it went wrong, correct course before entering the next step**. Compared to per-token intervention, the signal granularity rises from character-level to step-level.

#### Complementary Roles of ARM VM and Step-Level Intervention

The two are not mutually exclusive; they complement each other at different granularities:

| Scenario | Intervention Method | Trigger Timing |
|----------|--------------------|-|
| General Q&A / instruction following | ARM VM (per-token) | High-entropy token positions |
| Math / reasoning / code (think mode) | ARM VM + step-level intervention stacked | ARM VM: high-entropy tokens; step intervention: at each step end |

Step-level intervention only activates within `<think>` blocks (general conversation has no reasoning step structure and does not trigger).

#### How to Identify Step Boundaries

In Qwen3's think mode, after completing each reasoning step, the model naturally outputs **double newline `\n\n`** as a separator. This is a natural signal for step boundaries requiring no additional trained classifier.

RSD (ICML 2025) and ThinkPRM ([arxiv 2504.16828](https://arxiv.org/abs/2504.16828)) both use `\n\n` as the step separator, following the same approach.

#### Specific Intervention After Step End

**Core idea: don't roll back — instead "race three paths forward and pick the best."**

Intuitively, "this step went wrong" → roll back and regenerate this step sounds reasonable, but in vLLM streaming, the KV cache of already-generated content cannot be cheaply rolled back to the step's starting point (the cost is equivalent to re-prefilling the entire step). The practical approach is: don't modify already-generated content; instead, **at the moment of entering the next step, have the model simultaneously explore three different starting directions and continue with the best one**.

Specific flow:

```
Model outputs \n\n (current step ends)
  ↓
Pause; have the model independently generate 3 candidate continuations from the current position
  Each greedily expanded for L tokens (L is an experimental parameter; initial L=5, control groups L=10/L=20)
  ↓
Compare the "fluency score" of 3 candidates (avg log-prob — how confident the model is in generating these words)
  ↓
Select the highest-scoring candidate; discard the other two; continue normal generation
```

**Advantage**: Requires training no new model; directly reuses the existing vLLM forward pass; implementable today. `\n\n` trigger borrowed from RSD/ThinkPRM; B=3 candidates and log-prob scoring borrowed from AdaDec (FSE 2026, [arxiv 2506.08980](https://arxiv.org/abs/2506.08980), code generation scenario). **This specific combination has no end-to-end paper validation**; it is SIA's own exploration.

**Theoretical limitation**: Log-prob measures "how naturally the model writes in this direction," not "whether this reasoning step is logically correct." The model can perfectly well assign high probability to a step that sounds reasonable but is logically wrong (i.e., "confidently wrong"); if the model is already inclined to go in the wrong direction at this position, all three forked paths might be different versions of the same error, and the highest log-prob path is still wrong. **Expected: Route 1 can filter obviously off-track directions, but is powerless against "sounds reasonable but logically wrong" steps; effectiveness ceiling is limited.**

#### Comparison of Two Routes

| Route | Trigger | Candidate Length | Scoring Signal | Literature Validation | Training Required |
|-------|---------|-----------------|----------------|----------------------|-------------------|
| **Route 1** (do first) | `\n\n` | L=? (TBD by experiment) | LLM avg log-prob | No end-to-end validation, SIA exploration | **No** |
| **Route 2** (contingent on conclusions) | `\n\n` | Full step to `\n\n` | External PRM scoring | **RSD (ICML 2025) validated on MATH500/AIME ✓** | Requires PRM training |

Route 1's primary value is **zero-cost establishment of engineering infrastructure and quantification of "what we can get for free"**, not expecting high effectiveness — the conclusion will likely be "effectiveness is limited; Route 2 is recommended." Route 2 is the only approach with top-conference end-to-end validation on math reasoning; training data threshold can be lowered using the SP-PRM approach (automatically generating step labels from existing outcome preference data).

**Applicability note**: The top-conference conclusion that PRM outperforms ORM (Lightman et al., ICLR 2024, [arxiv 2305.20050](https://arxiv.org/abs/2305.20050)) is based on the MATH dataset; it has support for reasoning/code tasks but no direct validation in general conversation. Step-level intervention only commits to effectiveness in think-mode reasoning tasks.

#### This Month's Actual Work

1. **Implement Route 1 (step-level pause-and-rerank)**: In `SIALogitsProcessor`, detect `\n\n` within `<think>` blocks, trigger B=3 greedy fork, use LLM avg log-prob to select the best; candidate expansion length L as an experimental variable (initial L=5, control groups L=10/L=20), determine the optimal L for reasoning step boundaries
2. **A/B effectiveness evaluation**: MATH500 / AIME under think mode, three-way comparison: AdaDec baseline vs pure ARM VM vs noSIA, quantify the incremental gains from step-level intervention
3. **Throughput impact measurement**: At conc=16, actual throughput overhead when each `\n\n` triggers B=3 greedy fork
4. **Go/no-go on external PRM**: If AdaDec baseline effectiveness meets requirements, defer external PRM training beyond 6 months; if insufficient, output data requirements and effort estimate for PRM training

### Month 5 Success Criteria

| Metric | Month 5 Goal |
|--------|-------------|
| Route 1 implementation | `SIALogitsProcessor` `\n\n` trigger + B=3 greedy fork running end-to-end; optimal L determined through experiments |
| A/B effectiveness conclusion | MATH500 Route 1 vs pure ARM VM vs noSIA three-way comparison complete |
| Overhead measurement | Route 1 throughput impact at conc=16 measured |
| Go/no-go on external PRM | Clear conclusion on whether external PRM training is needed |
| Multimodal VM data | **Collection complete** |

---

## Month 6 (2026-12): Full Multimodal VM Delivery

**Primary track**: Complete full delivery of multimodal VM from training to production launch.

### Task 6.1: Multimodal VM Training

**Why multimodal VM is needed**: When the main LLM is Qwen3-VL-30B-A3B-Instruct and user input contains images, the current text-only VM completely cannot see image content — VM scoring becomes blind guessing, and intervention may be counterproductive.

**Solution**: Use a multimodal VLM (e.g., Qwen3-VL-4B) as the base and train a reward head, enabling the VM to understand both image and text context simultaneously.

- **Applicability**: Only relevant for VLM main inference LLMs (pure-text 0GM-VL-35B does not need this)
- **References**: Training procedure follows the reward head training approach from the original SIA paper (arxiv 2602.21215); data construction references the multi-dimensional preference annotation framework from ArmoRM (arxiv 2406.12845).
- **Supplementary references**:
  - **Skywork-VL Reward** ([arxiv 2505.07263](https://arxiv.org/abs/2505.07263), 2025): One of the strongest open-source vision-language RMs, directly usable as backbone selection reference or fine-tuning starting point.
  - **MSRL** ([arxiv 2603.25108](https://arxiv.org/abs/2603.25108), CVPR 2026): Multi-stage multimodal reward modeling, providing a complete VLM RM training framework.
  - **BaseReward** ([arxiv 2509.16127](https://arxiv.org/abs/2509.16127), 2025): A systematic analysis of key components in multimodal RM (modeling paradigm, reward head, training strategy, data filtering), achieving SOTA on multiple benchmarks including MM-RLHF-Reward Bench; serves as a systematic reference for VLM RM development.
- **Effort**: 2-3 weeks of training (data already prepared)

### Task 6.2: Multimodal VM Launch (Full Support for Qwen3-VL Scenarios)

Deploy multimodal VM; under Qwen3-VL-30B scenarios, SIA's intervention quality on image-text input upgrades from "blind guessing" to "truly understanding images."

> **Expected gains upon completion**: Multimodal support officially available; image-text mixed requests no longer bypass SIA; product differentiation advantage extended to visual understanding domain; Qwen3-VL-30B scenario and 0GM-VL-35B scenario achieve SIA capability parity.

**Month 6 additional note**: If Month 4's tiny judge PoC produced positive results, this month can additionally launch production-quality refinement of the tiny judge (1-2 weeks, in parallel with multimodal VM training).

### Month 6 Success Criteria

| Metric | Month 6 Goal |
|--------|-------------|
| Multimodal VM training | **Training complete, offline validation passed** |
| Qwen3-VL + SIA multimodal intervention | **Officially available** |
| Tiny judge productionization | **If Month 4 PoC was positive and bandwidth allows** |

---

## Final State Summary (Expected End of 2026-12)

| Metric | Current (2026-06) | Conservative Goal (CUDA graph still unavailable) | Optimistic Goal (CUDA graph restored) |
|--------|-------------------|---------------------------------------------------|---------------------------------------|
| conc=16 SIA tok/s | 369 | **≥ 550** | **≥ 750** |
| SIA/noSIA throughput ratio | 35% | **≥ 53%** (550÷1040) | **≥ 72%** (750÷1040) |
| SIA/noSIA ITL multiplier | 3.0× | **≤ 1.9×** | **≤ 1.5×** |
| AlpacaEval win-rate vs noSIA | 65.4% (Skywork judge, pending GPT-4 validation) | **≥ +5% (independently validated by GPT-4 judge)** |
| Multimodal VLM scenario support | ❌ | **✅** |
| Does VM need per-token calls | Yes (~20% of tokens) | **No (block-wise, ~5% or below; PRM TBD after Month 5 evaluation)** |

---

## Milestone Timeline

```
End of 2026-07  conc=16 tok/s ≥ 550; GPT-4 evaluation baseline established
End of 2026-08  Same-vocabulary VM training complete; two-stage coarse-filter PoC concluded
End of 2026-09  conc=16 tok/s ≥550 (conservative) / ≥750 (optimistic, CUDA graph restored); same-vocabulary VM launched; effectiveness quantifiable for first time
End of 2026-10  AlpacaEval win-rate ≥ +5%; optimal intervention mode determined; tiny judge PoC concluded
End of 2026-11  PRM feasibility evaluation complete; go/no-go decision concluded (ARM VM blind spot quantified + engineering overhead evaluated)
End of 2026-12  Multimodal VM training complete and launched
```

---

## Key Academic Research Findings (2026-06)

Before formulating this roadmap, a systematic literature review was conducted (covering top conferences including NeurIPS, ICML, ICLR, ACL, and others; 100+ papers; multiple rounds of cross-verification). The following three findings deserve particular attention:

### Finding 1: High-Concurrency Throughput Is a Research Gap in Token-Level Alignment Papers

Based on this review, token-level alignment papers (including top-conference works such as ICLR 2025's GenARM and ICML 2025's RSD) **only evaluate effectiveness in single-request or small-concurrency (≤4) scenarios** — no top-conference paper has systematically studied "throughput-effectiveness trade-offs for token-level intervention under high-concurrency continuous batching."

This means the throughput issue we observe in production (64% throughput drop at conc=16) is a **genuine engineering challenge with no academic solution yet**. If the SIA project systematically:

- Establishes evaluation benchmarks for high-concurrency scenarios (throughput vs alignment effectiveness Pareto)
- Validates block-wise scoring and AdaDec step-level intervention (including external PRM if Month 5 go/no-go passes) in high-concurrency settings
- Publishes the relevant results

This direction currently has no systematic academic research — it represents a potential engineering contribution.

---

### Finding 2: Judge Decoding's "Tiny Judge" Concept Has Major Potential for SIA, Yet Unexplored

Judge Decoding (arXiv 2025 preprint, arxiv 2501.19309) presents a counterintuitive result: replacing a large model's token acceptance criterion with **a linear projection layer of only 16,384 parameters (16.4k)**, using 500 triplet data samples and less than 1.5 hours of training, achieves **3.9–9.7× acceleration** on Llama-405B inference.

Current SIA uses a VM with 4B parameters for scoring, with ~30ms call latency.

**No work has yet explored**: Can the Judge Decoding tiny judge concept be transplanted to SIA — replacing the 4B VM with an extremely small scoring head (e.g., under 10M parameters), with training data likewise from preference pairs, achieving near-zero inference latency (400× parameter reduction, potentially 50–100× latency reduction)?

The cost of this direction is potential scoring precision loss, but for SIA's scenario where entropy gating already filters to ~20% of tokens triggering scoring, the tiny judge only needs to make coarse-grained good/bad judgments at critical positions — the precision requirement may not be high. **This direction has been incorporated into Month 4 Task 4.3 (contingent on bandwidth); if effectiveness meets the bar, Month 6 can further productionize it.**

---

### Finding 3: Effectiveness Improvement Numbers in This Field Are Pervasively Exaggerated — Treat with Caution

During the review, 25 specific performance claims were cross-verified, and **15 (60%) could not be substantiated from the original papers**. Typical examples:

> *Verification method: each claim was reviewed from three independent perspectives: reading the original paper, tracing experimental conditions, and cross-checking numbers — with "find counterarguments" as the default stance (not "find supporting evidence"); if at least two of three perspectives concluded the number could not be verified, the claim was marked as failing.*

| Paper Claim | Verification Result |
|-------------|-------------------|
| TARo: +22.4% improvement over baseline | ❌ Number not found in the original paper |
| TARo: +8.4% improvement over existing token-level methods | ❌ Number not found in the original paper |
| TITA: MMVet +8.6%, POPE +6.7% on LLaVA-1.5 | ⚠️ Only ICLR 2026 submission status, not peer-reviewed, cannot be independently verified |
| RSD: average +3.5 accuracy improvement over parallel decoding methods | ❌ Number not found in review records; verifiable numbers from RSD original paper: MATH500 achieves 88.0%, FLOPs reduced by up to 4.4× |
| GSI: 28% end-to-end latency reduction, 51% throughput improvement | ❌ Original paper's numbers don't match the claims in scope |

**Internal experiment corroboration**: The original SIA paper (arxiv 2602.21215) emphasizes that "sparse intervention only triggers at ~20% of high-entropy tokens, so inference overhead is far lower than full intervention." This claim is theoretically correct, but in our engineering practice, **the actual performance cost far exceeds the level implied by the paper**. The reason is that the paper completely ignores the execution overhead of the Value Model itself: each VM call requires a full GPU forward pass on the candidate sequences (~30ms/call), whereas the main LLM only takes ~9ms to generate one token. Even with only 20% of tokens triggering the VM, the VM call overhead already exceeds the main LLM itself, causing overall throughput at conc=16 to drop to 35% of noSIA — not the slight loss implied by the paper's framing of "sparse = low overhead." This gap is the most critical finding in engineering deployment and is the direct motivation for listing VM latency optimization as the highest priority in this roadmap.

**Internal experiment corroboration 2 (domain dependency of effectiveness)**: We used MMLU (a standard knowledge benchmark covering mathematics, physics, history, medicine, and other disciplines) as an effectiveness regression test. The rationale: the subjects covered by MMLU are almost entirely unrelated to the VM's training data (general instruction-following preference pairs) — it is a natural "out-of-domain test set." Results show that enabling SIA causes **no significant improvement or degradation** in MMLU accuracy. On the surface this seems positive (no regression), but the deeper implication is: **SIA's alignment effectiveness is highly sensitive to the domain coverage of VM training data** — if the types of questions users ask in production do not overlap with the VM training data's domain, SIA provides neither effectiveness improvement nor escapes the performance overhead of VM calls. This means that in vertical domains not covered by VM training data (e.g., specialized medical, legal, specific codebases), the benefits of deploying SIA are questionable; domain matching assessment should be conducted before deployment.

**Practical recommendation**: When evaluating externally published effectiveness improvements, do not directly cite their paper numbers. When communicating SIA's own effectiveness externally, ensure numbers come from reproducible independent evaluations (AlpacaEval, MMLU, and other standard benchmarks), not internal test sets.

---

### Finding 4: 2026 Academic New Trends

**(1) Sparse intervention paradigm independently validated by three groups**

GGRO (UAI 2026, arxiv 2606.09635), SeLaR (2604.08299), and AdaDec (FSE 2026, 2506.08980) — three groups working independently — all proposed "only intervene at high-entropy/low-confidence positions" strategies, fully consistent with SIA's `--entropy_threshold` design. This is **external academic endorsement** of SIA's core design intuition in 2026, while also signaling that this direction is no longer novel in academia. SIA's differentiated value must be demonstrated through systematic engineering research on high-concurrency throughput (see Finding 1).

**(2) Product-of-distributions fusion may improve E at low cost — already added to Month 3**

LLMdoctor (arxiv 2601.10416, 62.10% win vs. GenARM) changes additive logit biasing to product-of-distributions fusion; no VM retraining required; 1-2 days of engineering cost. **Already added to roadmap Month 3 Task 3.3.** This experiment's results also determine the necessity of Month 4 accept/reject experiments.

**(3) SAE Steering (DSPA) is a medium-term reconnaissance direction beyond the 6-month roadmap**

CMU's DSPA (arxiv 2603.21461) uses sparse autoencoders to directly apply alignment guidance in the LLM's activation space, **completely bypassing external VM/RM forward passes** (99.8% of activation values are zero). If the effect can match a 4B VM, L and T are fundamentally resolved, since the VM is completely removed from the critical path. Cost: SAE requires offline training; whether it can match a 4B VM remains unvalidated. **Recommended for evaluation as an independent research direction in 2027**; not included in the current 6-month execution plan.

---

## SIA Technical Limitations

While advancing the roadmap, it is important to clearly establish the current and foreseeable future technical boundaries of SIA, in order to set appropriate expectations.

| # | Limitation | Description |
|---|-----------|-------------|
| L1 | **Only applicable to self-hosted models** | SIA modifies the logit distribution at each step by injecting `LogitsProcessor` inside vLLM, requiring white-box access to the model's internal inference process. Cannot be used with external APIs such as OpenAI or Anthropic, nor with closed inference services that only expose a generation interface. |
| L2 | **Effectiveness highly dependent on VM training data coverage** | The Value Model can only provide reliable guidance in the domains and task types covered by its training data. For vertical domains not covered by training data (medicine, law, specific codebases, etc.), the VM's scoring may be noise or even negative guidance; SIA should not be expected to produce positive results in these scenarios. |
| L3 | **Multimodal input requires separately trained VM** | The original SIA paper and current implementation only support pure text input. When the main LLM is a vision-language model (VLM, e.g., Qwen3-VL-30B) and user input contains images, the text-only VM completely cannot see image content, and scoring degrades to blind guessing. Multimodal scenarios require separately training a VM supporting image-text input and re-evaluating effectiveness (roadmap Month 6 goal). |
| L4 | **Effectiveness is a statistical average; individual requests are not guaranteed** | SIA's alignment improvement is a statistical uplift over a large number of requests; it cannot guarantee the direction for any specific single request. For rare prompt types or domains not covered by VM training data, a single intervention may worsen the output. Effectiveness evaluation must rely on batch statistical metrics (e.g., AlpacaEval win-rate); single-instance results cannot be used for judgment. |
| L5 | **VM and main LLM share GPU memory, constraining each other** | The b2 inproc approach requires VM and main LLM to run on the same GPU cluster and share GPU memory. The larger the VM, the less memory available for the main LLM, and vice versa. This constrains: (a) the maximum context length supported by the main LLM; (b) the maximum concurrency the main LLM can serve; (c) the maximum scale of the VM. |
| L6 | **Tightly bound to inference framework version; upgrades are risky** | SIA's b2 inproc implementation deeply depends on vLLM's internal APIs; it has been confirmed that b2 inproc for 0GM-VL-35B completely fails on vLLM 0.19+. Every time the main LLM inference framework (vLLM version) is upgraded, the SIA layer needs re-adaptation and regression validation; upgrade costs are non-trivial. |

---

## Dependencies and Risks

| Risk | Affected Phase | Mitigation |
|------|---------------|------------|
| block-wise B=4 effectiveness loss exceeds expectations | Month 1 | Fall back to B=2 (sacrifice half the gain, preserve effectiveness) |
| Insufficient same-vocabulary VM training data | Month 2 | Supplement with synthetic preference data |
| MoE base VRAM too large (>10GB/GPU) | Month 2-3 | Fall back to dense 4B |
| Same-vocabulary VM training results below expectations | Month 3 | A/B comparison; keep old VM as fallback |
| Multimodal preference data difficult to obtain | Month 5-6 | Downgrade to exploratory experiment; no strong commitment |

---

## References

| Paper | Corresponding Task | Conference/Status |
|-------|-------------------|------------------|
| SIA original paper ([arxiv 2602.21215](https://arxiv.org/abs/2602.21215)) | Overall framework foundation / Task 6.1 | 2026 preprint |
| AlpacaEval 2.0 ([arxiv 2404.04475](https://arxiv.org/abs/2404.04475)) | Task 1.3 evaluation baseline | 2024 preprint |
| GenARM: Autoregressive Reward Model ([arxiv 2410.08193](https://arxiv.org/abs/2410.08193)) | Task 2.2 ARM training objective / Task 4.1 | **ICLR 2025** ✅ |
| Judge Decoding ([arxiv 2501.19309](https://arxiv.org/abs/2501.19309)) | Task 4.3 tiny judge PoC | 2025 preprint (ICLR 2025 not confirmed) |
| Let's Verify Step by Step / ORM vs PRM ([arxiv 2305.20050](https://arxiv.org/abs/2305.20050)) | Task 5.1 PRM applicability | **ICLR 2024** ✅ |
| Scaling LLM Test-Time Compute ([arxiv 2408.03314](https://arxiv.org/abs/2408.03314)) | Overall direction validation | NeurIPS 2024 Workshop ✅ |
| RSD: Reward-guided Speculative Decoding ([arxiv 2501.19324](https://arxiv.org/abs/2501.19324)) | Task 2.3 two-stage filter / Task 4.2 | **ICML 2025** ✅ |
| SSS: Stepwise Speculative Search ([arxiv 2508.15044](https://arxiv.org/abs/2508.15044)) | Task 2.3 two-stage filter basis | **EMNLP 2025** ✅ |
| GSI: Generative Speculative Inference ([arxiv 2506.04118](https://arxiv.org/abs/2506.04118)) | Task 2.3 two-stage filter basis | **ICLR 2026** ✅ |
| EASD: Entropy-Aware Speculative Decoding ([arxiv 2512.23765](https://arxiv.org/abs/2512.23765)) | Background reference (speculative decoding dual-entropy mechanism; **not applicable** to SIA's reward model scenario) | 2025 preprint |
| TARo: Token-level Adaptive Routing ([arxiv 2603.18411](https://arxiv.org/abs/2603.18411)) | Task 1.2 adaptive routing reference | 2026 preprint |
| LLMdoctor: Product-of-Distributions Fusion ([arxiv 2601.10416](https://arxiv.org/abs/2601.10416)) | Task 3.3 product-of-distributions fusion A/B | 2026 preprint |
| TITA: Token-level Inference-Time Alignment ([arxiv 2510.21794](https://arxiv.org/abs/2510.21794)) | Task 6.2 DPO distillation reference | 2025 preprint |
| GGRO: Gradient-Guided Reward Optimization ([arxiv 2606.09635](https://arxiv.org/abs/2606.09635)) | Finding 4 / Appendix A2 | **UAI 2026** ✅ |
| SeLaR: Soft Embedding Alignment at Low-Confidence ([arxiv 2604.08299](https://arxiv.org/abs/2604.08299)) | Finding 4: sparse intervention independent validation | 2026 preprint |
| AdaDec: Pause-and-Rerank at High Uncertainty ([arxiv 2506.08980](https://arxiv.org/abs/2506.08980)) | Finding 4: sparse intervention independent validation; **Task 5.1 AdaDec baseline implementation reference** (L=5 greedy fork + LLM log-prob) | **FSE 2026** ✅ |
| PRM as Unified Control Signal for Reasoning ([arxiv 2602.01070](https://arxiv.org/abs/2602.01070)) | Task 5.1 PRM complementary granularity support | 2026 preprint |
| Seesaw: PP↔TP Dynamic Parallelism Switching ([arxiv 2503.06433](https://arxiv.org/abs/2503.06433)) | Engineering optimization reference (TP/PP dynamic scheduling) | 2025 preprint |
| DSPA: SAE-based Activation Steering ([arxiv 2603.21461](https://arxiv.org/abs/2603.21461)) | Finding 4 / Appendix A1 | 2026 preprint (CMU) |
| ArmoRM: Multi-Objective Reward Model ([arxiv 2406.12845](https://arxiv.org/abs/2406.12845)) | Task 6.1 / Appendix C1 multi-objective VM | 2024 preprint |
| Token-level MDP Formalization ([arxiv 2602.02572](https://arxiv.org/abs/2602.02572)) | Theoretical background reference | **ICML 2026** ✅ |
| Nudging: Uncertainty-gated Sparse Intervention ([arxiv 2410.09300](https://arxiv.org/abs/2410.09300)) | Task 2.3 / direction validation | 2024 preprint |
| BatchLLM: Explicit Global Prefix Sharing ([arxiv 2412.03594](https://arxiv.org/abs/2412.03594)) | Engineering optimization reference | 2024 preprint |
| HybridFlow: ResourcePool LLM+RM Co-deployment ([arxiv 2409.19256](https://arxiv.org/abs/2409.19256)) | Appendix B4 independent GPU VM | **EuroSys 2025** ✅ |
| NEO: Asymmetric CPU-GPU Pipeline ([arxiv 2411.01142](https://arxiv.org/abs/2411.01142)) | Engineering optimization reference | 2024 preprint |
| STEP: Memory-triggered Search Tree Pruning ([arxiv 2601.09093](https://arxiv.org/abs/2601.09093)) | Engineering optimization reference (pressure-aware degradation) | 2026 preprint |
| RM Knowledge Distillation ([arxiv 2411.08302](https://arxiv.org/abs/2411.08302)) | Appendix B1 VM distillation basis / Task 2.2 training techniques | 2024 preprint |
| RM Distillation: Reward Model Compression ([arxiv 2405.19316](https://arxiv.org/abs/2405.19316)) | Appendix B1 VM distillation basis | 2024 preprint |
| RM Ensemble (common engineering practice) | Appendix B3 reward hacking protection | Common practice, no single citation |
| Cost-Effective RGTG: Vocabulary-wide Reward Head ([arxiv 2502.04517](https://arxiv.org/abs/2502.04517)) | Task 1.1 alternative / Task 2.2 VM architecture | **ICML 2025** ✅ |
| Low-Rank RM Parametrization ([arxiv 2407.04615](https://arxiv.org/abs/2407.04615)) | Task 2.2 VM scoring acceleration | **TMLR 2025** ✅ |
| From r to Q*: LLM as Q-Function ([arxiv 2404.12358](https://arxiv.org/abs/2404.12358)) | Task 2.2 VM cold-start theoretical basis (DPO ≡ Q-learning, log-prob difference encodes implicit reward, extended SIA derivation) | **COLM 2024** ✅ |
| Hydragen: High-Throughput Shared-Prefix Inference ([arxiv 2402.05099](https://arxiv.org/abs/2402.05099)) | Month 3 backup / T system optimization | **ICLR 2025** ✅ |
| To Intervene or Not: Probabilistic Gating ([arxiv 2606.11201](https://arxiv.org/abs/2606.11201)) | Task 1.2 dual entropy gating advanced | **ACL 2026** ✅ |
| Learning Adaptive LLM Decoding ([arxiv 2603.09065](https://arxiv.org/abs/2603.09065)) | Task 1.2 learned routing | 2026 preprint |
| Inference-Time Reward Hacking ([arxiv 2506.19248](https://arxiv.org/abs/2506.19248)) | Task 1.3 evaluation pipeline safety design | **NeurIPS 2025** ✅ |
| RMB: Reward Model Benchmark ([arxiv 2410.09893](https://arxiv.org/abs/2410.09893)) | Task 1.3 VM quality evaluation | **ICLR 2025** ✅ |
| RewardBench 2 ([arxiv 2506.01937](https://arxiv.org/abs/2506.01937)) | Task 1.3 VM quality evaluation | 2026 preprint |
| SP-PRM: ORM-derived Process Reward ([arxiv 2506.12446](https://arxiv.org/abs/2506.12446)) | Task 5.1 PRM data collection | 2025 preprint |
| ThinkPRM: Thinking Token Process Reward ([arxiv 2504.16828](https://arxiv.org/abs/2504.16828)) | Task 5.1 thinking mode PRM | 2025 preprint |
| DG-PRM: Dynamic Generalizable PRM ([arxiv 2507.17849](https://arxiv.org/abs/2507.17849)) | Task 5.1 cross-task PRM | **ACL 2025** ✅ |
| Skywork-VL Reward ([arxiv 2505.07263](https://arxiv.org/abs/2505.07263)) | Task 6.1 multimodal VM selection | 2025 preprint |
| MSRL: Multi-Stage Multimodal RM ([arxiv 2603.25108](https://arxiv.org/abs/2603.25108)) | Task 6.1 multimodal VM training framework | **CVPR 2026** ✅ |
| BaseReward: Multimodal RM Baseline ([arxiv 2509.16127](https://arxiv.org/abs/2509.16127)) | Task 6.1 multimodal VM evaluation reference | 2025 preprint |
| Reward Models Are Secretly Value Functions ([arxiv 2604.22981](https://arxiv.org/abs/2604.22981)) | Long-term vision C0a LLM-as-Q-Function | 2026 preprint |
| EntropyInfer: Rigid/Dynamic Attention Head Pruning ([arxiv 2606.09508](https://arxiv.org/abs/2606.09508)) | Long-term vision B2 VM attention head pruning | 2026 preprint |

---

## Appendix: Long-Term Research Directions (Beyond the 6-Month Plan)

> The following directions all have research basis, but have not been included in the 2026-07 to 2026-12 execution plan due to high engineering cost, uncertain effectiveness, or dependency on prior work completion. For reference when planning 2027 initiatives.

---

### A. Fundamental VM Replacement — Bypassing RM Forward Passes (2027 Research Direction)

These directions share a common goal: **freeing the main LLM from dependency on external VM for per-token scoring**, thereby fundamentally eliminating L (VM latency) and T (concurrent throughput loss).

**A1. SAE Activation Space Steering (DSPA, CMU, arxiv 2603.21461)**

Uses sparse autoencoders (SAE) to directly apply alignment guidance in the LLM's internal activation space, without calling any external VM (99.8% of activation values are zero). If the effect can approach a 4B VM, L/T are fundamentally resolved. Current challenge: SAE requires offline training; whether it can match a 4B VM remains unvalidated.  
**Recommendation**: Start with 1-2 weeks of paper reading + small-scale feasibility experiments, then decide whether to start a project.

**A2. RM Gradient Guidance (Exploratory direction, no direct literature support)**

Concept: at high-entropy positions, use the RM's **gradient information** (rather than scores) to guide token selection, theoretically avoiding full forward passes. Note: this direction currently has **no top-conference direct validation** — GGRO (UAI 2026, arxiv 2606.09635) is often cited, but that paper's actual contribution is validating the entropy-adaptive strategy of "only intervene at high-entropy positions" (consistent with SIA's `--entropy_threshold`), not gradient guidance. Furthermore, standard inference does not maintain gradient graphs; implementing this requires gradient checkpointing or approximation schemes, with far higher engineering cost than direct forward passes. Feasibility requires independent experimental validation before deciding whether to start a project.

---

### B. Incremental VM Efficiency Optimization (Natural Extension of the 6-Month Plan, 2027 Q1)

**B1. VM Distillation: 4B → 1.7B (NeurIPS 2024 Research Background)**

After the Month 3 same-vocabulary 4B ARM VM validates its effectiveness, use the 4B as a teacher to distill a 1.7B student VM. Relevant literature ([arxiv 2411.08302](https://arxiv.org/abs/2411.08302), [arxiv 2405.19316](https://arxiv.org/abs/2405.19316)) preliminarily shows that distilling a large RM to a small RM can retain ~80–90% of preference judgment capability, with ~2× latency reduction and halved VRAM usage — but these specific numbers **have not been independently verified internally** and should be treated as order-of-magnitude estimates, not definitive conclusions.  
**Recommended timing**: 2027 Q1, contingent on Month 3's 4B ARM VM meeting the effectiveness bar.

**B2. VM Internal Attention Head Pruning (EntropyInfer approach, arxiv 2606.09508)**

Categorize VM's internal attention heads into "structural (Rigid)" and "semantic decision-type (Dynamic)" — Rigid heads can be skipped or computed at reduced precision. Note: EntropyInfer's original 2.39× speedup was measured for **LLM prefill at 100k+ tokens**, while SIA's VM overhead is concentrated in **decode-phase short-sequence scoring** — the scenarios are very different; "20–40% latency reduction" is an extrapolation with **no direct experimental support**; actual gains require independent measurement on VM. Prerequisite: need to first conduct offline analysis of the current VM's (Qwen3.5-4B) attention activation patterns to confirm whether the proportion of Rigid heads is large enough; otherwise the gains will be even more limited.

**B3. RM Ensemble (Common Engineering Practice)**

Deploy 2–3 different VMs and average their scores, reducing single-VM reward hacking risk. After B1 is complete (1.7B VM available), two 1.7B VMs have lower VRAM requirements than one 4B VM, with manageable throughput impact.  
**Current prerequisite**: First establish reward hacking monitoring metrics (e.g., rate of samples with high VM scores but poor human ratings); introduce ensemble after evidence is available.

**B4. VM Independent GPU Cluster Deployment (HybridFlow, EuroSys 2025, arxiv 2409.19256)**

HybridFlow proposes a ResourcePool abstraction supporting distributed deployment of LLM + RM: VM has its own dedicated GPU cluster, its scoring runs truly in parallel with the main LLM's next decode step; **VM latency completely removed from LLM critical path**, with throughput ratio theoretically recovering from 35% to near 100%. Current b2 inproc is a colocated sequential mode; if additional GPUs become available in the future (1–2 dedicated A100s for VM), distributed mode is the fundamental solution.  
**Prerequisite**: Requires additional GPU resources; prioritize evaluation when capacity expands in 2027; not feasible under the current 4×A100 configuration.

---

### C. Deferred Lightweight Exploration Tasks (No Fixed Schedule Within 6 Months)

**C0a. LLM-as-Q-Function PoC (originally Task 2.4, ~1 week)**

From r to Q\* (COLM 2024, arxiv 2404.12358) and "Reward Models Are Secretly Value Functions" (arxiv 2604.22981, 2026) both propose that the LLM's token log-prob difference `log P(y|x, prefix) - log P(y|x)` theoretically approximates a Q-function, meaning the main model 0GM-VL-35B itself already carries sufficient reward signal — **potentially requiring no external VM at all** — with L and T fundamentally eliminated. Experiment cost is very low (add `--rm_backend self` mode, compare on 50 AlpacaEval questions), but priority is lower than tiny judge (2.3) — if 2.3 succeeds, 2.4's significance diminishes; if 2.3 fails, 2.4 can serve as an alternative direction for quick validation. Recommend following up after Month 4 tiny judge PoC conclusions are available.

**C0b. DPO Distillation Exploration (Exploratory, 2-4 weeks)**

Direction: collect generation trajectories preferred by the VM in the SIA system; use DPO to distill them into the 0GM-VL-35B main model, allowing the main model to internalize alignment signals and maintain alignment effectiveness without an external VM at inference time.

TITA (2025, arxiv 2510.21794) demonstrates that the inference-time log-ratio method (DPO equivalent form) is effective, but that is inference-time correction, not training-time distillation. Training-time DPO distillation **has no direct top-conference evidence** in this review; effectiveness is uncertain. Recommend evaluating as an independent exploratory project after the 6-month roadmap is complete with sufficient VM preference data available.

---

### E. Alignment Effectiveness Deepening (Coverage of More Task Types)

**C1. Multi-Objective VM (ArmoRM, arxiv 2406.12845, 2024)**

Train a multi-dimensional reward head (helpfulness / correctness / safety, etc.) with a gating network that dynamically weights dimensions based on prompt type: coding tasks favor correctness, conversational tasks favor helpfulness. The current VM is a single-objective ORM with uneven alignment effectiveness coverage across tasks.  
**Main obstacle**: Requires multi-dimensional preference annotation datasets; can simultaneously add multi-dimensional labels during Month 2 data collection as preparation for later.

**C2. Fusion of PRM and Token-Level VM**

If Month 5 feasibility evaluation concludes go, after PRM provides step-level scores, how to dimensionally reduce and distribute them to token-level (e.g., uniformly across tokens within a step), improving token-level intervention signal quality without additionally constructing token-level preference data, is an open question. Currently no top-conference direct validation; recommend running ablation experiments when PRM is actually deployed.

---

> **Long-term direction priority reference** (after prior work is complete): C0a/C0b (deferred lightweight exploration, can start anytime) > B1 VM distillation (low risk, expected gains) > E1 multi-objective VM (effectiveness coverage) > A1 DSPA (high potential, needs validation) ≥ A2 GGRO gradient (high potential, major engineering challenges) > B2/B3 (dependent on specific metrics).

---

## <a name="appendix-cuda-graph"></a>Appendix: 0GM-35B CUDA Graph Debugging History

**Conclusion**: RM CUDA graph (vllm PIECEWISE mode) is not viable on 0GM-35B; three rounds of systematic fixes all failed. This issue is unrelated to the tokenizer and is a fundamental constraint at the architectural level.

### Data Comparison

| Mode | Steady-State Latency (@500+ steps) | vs eager |
|------|------------------------------------|---------|
| eager (current production) | 40–50ms / step | Baseline |
| PIECEWISE fix3 (final attempt) | 130–140ms / step | **2.7–3.5× slower** |

### Three Rounds of Fixes

| Round | Fix Content | Result |
|-------|------------|--------|
| fix1 | CUDAGraph isolation: RM forward moved to independent CUDA stream to avoid conflict with main LLM graph | ❌ Failed: NaN chain contamination (§3.9) |
| fix2 | APC=0 (disable automatic prefix caching) + max_num_batched_tokens=65536 | ❌ Failed: piecewise completely degrades after first request |
| fix3 | Switch SDPA backend (flash_attn → math → xformers), tested one by one | ❌ Failed: first request succeeds, all subsequent requests fail (⚠️ root cause unresolved) |

### Root Cause Analysis

vllm PIECEWISE CUDA graph design assumptions: decode step batch_size=1, fixed number of tokens per step → fixed-shape graph can be precompiled.

RM's actual workload: topK=10 **complete candidate sequences** (prefix + candidate token); each step is equivalent to a prefill, with both batch_size and sequence length varying over time → cannot satisfy the fixed-shape assumption; graph execution degrades to eager fallback or triggers shape mismatch errors.

Same-vocabulary VM training does not change the prefill-heavy nature of RM, so it cannot resolve this issue.

### Impact

- 0GM-35B latency improvement can only come from eliminating cross-tokenizer overhead (~7ms), not from CUDA graph
- After same-vocabulary VM launches, b2_score_call p50 is expected to drop from ~30ms to ~23ms (not ~11ms)
- The ~11ms target is only achievable if the CUDA graph issue is fixed in a future vllm version
- VL-30B achieving 11ms is due to: ① same vocabulary (no cross-tokenizer overhead) ② CUDA graph available (VL-30B did not reproduce this bug)

**Reference experiment**: `alpaca-0gm35b-piecewise-fix3-20260610`; detailed debugging logs at `doc/0gm-35b-sia-perf-breakdown-20260609.md` §7 and `doc/cuda-graph-debugging-journal.md`.
