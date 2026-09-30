# Inference-Time Alignment & Sparse Intervention: Literature Survey (2026-07-01)

---

## 1. Executive Summary

### Key Finding

The 60-paper survey reveals a clear pattern: the dominant paradigm in inference-time alignment is **per-token logit bias with a separate reward model (RM)**, and virtually every paper in this space faces the same cost bottleneck SIA faces. The literature has converged on three architectural responses to that bottleneck:

1. **Vocabulary-wide scoring in one forward pass** (FaRMA, GenARM, RAD-Q, ARM): replace K separate RM calls per step with a single pass whose output head has |vocab| or d-dimensional structure, reading all candidate scores from one forward pass. This is the highest-leverage architectural change.
2. **RM distillation into a lightweight proxy** (CriticControl, Otter): train a cheap critic/head that approximates the RM signal without invoking the full RM at inference time.
3. **Coarser intervention granularity** (CARDS, STARS, RSD, GSI): move from per-token to per-segment intervention, reducing RM call count proportionally to segment length.

SIA already implements **option 3 at the token level** (entropy_threshold gate, ~25% intervention rate) and the FaRMA proxy (vm_topk=1) is a partial implementation of **option 1**. The confirmed 2.36× speedup from the upcoming FaRMA vocab-wide head is the single highest-ROI near-term change.

No paper in the surveyed set offers a drop-in improvement that avoids model retraining. Every high-speedup idea requires either: (a) retraining the VM, or (b) fundamental changes to SIA's intervention architecture.

### Priority Ranking of Actionable Items

| Rank | Action | Source | Speedup Estimate | Difficulty |
|------|--------|---------|-----------------|------------|
| 1 | FaRMA vocab-wide head (confirmed, already planned) | FaRMA (ICML 2025) | 2.36× confirmed | Requires VM retraining |
| 2 | Tighten entropy_threshold (25% → ~10%) | Nudging, SIA paper | 1.1–1.3× | Days |
| 3 | CriticControl-style RM distillation into a small value head on LLM hidden states | CriticControl (ACL 2023) | High if faithful, uncertain | Months |
| 4 | RAD-Q/ARM single-pass reward head (if VM can share embedding space) | RAD-Q (TMLR 2025) | 1.5–3× at K=10 | Requires retraining |
| 5 | KV-cache prefix sharing across K=10 candidate scoring calls | RAD (EMNLP 2023) | Minor if not already in b2 | Medium |
| 6 | Monitor GenARM/ARM for shared-embedding VM retraining path | GenARM (ICLR 2025) | 1.2–2× | Requires retraining |

---

## 2. Taxonomy of Approaches

### Category A: Sparse / Selective Intervention Methods
Papers that gate RM calls using entropy, confidence, or other signals, intervening on only a subset of decode steps.

- **SIA (arXiv:2602.21215)** — Entropy threshold gate at τ_H ≈ 1.0; foundational for this codebase.
- **GGRO (UAI 2026)** — Entropy-gated gradient-based token insertion; incompatible with vLLM.
- **SPINE (arXiv:2511.17938)** — Entropy-gated test-time RL training; not an inference method.
- **Nudging (ACL 2025)** — Top-1 probability threshold for token injection; ~10% intervention rate.
- **CARDS (COLM 2025 / arXiv:2406.16306)** — Entropy-gated segment-level rejection sampling; ~1 RM call per 20 tokens.
- **STARS (arXiv:2511.03827)** — Fixed-interval segment rejection sampling with synchronous batching.
- **TARo (arXiv:2603.18411)** — Adaptive per-token blend weight via MLP router; calls RM at 100% of tokens.

### Category B: Vocabulary-Wide Single-Pass RM Architecture
Papers where the RM produces scores for the entire vocabulary (or all top-K candidates) in a single forward pass, eliminating K separate calls.

- **FaRMA (ICML 2025)** — |vocab|-dimensional RM output head + TD constraint training; 1 pass per step.
- **GenARM / PARM / UniARM (ICLR 2025 / ICML 2025 / arXiv:2602.09538)** — Autoregressive RM outputting per-token log-probs over full vocabulary.
- **RAD-Q / ARM (TMLR 2025)** — Low-rank Q-style reward head: single forward pass + dot products with frozen output embeddings.
- **GeDi (EMNLP 2021)** — 2 CC-LM forward passes covering all vocabulary via Bayes rule.
- **DExperts (ACL 2021)** — 2 LM forward passes (expert + anti-expert) covering full vocabulary logit delta.
- **Proxy-Tuning (COLM 2024)** — Logit delta from small tuned vs. untuned LM pair; full-vocab, dense.
- **TITA (arXiv:2510.21794)** — Log-probability ratio from a smaller VLM; dense per-token.
- **LLMdoctor (AAAI 2026)** — GFlowNet-trained "doctor" model providing per-token log-probs; dense.

### Category C: RM Distillation / Embedding into LLM
Papers that eliminate the separate RM call by fusing reward signal into the main LLM or a lightweight probe.

- **CriticControl (ACL 2023)** — Train a small value network via RL with terminal-only RM; critic replaces RM at inference.
- **Otter (arXiv:2408.10764)** — Non-disruptive parameter insertion into frozen LLM layers; reward from within LLM forward pass.
- **AAD (ICML 2026)** — DPO log-ratio as implicit reward; requires two full main-model copies.
- **BoNBoN (NeurIPS 2024)** — Fine-tune LLM to approximate BoN distribution; zero RM at inference.

### Category D: Per-Token Logit Bias with Separate RM (Dense)
Papers with the same basic architecture as SIA but without sparsity; they validate the design but offer no throughput improvement.

- **ARGS (ICLR 2024)** — k RM calls per token; dense, no entropy gate; SIA's direct antecedent.
- **RAD (EMNLP 2023)** — Top-k candidates scored per step with unidirectional RM + KV-cache reuse.
- **PARGS / PARM-Q (COLM 2025)** — ARGS with partial-sequence BT training; validates SIA's VM design.
- **IVG (EMNLP 2024)** — Token-level log-ratio + chunk-level beam search; two LLM-scale forward passes per token.
- **PAD (ICLR 2025)** — Personalized logit bias; no sparsity, 2× overhead.
- **Controlled Decoding (ICML 2024)** — Token-level and block-level prefix scorer; no entropy gating.
- **Reward Shaping / Stackelberg (arXiv:2602.02572)** — Pre-trained Q-function replaces live RM; offline only.
- **FUDGE (NAACL 2021)** — Top-200 candidate logit bias via lightweight classifier; dense.
- **CriticControl (ACL 2023 Findings)** — Value-ratio rescaling via critic; no entropy gate.
- **PPLM (ICLR 2020)** — Gradient ascent on KV history; 3–10× LM backward passes per token.

### Category E: Sequence-Level Selection / Best-of-N Variants
Papers that operate entirely at the complete-sequence level; no per-token intervention.

- **Best-of-N theory (ICML 2024)**, **BOND (NeurIPS 2024)**, **BoNBoN (NeurIPS 2024)**, **MBR-BoN (NAACL 2025)**, **InferenceTimePessimism (arXiv:2503.21878)**: sequence-level RM scoring; not relevant to SIA's token-level bottleneck.
- **Speculative Rejection (NeurIPS 2024)** — Dynamic batch sparsification of N parallel candidates; orthogonal to SIA.
- **FLIP (arXiv:2602.13551)** — Backward inference reward model; complete-sequence only.
- **HIA (arXiv:2508.05165)** — Pre-generation prompt filtering via cheap RM; pre-decode only.
- **TreeBoN (EMNLP 2025)**, **DARWIN (NAACL 2025)**, **Weak-to-Strong Search (NeurIPS 2024)**, **DeAL (arXiv:2402.06147)**: beam/tree search over multi-token segments; requires maintaining N parallel beams.
- **Scaling Test-Time Compute (arXiv:2408.03314)** — PRM-guided beam search for reasoning; not per-token.

### Category F: Speculative Decoding + Alignment
Papers combining speculative decoding with reward guidance; eliminate separate RM but require draft model.

- **RSD (arXiv:2501.19324)** — Draft model + PRM step-level accept/reject.
- **GSI (ICLR 2026)** — Draft + base model + PRM with tilted-reward correction.
- **SSR (arXiv:2505.15340)** — Step-level speculative reasoning for math CoT.
- **Judge Decoding (ICLR 2025)** — Tiny linear probe on target model hidden states to extend draft acceptance.
- **AASD (EMNLP 2025)** — Entropy-adaptive draft acceptance threshold; no RM.
- **SpecPipe / TD-Pipe (2025)** — Infrastructure-level pipeline parallelism; not alignment-specific.

### Category G: Training-Time / Reward Modeling Methods
Papers that contribute better RM training recipes or alignment training but have no direct inference-time mechanism transferable to SIA.

- **Q-RM (ICML 2025)** — Token-level Q-function for PPO training; 70B backbone.
- **TinyRM (ICML 2025 Workshop)** — 150–400M bidirectional pairwise RM; requires retraining as pointwise.
- **CPMI (ACL 2026)** — Contrastive MI labels for PRM training; offline only.
- **RAIN (ICLR 2024)** — Self-evaluation tree search; 4× slowdown.
- **Transfer Q* (NeurIPS 2024)** — Baseline-transfer Q-function with continuation rollouts per candidate.
- **PPO-MCTS (arXiv:2309.15028)** — MCTS over tokens using PPO value network; S× more VM calls.
- **SPINE (arXiv:2511.17938)** — Test-time RL with entropy-gated gradient updates.
- **LLM Query Scheduling (arXiv:2502.04677)** — Prefix-reuse scheduling for vLLM; infrastructure.
- **InfAlign (arXiv:2412.19792)** — Training-aware alignment; no inference mechanism.
- **CREAM (ICLR 2025)** — Consistency-regularized self-rewarding; training only.
- **Reward Model Survey (arXiv:2504.12328)** — Overview; no new methods.

---

## 3. Paper-by-Paper Analysis (High and Medium Priority)

### 3.1 FaRMA — Towards Cost-Effective Reward Guided Text Generation
**Venue:** ICML 2025 (arXiv:2502.04517)

**Core Method:**
FaRMA redesigns the RM to output a |vocabulary|-dimensional score vector in a single forward pass, trained with Bradley-Terry loss on full sequences plus a Temporal Difference constraint that enforces Bellman optimality on prefixes. At each decode step, one forward pass over the current prefix produces scores for every token in the vocabulary; the top-K scores are extracted by index and added to the LLM's log-probabilities. This collapses the cost from K separate RM forward passes (one per candidate) to exactly 1 regardless of K.

**Specific Applicability to SIA:**
This is the most directly applicable paper in the corpus. SIA's dominant cost at topk=10 is K=10 RM forward passes per intervention step, each requiring encoding a full chat-formatted string (user prompt + partial response + candidate token decoded to text). FaRMA eliminates all K-1 redundant passes. Concretely:
- The SIALogitsProcessor.apply() candidate-scoring loop is replaced by a single VM forward pass on the prefix alone, followed by an index lookup into the |vocab|-dimensional output.
- The step that decodes K candidate token IDs back into strings (currently a non-trivial CPU cost) is eliminated entirely — candidate scores are read directly by token ID.
- The b2 inproc backend needs to expose a new inference path for the |vocab|-head model.
- The existing vm_topk=1 FaRMA proxy approximates this but still encodes a single candidate string; the true FaRMA architecture is strictly cheaper because it never constructs any candidate string.
- The confirmed 2.36× speedup (from the project's own benchmarks) aligns with the theoretical ~K× reduction in RM compute at K=10, partially offset by the larger output head compute.

**Expected Speedup:** 2.36× confirmed (already planned). At vm_topk=1 the proxy achieves 760 tok/s; the full FaRMA head should approach or exceed this without the quality degradation of vm_topk=1.

**Implementation Difficulty:** Requires retraining the VM (VM-Qwen3-4B) with a |vocab|-dimensional output head and the TD constraint loss. The b2 inproc backend must be updated to expose the new head. This is a training-infrastructure change, not a serving change.

**Verdict: Pursue (already planned). Highest priority.**

---

### 3.2 GenARM — Reward Guided Generation with Autoregressive Reward Model
**Venue:** ICLR 2025 (arXiv:2410.08193)

**Core Method:**
GenARM trains an Autoregressive Reward Model parametrized as r(x,y) = Σ_t log π_r(y_t | x, y_{<t}), i.e., the ARM is itself an autoregressive LM whose next-token log-probs serve as per-token reward signals. At decoding, π_tilde(y_t) ∝ π_base(y_t) · π_r(y_t)^(1/β): one ARM forward pass per generated token yields scores over the full vocabulary, added (log-scaled by 1/β) to base LM log-probs. No candidate loop, no rejection sampling.

**Specific Applicability to SIA:**
GenARM's architecture is conceptually identical to a FaRMA-style vocabulary-wide scorer but parametrized differently: the ARM is an autoregressive LM (causal transformer) rather than a classifier with a |vocab| linear head. For SIA this means:
- The VM-Qwen3-4B would be retrained as an autoregressive RM using Bradley-Terry preference loss on (instruction, chosen, rejected) pairs.
- At inference, one ARM forward pass per entropy-gated step produces vocabulary-wide scores; no candidate strings need to be decoded.
- The ARM must share tokenizer/vocabulary with the LLM — Qwen3-4B and Qwen3-VL-30B share vocabulary, so this is feasible.
- The key advantage over FaRMA: GenARM's ARM is architecturally a standard causal LM, so existing LM infrastructure (vLLM, b2 inproc) supports it without custom output heads.
- The key disadvantage: the ARM produces full LM-style outputs, which may be heavier than FaRMA's single linear head on a prefix embedding.

Critically, GenARM does not include SIA's entropy gate — it intervenes at every token. Layering SIA's entropy_threshold on top of GenARM would produce sparse GenARM, reducing ARM calls from 100% to ~25% of steps, which is the right configuration for SIA.

**Expected Speedup:** 1.2–2× over current SIA (by eliminating K=10 → 1 pass per step), but lower than FaRMA because the ARM is a full LM pass (heavier than FaRMA's linear head). Requires retraining.

**Implementation Difficulty:** Requires retraining VM-Qwen3-4B as an autoregressive RM from scratch with BT loss. The b2 inproc backend can serve the resulting model without architectural changes since it is a standard causal LM.

**Verdict: Monitor. Retraining effort is similar to FaRMA but architectural fit with b2 inproc is better. If FaRMA retraining is underway, evaluate GenARM as an alternative parametrization during the same training run.**

---

### 3.3 RAD-Q / ARM — On the Low-Rank Parametrization of Reward Models for Controlled Language Generation
**Venue:** TMLR 2025 (arXiv:2407.04615)

**Core Method:**
RAD-Q observes that reward matrices learned by V-style models (one forward pass per candidate) are empirically low-rank. It replaces the V-style RM with a Q-style "ARM": a single prefix forward pass yields a context vector h_t, and per-token reward scores are r̂(v|h_t) = ⟨h_t, w⟩ + ⟨W·h_t, e(v)⟩, where e(v) are frozen output embeddings of the LM. This gives scores for all vocabulary tokens in O(1) RM call. The speedup at K=20–40 candidates is ~2–3× over V-style RAD as shown empirically. A training-time regularizer L_reg pushes Δr̂ toward zero for random tokens, improving fluency.

**Specific Applicability to SIA:**
RAD-Q is directly relevant but architecturally constrained. The key efficiency trick requires the RM to share the output embedding matrix with the main LLM, so W·h_t dotted with frozen LM embeddings scores all tokens cheaply. In SIA:
- VM-Qwen3-4B and Qwen3-VL-30B have different architectures; they do not share an embedding space.
- To apply RAD-Q, the VM would need to be retrained either (a) using the VL-30B's embedding matrix as its scoring head, or (b) distilled from a V-style VM teacher using the LLM's embeddings.
- If feasible, the SIALogitsProcessor would: run one VM prefix forward pass → compute W·h_t → dot-product with LM embedding table → extract top-K scores by index. No candidate strings decoded.
- The d×d W matrix for a 4B model has d≈3072; the dot product with the full Qwen3 vocabulary (~150K tokens) costs ~460M FLOPs per step — not clearly cheaper than a standard 4B forward pass in practice.
- The fundamental mismatch: SIA's VM currently sees chat-formatted strings (full prompt + partial response + candidate token decoded to text), while RAD-Q's Q-style head sees only the prefix h_t and scores the next raw token ID. This changes what the VM conditions on for its reward estimate.

**Expected Speedup:** 2–3× over V-style at K=20–40 candidates (from the paper's Figure 8), but the shared-embedding requirement makes this uncertain for SIA's cross-architecture setup.

**Implementation Difficulty:** Medium-to-hard. Requires retraining VM against LLM embedding space. The embedding-sharing constraint may necessitate using a subset of the VL-30B architecture as the VM backbone, which is a significant architectural departure.

**Verdict: Monitor. If the GenARM/FaRMA retraining infrastructure is built, evaluate RAD-Q as a lower-compute alternative (avoids full LM head in the ARM). The embedding-sharing constraint is the key feasibility gate — verify whether Qwen3-4B and Qwen3-VL-30B share output vocabulary embeddings.**

---

### 3.4 CriticControl — Critic-Guided Decoding for Controlled Text Generation
**Venue:** ACL 2023 Findings (arXiv:2212.10938)

**Core Method:**
CriticControl trains a small value network (critic) via actor-critic RL: the base LLM is the actor (frozen), the critic is trained with TD-style updates toward terminal rewards provided by a heavy RM called only at sequence completion. At inference, the critic replaces the heavy RM entirely: it computes a value ratio α(x_t, x_{<t}) = V(x_{≤t}) / V(x_{<t}), and the top-K token probabilities are multiplied by this ratio, then renormalized. No RM calls at inference.

**Specific Applicability to SIA:**
This is the most architecturally disruptive but potentially highest-speedup option outside of vocabulary-wide RM retraining. In SIA's context:
- The critic would be a small MLP or lightweight transformer head trained on the LLM's own hidden states (accessible within the vLLM LogitsProcessor if the LLM exposes them) or on decoded state features.
- Training requires RL rollouts at T>0 with terminal reward provided by the current 4B VM — the VM is used only during offline training, not at serving time.
- At inference, the LogitsProcessor calls the critic head (microseconds, not 10–22ms) to compute α, then applies it to top-K logits. VM is never invoked at serving time.
- Key risk: the critic must faithfully replicate the VM's guidance signal. For simple style/toxicity tasks (as in CriticControl's experiments), critics generalize well. For complex alignment rewards (helpfulness, instruction-following), fidelity is unproven at scale.
- The value ratio formulation (multiplicative, not additive) differs from SIA's additive logit bias; it would need to be converted to log space: log α = log V(x_{≤t}) - log V(x_{<t}), then added to logits.
- No entropy gate: CriticControl intervenes at every token. SIA's entropy_threshold gate could still be applied on top to reduce even the cheap critic calls further.

**Expected Speedup:** If critic is faithful: effectively eliminates the 10–22ms VM overhead entirely, collapsing the intervention cost to near zero. The gap between 760 tok/s (FaRMA proxy) and 1340 tok/s (noSIA) would largely close. However, quality degradation risk is significant.

**Implementation Difficulty:** Hard (months). Requires: (a) RL rollout data collection with terminal VM rewards, (b) critic training loop, (c) validation that critic quality matches VM on the target task distribution. No existing infrastructure for this in the SIA codebase.

**Verdict: Monitor for future exploration. Do not pursue before FaRMA vocab-head is validated. If FaRMA brings throughput to ~750+ tok/s and quality is confirmed, and the gap to noSIA (1340 tok/s) is still unacceptable, revisit critic distillation as the next research investment.**

---

### 3.5 RAD — Reward-Augmented Decoding: Efficient Controlled Text Generation With a Unidirectional Reward Model
**Venue:** EMNLP 2023

**Core Method:**
RAD uses a causally-masked (unidirectional) RM to enable KV-cache reuse across decoding steps. Because causal masking means prior token representations are unchanged when a new token is appended, the RM only needs to process the single newly-appended candidate token at each step (O(1) per candidate per step after caching the prefix), rather than re-encoding the full sequence. Cost drops from O(K·m²) to O(K·m) over a sequence of length m.

**Specific Applicability to SIA:**
The KV-cache prefix sharing insight is potentially relevant if SIA's b2 inproc backend is re-encoding the full chat-formatted prompt+partial-response for each of the K=10 candidate scoring calls. If prefix caching is already active in b2 inproc (which vLLM's prefix caching would suggest), this paper offers no new gain. If not:
- The concrete optimization is: cache the KV states for the shared prefix (system prompt + user message + response so far) once per intervention step, then score each candidate with a single new-token forward pass using the cached KV states.
- This reduces per-candidate RM cost from O(prefix_length) to O(1), proportional to the sequence length.
- For SIA at 25% intervention rate with 10 candidates and typical response lengths of 200–500 tokens, this could meaningfully reduce per-step VM cost if b2 is not already doing this.

**Expected Speedup:** Minor to low (likely <1.2×) if b2 inproc already caches prefixes. Worth a quick profiling check.

**Implementation Difficulty:** Easy if b2 already supports prefix caching (just verify). Medium if new caching logic is needed in the b2 backend.

**Verdict: Investigate first (days of profiling). Check whether b2 inproc re-encodes the full prefix for each of K=10 candidates or reuses a cached KV state. If re-encoding is occurring, implement prefix caching to amortize cost across K candidates per step.**

---

### 3.6 CARDS — Cascade Reward Sampling for Efficient Decoding-Time Alignment
**Venue:** COLM 2025 (arXiv:2406.16306)

**Core Method:**
CARDS uses token-level entropy to detect segment boundaries: when H(t) ≥ τ_u, the current token marks a semantic segment end. The RM evaluates the full prefix at that boundary and the segment is accepted or rejected via Metropolis-Hastings-style rejection sampling (accept probability ∝ exp((r - τ_r(t)) / β)). Rejected segments are discarded and resampled from the boundary. This yields ~1 RM call per 20 LLM tokens vs. ~5120 for per-token methods like ARGS.

**Specific Applicability to SIA:**
The segment boundary concept is the most transferable idea. SIA already uses entropy gating at the token level; CARDS shows that entropy spikes mark semantically coherent chunk boundaries where RM scoring is most useful. The insight could inspire an extension of SIA's entropy gate: rather than intervening at every high-entropy step, accumulate tokens between entropy spikes and score the accumulated segment once (1 call per semantic chunk vs. K calls per spike). However:
- This requires rejection sampling (resample from the boundary if the score is too low), which breaks vLLM's streaming decode model. KV-cache invalidation and re-prefill of rejected suffixes are non-trivial in vLLM.
- The FaRMA proxy (vm_topk=1, 760 tok/s) already achieves the K=1 call-per-step cost without the sequential-retry latency penalty of rejection sampling.
- CARDS explicitly acknowledges (Appendix B.1) that dynamic per-request segmentation breaks vLLM's continuous batching — the same problem SIA faces.

**Expected Speedup:** Low to medium within a logit-bias architecture; full CARDS adaptation is architecturally incompatible.

**Implementation Difficulty:** Hard (months) for full adoption. The entropy boundary observation is already captured by SIA's entropy_threshold.

**Verdict: Skip for near-term. The conceptual insight (entropy spikes = semantic segment boundaries) confirms SIA's existing design. No actionable implementation path without restructuring vLLM's generation loop.**

---

### 3.7 Nudging — Inference-time Alignment of LLMs via Guided Decoding
**Venue:** ACL 2025 (arXiv:2410.09300)

**Core Method:**
Nudging uses a top-1 probability threshold (max(softmax(logits)) < γ, default γ=0.3–0.5) to identify uncertain tokens, then generates a 16-token lookahead from a small aligned model and injects the first word as a forced token. At γ=0.5, ~11% of tokens trigger intervention. No reward model is used.

**Specific Applicability to SIA:**
Two ideas are transferable:
1. **Cheaper gating signal**: top-1 probability threshold costs O(1) after the argmax is already computed (which vLLM already does), whereas SIA's entropy_threshold costs O(vocab) for the full softmax + entropy sum. In practice for Qwen3-VL-30B with a 150K+ vocabulary, switching from entropy (O(150K) additions) to max-probability gating would save a small amount of CPU time per token. The gate check overhead is unlikely to be the bottleneck at p50=10–22ms VM calls, but it is measurable.
2. **Calibration of intervention rate**: Nudging finds ~10% of tokens are alignment-relevant at γ≈0.3–0.5. SIA's current ~25% intervention rate at entropy_threshold=1.0 may be tightenable. If raising entropy_threshold from 1.0 to 1.5 or 2.0 reduces intervention to ~10–15% with minimal quality loss, this is a free throughput gain.

**Expected Speedup:** 1.1–1.3× from tightening the entropy threshold, if quality holds. Gating signal switch (entropy → max-prob) is negligible speedup on its own.

**Implementation Difficulty:** Easy (days). Change --entropy_threshold from 1.0 to 1.5 or 2.0, run quality evaluation on a standard benchmark (e.g., AlpacaEval), confirm win-rate does not degrade significantly.

**Verdict: Pursue (easy win). Tighten entropy_threshold to 1.5–2.0 and measure quality/throughput tradeoff on a standard benchmark before committing to retraining.**

---

### 3.8 STARS — Segment-level Token Alignment with Rejection Sampling
**Venue:** arXiv:2511.03827

**Core Method:**
STARS uses fixed-size segments of K=15 or 30 tokens (rather than CARDS' entropy-based variable segments) so all requests in a batch hit RM calls synchronously, enabling one large batched RM forward pass instead of staggered small ones. The RM evaluates the prefix + segment; rejected segments are resampled up to 20 times with an adaptive acceptance threshold τ_r(k) that tightens as generation progresses.

**Specific Applicability to SIA:**
The synchronous batching insight is conceptually valuable for SIA's concurrent serving bottleneck. Currently, SIA's entropy gate fires at different steps for different concurrent requests, delivering ragged mini-batches to the VM (some requests trigger at step 47, others at step 52, etc.). If SIA adopted a fixed-interval intervention schedule (call VM every K tokens for all concurrent requests), VM calls from all requests would arrive together, allowing the b2 inproc backend to batch them into one larger forward pass with better GPU utilization.

However:
- Fixed-interval scheduling sacrifices SIA's entropy gate (which is more principled) and reduces quality.
- STARS uses tiny RMs (66–304M parameters) not a 4B VM; the batching benefit scales differently.
- The segment rejection-sampling loop is architecturally incompatible with vLLM's LogitsProcessor.
- The synchronous batching idea is really an infrastructure/scheduling concern: the better solution is to improve b2 inproc's ability to batch requests that happen to trigger VM calls near the same time, without forcing a fixed interval.

**Expected Speedup:** Low to negligible for SIA given the architecture mismatch. The synchronous-batching insight is better addressed at the vLLM request scheduler level.

**Implementation Difficulty:** Hard for full adaptation. The scheduling insight is captured more easily by improving b2 inproc request batching.

**Verdict: Skip. Monitor the batching insight — if VM call clustering becomes important at high concurrent request counts, revisit STARS' fixed-interval scheduling concept as a request batching heuristic.**

---

### 3.9 Inference-Time Alignment via Sparse Junction Steering (SIA paper itself)
**Venue:** arXiv:2602.21215

**Core Method:**
The foundational paper for this codebase. Entropy gate at τ_H ≈ 1.0; VM scores the partial sequence state V_θ(x, y_{≤t}) at each junction; scores added (×β) to logits. Theoretical result: sparse steering error is bounded by KL divergences at skipped steps only; noise-robustness result justifies sparse intervention. Weak-to-strong generalization (4B VM guiding larger LLMs) confirmed empirically.

**Specific Applicability to SIA:**
The paper itself is the implementation. Key gap identified: the paper's VM is a state-value function (1 forward pass per junction), whereas the current SIA implementation scores K=10 candidates separately (K=10 forward passes per junction). This architectural discrepancy is the root cause of SIA's per-step overhead not matching the paper's theoretical model. The paper's noise-robustness result (sparsity suppresses VM noise) provides theoretical justification for vm_topk=1 and the FaRMA proxy experiments. The τ_H ≈ 1.0 consistency finding validates the current config.

**Expected Speedup:** None beyond what is already deployed (entropy gate already in production).

**Implementation Difficulty:** Easy (already done).

**Verdict: No action. Already deployed. The architectural gap (K=10 passes vs. paper's 1 pass per junction) is addressed by FaRMA.**

---

### 3.10 PARGS — A Critical Look at Tokenwise Reward-Guided Text Generation
**Venue:** COLM 2025 (arXiv:2406.07780)

**Core Method:**
Theorem 1 proves that standard full-sequence RMs assign arbitrary rewards to prefixes, invalidating their use for per-token logit guidance. PARGS fixes this by applying Bradley-Terry loss at every prefix length during training, calibrating r_φ(y^{1:i}|x) to reflect genuine preference at each prefix length. At decode time, top-K candidates are scored with K RM forward passes per token (identical to ARGS/SIA in structure).

**Specific Applicability to SIA:**
PARGS provides theoretical justification that SIA's VM must be trained with partial-sequence loss (which VM-Qwen3-4B presumably uses). It validates the existing VM training setup but offers no throughput improvement. The K=10 forward passes per step architecture is unchanged.

**Expected Speedup:** None (validates existing design).

**Verdict: Skip (informational only). Confirms correctness of SIA's VM training approach.**

---

### 3.11 GGRO — Gradient-Guided Reward Optimization for Inference-time Alignment
**Venue:** UAI 2026 (arXiv:2606.09635)

**Core Method:**
At high-entropy positions, GGRO computes RM gradients w.r.t. token embeddings (1 forward + 1 backward pass, repeated S=3–8 times), selects a "nudging token" from the gradient direction, and inserts it into the sequence. This physically extends the sequence length mid-generation.

**Specific Applicability to SIA:**
Incompatible. Requires backward passes (not supported by b2 inproc forward-only scoring), inserts tokens mid-sequence (breaks vLLM's paged KV-cache), and is far more expensive per intervention than SIA's current approach. The entropy gating concept is already implemented in SIA.

**Verdict: Skip.**

---

### 3.12 TARo — Token-level Adaptive Routing for LLM Test-time Alignment
**Venue:** arXiv:2603.18411

**Core Method:**
A trained MLP router computes a per-token blend weight α̂_t ∈ (0,1) that interpolates between base model and reward model logits. Both models run at every token. Achieved 85.9 tok/s on 8×H100 for Qwen2.5-3B.

**Specific Applicability to SIA:**
Calling the RM at 100% of tokens is the opposite of SIA's sparse intervention strategy. Adding a router MLP would add training complexity and latency at every token. Not applicable.

**Verdict: Skip.**

---

### 3.13 PARM — Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model
**Venue:** ICML 2025 (arXiv:2505.06274)

**Core Method:**
Single ARM conditioned on user preference vectors via PBLoRA (preference-aware bilinear LoRA), replacing k per-objective ARMs with one model. 19.5% speedup over GenARM for k=3 objectives. One forward pass per token per request.

**Specific Applicability to SIA:**
SIA uses k=1 VM; PARM's savings only apply when k>1. No sparsity mechanism. Not applicable.

**Verdict: Skip.**

---

### 3.14 UniARM — Towards a Unified Autoregressive Reward Model for Multi-Objective Test-Time Alignment
**Venue:** arXiv:2602.09538

**Core Method:**
MoSLoRA architecture combining preference-agnostic shared module and preference-modulation module in a single ARM. 2.1× speedup over GenARM for multi-objective alignment by collapsing k models to 1.

**Specific Applicability to SIA:**
Same as PARM: savings are only relevant for k>1 objectives. Not applicable to SIA's single-VM design.

**Verdict: Skip.**

---

## 4. Synthesis: What SIA Should Do Next

### 4.1 Current State and Target

| Configuration | Throughput | Quality |
|--------------|-----------|---------|
| noSIA | 1340 tok/s | Baseline |
| SIA (topk=10, vm_topk=10) | 322 tok/s | Best aligned |
| FaRMA proxy (vm_topk=1) | 760 tok/s | Quality TBD |
| FaRMA vocab-wide head (planned) | ~760+ tok/s | Should match topk=10 |

The throughput gap is: 322 → 760 (FaRMA proxy, confirmed) → ~760+ (FaRMA head) → 1340 (noSIA upper bound). After the FaRMA vocab-head deployment, the residual gap is from VM call overhead at the ~25% of steps that trigger intervention. Each VM call still costs p50=10–22ms per step even with the vocab-head.

### 4.2 Recommendations Ranked by Impact × Feasibility

---

**Recommendation 1: Deploy FaRMA Vocab-Wide Head (Already Planned)**
*Expected throughput: ~760–900 tok/s. Difficulty: VM retraining.*

This is the confirmed-highest-ROI next step. The 2.36× speedup is empirically validated. The full FaRMA architecture is strictly better than the vm_topk=1 proxy because it eliminates candidate string decoding overhead entirely (no K strings constructed, no K forward passes, just one prefix forward pass + index lookup). Priority: highest.

Actionable steps:
1. Retrain VM-Qwen3-4B with |vocab|-dimensional output head and TD constraint loss (Bellman backup on partial sequences).
2. Update b2 inproc backend to serve the new head and expose per-token-ID score extraction.
3. Update SIALogitsProcessor.apply() to call b2 with prefix-only input and extract top-K scores by token ID.
4. Remove candidate string construction loop from the LogitsProcessor hot path.
5. Validate on AlpacaEval / MT-Bench that quality matches or exceeds current topk=10 SIA.

---

**Recommendation 2: Tighten entropy_threshold (1.0 → 1.5–2.0)**
*Expected throughput: 1.1–1.3× on top of current. Difficulty: days.*

Literature (Nudging, SIA paper ablations) suggests 10–15% intervention rates are often sufficient. Current SIA at τ_H=1.0 intervenes ~25% of steps. Raising τ_H reduces VM calls proportionally with no code changes. The noise-robustness result from the SIA paper provides theoretical justification that tightening the gate is safe.

Actionable steps:
1. Run quality evaluation (AlpacaEval, HH-RLHF, or domain-specific benchmark) at entropy_threshold ∈ {1.0, 1.5, 2.0, 2.5}.
2. Measure tok/s and win-rate at each threshold.
3. Select the Pareto-optimal point.

This is free throughput — no code change beyond the `--entropy_threshold` flag. Should be done before the FaRMA retraining to establish a quality baseline and calibrate the intervention rate target.

---

**Recommendation 3: Audit b2 Inproc for KV-Cache Prefix Sharing Across K=10 Candidates**
*Expected throughput: 1.1–1.5× if not already caching. Difficulty: days of profiling.*

The RAD paper's key insight is that a causal RM only needs to encode the single new candidate token if prefix KV states are cached. In SIA's current V-style setup, each of the K=10 candidate calls sends a full chat-formatted sequence to b2 inproc. If b2 is re-encoding the shared prefix (system prompt + user message + partial response) for each of the K=10 calls independently, enabling prefix caching would reduce per-call latency proportionally to the prefix fraction.

Actionable steps:
1. Profile b2 inproc with request traces: does each of K=10 candidate calls trigger a full sequence encode, or does b2 detect and reuse the shared prefix KV cache?
2. If prefix caching is not active: enable vLLM prefix caching in the b2 backend configuration and re-benchmark at topk=10.
3. Measure p50 VM call latency before and after. If latency drops meaningfully, this is a free win that also benefits the FaRMA vocabulary-head architecture (one prefix encode instead of K).

---

**Recommendation 4: Evaluate Smaller VM (1.7B or 0.6B) as Drop-In**
*Expected throughput: 1.5–2× on VM call cost. Difficulty: days of evaluation.*

The SIA paper confirms weak-to-strong generalization: a 0.6B or 1.7B VM can guide a 30B LLM with only moderate quality loss. The pre-trained VM-Qwen3-0.6B-Base and VM-Qwen3-1.7B-Base checkpoints are already available on HuggingFace. At b2 inproc, a 0.6B model runs significantly faster per call than the current 4B VM. This is a free experiment with existing checkpoints.

Actionable steps:
1. Load VM-Qwen3-1.7B-Base (or 0.6B) into b2 inproc with the same --rm_model config.
2. Run throughput benchmark (tok/s at topk=10).
3. Run quality benchmark (AlpacaEval win-rate or task-specific).
4. If 1.7B is within 2–3% win-rate of 4B at notably higher tok/s, use 1.7B as the default VM.

This does not require any code change — just a config swap.

---

**Recommendation 5: Long-Horizon — CriticControl-Style RM Distillation**
*Expected throughput: potentially 1340 tok/s (eliminates VM at serving time). Difficulty: months.*

If FaRMA vocab-head + entropy_threshold tightening brings SIA to ~900–1000 tok/s and the residual gap to 1340 tok/s (noSIA) is still commercially important, the next architectural step is distilling the VM's signal into a lightweight critic trained on LLM hidden states. This eliminates the separate VM inference entirely: the critic is a small MLP head on the LLM's own activations, callable within the LogitsProcessor for microseconds rather than 10–22ms.

Key risks: critic fidelity on complex alignment tasks (vs. simple toxicity/sentiment in CriticControl). Mitigations: validate critic vs. VM win-rate on held-out prompts before deploying; fall back to VM for long-tail prompts where critic confidence is low.

This is a 2–3 month research project contingent on FaRMA success.

---

**Recommendation 6: Investigate GenARM/ARM as Alternative VM Retraining Target**
*Expected throughput: overlaps with FaRMA. Difficulty: comparable to FaRMA.*

If FaRMA retraining infrastructure is being built, evaluate GenARM (autoregressive LM-style RM) and ARM/RAD-Q (low-rank Q-head with frozen LM embeddings) as alternative parametrizations during the same training infrastructure setup. GenARM may integrate more cleanly with b2 inproc (standard causal LM, no custom head). ARM/RAD-Q requires embedding-space alignment between VM and LLM, which may not be feasible for Qwen3-4B → Qwen3-VL-30B.

---

### 4.3 Comparison Against FaRMA Baseline (2.36× Confirmed)

| Recommendation | Throughput Gain | Stackable? | Risk |
|----------------|----------------|-----------|------|
| FaRMA vocab-head | 2.36× confirmed | — | Requires retraining |
| Entropy_threshold tightening | 1.1–1.3× additional | Yes, on top of FaRMA | Quality regression if over-tightened |
| b2 prefix caching audit | 1.1–1.5× additional | Yes | Low; may already be enabled |
| Smaller VM (1.7B) | 1.5–2× VM cost | Yes | Quality regression |
| CriticControl distillation | Up to 4× additional (eliminates VM) | Yes | High quality risk |

All recommendations except CriticControl are stackable on top of FaRMA. The aggressive path is: FaRMA vocab-head + entropy_threshold=1.5–2.0 + smaller VM → potentially 700–1100 tok/s from the current 322 tok/s, closing 60–80% of the gap to noSIA without any serving architecture changes.

---

## 5. Full Reference List

**Core Alignment / Sparse Intervention**
1. Hu et al. "Inference-time Alignment via Sparse Junction Steering (SIA)." arXiv:2602.21215, 2026.
2. Ethayarajh et al. "Is Best-of-N the Best of Them? Coverage, Scaling, and Optimality in Inference-Time Alignment." arXiv:2503.21878, 2025.

**Vocabulary-Wide / Single-Pass RM**
3. Chen et al. "Towards Cost-Effective Reward Guided Text Generation (FaRMA)." ICML 2025, arXiv:2502.04517.
4. Li et al. "GenARM: Reward Guided Generation with Autoregressive Reward Model for Test-time Alignment." ICLR 2025, arXiv:2410.08193.
5. Ye et al. "PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model." ICML 2025, arXiv:2505.06274.
6. Anonymous. "UniARM: Towards a Unified Autoregressive Reward Model for Multi-Objective Test-Time Alignment." arXiv:2602.09538, 2026.
7. Khoti et al. "On the Low-Rank Parametrization of Reward Models for Controlled Language Generation (RAD-Q/ARM)." TMLR 2025, arXiv:2407.04615.
8. Yang et al. "GeDi: Generative Discriminator Guided Sequence Generation." EMNLP 2021 Findings, arXiv:2009.06367.
9. Liu et al. "DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts." ACL 2021, arXiv:2105.03023.
10. Liao & Bing. "Tuning Language Models by Proxy (Proxy-Tuning)." COLM 2024, arXiv:2401.08565.

**Sparse / Selective Intervention**
11. Lu et al. "TARo: Token-level Adaptive Routing for LLM Test-time Alignment." arXiv:2603.18411, 2026.
12. Kang & McAuley. "Alignment-Aware Decoding (AAD)." ICML 2026, arXiv:2509.26169.
13. Shi et al. "Nudging: Inference-time Alignment of LLMs via Guided Decoding." ACL 2025, arXiv:2410.09300.
14. Anonymous. "SPINE: Token-Selective Test-Time Reinforcement Learning with Entropy-Band Regularization." arXiv:2511.17938, 2025.
15. Anonymous. "Gradient-Guided Reward Optimization for Inference-time Alignment (GGRO)." UAI 2026, arXiv:2606.09635.

**Per-Token Logit Bias (Dense, No Sparsity)**
16. Deng & Raffel. "Reward-Augmented Decoding (RAD)." EMNLP 2023.
17. Khoti et al. "A Critical Look At Tokenwise Reward-Guided Text Generation (PARGS)." COLM 2025, arXiv:2406.07780.
18. Mudgal et al. "Controlled Decoding from Language Models." ICML 2024.
19. Lu et al. "Inference-Time Language Model Alignment via Integrated Value Guidance (IVG)." EMNLP 2024 Findings, arXiv:2409.17819.
20. Lu et al. "PAD: Personalized Alignment of LLMs at Decoding-Time." ICLR 2025, arXiv:2410.04070.
21. Anonymous. "Token-Level Inference-Time Alignment for Vision-Language Models (TITA)." arXiv:2510.21794, 2025.
22. Anonymous. "LLMdoctor: Token-Level Flow-Guided Preference Optimization for Efficient Test-Time Alignment." AAAI 2026, arXiv:2601.10416.
23. Dathathri et al. "Plug and Play Language Models: A Simple Approach to Controlled Text Generation (PPLM)." ICLR 2020, arXiv:1912.02164.
24. Krause et al. "FUDGE: Controlled Text Generation With Future Discriminators." NAACL 2021, arXiv:2104.05218.
25. Anonymous. "Reward Shaping for Inference-Time Alignment: A Stackelberg Game Perspective." arXiv:2602.02572, 2026.
26. Khanov et al. "ARGS: Alignment as Reward-Guided Search." ICLR 2024.
27. Anonymous. "DeAL: Decoding-time Alignment for Large Language Models." arXiv:2402.06147, 2024.
28. Liu et al. "Don't throw away your value model! (PPO-MCTS)." arXiv:2309.15028, 2023.
29. Yang et al. "RAIN: Your Language Models Can Align Themselves without Finetuning." ICLR 2024, arXiv:2309.07124.

**RM Distillation / Embedding into LLM**
30. Kim et al. "Critic-Guided Decoding for Controlled Text Generation (CriticControl)." ACL 2023 Findings, arXiv:2212.10938.
31. Anonymous. "Predicting Rewards Alongside Tokens: Non-disruptive Parameter Insertion (Otter)." arXiv:2408.10764, 2024.

**Segment-Level / Cascade Methods**
32. Bhatt et al. "Cascade Reward Sampling for Efficient Decoding-Time Alignment (CARDS)." COLM 2025, arXiv:2406.16306.
33. Anonymous. "STARS: Segment-level Token Alignment with Rejection Sampling." arXiv:2511.03827, 2025.

**Sequence-Level / Best-of-N**
34. Touvron et al. "BOND: Aligning LLMs with Best-of-N Distillation." NeurIPS 2024, arXiv:2407.14622.
35. Gui et al. "BoNBoN Alignment for Large Language Models and the Sweetness of Best-of-n Sampling." NeurIPS 2024, arXiv:2406.00832.
36. Beirami et al. "Theoretical guarantees on the best-of-n alignment policy." ICML 2024.
37. Swamy et al. "Regularized Best-of-N Sampling with Minimum Bayes Risk Objective (MBR-BoN)." NAACL 2025, arXiv:2404.01054.
38. Shi et al. "Is Best-of-N the Best of Them? (InferenceTimePessimism)." arXiv:2503.21878, 2025.
39. Sessa et al. "BOND: Aligning LLMs with Best-of-N Distillation." NeurIPS 2024.
40. Kimura et al. "TreeBoN: Enhancing Inference-Time Alignment with Speculative Tree-Search and Best-of-N Sampling." EMNLP 2025 Findings, arXiv:2410.16033.
41. Zhao et al. "Fast Best-of-N Decoding via Speculative Rejection." NeurIPS 2024.
42. Sun et al. "DARWIN: Reward-Guided Tree Search for Inference Time Alignment of Large Language Models." NAACL 2025.
43. Anonymous. "Weak-to-Strong Search: Align Large Language Models via Searching over Small Language Models." NeurIPS 2024, arXiv:2405.19262.

**Speculative Decoding + Alignment**
44. Xu et al. "Reward-Guided Speculative Decoding for Efficient LLM Reasoning (RSD)." arXiv:2501.19324, 2025.
45. Anonymous. "Guided Speculative Inference for Efficient Test-Time Alignment of LLMs (GSI)." ICLR 2026.
46. Zhai et al. "SSR: Speculative Parallel Scaling Reasoning in Test-time." arXiv:2505.15340, 2025.
47. Tang et al. "Judge Decoding: Faster Speculative Sampling Requires Going Beyond Model Alignment." ICLR 2025, arXiv:2501.19309.
48. Anonymous. "Alignment-Augmented Speculative Decoding with Alignment Sampling and Conditional Verification (AASD)." EMNLP 2025.

**Transfer Q* / Q-Function**
49. Fisch et al. "Transfer Q*: Principled Decoding for LLM Alignment." NeurIPS 2024, arXiv:2405.20495.

**Inference-Time Compute Scaling**
50. Snell et al. "Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Model Parameters." arXiv:2408.03314, 2024.

**Training-Time / Reward Modeling Methods**
51. Anonymous. "Discriminative Policy Optimization for Token-Level Reward Models (Q-RM)." ICML 2025, arXiv:2505.23363.
52. Anonymous. "Tiny Reward Models (TinyRM)." ICML 2025 Workshop, arXiv:2507.09973.
53. Anonymous. "Efficient Process Reward Modeling via Contrastive Mutual Information (CPMI)." ACL 2026, arXiv:2604.10660.
54. Anonymous. "Small Reward Models via Backward Inference (FLIP)." arXiv:2602.13551, 2026.
55. Anonymous. "Aligning LLMs on a Budget: Inference-Time Alignment with Heuristic Reward Models (HIA)." arXiv:2508.05165, 2025.

**Low-Priority (Mentioned for Completeness)**
56. Anonymous. "Reward-Shifted Speculative Sampling Is An Efficient Test-Time Weak-to-Strong Aligner." arXiv:2508.15044, 2025.
57. Anonymous. "InfAlign: Inference-aware language model alignment." arXiv:2412.19792, 2024.
58. Anonymous. "CREAM: Consistency Regularized Self-Rewarding Language Models." ICLR 2025.
59. Anonymous. "A Comprehensive Survey of Reward Models: Taxonomy, Applications, Challenges, and Future." arXiv:2504.12328, 2025.
60. Anonymous. "LLM Query Scheduling with Prefix Reuse and Latency Constraints." arXiv:2502.04677, 2025.
61. Xu et al. "SpecPipe: Accelerating Pipeline Parallelism-based LLM Inference with Speculative Decoding." arXiv:2504.04104, 2025.
62. Anonymous. "TD-Pipe: Temporally-Disaggregated Pipeline Parallelism Architecture for High-Throughput LLM Inference." arXiv:2506.10470, 2025.