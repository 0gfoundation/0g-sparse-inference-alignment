# SIA Product Roadmap (H2 2026)

## What Is SIA

SIA is a system that improves AI response quality in real time \[1\]. As the AI generates a response, SIA deploys a companion **Value Model**—a live "quality inspector"—that evaluates each candidate word and steers the AI toward higher-quality outputs, without modifying the main model itself and operating entirely at runtime.

In comprehensive dialogue benchmarks, responses generated with SIA enabled are rated better **65% of the time** compared to those without SIA.

**Current scope**: The research has validated SIA on general-purpose dialogue tasks (helpfulness, harmlessness, honesty), and it works with both base and instruction-tuned models. The Value Model does not need to match the main model in size. The following areas currently have limited effectiveness or are not yet supported: **math, reasoning, and code** tasks that require multi-step logic—SIA intervenes word by word and cannot detect when the AI goes off track in its overall reasoning direction (addressed in Month 5); **image-text mixed inputs**—the Value Model is text-only and cannot understand image content (addressed in Month 6). Additionally, if the Value Model's training data is far removed from the domain of actual use, the quality of the intervention signal degrades.

---

## Current Pain Points

Enabling SIA currently comes with three costs that affect product experience and service scale:

1. **Slower responses**: Single-request throughput with SIA enabled is approximately 60% of baseline
2. **Reduced serving capacity**: In high-concurrency scenarios, the number of users that can be served simultaneously drops to roughly **35%** of baseline
3. **Limited scenario coverage**: Image-text requests bypass SIA entirely; math and reasoning tasks see limited benefit because SIA cannot track overall reasoning direction

The next six months are focused on systematically addressing all three issues.

---

## 6-Month Plan

### Month 1 (July 2026): Dramatically Reduce Value Model Overhead

**Background**: SIA currently intervenes at each "key decision point," which accounts for roughly 20% of all output tokens. Research \[1\] shows that intervening at only these ~20% high-entropy positions (where the AI is most uncertain) already matches the quality of full intervention. This month, we push further in two parallel directions—reducing how often the Value Model is called and reducing how long each call takes—to improve serving capacity.

**Key Work**:

- **Dual-gating mechanism** \[7\]: On top of the existing rule "only intervene when the AI is uncertain," we add a second filter: skip the Value Model call when all candidate tokens are roughly equivalent in quality (i.e., the intervention would have negligible impact). Both conditions must hold to trigger a skip. This filters out low-value calls, reducing them from ~20% to ~5% of tokens.
- **Tiny value head Proof of Concept** \[4\]\[8\]: We explore whether a tiny scoring module—under 10 million parameters, roughly 400× smaller than the current 4B Value Model—can replace the existing approach. This module would run directly inside the main model with near-zero latency. The goal this month is a clear feasibility verdict. Two candidate approaches will be tested in parallel: ① train a tiny classifier that reads the main model's internal representations to judge candidate token quality \[4\]; ② train a minimal linear layer on the same internal representations to directly output a score (under 0.01% of the main model's parameters) \[8\]. Both share the same core idea; we run small-scale experiments and pick the more promising one.

**Expected Outcomes**:
- Dual-gating: Value Model call frequency drops from ~20% to ~5%; concurrent serving capacity increases from 35% to **~50%**
- Tiny value head (if PoC succeeds): Value Model latency drops from ~30ms to <0.1ms; serving capacity could approach no-SIA levels
- No meaningful degradation in response quality

---

### Month 2 (August 2026): Train a Vocabulary-Aligned Value Model, Validate Speedup

**Background**: The current Value Model and main AI model use different vocabularies—the main model has 248,000 vocabulary entries while the Value Model has only 151,000. This mismatch prevents the two models from sharing an inference cache, forcing the Value Model to recompute from scratch on every call and adding unnecessary latency. This month's goal is to rigorously quantify how much speed is gained by switching to a vocabulary-aligned Value Model, while confirming that quality is preserved.

**Key Work**:

- **Obtain the original NTU training setup**: Request the complete training data and hyperparameters from the original paper authors (NTU), ensuring the only variable between old and new Value Models is the vocabulary—enabling a clean A/B comparison.
- **Train a vocabulary-aligned Value Model**: Starting from a base model with the exact same vocabulary as the main model, train using the identical data and settings as the original paper, eliminating all other confounding variables.

**Expected Outcomes**:
- Measurable reduction in per-call Value Model latency (shared vocabulary enables inference cache reuse)
- Response quality matches the original Value Model (quantified via A/B testing)

---

### Month 3 (September 2026): Improve Scoring Quality on Top of Vocabulary Alignment

**Background**: After Month 2 quantifies the speedup from vocabulary alignment, we pick one or two of the following candidate directions—based on experimental results—to further improve the Value Model's quality or efficiency.

**Candidate Directions** (decided after Month 2 results are in):

- **Higher-quality training data**: Incorporate high-quality public preference datasets and use the main model to auto-generate response pairs with quality labels, then retrain on the vocabulary-aligned foundation to improve scoring accuracy.
- **Vocabulary-wide scoring head** \[2\]: The current approach scores each candidate token in a separate forward pass (e.g., 10 candidates = 10 passes). The new design outputs scores for all vocabulary entries in a single pass—theoretical speedup up to 10×, measured at ~6× in practice.
- **Multiplicative fusion A/B experiment** \[3\]: SIA currently adds the Value Model's signal on top of the AI's output probability distribution. Multiplicative fusion instead multiplies the two distributions together for more precise steering. Literature shows a 10%+ win-rate improvement over additive fusion.
- **Adaptive fusion weight** \[9\]: The current `--weight` parameter is a fixed global value. We make it dynamic—automatically higher at high-entropy positions and lower when the AI is already confident. No retraining required; a 1–2 day code change that can be combined with multiplicative fusion for comparison.

**Expected Outcomes**:
- Measurable improvement in scoring quality or efficiency
- Concurrent serving capacity maintained at ≥ 50% or better

---

### Month 4 (October 2026): Push the Quality Ceiling Further

**Background**: After Month 3 delivers quality improvements and quantifies results, we decide whether further upgrades are needed and lock in the best intervention strategy.

**Key Work**:

- **Value Model scale decision**: If the current Value Model already meets the quality bar (Skywork win rate ≥ 65%), skip the upgrade. Otherwise, scale the Value Model from 4B to 8B parameters for stronger scoring capability. Because Month 1 will have already reduced call frequency to ~5%, the latency cost of a larger model remains acceptable.
- **Accept/reject intervention mode experiment** (inspired by \[6\]): The current approach softly biases the AI's output distribution. The new mode is a hard intervention: directly accept a token if the Value Model strongly endorses it; otherwise resample from the candidates. We evaluate which mode performs better.
- **Token-level reward redistribution for better training data (RED)** \[10\]: The Value Model is currently trained on sequence-level preference data (only knowing which response is better overall), which limits its precision. RED uses attention-based credit assignment to automatically break down a sequence-level rating into per-token signals—no additional human annotation required, just an offline pass over the existing preference dataset. Retraining the Value Model with these finer-grained labels is expected to improve scoring accuracy.

**Expected Outcomes**:
- Skywork win rate **≥ 65%**
- Best intervention strategy confirmed, providing a solid foundation for future iterations

---

### Month 5 (November 2026): Explore Improvements for Math and Reasoning Tasks

**Background**: SIA currently intervenes token by token, creating a blind spot for math problems, code generation, and other tasks that require multi-step reasoning. When the AI goes off track in its overall reasoning direction, per-token corrections cannot catch it in time.

**Key Work**:

- **Step-level quality evaluation (pause-and-rerank)** \[5\]: At each completed reasoning step, automatically pause and generate 3 alternative continuations, then select the best one to proceed. No new model training required—reuses the main model's existing reasoning capability.
- **Process Reward Model feasibility assessment (PRM)** \[6\]: If the no-training approach above shows limited gains, assess whether it's worth training a Value Model specifically designed to evaluate reasoning steps. This direction has been validated on math benchmarks in top-tier published research, but requires additional training cost. The deliverable for this month is a clear go/no-go decision.

**Expected Outcomes** (if experiments are positive):
- Measurable accuracy gains on math, code, and reasoning tasks
- Deliverable is "a clear experimental conclusion"—no firm commitment to launch

---

### Month 6 (December 2026): Support Image-Text Mixed Inputs

**Background**: When the main model processes requests that include images, the current Value Model cannot see the image content at all—effectively evaluating blind—which can make the intervention counterproductive. As a result, image-text requests currently bypass SIA entirely.

**Key Work**:

- **Train a multimodal Value Model**: Starting from a vision-language model base (Qwen3-VL-4B) that understands both images and text, collect image-text preference training data and train a Value Model capable of evaluating multimodal content.
- **Full deployment for image-text scenarios**: Connect the multimodal Value Model to the VL main model's inference pipeline so that image-text mixed requests can finally benefit from SIA quality improvements.

**Expected Outcomes**:
- Visual Q&A, image analysis, and other image-text scenarios benefit from SIA quality improvements
- Full support for the VL main model; image-text requests no longer bypass SIA

---

## 6-Month Goals Summary

| Dimension | Now (June 2026) | Target in 6 Months |
|---|---|---|
| Concurrent serving capacity (vs. no SIA) | 35% | **≥ 53%** (conservative) / ≥ 72% (optimistic) |
| Response speed impact (Value Model latency) | ~30ms | **~23ms** (conservative) / ~11ms (optimistic) |
| Dialogue quality (vs. no SIA) | 65% win rate | **Skywork win rate ≥ 65%** (see note) |
| Scenario coverage (image-text + math/reasoning) | Insufficient | **Image-text supported; math/reasoning conclusion reached** |

---

**Note**: Skywork is an industry-standard automated evaluation tool for AI dialogue quality. It works by having an evaluation model compare two responses head-to-head to determine which is better. A win rate ≥ 65% means that responses generated with SIA enabled are judged as superior in 65% of comparisons.

---

## References

\[1\] Runyi Hu et al. *Inference-time Alignment via Sparse Junction Steering*. arxiv 2602.21215. https://arxiv.org/abs/2602.21215

\[2\] *Cost-Effective RGTG: Vocabulary-wide Reward Head*. ICML 2025. https://arxiv.org/abs/2502.04517

\[3\] *LLMdoctor: Product-of-Distributions Fusion for Inference-time Alignment*. 2026. https://arxiv.org/abs/2601.10416

\[4\] *Judge Decoding: Tiny Judge for Speculative Decoding*. ICLR 2025. https://arxiv.org/abs/2501.19309

\[5\] *AdaDec: Adaptive Decoding via Pause-and-Rerank*. FSE 2026. https://arxiv.org/abs/2506.08980

\[6\] *RSD: Reward-guided Speculative Decoding*. ICML 2025. https://arxiv.org/abs/2501.19324

\[7\] *To Intervene or Not: Probabilistic Gating for Inference-time Alignment*. ACL 2026. https://arxiv.org/abs/2606.11201

\[8\] *SWIFT: Mining Intrinsic Rewards from LLM Hidden States for Efficient Best-of-N*. KDD 2026. https://arxiv.org/abs/2505.12225

\[9\] *AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation*. ACL 2025. https://arxiv.org/abs/2503.02832

\[10\] *RED: Unleashing Token-Level Rewards from Holistic Feedback via Reward Decomposition*. EMNLP 2025. https://arxiv.org/abs/2411.08302
