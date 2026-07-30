# vocab_lowrank Head Underperformance — Root Cause Analysis & Literature-Informed Fix Plan

**Date:** 2026-07-14
**Scope:** Why the `vocab_lowrank` (FaRMA-style) Value Model head underperforms the original `scalar` head on both 0GM-1.0-35B-A3B (no improvement / regression) and Qwen3-VL-30B-A3B-Instruct (positive but ~half the effect), and which of three candidate fixes is best supported by prior academic work.

---

## Part 1 — Root Cause Analysis

### Summary

The root cause is not a bug, not the cross-tokenizer mapping table, and not backbone quality — those were already diagnosed and fixed in earlier work (see `doc/vocab-lowrank-cross-tokenizer-mapping-20260703.md`, `exp/alpaca-vl30b-sia-backbone-frozen-20260707/report.md`). What remains is a genuine **training-objective / capacity mismatch intrinsic to the `vocab_lowrank` head as currently trained**, reproduced independently on two unrelated model families.

### Evidence chain

Token-weighted `top1_flip` rate (fraction of intervention steps where SIA's reward-adjusted top-1 candidate differs from the LLM's own top-1 candidate), computed directly from server logs:

| Deployment | intervention rate | **top1_flip rate** | "healthy" reference range (from this repo's own prior docs) |
|---|---|---|---|
| 0GM-35B, scalar head (`oldvm` control) | 16.72% | **66.29%** | 50–80% ✅ |
| 0GM-35B, vocab_lowrank (correct backbone, clean cross-tokenizer mapping) | 16.08% | **42.86%** | below range ✗ |
| VL-30B, vocab_lowrank new head (805Q, computed from `exp/docker_log_20260710_newhead.txt`) | — | **44.08%** (132,540 intervened steps, 58,421 flips) | below range ✗ |

Three observations:

1. **Intervention rate is nearly identical between heads (~16%)** — the entropy gate fires the same amount; the difference is entirely in what happens when it fires.
2. **vocab_lowrank's flip rate (42.86% / 44.08%) is remarkably consistent across two completely unrelated LLMs** (0GM-35B: Qwen3.5-MoE, 248k vocab; VL-30B: Qwen3-VL-MoE, 152k vocab, no cross-tokenizer mapping involved at all). This consistency indicates a property of the head itself, not a per-deployment artifact, cross-tokenizer noise, or model-family mismatch.
3. Cross-tokenizer mapping-table diagnostics on 0GM-35B came back clean (`vm_map_miss_rate`/`vm_map_empty_rate` = 0.00%, collision ≤ 0.27%), ruling out mapping quality as the explanation for 35B being worse than 30B.

In plain terms: when vocab_lowrank intervenes, its reward differential across the candidate set is too flat/weak to actually change the model's top-1 choice roughly 57% of the time — the intervention is often close to a no-op. The scalar head only fails to change the choice 20–34% of the time. On 30B, a mostly-inert intervention still nudges the average positively (+4.55%, about half of scalar head's +8.73%). On 35B, the same weak signal isn't enough and nets out negative (-2.28%).

### Mechanistic explanation (from `src/value_model/model.py` / `train.py`)

- **Only 1 of 151,936 rows of `score_B` gets gradient per training position** — the row corresponding to whatever token was actually generated (`W_B[input_ids[t+1]]`). Every other candidate token the LLM could have generated at that position — exactly the tokens SIA needs to rank at inference — gets zero direct gradient from that step.
- **The "TD" loss actually implemented is not FaRMA's Bellman-max loss.** The paper's TD term is `0.5·(V(prefix) − max_{y'} V(prefix+y'))²` (max over the whole vocab, implicitly suppressing every non-chosen token). What's implemented is `MSE(V(t), R_win_constant)` on the winning sequence only — a much weaker "regress toward a scalar reward" signal with no explicit comparison between the true token and alternatives.
- **BT loss operates at the whole-trajectory level** (average V over the sequence): "this whole response is better than that whole response," not "at this exact decoding step, token A beats token B" — a mismatch with what SIA does at inference (rank ~10 specific candidates at one position).
- **`head_rank=64`** is a 40× compression of the 2560-dim hidden state into a single subspace that must carry reward-relevant direction for the entire 151,936-token vocabulary — plausible additional ceiling on discriminability, though (see Part 2) less load-bearing than the objective mismatch above.

None of this is a coding bug — `doc/vocab-lowrank-loss-code-analysis-20260707.md` already flags it as a deliberate simplification (materializing the true full-vocab max is a ~5GB tensor, and unstable without a target network). The net effect: the head is trained to be directionally correct on average, not sharply discriminative among specific candidates — exactly the resolution the flip-rate numbers show is missing.

---

## Part 2 — Literature Review of the Three Candidate Fixes

Three fixes were proposed following the Part 1 analysis:

1. Distill the (working) scalar head's per-candidate scores into vocab_lowrank.
2. Bounded/"Top-K" TD loss — approximate the true Bellman max over a manageable candidate set instead of the full 151,936-way max.
3. Increase `head_rank` (e.g. 64 → 128/256).

### Fix 1 — Distillation from the scalar head: **strongest literature support**

Four independent, structurally similar precedents, all pointing the same direction — dense, per-position supervision from an existing (more expensive/accurate) signal beats sparse trajectory-level preference training:

- **AlignDistil** ([arXiv:2503.02832](https://arxiv.org/abs/2503.02832), 2025): trains a student against a token-level distributional teacher signal (synthesized from a DPO model and a reverse-DPO model) via per-token KL divergence, instead of trajectory-level scalar reward. Reports **>2× faster convergence** vs token-level scalar-reward methods, and reaches **12.93%** on AlpacaEval 2.0 vs ~8–9% for the scalar-token-level baseline.
- **Teacher Value-based Knowledge Distillation (TVKD)**: uses a teacher's value function directly as the reward proxy for distillation — structurally close to "use the scalar head's per-candidate score as the vocab_lowrank training target."
- **RAD** ([arXiv:2310.09520](https://arxiv.org/abs/2310.09520), Deng & Raffel 2023): the original reward-augmented-decoding reward model is trained with dense, per-position supervision (not sparse trajectory preference pairs) — supporting the general claim that dense supervision beats sparse trajectory-level training for this task shape.
- **Math-Shepherd** ([arXiv:2312.08935](https://arxiv.org/abs/2312.08935)): automatically generates dense step-level labels via Monte Carlo rollout (an expensive signal) to train a cheap process reward model — the same "use an expensive signal as an automatic labeler for a cheap approximator" pattern, validated in a different domain (math reasoning), with 86% human-verified label accuracy.

These four works independently converge on: **dense, per-token/per-candidate supervision from an existing signal outperforms sparse trajectory-level preference training for token/candidate-level reward heads.** This is exactly Fix 1's mechanism, and this project has a practical advantage over all four cited works: no new teacher needs to be trained — the scalar head is already deployed and validated on both target LLMs, and its own per-candidate scores at intervention time are free supervision to log.

### Fix 2 — Bounded/"Top-K" TD loss: **moderate support, direction validated, exact trick unstudied**

- **ILQL** ([arXiv:2206.11871](https://arxiv.org/abs/2206.11871), Snell et al., ICLR 2023): uses "support-constrained Bellman backups" plus an explicit conservatism loss term rather than a naive max over the full action space — the same design philosophy as truncating the full-vocab max to top-K candidates (avoid unreliable extrapolation over out-of-support candidates).
- **From r to Q\*** ([arXiv:2404.12358](https://arxiv.org/abs/2404.12358), Rafailov et al. 2024): proves DPO is equivalent to an inverse Q-learning algorithm satisfying the Bellman equation in the token-level MDP — theoretical grounding for learning a token-level implicit value/Q function from preference data at all.
- **TDPO** ([arXiv:2404.11999](https://arxiv.org/abs/2404.11999), Zeng et al., ICML 2024): pushes the training objective from trajectory-level to token-level (per-token forward-KL constraints), and outperforms both DPO and PPO on controlled sentiment generation and single-turn dialogue — direct validation that pushing the objective down to token level (in general) is effective and reproducible.

None of these three papers directly ablates "truncate the Bellman max to top-K candidates" with reported numbers for different K — the support is at the level of design philosophy (bounded/support-constrained backups are theoretically sound and empirically validated in adjacent settings), not a directly transferable hyperparameter recipe. K and the exact truncation scheme would need to be tuned empirically for this codebase.

### Fix 3 — Increase `head_rank`: **weakest support — evidence argues against this being the primary fix**

- **Breaking the Softmax Bottleneck** (Yang, Dai & Cohen, 2017): proves a fundamental capacity ceiling for any `hidden_dim → vocab_size` low-rank linear map used to model a conditional distribution — mathematically the same structure as `score_A`/`score_B`. Follow-ups (Mixtape, NeurIPS 2019; Sigsoftmax) offer higher-rank/alternative-nonlinearity fixes. This *sounds* like direct support for Fix 3.
- However, the most directly relevant empirical precedent argues the opposite: **RAD-Q** ([arXiv:2407.04615](https://arxiv.org/abs/2407.04615), Troshin, Niculae & Fokkens, TMLR 2024/2025) studies *exactly this task shape* — a low-rank factorized head predicting per-candidate reward scores in a single forward pass, the same efficiency motivation as vocab_lowrank/FaRMA — and finds that the **low-rank variant performs on par with the more flexible (higher-rank) RAD** on detoxification and sentiment-control tasks, while needing only one RM call per token. This suggests that for *reward prediction* specifically (a smoothed scalar function of context+candidate), the softmax-bottleneck capacity ceiling that matters for full generative *distribution* modeling is much less binding — reward is a simpler function than the full conditional word distribution, and low rank appears sufficient when paired with the right training signal.
- This is consistent with this project's own data: **vocab_lowrank's top1_flip rate is nearly identical (42.86% vs 44.08%) across two architecturally unrelated LLMs** with very different candidate-token distributions. If rank capacity relative to a specific model's candidate complexity were the bottleneck, more variance between the two deployments would be expected. The consistency instead points to the shared training method (BT + weak TD, sparse trajectory-level, single-row gradient per position) as the limiting factor, not per-model capacity exhaustion.

**Conclusion: Fix 3 is unlikely to be the primary fix.** It remains a near-zero-cost ablation worth running in parallel (a day's work), but should not be prioritized or relied upon as the main lever.

---

## Recommendation

Given limited remaining time, resource allocation should not be split evenly across all three:

1. **Prioritize Fix 1 (distillation from the scalar head).** It has the strongest, most directly quantified literature support (AlignDistil: >2× faster convergence, 8–9%→12.93% on AlpacaEval 2.0) and the lowest implementation cost in this specific project — no new labeled data is needed; log the scalar head's own per-candidate scores at intervention time (already running in production on both target LLMs) and train `score_A`/`score_B` (backbone frozen) to regress/rank against them.
2. **Layer Fix 2 on top of Fix 1, not as a separate track.** A bounded/Top-K TD regularization term can be added to the distillation loss without conflict — it is complementary, not competing.
3. **Run Fix 3 in parallel at near-zero cost** (retrain the frozen-backbone head at `head_rank=128`/`256`, same everything else, check val R² and top1_flip before committing GPU-hours to a full AlpacaEval run) but do not treat it as the primary bet — the RAD-Q precedent and this project's own cross-model flip-rate consistency both argue rank is not the dominant bottleneck.
4. **Use `top1_flip` rate as the fast iteration signal**, not full 805Q AlpacaEval generation+scoring. It's computable from a short smoke-test run (minutes, not hours) and is the metric directly diagnostic of the problem identified in Part 1. Reserve full AlpacaEval runs for confirming a change once flip rate looks healthy (target: 50–80%, matching the scalar head's empirically observed range).

---

## Sources

- [AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation](https://arxiv.org/abs/2503.02832)
- [On the Low-Rank Parametrization of Reward Models for Controlled Language Generation (RAD-Q)](https://arxiv.org/abs/2407.04615)
- [Reward-Augmented Decoding: Efficient Controlled Text Generation With a Unidirectional Reward Model (RAD)](https://arxiv.org/abs/2310.09520)
- [Offline RL for Natural Language Generation with Implicit Language Q Learning (ILQL)](https://arxiv.org/abs/2206.11871)
- [From r to Q*: Your Language Model is Secretly a Q-Function](https://arxiv.org/abs/2404.12358)
- [Token-level Direct Preference Optimization (TDPO)](https://arxiv.org/abs/2404.11999)
- [Breaking the Softmax Bottleneck: A High-Rank RNN Language Model](https://www.semanticscholar.org/paper/Breaking-the-Softmax-Bottleneck:-A-High-Rank-RNN-Yang-Dai/ef9ddbc35676ce8ffc2a8067044473727839dbac)
- [Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://arxiv.org/abs/2312.08935)
- [GenARM: Reward Guided Generation with Autoregressive Reward Model for Test-time Alignment](https://arxiv.org/abs/2410.08193)

## Related repo docs

- `doc/vocab-lowrank-loss-code-analysis-20260707.md` — exact loss formulas and gradient-coverage analysis this Part 1 builds on.
- `doc/vocab-lowrank-cross-tokenizer-mapping-20260703.md` — cross-tokenizer mapping design (ruled out as the 0GM-35B bottleneck).
- `doc/vocab-reward-head-training-loss-survey-20260706.md` — 94-paper survey establishing BT+TD (not cross-entropy) as the standard training objective for this head type.
- `exp/alpaca-vl30b-sia-backbone-frozen-20260707/report.md` — the VL-30B experiment series that isolated backbone quality from head architecture.
- `doc/alpaca-eval-vl30b-3way-805q-20260710.md` — the 805Q new-head-vs-old-head-vs-noSIA comparison that reproduced the ~half-effect gap at full sample size.
