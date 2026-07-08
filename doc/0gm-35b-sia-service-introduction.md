# 0GM-1.0-35B-A3B-0427-SIA — Inference-Time Aligned Language Model Service

## What It Is

0GM-1.0-35B-A3B-0427-SIA is a production serving configuration of the 0GM-1.0-35B-A3B-0427 model, augmented with **SIA (Sparse Inference-time Alignment)**. SIA is a real-time token-level guidance system that steers the model's generation toward higher-quality responses at decode time — with no fine-tuning or retraining of the base model required.

At each generation step, a compact Value Model scores the top candidate tokens against learned human preference signals and biases token selection toward higher-reward choices. Critically, the intervention is **sparse**: it only activates at high-entropy "critical junction" tokens (typically 10–30% of decode steps), leaving the model's confident decisions untouched. This preserves the base model's fluency and reasoning depth while systematically improving output quality.

## Base Model Highlights

0GM-1.0-35B-A3B is a **35B Mixture-of-Experts model with only 3B active parameters per token**, delivering the per-token compute cost of a 3B model while retaining the knowledge capacity of a 35B-scale architecture. The model natively supports extended chain-of-thought reasoning (thinking mode), which is the recommended configuration and the basis for all performance figures below.

## Value Model Training

The Value Model that guides SIA is trained on ~157,000 instruction-response pairs drawn from two public datasets, covering the three core dimensions of human alignment:

| Dataset | Samples | Alignment Focus |
|---|---|---|
| **WildGuardMix** | ~38,000 | **Harmlessness** — safety-centric prompts and responses |
| **UltraFeedback (ShareGPT + UltraChat)** | ~119,000 | **Helpfulness & Honesty** — diverse open-ended instructions with multi-LLM responses |

Each response in the training data is scored by **Skywork-Reward-V2-Qwen3-8B**, a state-of-the-art trajectory-level reward model, providing the supervision signal the Value Model learns from. The result is a Value Model that has internalized human preferences across safety, helpfulness, and honesty — and applies this judgment live at every generation step to steer the main model's outputs.

## Key Differentiators vs. Market Alternatives

| | Standard instruction-tuned LLM | RLHF / DPO fine-tuned LLM | **0GM-35B-SIA** |
|---|---|---|---|
| Alignment method | Prompt engineering | Offline fine-tuning | **Live inference-time steering** |
| Updatable without retraining | — | ❌ | ✅ |
| Risk of catastrophic forgetting | — | ✅ present | ❌ none |
| Open-ended task quality gain | baseline | varies | **+15.3% vs unaligned baseline** |
| Output efficiency | normal | normal | **Fewer tokens, higher scores** |

Unlike RLHF or DPO, SIA applies alignment **at inference time**. The base model is never modified, so there is no risk of catastrophic forgetting, no distribution shift, and the alignment strength can be tuned at serving time by adjusting a single weight parameter.

## Output Quality

All figures use **thinking mode** (chain-of-thought enabled), the production configuration.

**AlpacaEval — Open-ended Instruction Following (Skywork reward scoring):**

- **SIA mean reward score: 27.69**
- **noSIA mean reward score: 24.01**
- **Improvement: Δ = +3.68 points (+15.3% relative)**
- SIA wins 64% of head-to-head paired comparisons (122 wins / 64 losses / 5 ties)
- SIA generates **fewer tokens on average** (1,573 vs 1,736) yet consistently scores higher — more concise and more aligned simultaneously

## What Problems It Excels At

- **Open-ended instruction following**: writing, analysis, summarization, explanation, Q&A — any task where response quality is judged by human preference
- **Tasks that benefit from extended thinking**: problems where the model's deliberation is guided toward more accurate and helpful conclusions
- **Quality-critical applications**: scenarios where consistent, high-quality output matters and mediocre responses are unacceptable

## Throughput

- **SIA: 259 tokens/s** output throughput (thinking mode)
- **noSIA baseline: 648 tokens/s** output throughput
- SIA overhead is approximately 2.5× — a deliberate trade-off for meaningfully better output quality on every request
