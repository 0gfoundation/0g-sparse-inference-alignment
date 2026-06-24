# Alignment Methods Overview: Taxonomy and Comparison from the Paper

This document summarizes the taxonomy, evaluation, and quantitative comparison of various alignment methods from the SIA paper (Inference-time Alignment via Sparse Junction Steering).

---

## 1. Training-time Methods

| Method | Mechanism | Limitations |
|------|------|--------|
| SFT (Supervised Fine-tuning) | Direct fine-tuning on high-quality samples | Can only learn pre-selected behaviors; poor flexibility |
| RLHF | Train a Reward Model, then optimize policy with RL | Requires extensive human annotation; extremely high compute cost |
| DPO | Train policy directly on preference pairs, bypassing explicit RM | Still requires an offline training phase |

Paper summary: All three methods rely on "large-scale parameter updates," incurring high compute costs. Once trained, the model is frozen and cannot flexibly adapt to new objectives. SIA, as an inference-time method, bypasses this cost entirely.

**Best suited for**: One-time large-scale alignment where latency is not a concern and sufficient compute and annotation resources are available.

---

## 2. Inference-time Methods — Prompt-based

**Examples**: Constitutional AI, in-context learning

**Mechanism**: Embed alignment objectives directly into the prompt and guide model behavior through language instructions.

**Limitation** (from the paper):

> "effectiveness is limited by the expressiveness of natural language prompts"

The ceiling is limited by how complex an objective natural language can express; difficult to precisely convey complex alignment requirements.

**Best suited for**: Lowest cost; suitable for simple behavioral adjustments (e.g., "please respond politely"), not for fine-grained value alignment.

---

## 3. Inference-time Methods — Search-based

This is the main comparison target in the paper, covering four variants:

| Method | Mechanism | Issues identified in the paper |
|------|------|----------------|
| **Best-of-N (BoN)** | Generate N complete candidates, select the one with highest reward | Must generate complete sequences before scoring; worst efficiency; BoN-8 requires ~4× the compute of SIA |
| **ARGS** | Adjust logits at each decoding step using trajectory reward | "noisy local signals" — intermediate steps are insensitive to reward; high signal noise |
| **Transfer Q\*** | Estimate Q-values via multi-token forward simulation | High compute cost; high variance |
| **CBS (Chunk-based Search)** | Search at the chunk level | More granular than BoN, but still requires 2–4× the compute of SIA |

**SIA's quantitative advantages** (from paper Table 1):

- vs BoN-8: ~**4×** lower compute overhead, with equal or better reward
- vs CBS-8: **2–4×** efficiency improvement
- vs ARGS / Transfer Q\*: Noticeably higher reward in the dense token intervention scenario

**Best suited for**: Offline scenarios with ample compute where only the final result matters; not suitable for latency-sensitive online inference.

---

## 4. Inference-time Methods — Refine-based

**Examples**: Aligner, DIFFPO, SEA

**Mechanism**: Generate an initial response, then iteratively improve it through multiple rounds of feedback.

**Limitation** (from the paper):

> "typically require additional inference rounds, leading to increased latency"

Multi-round generation multiplies latency and results in very low throughput.

**Best suited for**: Scenarios requiring extremely high quality where latency is not a concern (e.g., document generation, code review); not suitable for interactive dialogue.

---

## 5. Inference-time Methods — Token-level Steering

The category SIA belongs to. The paper's core criticism of existing token-level steering methods: they assume **every decoding step contributes equally to alignment** and apply intervention to all tokens uniformly (dense steering).

From the paper:

> "Dense intervention creates an inherent conflict between the original LLM distribution and external reward signals, ultimately degrading overall generation quality."

**SIA's improvement**: Intervene only at high-entropy "junction" critical decision points (sparse), achieving equal or better performance than full dense intervention with just 20%–80% intervention rate, while avoiding degradation of generation quality.

---

## 6. Summary Comparison

| Method Category | Examples | Advantages | Disadvantages | Best Suited For |
|----------|------|------|------|----------|
| Training-time | RLHF / DPO / SFT | Solid effectiveness | High training cost; hard to adjust after fixing | One-time large-scale alignment |
| Prompt-based | Constitutional AI | Zero extra cost | Limited precision; complex objectives hard to express | Simple behavioral adjustments |
| Search-based | BoN / ARGS / CBS | Good effectiveness | Compute multiplied (2–4×) | Offline, compute-rich scenarios |
| Refine-based | Aligner / SEA | Quality can be iteratively improved | Latency multiplied; multi-round inference | Offline high-quality generation |
| Dense token steering | ARGS (full) | Fine-grained intervention | Degrades original generation quality | — |
| **SIA (sparse token steering)** | — | Low overhead; no quality degradation; plug-and-play | Cannot correct low-entropy errors | Online inference; latency-sensitive scenarios |
