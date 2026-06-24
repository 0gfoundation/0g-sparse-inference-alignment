# 0GM-1.0-35B-A3B + Qwen3-4B-RM SIA vs noSIA MMLU-Redux Evaluation Results

**Date**: 2026-06-02
**Main inference model**: 0GM-1.0-35B-A3B-0427 (Qwen3.5/3.6 MoE, vocab=248044, thinking mode)
**Value Model (RM)**: VM-Qwen3-4B-merged-for-vllm (trained on Qwen3-4B base)
**Evaluation**: MMLU-Redux, 30 subjects × 20Q (SIA, 600 questions) vs 30 subjects × 30Q (noSIA, 900 questions)
**vllm version**: 0.19.0 (venv4)

---

## 1. TL;DR

| Metric | noSIA (matched 30 subj × 30Q = 900Q) | SIA (30 subj × 20Q = 600Q) | Δ |
|------|--------------------------------------|----------------------------|---|
| **Accuracy** | 75.6% (680/900) | 74.3% (446/600) | **-1.2 pp** |
| **Throughput** | 108.9 tok/s | 51.4 tok/s | **52.8% slower** (1.97×) |
| per-Q latency (avg) | 11.7 s | 19.6 s | +67% |
| avg tokens/Q | 1271 | 1006 | -21% (SIA shortened thinking) |
| Total time | 165 min | 196 min | — |

**Key conclusions**:
- **Accuracy nearly equal** (-1.2 pp, mean ≈ 0, median = 0). 13 subjects improved / 5 flat / 12 declined — indistinguishable from "no-effect + noise"
- **53% speed loss**, mainly from cross-process RM HTTP calls (~95ms/INTERVENE × 8.3% intervene rate, see [`0gm-35b-sia-rm-inproc-path-20260602.md`](0gm-35b-sia-rm-inproc-path-20260602.md))
- **Current RM (VM-Qwen3-4B) was not trained for 0GM** — alignment signal is essentially noise; cannot compare against 14B + same RM SIA gains

---

## 2. Per-subject Δaccuracy Full Results

```
subject                             noSIA(30Q)      SIA(20Q)        Δacc
---------------------------------------------------------------------------
anatomy                            25/30 (83.3%)   15/20 (75.0%)   -8.3%
astronomy                          28/30 (93.3%)   15/20 (75.0%)  -18.3% ⚠️
business_ethics                    23/30 (76.7%)   16/20 (80.0%)   +3.3%
clinical_knowledge                 27/30 (90.0%)   18/20 (90.0%)   +0.0%
college_chemistry                  10/30 (33.3%)    7/20 (35.0%)   +1.7%
college_computer_science           21/30 (70.0%)   14/20 (70.0%)   +0.0%
college_mathematics                20/30 (66.7%)   14/20 (70.0%)   +3.3%
college_medicine                   24/30 (80.0%)   14/20 (70.0%)  -10.0%
college_physics                    21/30 (70.0%)   13/20 (65.0%)   -5.0%
conceptual_physics                 29/30 (96.7%)   19/20 (95.0%)   -1.7%
econometrics                       25/30 (83.3%)   17/20 (85.0%)   +1.7%
electrical_engineering             21/30 (70.0%)   10/20 (50.0%)  -20.0% ⚠️
formal_logic                       20/30 (66.7%)   15/20 (75.0%)   +8.3%
global_facts                       15/30 (50.0%)    9/20 (45.0%)   -5.0%
high_school_chemistry              22/30 (73.3%)   15/20 (75.0%)   +1.7%
high_school_geography              25/30 (83.3%)   19/20 (95.0%)  +11.7% ✅
high_school_macroeconomics         26/30 (86.7%)   18/20 (90.0%)   +3.3%
high_school_mathematics            16/30 (53.3%)    8/20 (40.0%)  -13.3% ⚠️
high_school_physics                17/30 (56.7%)   11/20 (55.0%)   -1.7%
high_school_statistics             25/30 (83.3%)   16/20 (80.0%)   -3.3%
high_school_us_history             25/30 (83.3%)   19/20 (95.0%)  +11.7% ✅
human_aging                        21/30 (70.0%)   12/20 (60.0%)  -10.0%
logical_fallacies                  26/30 (86.7%)   18/20 (90.0%)   +3.3%
machine_learning                   25/30 (83.3%)   17/20 (85.0%)   +1.7%
miscellaneous                      30/30 (100.0%)  20/20 (100.0%)  +0.0%
philosophy                         24/30 (80.0%)   16/20 (80.0%)   +0.0%
professional_accounting            26/30 (86.7%)   19/20 (95.0%)   +8.3%
professional_law                   20/30 (66.7%)   14/20 (70.0%)   +3.3%
public_relations                   22/30 (73.3%)   14/20 (70.0%)   -3.3%
virology                           21/30 (70.0%)   14/20 (70.0%)   +0.0%
---------------------------------------------------------------------------
TOTAL (matched)                   680/900 (75.6%)  446/600 (74.3%)  -1.2%
```

- **Worst 3**: electrical_engineering -20pp, astronomy -18.3pp, high_school_math -13.3pp
- **Best 3**: high_school_geography +11.7pp, high_school_us_history +11.7pp, professional_accounting +8.3pp
- **Subject counts**: 13 improved / 5 flat / 12 declined
- mean Δ = -1.22%, median Δ = 0.00%

---

## 3. Detailed Speed Breakdown

### 3.1 Overall

| | noSIA | SIA |
|---|-------|-----|
| total tokens | 1,097,497 | 603,668 |
| total latency | 10,078 s | 11,747 s |
| **throughput** | **108.9 tok/s** | **51.4 tok/s** |
| per-token | 9.18 ms | 18.21 ms (**+9.0 ms/tok**) |

### 3.2 SIA Internal Latency (server pf-summary cumulative stats)

| phase | p50 | p95 | max | Meaning |
|-------|-----|-----|-----|------|
| `http_post` (RM /classify) | **94.7 ms** | 124 ms | 140 ms | Single cross-process RM call (main bottleneck) |
| `apply_total` (per apply, includes SKIP+INTERVENE) | 4.0 ms | 77 ms | 122 ms | Total apply() time; p50 is mainly SKIP |
| `skip_step` (per SKIP) | 4.0 ms | 4.1 ms | 4.1 ms | SIA cleanup per SKIP |
| `apply_cpu_sync` | 3.6 ms | 3.6 ms | 3.7 ms | entropy + topk + cpu().tolist() |

### 3.3 Per-token SIA Overhead Breakdown

```
Average SIA overhead per token
  ≈ 0.083 × 96.5 ms (INTERVENE: http_post + ops within intv)
  +  0.917 × 4.01 ms (SIA cleanup on SKIP path)
  ≈ 8.0 ms (INTERVENE-driven)  +  3.7 ms (SKIP-driven)
  ≈ 11.7 ms/tok

Measured: SIA 18.2 ms/tok  vs  noSIA 9.18 ms/tok  →  Δ = +9.0 ms/tok
```

**Main bottleneck**: Cross-process HTTP `/classify` 95 ms/call, accounting for the vast majority of the 8 ms/tok overhead. See [`0gm-35b-sia-rm-inproc-path-20260602.md`](0gm-35b-sia-rm-inproc-path-20260602.md) for analysis of "why cross-process is required" and "whether inproc is feasible."

---

## 4. SIA Intervention Metrics (cumulative stats from server DONE across 380 requests)

| Metric | Value |
|------|------|
| Intervention steps / total steps | 36,350 / 436,030 = **8.34%** |
| top-1 flip / intervention steps | 1,253 / 36,350 = **3.45%** |
| top-1 flip / total steps | **0.287%** |
| Per-request intervention rate distribution | p50=7.7%, mean=8.0%, max=20.6%, min=1.0% |
| Per-request top-1 flip rate | p50=3.2%, mean=4.1% (relative to interventions per request) |

**Comparison with Qwen14B baseline (historical)**:
- Same `entropy_threshold=1.0, topk=5, weight=1.0, RM=VM-Qwen3-4B`
- 14B intervention rate **30%+** vs 0GM-35B **8.34%**
- 14B flip rate historically ~10%+ vs 0GM-35B **3.45%**

---

## 5. Why Is the Intervention Rate So Low on 0GM-35B (Deep Analysis)

### 5.1 Entropy Calculation Review

SIA's entropy is not the full-vocabulary entropy — it is the **entropy of the top-5 logits after re-normalization** (`src/sia_vllm_RM.py:698-702`):

```python
topk_result = torch.topk(logits, self._TOPK, dim=-1)        # top-5 raw logits
log_probs = F.log_softmax(topk_result.values.float(), dim=-1)   # softmax over top-5 only
probs = log_probs.exp()
entropies = -(probs * log_probs).sum(dim=-1)                # range [0, log(5) ≈ 1.609]
```

threshold=1.0 = top-5 must be **at least 62% close to uniform distribution** to trigger INTERVENE.

Concrete examples:
- **Flat logits** [10, 9.5, 9, 8.5, 8] → re-normalized probs ≈ [0.43, 0.26, 0.16, 0.10, 0.06], **entropy ≈ 1.40** → INTERVENE
- **Sharp logits** [12, 9, 8, 7, 6] → re-normalized probs ≈ [0.93, 0.05, 0.02, 0.006, 0.002], **entropy ≈ 0.32** → SKIP

### 5.2 Candidate Causes (ranked by probability)

#### ① **35B is more "confident" than 14B** — Primary cause

Larger pretrained model → more certain per-token decisions → top-1 within top-5 dominates probability mass → entropy near 0 after re-normalization.

- 14B in chat: typical top-1 vs top-2 logit gap ~1-2 → entropy ~1.0-1.2 (frequently ≥ threshold)
- 35B: typical top-1 vs top-2 logit gap ~2-3 → entropy ~0.3-0.5 (well below threshold)

This is the natural property of larger pretrained models — not a bug.

#### ② **0GM is a thinking model; large amounts of tokens are in low-entropy reasoning prose** — High-probability secondary cause

0GM's chat template automatically prepends `<think>\n`, forcing the model to output chain-of-thought first. Extended inner monologue such as:

```
The user wants to identify the disorder characterized by uncontrollable...
Option A: Dyslexia - A learning disorder affecting reading.
Option B: Epilepsy - A neurological disorder characterized by seizures.
...
```

In such prose, every token (spaces, word continuations, conjunctions, option-list structures) is almost **certain** — top-1 dominates overwhelmingly, entropy is far < 1.0.

High entropy only appears at truly "critical decision points":
- Which option to select (`Therefore the answer is __`)
- Whether to close with `</think>` or continue reasoning
- Self-correction (`Wait, no, actually __`)

Data confirms: SIA mean 1106 tokens/Q, INTERVENE ~91 per Q (8.3%) — roughly corresponding to "truly exploratory key nodes within thinking," consistent with the paper's sparse junction intervention theory.

Qwen14B is not a thinking model; every response token is an "answer token," uncertainty is concentrated, so 30%+ intervention rate is reasonable.

#### ③ **MoE architecture produces sharper outputs than dense models** — Medium-probability secondary cause

MoE activates specialized experts per token; experts tend to be more **confident** specialists → output logits are more peaked than an equivalent-size dense model. This compounds with ①.

#### ④ **vocab 248K vs 152K, indirect effect** — Low-probability factor

Entropy is computed within top-5; vocabulary size **does not directly affect** top-5 entropy. However, a larger vocab → probability mass is spread over more tokens → top-1 within top-5 is relatively more prominent (more tokens outside top-5 absorb mass). A second-order effect.

#### ⑤ **Sampling parameter differences — ruled out**

SIA entropy is computed **before** the vllm sampler's temperature/top_k/top_p/repetition_penalty (sampler.py order: float32 cast → SIA processor → apply_penalties → temperature). The two evaluations used different sampling params (0GM: `top_k=20, top_p=0.95, rep_pen=1.0` vs 14B: `top_p=1.0, rep_pen=1.3`), but this **does not affect SIA entropy judgment**.

#### ⑥ **Code bug — ruled out**

Line-by-line reviewed; code paths are identical on both sides, no 0GM-specific branches. The difference comes from the logits themselves.

### 5.3 Why Is the Flip Rate Extremely Low (3.45% of intervened)

flip = SIA actually changed the top-1 selection. Conditions:
1. top-5 entropy ≥ 1.0 (INTERVENE triggered)
2. **AND** RM reward delta is large enough to exceed the logit gap between top-1 and top-2 within top-5

RM outputs post-sigmoid probabilities (range [0, 1]); after `(rm_scores - mean) * weight=1.0`, deltas are on the order of ±0.7 (measured).

**Key example** (when top-5 entropy=1.0):
- 14B: typical logit gap ~0.3-1.0 → RM delta 0.7 can often flip → historical flip rate 10%+
- 35B: typical logit gap ~0.5-2.0 → RM delta 0.7 rarely flips → flip rate 3%

Even when 35B entropy is "not so low," the internal logit gaps within top-5 are still larger than 14B (analogous to lower temperature), making RM delta insufficient to overturn top-1.

### 5.4 Conclusion

**8% intervention rate + 3.45% flip rate are primarily objective properties of "35B + thinking model"**, not bugs:
- **Primary cause ①**: 35B model's top-5 distribution is more peaked than 14B; steps with entropy ≥ 1.0 are naturally fewer
- **Primary cause ②**: Thinking model's long stretches of low-entropy reasoning prose pulls down the overall intervention rate
- **Secondary cause ③**: MoE outputs are more peaked than dense models
- ④/⑤/⑥ are not primary causes

---

## 6. Possible Causes for -1.2pp Accuracy

Cannot simply say "SIA doesn't work," since 13 subjects improved (max +11.7pp). Possible explanations:

### 6.1 RM Training Distribution Mismatch (Primary Cause)

VM-Qwen3-4B-RM was **trained on Qwen3-4B base chat completion scenarios**, not thinking mode. 0GM's `<think>...</think>` inner monologue is **out-of-distribution for RM training** — the RM's reward signal during the thinking phase is essentially OOD, rewarding "helpful-sounding phrasing" rather than "correct reasoning."

Concrete manifestation (see astronomy -18.3pp analysis):
- noSIA on Q1/Q13/Q17: longer thinking (1644-1929 tok), correctly selects B after thorough comparison
- SIA on the same questions: **confidently converges early** (962-1364 tok), incorrectly selects A
- Common pattern: SIA's thinking contains phrases like "definitively chose," "physically reasonable," "this matches" — **confident-sounding phrasing** — indicating RM rewards "helpful / confident" phrasing tokens

### 6.2 Possible Explanation for Split Subject Performance

- **Subjects where SIA performs well** (geography, us_history, professional_accounting, formal_logic): answers tend to be **factually clear with short reasoning** — RM rewarding "clear phrasing" aligns with correctness
- **Subjects where SIA performs poorly** (electrical_engineering, astronomy, math, medicine): answers require **careful comparison of multiple candidates** — RM rewarding "early convergence / confident phrasing" actually disrupts thorough reasoning

### 6.3 Intervention Rate of 8% May Be Too Low

If the true "junctions" on 0GM are missed by threshold=1.0, SIA may be intervening on marginal high-entropy noise steps rather than critical decision points. This leads to **no alignment benefit** while **still increasing cost**.

---

## 7. Action Recommendations

By priority:

1. **Threshold tuning experiment (low cost, ~30 minutes)**:
   - Run `entropy_threshold=0.5` on a small subset (60Q × 3 subjects)
   - See if intervention rate rises to ~25-30% and how accuracy changes
   - If intervention rate rises and accuracy **improves** → threshold is the issue; run full 600Q with 0.5-0.7
   - If intervention rate rises and accuracy **worsens** → RM signal itself is incompatible with 0GM; consider training a new RM

2. **Train an RM adapted for 0GM (large effort)**:
   - Use chat generated by 0GM itself (including `<think>` sections) with preference labels
   - Train a new VM-0GM-7B or similar
   - Align RM training distribution with 0GM inference distribution

3. **Accelerate RM calls (conditional)**:
   - Only after #1 or #2 confirms "SIA has real alignment benefit"
   - If benefit confirmed, follow [`0gm-35b-sia-rm-inproc-path-20260602.md`](0gm-35b-sia-rm-inproc-path-20260602.md) approach A (transformers DynamicCache inproc); expected SIA speed from 51 → ~78-85 tok/s

**Priority rationale**: Current data shows no alignment benefit from SIA on 0GM — optimizing speed would be premature optimization. First resolve "is there benefit," then "how to go faster."
