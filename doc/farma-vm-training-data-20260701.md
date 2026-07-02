# FaRMA VM Training Data Details (2026-07-01)

Supplementary notes on the SIA paper's training data specification.
See `doc/farma-vm-training-plan-20260701.md` for the full training plan.

---

## 1. Paper References

### Section 4.2.2 Empirical Evaluation — Settings (p.4)

Original:

> "Our training corpus is derived from WildGuardMix (Han et al., 2024) and UltraFeedback
> (Cui et al., 2024), encompassing a diverse range of prompts and responses centered on the
> 3H alignment criteria: Harmlessness, Helpfulness, and Honesty. We employ
> Skywork-Reward-V2-Qwen3-8B (Liu et al., 2025a) as the trajectory-level reward model to
> provide supervision signals during the training of our value model. To ensure parameter
> efficiency and maintain the fundamental capabilities of the base LLM, we utilize LoRA
> (Hu et al., 2022) for the training of V_θ. We train a corresponding value model V_θ of
> the same parameter scale as the backbone LLM. Detailed experimental setups and
> hyperparameters are provided in App. C."

中文：

> 我们的训练语料来自 WildGuardMix（Han et al., 2024）和 UltraFeedback（Cui et al.,
> 2024），涵盖了多样化的提示与回复，围绕 3H 对齐标准展开：无害性（Harmlessness）、
> 有用性（Helpfulness）和诚实性（Honesty）。我们采用 Skywork-Reward-V2-Qwen3-8B
> （Liu et al., 2025a）作为轨迹级奖励模型，在价值模型训练过程中提供监督信号。为确保
> 参数效率并保留基础 LLM 的核心能力，我们使用 LoRA（Hu et al., 2022）对 V_θ 进行
> 微调。我们训练一个与主干 LLM 参数规模相同的价值模型 V_θ。详细的实验设置和超参数
> 见附录 C。

---

### Appendix C.1 Value Model Training — Datasets (p.13)

Original:

> "Our training pipeline incorporates two primary datasets: WildGuardMix (Han et al., 2024)
> and UltraFeedback (Cui et al., 2024). WildGuardMix is a comprehensive safety-centric
> dataset designed to evaluate harmlessness; we utilize its complete set of 37,976
> instruction-response pairs. UltraFeedback, conversely, provides diverse feedback for
> evaluating helpfulness, truthfulness, and honesty. We specifically curate the ShareGPT
> and UltraChat subsets from UltraFeedback, containing 19,949 and 9,929 instructions,
> respectively. Each instruction in UltraFeedback is accompanied by four distinct responses
> generated from a pool of 17 diverse LLMs. In total, the combined training corpus comprises
> approximately 157k instruction-response pairs, providing a balanced coverage of both safety
> and utility dimensions."

中文：

> 我们的训练流程使用两个主要数据集：WildGuardMix（Han et al., 2024）和 UltraFeedback
> （Cui et al., 2024）。WildGuardMix 是一个以安全为核心的综合数据集，专为评估无害性
> 而设计；我们使用其完整的 37,976 条指令-回复对。UltraFeedback 则提供多样化的反馈，
> 用于评估有用性、真实性和诚实性。我们从 UltraFeedback 中专门筛选了 ShareGPT 和
> UltraChat 两个子集，分别包含 19,949 和 9,929 条指令。**UltraFeedback 中每条指令均
> 附有四个不同的回复，这些回复由 17 个不同的 LLM 生成。** 综合来看，完整训练语料共
> 包含约 157k 条指令-回复对，在安全性和通用性两个维度上提供了均衡的覆盖。

---

## 2. How the 157k Figure Is Derived

The 157k figure comes from expanding each UltraFeedback instruction's **4 responses**:

| Dataset | Instructions | Responses per instruction | Pairs |
|---------|-------------|--------------------------|-------|
| WildGuardMix | 37,976 | 1 | 37,976 |
| UltraFeedback (ShareGPT) | 19,949 | 4 | 79,796 |
| UltraFeedback (UltraChat) | 9,929 | 4 | 39,716 |
| **Total** | | | **≈ 157,488 ≈ 157k** |

---

## 3. preprocess.py Already Handles All 4 Responses Correctly

After reading the raw official repo at `/workspace/SIA/raw_SIA/SIA/src/value_model/preprocess.py`,
the script already iterates all completions per instruction (line 85):

```python
for completion in completions:
    response = completion.get("response")
    ...
    results.append(result_dict)
```

No modification is needed. Running the script as-is against UltraFeedback naturally
produces the ~157k sample count described in the paper.

---

## 4. Code vs. Paper Consistency Audit (2026-07-01)

Full audit of `/workspace/SIA/raw_SIA/SIA` against the SIA paper training description.

### 4.1 Complete Training Pipeline (code is internally self-consistent)

The pipeline requires three scripts in sequence — this is NOT documented in the README but
is the only way to produce valid training data:

```
Step 1: src/value_model/preprocess.py
        Input:  raw UltraFeedback / WildGuardMix data
        Output: vm_data.json
                fields: id, prompt, result, rm_prompt,
                        rm_prompt_tokens_ids, generated_tokens_ids,
                        rm_guided_tokens_mask, intervene_ratio
                NOTE: NO "reward" field yet

Step 2: src/measure_reward.py
        Input:  vm_data.json (from step 1)
        Output: vm_data_with_reward.json
                Adds "reward" field via teacher RM forward pass

Step 3: src/value_model/train.py
        Input:  vm_data_with_reward.json (from step 2)
        Loss:   MSELoss( avg(token_rewards at masked positions), global_reward )
```

### 4.2 Discrepancy 1 — Teacher RM Model (IMPORTANT, worth asking authors)

| Source | Teacher RM |
|--------|-----------|
| `scripts/measure_reward.sh` (code) | `Skywork-Reward-V2-**Llama-3.1-8B**` |
| SIA paper §4.2.2 | `Skywork-Reward-V2-**Qwen3-8B**` |
| FaRMA paper Appendix C.1 | `Skywork-Reward-V2-**Qwen3-8B**` |

The shell script references a Llama-3.1 backbone; the published paper specifies Qwen3.
These are different model families with different reward score distributions, which
directly affects VM training quality. The shell script is likely stale (written before
the paper finalized on Qwen3-8B).

**Action**: Ask authors which teacher RM was actually used in the published experiments.

### 4.3 Discrepancy 2 — UltraFeedback Preprocessing Not Shown

`scripts/preprocess_vm_data.sh` only demonstrates WildGuardMix preprocessing.
There is no example command for the UltraFeedback (ShareGPT + UltraChat) subsets,
even though `preprocess.py` fully supports the `ultrafeedback` mode.

The full data production commands are not provided in the repo.

**Action**: Ask authors for the exact preprocessing commands used for UltraFeedback.

### 4.4 Training Objective — Consistent with Paper

The MSE loss in `train.py` (lines 255–261) exactly matches the SIA paper description:
- Predict trajectory-level reward at every generated token position
- Loss target: global `reward` from teacher RM
- All generated tokens get mask=1 (uniform weighting)

No discrepancy here.

### 4.5 LoRA Hyperparameters — Consistent with Paper

`scripts/train_vm.sh` defaults (r=16, α=32, dropout=0.1, lr=1e-4, epochs=3, max_len=1024)
match Appendix C.1 exactly.

### 4.6 Summary: Neither Question Needs to Be Asked

**Question 1 (teacher RM):** The paper is correct; the shell script is stale.

- Both the SIA paper §4.2.2 and FaRMA paper Appendix C.1 independently specify
  `Skywork-Reward-V2-Qwen3-8B` — two peer-reviewed publications are unlikely to both
  be wrong.
- `preprocess.py` tokenizer detection (line 24) checks `"Qwen3" in tokenizer_path` first,
  confirming the codebase was designed around Qwen3.
- Qwen3 was released after Llama-3.1; the shell script was simply written in an earlier
  development phase and never updated.
- **Decision: use `Skywork-Reward-V2-Qwen3-8B`, follow the paper.**

**Question 2 (UltraFeedback preprocessing):** Fully inferable from the code.

- `preprocess.py`'s `ultrafeedback` mode handles any JSONL with `instruction` + `completions`
  fields — exactly matching the public `openbmb/UltraFeedback` dataset schema.
- ShareGPT and UltraChat are simply two source-filtered subsets of that dataset.
  Filter by `source` field → two JSONL files → run `preprocess.py` twice.
- Inferred commands:

```bash
python3 src/value_model/preprocess.py ultrafeedback \
    --dataset_path /path/to/ultrafeedback-sharegpt.jsonl \
    --tokenizer_path /path/to/Skywork-Reward-V2-Qwen3-8B \
    --output_path assets/vm_data/sharegpt.json

python3 src/value_model/preprocess.py ultrafeedback \
    --dataset_path /path/to/ultrafeedback-ultrachat.jsonl \
    --tokenizer_path /path/to/Skywork-Reward-V2-Qwen3-8B \
    --output_path assets/vm_data/ultrachat.json
```

- **Decision: self-solvable, no need to ask.**

**Conclusion: no need to contact the authors at all for the training pipeline.**

Note: asking authors for their trained VM checkpoint is also not useful — their checkpoint
uses `Linear(hidden_size, 1)` (scalar head), while the FaRMA VM we need uses
`Linear(hidden_size, vocab_size)`. The architectures are incompatible; the checkpoint
cannot be reused.

---

## 5. Appendix C.1 Training Hyperparameters (original text, p.13)

> "The value models are trained using LoRA with the following hyperparameters: a rank
> r = 16, α = 32, and a dropout rate of 0.1. We employ a learning rate of 1 × 10⁻⁴ and
> a batch size of 16. The maximum sequence length for responses is truncated at 1024 tokens.
> All models are trained for 3 epochs on a single NVIDIA H200 GPU, with the training process
> for each model scale typically concluding within a few hours."

Note: batch size = 16 here is the **effective** batch size (local batch 8 × gradient
accumulation 2), consistent with `scripts/train_vm.sh` defaults.
