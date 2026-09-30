# VM Training Data Preprocessing (2026-07-02)

This document records all commands executed to preprocess and score the Value Model training data,
following the official SIA pipeline (arxiv 2602.21215, code: `/workspace/sia-repo/SIA`).

---

## Dataset Clarification

**3 data sources are used** (2 datasets, UltraFeedback contributes 2 specific subsets):

| Source | HuggingFace ID | Local Path | Samples | Purpose |
|--------|---------------|------------|---------|---------|
| WildGuardMix | `allenai/wildguardmix` | `/workspace/sia-repo/datasets/WildGuardMix` | 37,934 | Harmlessness |
| UltraFeedback — ShareGPT | `openbmb/UltraFeedback` | `…/UltraFeedback/sharegpt.jsonl` | 79,792 (19,949 × ~4) | Helpfulness |
| UltraFeedback — UltraChat | `openbmb/UltraFeedback` | `…/UltraFeedback/ultrachat.jsonl` | 39,716 (9,929 × ~4) | Helpfulness |
| **Total** | | | **157,442** | |

Paper quote (§4.1): *"we specifically curate the ShareGPT and UltraChat subsets from UltraFeedback,
containing 19,949 and 9,929 instructions, respectively."*

**NOT used for VM training** (confirmed against paper and official code):
- `HelpSteer2` — mentioned only in the 6-month roadmap for future vocabulary-aligned VM retraining
- `Skywork-Reward-Preference-80K` — training data for the Skywork RM itself, not for SIA VM
- Other UltraFeedback subsets: `evol_instruct.jsonl`, `flan.jsonl`, `false_qa.jsonl`, `truthful_qa.jsonl`

---

## Scoring Model

`Skywork/Skywork-Reward-V2-Qwen3-8B` — trajectory-level RM used to annotate each
instruction-response pair with a scalar reward score. Same model serves as the VM backbone.

Local path: `/workspace/sia-repo/models/Skywork-Reward-V2-Qwen3-8B`

Evaluation RM (separate, not used here): `Skywork/Skywork-Reward-V2-Llama-3.1-8B`

---

## Environment Setup

```bash
# Install required Python packages (system-level, no venv available)
pip3 install transformers datasets tqdm accelerate peft --break-system-packages
pip3 install flash-attn --no-build-isolation --break-system-packages

# Versions used
# transformers 5.12.1 | datasets 5.0.0 | peft 0.19.1 | flash-attn 2.8.3.post1
# torch 2.11.0+cu128 | GPU: NVIDIA H200 143GB
```

---

## Step 1 — Format Conversion

Converts raw datasets into the token-sequence format expected by `measure_reward.py` and the trainer.
Each sample gets fields: `id`, `prompt`, `result`, `rm_prompt`, `rm_prompt_tokens_ids`,
`generated_tokens_ids`, `rm_guided_tokens_mask`, `intervene_ratio`.

Tokenizer used: `Skywork-Reward-V2-Qwen3-8B` (detected as Qwen3 → applies Qwen3 chat template).

Working directory for all commands: `/workspace/sia-repo/SIA`

```bash
# Create output directory
mkdir -p /workspace/sia-repo/vm-training-data

cd /workspace/sia-repo/SIA

# WildGuardMix
python3 src/value_model/preprocess.py wildguardmix \
    --dataset_path /workspace/sia-repo/datasets/WildGuardMix \
    --split wildguardtrain \
    --tokenizer_path /workspace/sia-repo/models/Skywork-Reward-V2-Qwen3-8B \
    --output_path /workspace/sia-repo/vm-training-data/wildguardmix-Qwen3.json

# UltraFeedback — ShareGPT subset only
python3 src/value_model/preprocess.py ultrafeedback \
    --dataset_path /workspace/sia-repo/datasets/UltraFeedback/sharegpt.jsonl \
    --tokenizer_path /workspace/sia-repo/models/Skywork-Reward-V2-Qwen3-8B \
    --output_path /workspace/sia-repo/vm-training-data/ultrafeedback-sharegpt-Qwen3.json

# UltraFeedback — UltraChat subset only
python3 src/value_model/preprocess.py ultrafeedback \
    --dataset_path /workspace/sia-repo/datasets/UltraFeedback/ultrachat.jsonl \
    --tokenizer_path /workspace/sia-repo/models/Skywork-Reward-V2-Qwen3-8B \
    --output_path /workspace/sia-repo/vm-training-data/ultrafeedback-ultrachat-Qwen3.json
```

**Output** (in `/workspace/sia-repo/vm-training-data/`):

| File | Samples | Size |
|------|---------|------|
| `wildguardmix-Qwen3.json` | 37,934 | 206 MB |
| `ultrafeedback-sharegpt-Qwen3.json` | 79,792 | 403 MB |
| `ultrafeedback-ultrachat-Qwen3.json` | 39,716 | 247 MB |

---

## Step 2 — RM Score Annotation

Loads `Skywork-Reward-V2-Qwen3-8B` (BF16, Flash Attention 2) and appends a `"reward"` scalar field
to each sample. Samples exceeding 2048 tokens are skipped. All 3 jobs run in parallel on cuda:0
(H200 has 143GB VRAM; 3 × ~16GB = ~48GB).

**Note on output filename convention:** The README names output files with the RM model embedded
(e.g. `wildguardmix-Qwen3-Skywork-Reward-V2-Qwen3-8B.json`). We use `-scored.json` as a shorter
suffix. If you later score with a different RM, rename accordingly to avoid confusion.

```bash
cd /workspace/sia-repo/SIA

# WildGuardMix
python3 src/measure_reward.py \
    --input_file  /workspace/sia-repo/vm-training-data/wildguardmix-Qwen3.json \
    --output_file /workspace/sia-repo/vm-training-data/wildguardmix-Qwen3-scored.json \
    --rm /workspace/sia-repo/models/Skywork-Reward-V2-Qwen3-8B \
    --device cuda:0

# UltraFeedback — ShareGPT
python3 src/measure_reward.py \
    --input_file  /workspace/sia-repo/vm-training-data/ultrafeedback-sharegpt-Qwen3.json \
    --output_file /workspace/sia-repo/vm-training-data/ultrafeedback-sharegpt-Qwen3-scored.json \
    --rm /workspace/sia-repo/models/Skywork-Reward-V2-Qwen3-8B \
    --device cuda:0

# UltraFeedback — UltraChat
python3 src/measure_reward.py \
    --input_file  /workspace/sia-repo/vm-training-data/ultrafeedback-ultrachat-Qwen3.json \
    --output_file /workspace/sia-repo/vm-training-data/ultrafeedback-ultrachat-Qwen3-scored.json \
    --rm /workspace/sia-repo/models/Skywork-Reward-V2-Qwen3-8B \
    --device cuda:0
```

**Actual output** (in `/workspace/sia-repo/vm-training-data/`):

| File | Scored Samples | Skipped (>2048 tok) | Avg Reward | Reward Range |
|------|---------------|---------------------|------------|-------------|
| `wildguardmix-Qwen3-scored.json` | 37,912 | 22 | -0.358 | [-31.5, +33.3] |
| `ultrafeedback-sharegpt-Qwen3-scored.json` | 79,522 | 270 | +2.316 | [-21.9, +32.3] |
| `ultrafeedback-ultrachat-Qwen3-scored.json` | 39,672 | 44 | +4.590 | [-22.5, +35.5] |
| **Total** | **157,106** | **336** | — | — |

All samples validated: 0 missing rewards, 0 NaN values, 0 zero-reward entries.

---

## Output Directory

`/workspace/sia-repo/vm-training-data/`

The `-scored.json` files are the final training-ready artifacts. Pass them to
`src/value_model/train.py` via `--data_file`.

---

## Reference

- Paper: https://arxiv.org/abs/2602.21215
- Official code: `/workspace/sia-repo/SIA` (github.com/hurunyi/SIA)
- Preprocessing script: `SIA/src/value_model/preprocess.py`
- Scoring script: `SIA/src/measure_reward.py`
