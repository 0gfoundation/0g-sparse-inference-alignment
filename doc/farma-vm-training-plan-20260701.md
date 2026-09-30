# FaRMA VM Training Plan (2026-07-01)

Training the Qwen3-4B Value Model with a vocabulary-wide scoring head (FaRMA architecture),
enabling single-forward scoring of all K candidates instead of K separate forwards.

Reference: SIA paper arxiv 2602.21215, local training code at `/workspace/SIA/git/SIA/`.

---

## 1. Background

### 1.1 Current VM architecture

The existing VM-Qwen3-4B checkpoint uses a scalar score head:

```
Linear(hidden_size=2560, out_features=1)
```

At each intervention step, SIA calls the VM once per candidate token, or batches K candidates
into one VM forward. Either way, the RM input grows linearly with K (topk=10 → 10 prompts).

### 1.2 FaRMA architecture

Replace the scalar head with a vocabulary-wide head:

```
Linear(hidden_size=2560, out_features=vocab_size=151936)
```

The VM runs one forward on the current prefix and outputs a reward score for every token in
the vocabulary. SIA reads off the K candidate scores by index — no extra VM forward needed.

Measured throughput benefit: **2.36× system throughput** at conc=16 (see `farma-proxy-vm-topk1-20260630.md`).

### 1.3 Relation to ARM training objective

The vocab-wide head naturally expresses the ARM (Autoregressive Reward Model) objective
(GenARM, ICLR 2025): the head predicts the reward contribution of each possible next token,
conditioned on the current prefix. This is a better training target for SIA than ORM (which
assigns a single reward to the full sequence) because SIA intervenes at the token level.

FaRMA + ARM training are complementary and should be implemented together.

---

## 2. Training Data

### 2.1 Datasets — all public, no author contact needed

| Dataset | Samples | Source |
|---------|---------|--------|
| WildGuardMix | 37,976 pairs | HuggingFace: `allenai/wildguardmix` |
| UltraFeedback (ShareGPT subset) | 19,949 instructions | HuggingFace: `openbmb/UltraFeedback` |
| UltraFeedback (UltraChat subset) | 9,929 instructions | HuggingFace: `openbmb/UltraFeedback` |

Note: the SIA paper states ~157k total pairs; actual dataset counts add to ~68k. The discrepancy
is unexplained in the paper. The three datasets above are what the preprocessing scripts support.

### 2.2 Teacher reward model

`Skywork-Reward-V2-Qwen3-8B` (public, HuggingFace: `Skywork/Skywork-Reward-V2-Qwen3-8B`)
is used to score each instruction-response pair and provide a trajectory-level scalar reward.
This reward becomes the training supervision signal.

### 2.3 Training data format

Each training sample (after preprocessing + scoring):

```json
{
  "id": 42,
  "prompt": "Human:\n...\nAssistant:\n",
  "result": "...",
  "rm_prompt": "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n",
  "rm_prompt_tokens_ids": "1234,5678,...",
  "generated_tokens_ids": "9012,3456,...",
  "rm_guided_tokens_mask": "1,1,1,...",
  "reward": 3.14
}
```

Fields consumed by `train.py`:
- `rm_prompt_tokens_ids` — tokenized conversation prefix (prompt only)
- `generated_tokens_ids` — tokenized response
- `rm_guided_tokens_mask` — which response token positions contribute to loss (all 1s by default)
- `reward` — scalar from teacher RM

---

## 3. Data Preparation Pipeline

All scripts are at `/workspace/SIA/git/SIA/`.

### Step 1 — Format conversion (no scoring yet)

```bash
cd /workspace/SIA/git/SIA

# WildGuardMix
python3 src/value_model/preprocess.py wildguardmix \
    --dataset_path allenai/wildguardmix \
    --split wildguardtrain \
    --tokenizer_path /path/to/Qwen3-4B-Base \
    --output_path assets/vm_data/wildguardmix.json

# UltraFeedback (point at the JSONL directory for ShareGPT and UltraChat subsets)
python3 src/value_model/preprocess.py ultrafeedback \
    --dataset_path /path/to/ultrafeedback_sharegpt/ \
    --tokenizer_path /path/to/Qwen3-4B-Base \
    --output_path assets/vm_data/ultrafeedback_sharegpt.json

python3 src/value_model/preprocess.py ultrafeedback \
    --dataset_path /path/to/ultrafeedback_ultrachat/ \
    --tokenizer_path /path/to/Qwen3-4B-Base \
    --output_path assets/vm_data/ultrafeedback_ultrachat.json
```

Output: JSON files with tokenized prompts and responses, but **no `reward` field yet**.

### Step 2 — Score with teacher RM (adds `reward` field)

```bash
for split in wildguardmix ultrafeedback_sharegpt ultrafeedback_ultrachat; do
    python3 src/measure_reward.py \
        --input_file  assets/vm_data/${split}.json \
        --output_file assets/vm_data/${split}_scored.json \
        --rm /path/to/Skywork-Reward-V2-Qwen3-8B \
        --device cuda:0
done
```

This step is compute-intensive (runs the 8B teacher RM on every sample). Can be parallelized
across multiple GPUs by splitting the data files.

### Step 3 — Train

```bash
bash scripts/train_vm.sh \
    --data_file \
        assets/vm_data/wildguardmix_scored.json \
        assets/vm_data/ultrafeedback_sharegpt_scored.json \
        assets/vm_data/ultrafeedback_ultrachat_scored.json \
    --base_model_path /path/to/Qwen3-4B-Base \
    --output_dir /path/to/vm_output
```

---

## 4. Training Hyperparameters

Sourced from `scripts/train_vm.sh` defaults, `src/value_model/train.py`, and the
`model_config.json` of the existing VM-Qwen3-4B-Base checkpoint.

| Parameter | Value | Source |
|-----------|-------|--------|
| Base model | Qwen3-4B-Base | adapter_config.json |
| Optimizer | AdamW | train.py:796 |
| Learning rate | 1e-4 | train_vm.sh |
| Weight decay | 1e-4 | train_vm.sh |
| Batch size | 8 | train_vm.sh |
| Gradient accumulation | 2 → effective batch = **16** | train_vm.sh |
| Epochs | 3 | train_vm.sh |
| Max sequence length | 1024 tokens | train_vm.sh |
| Gradient clipping | max_norm=1.0 | train.py:297 |
| LR scheduler | none (constant) | not set in train.py |
| Train / val split | 99% / 1% | train_ratio=0.99 |
| Hardware (paper) | single H200 | paper §4 |
| Training duration (paper) | "within a few hours" | paper §4 |

Achieved val R² = **0.9486** on best epoch (epoch 3) for existing VM-Qwen3-4B checkpoint.

### LoRA configuration (from adapter_config.json)

| Parameter | Value |
|-----------|-------|
| rank (r) | 16 |
| alpha | 32 |
| dropout | 0.1 |
| target modules | q_proj, k_proj, v_proj, o_proj |
| task type | FEATURE_EXTRACTION |
| bias | none |

---

## 5. Loss Function

ORM distillation with MSE (paper Eq. 6):

```
L(θ) = E[(1/T Σ_t V_θ(x, y≤t) − R(x, y))²]
```

Implementation in `train.py` (lines 241–261):
1. Forward the full sequence (prompt + response) through the VM
2. At each masked position, read the scalar score from `token_reward_head`
3. Compute a weighted average over masked positions → `avg_prediction`
4. `loss = MSE(avg_prediction, target_reward)`

For FaRMA (vocab-wide head), the forward changes to indexing:

```python
# vocab_rewards: (batch, seq_len, vocab_size)
vocab_rewards = self.token_reward_head(hidden_states)
# Index with the actual next token at each position
token_scores = vocab_rewards[:, :-1, :].gather(
    -1, input_ids[:, 1:].unsqueeze(-1)
).squeeze(-1)
# Mask and average → same MSE loss
```

This is the ARM objective: train the head to predict, at each prefix position, the reward
of the token that was actually generated next.

---

## 6. Code Changes for FaRMA

### 6.1 `src/value_model/model.py`

One line change in `ValueModel.__init__` (line 111):

```python
# Current (scalar head):
self.token_reward_head = nn.Linear(self.hidden_size, 1)

# FaRMA (vocabulary-wide head):
self.token_reward_head = nn.Linear(self.hidden_size, vocab_size)
# vocab_size must be passed in or read from base_model.config.vocab_size
```

Forward method changes for training (replace the token_rewards computation):

```python
# Current:
token_rewards = self.token_reward_head(hidden_states.float())  # (B, T, 1)
token_rewards = token_rewards.squeeze(-1)                       # (B, T)

# FaRMA training (ARM-style: index vocab head with next token):
vocab_rewards = self.token_reward_head(hidden_states.float())   # (B, T, V)
# Shift: at position t, predict reward of token t+1
next_tokens = input_ids[:, 1:]                                  # (B, T-1)
token_rewards = vocab_rewards[:, :-1, :].gather(
    -1, next_tokens.unsqueeze(-1)
).squeeze(-1)                                                   # (B, T-1)
```

Forward method for inference (used by SIA):

```python
# FaRMA inference: single forward on prefix, return vocab-wide rewards
last_hidden = hidden_states[:, -1, :].float()   # (B, hidden_size)
vocab_rewards = self.token_reward_head(last_hidden)  # (B, vocab_size)
# SIA indexes: vocab_rewards[0, candidate_token_ids]  # (K,)
```

### 6.2 SIA scoring integration (`src/sia_vllm_RM.py`)

New scoring path for FaRMA: instead of calling `score_candidates_batch` with K candidate
sequences, call a new `score_prefix_vocab_wide(sid)` that:
1. Advances the KV cache to the current prefix (same `_prepare_b2_session`)
2. Runs one VM forward on the prefix
3. Returns the vocab-wide reward vector
4. SIA indexes it for all K candidates in a single `torch.index_select`

No GPU→CPU sync for candidate indices needed — scoring and indexing happen on GPU.

### 6.3 Memory budget

| Component | Size |
|-----------|------|
| Qwen3-4B base weights (bf16) | ~8 GB |
| Vocab head Linear(2560, 151936) (bf16) | ~778 MB |
| LoRA adapter weights | ~50 MB |
| KV cache (runtime) | variable |

The vocab head adds ~778MB vs the current near-zero scalar head. Ensure the VM's GPU memory
allocation (`--rm_b2_gpu_mem`) has enough headroom.

---

## 7. Conversion for vLLM Inference

After training, the checkpoint must be converted to vLLM-compatible format before use with
the b2 inproc backend (same as the current VM):

```bash
python scripts/convert_rm_for_vllm.py \
    --rm /path/to/Qwen3-4B-Base \
    --rm_lora /path/to/farma_vm_checkpoint \
    --output /path/to/VM-Qwen3-4B-FaRMA-merged-for-vllm
```

The FaRMA vocab head (`token_reward_head.pt`) needs to be included in this conversion.
The existing `convert_rm_for_vllm.py` may need a small update to handle the vocab-wide
head shape.

---

## 8. Open Questions

1. **Dataset size discrepancy**: paper claims ~157k pairs, scripts cover ~68k. Worth clarifying
   whether additional data sources (e.g., more UltraFeedback subsets) are needed to match paper quality.

2. **LR scheduler**: train.py currently uses no scheduler (constant LR). A cosine decay schedule
   may help convergence for the larger vocab head output.

3. **ARM training mask shift**: the current `rm_guided_tokens_mask` aligns with `generated_tokens_ids`.
   For ARM-style training, the mask must be shifted by 1 position (predict token t+1 from hidden state
   at t). The train.py collate and loss code needs updating for this.

4. **Vocab head initialization**: a zero-initialized head (default `nn.Linear`) or a small-variance
   init may both work. The current scalar head uses default PyTorch init; the vocab head may benefit
   from `nn.init.normal_(weight, std=0.01)` to avoid large initial logit shifts.

5. **convert_rm_for_vllm.py compatibility**: the current script merges LoRA + scalar head into a
   vLLM-pooling-compatible format. FaRMA's vocab-wide head changes what gets merged and how the
   b2 backend's `score_candidates_batch` API is called. Plan a separate integration pass.

---

## 9. GPU Memory Requirements

### 9.1 Component breakdown

Calculated from Qwen3-4B-Base config (hidden=2560, vocab=151936, layers=36) and
`scripts/train_vm.sh` defaults (batch=8, seq=1024, gradient checkpointing + Flash Attention).

| Component | Memory | Notes |
|-----------|--------|-------|
| Qwen3-4B base model (bf16, frozen) | 7.1 GB | No gradients |
| LoRA weight + AdamW m/v + grad (fp32) | 0.14 GB | Only 9.6M trainable params |
| **Vocab head weight (fp32)** | **1.45 GB** | Linear(2560, 151936) |
| **Vocab head AdamW m+v (fp32)** | **2.90 GB** | Primary new cost |
| **Vocab head gradient (fp32)** | **1.45 GB** | Primary new cost |
| `vocab_rewards` forward (B=8, T=1024) | 2.32 GB | shape=(8, 1024, 151936) bf16 |
| `vocab_rewards` backward peak | 2.32 GB | Forward + backward overlap at peak |
| Transformer activations (grad checkpoint) | 0.16 GB | Recomputed per layer |
| CUDA allocator overhead | 1.00 GB | |
| **Total (batch=8)** | **~18.8 GB** | |
| **Total (batch=4)** | **~16.4 GB** | |

Comparison: current scalar head training (batch=8) ≈ **8.4 GB**.
FaRMA vocab head adds **~10.4 GB** — split roughly equally between head optimizer states
(5.8 GB) and the `vocab_rewards` activation tensor peak (4.6 GB).

### 9.2 Feasibility by GPU size

| GPU memory | Feasibility | Notes |
|-----------|-------------|-------|
| 16 GB | ❌ Not feasible | batch=4 requires 16.4 GB → OOM |
| 24 GB | ✅ Feasible | batch=8 requires 18.8 GB, ~5 GB headroom |
| 40 GB | ✅ Comfortable | Can omit gradient checkpointing if desired |
| 80 GB | ✅ Ample | Same as paper environment (H200) |

**Minimum: 24 GB. Practical recommendation: 40 GB.**

### 9.3 Required training flags for 24 GB

Two settings must be enabled in `train.py` before training starts:

```python
# Enable gradient checkpointing (reduces transformer activation memory ~10×)
base_model.gradient_checkpointing_enable()

# Enable Flash Attention (avoids storing full attention matrices)
model = AutoModelForCausalLM.from_pretrained(
    ..., attn_implementation="flash_attention_2"
)
```

Without gradient checkpointing, transformer layer activations grow to ~19 GB for
standard attention (seq_len=1024, batch=8, 36 layers × 536 MB attention matrices).
With Flash Attention + gradient checkpointing, this collapses to ~0.16 GB.

The `vocab_rewards` tensor (2.32 GB) is unavoidable even with gradient checkpointing —
it must be materialized for the ARM gather operation and its backward pass.

### 9.4 Optional: chunked vocab head forward (enables 16 GB training)

The `vocab_rewards` tensor (B, T, vocab_size) is the single largest temporary allocation.
Computing it in chunks eliminates the peak without changing the math:

```python
chunk_size = 64   # process 64 token positions at a time
loss_accum = 0.0
for start in range(0, seq_len, chunk_size):
    h_chunk = hidden_states[:, start:start+chunk_size, :]   # (B, chunk, hidden)
    rewards_chunk = token_reward_head(h_chunk)               # (B, chunk, vocab)
    # immediately gather and accumulate loss — no full tensor stored
    next_tokens_chunk = input_ids[:, start+1:start+chunk_size+1]
    scores_chunk = rewards_chunk.gather(-1, next_tokens_chunk.unsqueeze(-1)).squeeze(-1)
    mask_chunk = reward_mask[:, start:start+chunk_size]
    if mask_chunk.any():
        loss_accum += (scores_chunk * mask_chunk).sum()
loss = mse(loss_accum / mask.sum(), target_reward)
```

With chunk=64, peak `vocab_rewards` drops from 2.32 GB to **148 MB**
(4 × 64 × 151936 × 2 bytes), reducing total training memory to ~**13 GB** (batch=4).
This enables 16 GB GPUs. Requires modifying the loss loop in `train.py`.
