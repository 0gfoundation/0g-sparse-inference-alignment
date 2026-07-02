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

| Dataset | Instructions | Pairs | HuggingFace |
|---------|-------------|-------|-------------|
| WildGuardMix | 37,976 | 37,976 × 1 = **37,976** | `allenai/wildguardmix` (gated) |
| UltraFeedback (ShareGPT subset) | 19,949 | 19,949 × 4 = **79,796** | `openbmb/UltraFeedback` |
| UltraFeedback (UltraChat subset) | 9,929 | 9,929 × 4 = **39,716** | `openbmb/UltraFeedback` |
| **Total** | | **≈ 157,488 ≈ 157k** | |

The 157k figure matches the paper exactly. UltraFeedback has 4 completions per instruction
(from 17 different LLMs); each completion becomes a separate training pair.

Downloaded raw data lives at `/workspace/git-gaoteng/vm-training-data/`.

### 2.2 Dataset Field Details

**WildGuardMix** (safety/harmlessness dataset):

| Field | Used by train pipeline? | Content |
|-------|------------------------|---------|
| `prompt` | ✅ yes | User instruction (may contain harmful requests) |
| `response` | ✅ yes | Model response |
| `adversarial` | ❌ ignored | Whether prompt is adversarially constructed |
| `prompt_harm_label` | ❌ ignored | harmful / unharmful |
| `response_refusal_label` | ❌ ignored | refusal / compliance |
| `response_harm_label` | ❌ ignored (but used as filter) | harmful / unharmful / None |
| `subcategory` | ❌ ignored | benign / violence / hate_speech / … |

Only `prompt` and `response` are passed to `preprocess.py`; all other fields are discarded.
Harmful responses are NOT filtered — they enter training with low teacher RM scores,
which teaches the VM to assign them low token rewards.

**WildGuardMix filter (86,759 → 37,976):**
The `wildguardtrain` config has 86,759 rows total, but 48,783 have `response_harm_label=None`
(prompt-only entries with no response). Filtering to rows where `response_harm_label is not None`
yields exactly 37,976 pairs, matching the paper.

**UltraFeedback** (helpfulness/honesty dataset):

| Field | Used? | Content |
|-------|-------|---------|
| `instruction` | ✅ yes | User prompt |
| `completions` | ✅ yes (iterates all 4) | List of 4 dicts, each with `response` + annotations |
| `source` | used for filtering only | sharegpt / ultrachat / evol_instruct / … |
| `models` | ❌ ignored | Names of 4 LLMs that generated completions |
| `correct_answers` / `incorrect_answers` | ❌ ignored | GT labels (some tasks) |

Each `completion` dict also contains `annotations`, `overall_score`, `fine-grained_score`,
`principle`, `critique` — all ignored by `preprocess.py`, which only reads `completion["response"]`.

Completion counts (verified): ShareGPT 19,948/19,949 have exactly 4 (1 has 0, skipped);
UltraChat all 9,929 have exactly 4.

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

**How the RM input is assembled** (`measure_reward.py` + `utils.py`):

```
preprocess.py output:
  prompt = "Human:\n{instruction}\nAssistant:\n"
  result = "{response}"

measure_reward.py reassembles:
  text = prompt + result
       = "Human:\n{instruction}\nAssistant:\n{response}"

ConversationProcessor.parse_conversation_to_format(text) splits on (Human|Assistant):
  → [{"role": "user",      "content": "{instruction}"},
     {"role": "assistant", "content": "{response}"}]

tokenizer.apply_chat_template(conversations) applies Qwen3 chat template:
  → "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"

This string is tokenized and fed to Skywork-Reward-V2-Qwen3-8B.
The RM outputs logits[0][0] (scalar) as the reward score.
Samples with tokenized length ≥ 2048 are skipped.
```

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
| Base model | **Qwen3-4B-Base** (paper) / **Skywork-Reward-V2-Qwen3-4B** (code) — see note | paper App.C.1 / train_vm.sh |
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

**Note — Base model discrepancy (must confirm with authors):**
- Paper Appendix C.1: "VM-Qwen3-4B is initialized from **Qwen3-4B-Base**"
- `scripts/train_vm.sh` line 8, `README.md` line 192, `train.py` help text: all use **Skywork-Reward-V2-Qwen3-4B**
- The two differ: Skywork-Reward-V2-Qwen3-4B has been fine-tuned on 26M preference pairs;
  Qwen3-4B-Base has not. The transformer backbone weights are different.
- Both HF models exist: `Qwen/Qwen3-4B-Base` and `Skywork/Skywork-Reward-V2-Qwen3-4B`.
- Pending author confirmation (see also: teacher RM discrepancy in `farma-vm-training-data-20260701.md`).

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

**Step-by-step training mechanism** (`train.py` lines 225–261):

**Step 1 — Assemble input sequence**
```python
input_ids = rm_prompt_ids + generated_ids
# [P P P P P P R R R R R R R]
#  ← prompt →  ← response →

full_mask = [0]*len(rm_prompt_ids) + [1]*len(generated_ids)
# [0 0 0 0 0 0 1 1 1 1 1 1 1]   ← 1 = participates in loss
```

**Step 2 — VM forward: scalar reward at every token position**
```python
outputs = model(input_ids)
token_rewards = outputs.token_rewards  # shape: (batch, seq_len)
# [* * * * * * v1 v2 v3 v4 v5 v6 v7]
#               ↑ only these are used
```

**Step 3 — Average masked positions**
```python
masked_predictions = token_rewards[mask_positions]  # response tokens only
avg_prediction = masked_predictions.mean()
# ≈ average reward the VM predicts for the full response
```

**Step 4 — MSE against teacher RM score**
```python
loss = MSELoss(avg_prediction, target_reward)
# target_reward = score Skywork-Reward-V2-Qwen3-8B gave the full response
```

The training objective pushes the VM to distribute the trajectory-level reward evenly across
all response tokens, so that at inference time any prefix can be scored by averaging its
token rewards — without needing the full response.

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

1. ~~**Dataset size discrepancy**~~: **Resolved.** 157k = 37,976 (WildGuardMix × 1) +
   19,949 × 4 + 9,929 × 4 = 157,488. Each UltraFeedback instruction has 4 completions;
   `preprocess.py` already iterates all 4. No additional data sources needed.

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

---

## 10. Complete Command Lines

Assumed models (best guess — see §4 discrepancy notes):

| Role | Model |
|------|-------|
| Teacher RM | `Skywork/Skywork-Reward-V2-Qwen3-8B` (paper text wins over stale shell script) |
| VM base | `Skywork/Skywork-Reward-V2-Qwen3-4B` (code/README/scripts all consistent) |

### 10.0 Path variables

```bash
VM_BASE=/path/to/Skywork-Reward-V2-Qwen3-4B
TEACHER_RM=/path/to/Skywork-Reward-V2-Qwen3-8B
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  # placeholder — real token was invalidated after being exposed in git history

SIA=/workspace/SIA/raw_SIA/SIA
RAW=/workspace/git-gaoteng/vm-training-data
PROC=/workspace/git-gaoteng/vm-training-data/processed
CKPT=/workspace/git-gaoteng/vm-checkpoints

mkdir -p $PROC $CKPT
```

### 10.1 Step 1 — Preprocess (tokenize, no scoring yet)

```bash
cd $SIA

# WildGuardMix（从 HuggingFace 加载；内部 None response 行会被自动跳过 → ~37,976 条）
HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
python3 src/value_model/preprocess.py wildguardmix \
    --dataset_path allenai/wildguardmix \
    --split wildguardtrain \
    --tokenizer_path $VM_BASE \
    --output_path $PROC/wildguardmix.json

# UltraFeedback — ShareGPT subset（已下载的 JSONL）
python3 src/value_model/preprocess.py ultrafeedback \
    --dataset_path $RAW/ultrafeedback-sharegpt/sharegpt.jsonl \
    --tokenizer_path $VM_BASE \
    --output_path $PROC/sharegpt.json

# UltraFeedback — UltraChat subset
python3 src/value_model/preprocess.py ultrafeedback \
    --dataset_path $RAW/ultrafeedback-ultrachat/ultrachat.jsonl \
    --tokenizer_path $VM_BASE \
    --output_path $PROC/ultrachat.json
```

产物：每个 JSON 含 `rm_prompt_tokens_ids` / `generated_tokens_ids` / `rm_guided_tokens_mask`，尚无 `reward` 字段。

### 10.2 Step 2 — Teacher RM scoring（写入 `reward` 字段）

```bash
cd $SIA

for SPLIT in wildguardmix sharegpt ultrachat; do
    python3 src/measure_reward.py \
        --input_file  $PROC/${SPLIT}.json \
        --output_file $PROC/${SPLIT}_scored.json \
        --rm $TEACHER_RM \
        --device cuda:0
done
```

注：8B 模型逐条打分，~157k 条在单卡约需数小时，可并行到多卡（分别指定 `--device cuda:1` 等）。

### 10.3 Step 3 — Train VM

```bash
cd $SIA

bash scripts/train_vm.sh \
    --data_file \
        $PROC/wildguardmix_scored.json \
        $PROC/sharegpt_scored.json \
        $PROC/ultrachat_scored.json \
    --base_model_path $VM_BASE \
    --output_dir $CKPT/VM-Qwen3-4B \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --num_epochs 3 \
    --device cuda:0 \
    --lora_r 16 \
    --lora_alpha 32 \
    --save_interval_steps 5000
```

默认参数（与论文一致）：batch=8，grad_accum=2，lr=1e-4，epochs=3，max_len=1024，LoRA r=16 α=32。

### 10.4 预期产物

```
$CKPT/VM-Qwen3-4B/
├── lora_weights/          ← PEFT LoRA adapter（接 --rm_lora 使用）
├── token_reward_head.pt   ← Linear(hidden_size, 1) 权重
├── model_config.json      ← 配置与最优 val R²
└── training_curves.png    ← loss / R² 曲线
```
