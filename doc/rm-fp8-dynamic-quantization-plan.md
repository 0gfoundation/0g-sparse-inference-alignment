# Value Model FP8_DYNAMIC Quantization Plan (A2 Route, Complete Version)

**Date**: 2026-05-26
**Goal**: Quantize SIA's Value Model (`VM-Qwen3-4B-merged-for-vllm`) to FP8, reducing RM forward compute time from ~50ms/call to ~30-35ms, with expected end-to-end throughput improvement from G3's 29.65 tok/s to ~33-36 tok/s (+10-22%).
**Constraint**: **No degradation in intervention quality** (reward ranking, SIA accuracy equivalent to BF16).
**Key difference from G4**: G4 uses vLLM's runtime `--quantization fp8 --kv-cache-dtype fp8`, where activation scale defaults to 1.0 triggering a degraded path; this plan uses offline `llm-compressor` to produce a **FP8_DYNAMIC** checkpoint with **dynamic activation quantization (per-token real-time scale computation)**, no static calibration dependency.

---

## 0. Direct Answer to the Key User Question

> "When using an MMLU evaluation subset for calibration, does that assume future data must also match this dataset's text distribution?"

**For static quantization: yes. For FP8_DYNAMIC: no.**

There are two types of scales in FP8 quantization:

| Scale Type | Source | Data Distribution Dependency |
|---|---|---|
| **Weight scale** | Computed directly from weight tensor statistics | No dependency |
| **Activation scale (static)** | Run calibration data once, fix a scale offline | **Strong dependency**: distribution shift causes quality degradation |
| **Activation scale (dynamic)** | Computed from current activations at each forward step | **Zero dependency**: adaptive to any input |

**FP8_DYNAMIC** = static per-channel FP8 weights + **dynamic per-token** FP8 activations.

- Weight scale only looks at the weight tensor itself, completely independent of calibration data
- Activation scale is computed on-the-fly at each forward pass; **even if calibration data is biased, it only affects "which layers quantization ops are inserted into", not the scale values**
- Therefore, the assumption "calibration set = deployment set" **does not exist**

The `llm-compressor` oneshot flow under FP8_DYNAMIC requires a few dozen samples, but their purpose is **to let the modifier traverse the model graph and correctly insert quantization ops** — not to fit a distribution. Passing MMLU, AlpacaEval, or random text **produces an identical quantized model**.

---

## 0.5 Hardware Prerequisites

**FP8 E4M3 operators require NVIDIA GPU compute capability sm_89 or higher**:

| GPU | sm version | Works with this plan? |
|---|---|---|
| H100, H200 | sm_90 | Yes |
| RTX 4090, L40, L40S, RTX 6000 Ada | sm_89 | Yes |
| A100 (40/80GB) | sm_80 | No (will error or fall back to emulation) |
| V100, T4, RTX 3090 | sm_70 / sm_75 / sm_86 | No |

This project deploys on **H200 without issues**. If migrating to A100, this plan cannot be used directly — the INT8 / AWQ route is required instead.

Pre-launch self-check:
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# Expected: NVIDIA H200, 9.0  (or H100 9.0 / RTX 4090 8.9 / L40 8.9, etc.)
```

## 1. Background: Why A2, and Not Something Else

After 17 experimental groups and C1/C2 measurements, the bottleneck is clearly identified:

```
per-RM-call ≈ 78ms (G3) ≈ 60ms (C1+C2 profile total)

Breakdown:
  RM actual forward compute: ~40-50ms  ← this is the dominant term; FP8 can help here
  HTTP/scheduling/parse:     ~10-15ms  ← C1/C2 already explored; little headroom remains
  Client-side tokenize:      ~5-7ms    ← C2 moved this from server to client; zero net gain
```

The reason C1+C2 yield near-zero gains (measured 29.65 → 29.43 tok/s): they only touched the "protocol layer", which cannot reach the core compute. **A2 directly attacks RM forward compute** — the only remaining lever under the current shared-GPU deployment with zero quality loss.

See `doc/profiling-bottleneck-analysis.md` for detailed analysis.

---

## 2. Plan Details

### 2.1 Quantization Target Layers

Layer structure of `VM-Qwen3-4B-merged-for-vllm` (confirmed from config.json):

```
36 transformer decoder layers, each with 7 Linear layers:
  self_attn: q_proj, k_proj, v_proj, o_proj
  mlp:       gate_proj, up_proj, down_proj
= 252 Linear layers → all quantized to FP8_DYNAMIC
+ embed_tokens (Linear-like, not quantized)
+ score head (Linear(2560, 1), not quantized ← critical for reward precision)
```

### 2.2 ignore list (Critical)

```python
ignore = [
    "score",         # classifier head, reward precision
    "lm_head",       # fallback (even if this model has no lm_head in practice)
    "embed_tokens",  # embedding lookup does not go through FP8 GEMM path
]
```

**If "score" is not ignored, reward numerical precision will degrade significantly**, because the score head is an extremely small `Linear(2560, 1)` matrix where FP8 quantization error gets amplified.

### 2.3 KV Cache Remains BF16

**Do not touch KV cache**. Reasons:
- vLLM 0.10.1.1's `--kv-cache-dtype fp8` triggers a degraded path when q/k/v/prob scales are uncalibrated (G4 measured -7%)
- Calibrating KV scales requires an additional workflow, adding significant complexity
- Quantizing weights + dynamic activations alone already captures most of the FP8 compute benefit

### 2.4 Known Items to Verify

The following **has not been tested in this repository**; verify immediately after quantization succeeds:

> **Can vLLM 0.10.1.1's `--runner pooling --convert classify` correctly attach a classifier head on a compressed-tensors format checkpoint?**

G4 previously used vLLM's runtime FP8 (`--quantization fp8`) without going through the compressed-tensors path. This plan switches to an offline compressed-tensors checkpoint, making `--convert classify` a new combination with two possible outcomes:

1. Normal: vLLM loads the compressed-tensors weights and still applies the score head pooler behavior per `--convert classify`
2. Failure: vLLM's loading logic makes the two flags mutually exclusive (compressed-tensors already treats the model as having a built-in classification head, causing `--convert classify` to error)

→ Confirm from the Phase 2 startup log immediately. If it fails, proceed to §4 R3 mitigation.

### 2.5 Quantization Configuration

```python
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    targets=["Linear"],   # use list form; old versions accept str, new versions require list; list is backward/forward compatible
    scheme="FP8_DYNAMIC",
    ignore=["score", "lm_head", "embed_tokens"],
)
```

**`FP8_DYNAMIC` scheme internal behavior**:
- Weights: `static FP8 E4M3` per-output-channel, scale computed from weight statistics
- Activations: `dynamic FP8 E4M3` per-token, scale computed from the current batch at forward time
- KV cache: unchanged (not within the Linear quantization scope)

---

## 3. Complete Workflow

### Phase 1: Offline Quantization (~20 minutes, one-time)

#### Step 1.0 — **Verify the actual module name of the score head** (mandatory; if missed, ignore list is incomplete = catastrophic reward precision degradation)

`ignore=["score"]` is written based on assumption. The `Qwen3ForSequenceClassification` classifier head may actually be named `score`, `model.score`, `classifier`, `score.dense`, etc. **An unmatched ignore = the entire head gets quantized = catastrophic reward precision degradation**. Confirm the actual name before quantizing:

```python
from transformers import AutoModelForSequenceClassification
import torch.nn as nn

m = AutoModelForSequenceClassification.from_pretrained(
    "/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm",
    torch_dtype="auto", trust_remote_code=True,
)
print("=== All nn.Linear modules (including classifier head candidates) ===")
for name, mod in m.named_modules():
    if isinstance(mod, nn.Linear):
        out_features = mod.out_features
        # classifier head characteristic: very small output dimension (typically 1 or num_labels)
        if out_features <= 16:
            print(f"  *** classifier head candidate: {name} -> Linear({mod.in_features}, {out_features})")

print("\n=== Full name of embedding module ===")
for name, mod in m.named_modules():
    if isinstance(mod, nn.Embedding):
        print(f"  embedding: {name} -> Embedding({mod.num_embeddings}, {mod.embedding_dim})")
```

Expected output:
```
*** classifier head candidate: score -> Linear(2560, 1)
embedding: model.embed_tokens -> Embedding(151936, 2560)
```

**Use these two actual names (full module paths without stripping the prefix) to replace the placeholder `"score"` and `"embed_tokens"` in the `ignore=[...]` in Step 1.1**. Common Qwen3 actual forms:

| Placeholder | Likely actual name |
|---|---|
| `"score"` | `"score"` or `"classifier"` |
| `"embed_tokens"` | `"model.embed_tokens"` (with `model.` prefix) |
| `"lm_head"` | `"lm_head"` (kept as a safety fallback, even if this model may not have lm_head) |

#### Step 1.1 — Write the quantization script

Create `scripts/quantize_rm_fp8_dynamic.py`:

```python
"""
Quantize VM-Qwen3-4B-merged-for-vllm to FP8_DYNAMIC, keeping the score head in BF16.

Usage:
    python scripts/quantize_rm_fp8_dynamic.py
        [--src /path/to/source] [--dst /path/to/output] [--n_samples 50]
"""
import argparse, random
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def build_calibration_dataset(tok, n_samples: int = 50, max_len: int = 1024):
    """Build calibration samples. FP8_DYNAMIC does not depend on distribution, but oneshot needs
    some samples to traverse the model graph.

    Sample format must match the actual text the RM sees at deployment: chat-formatted user+assistant turn.
    """
    # 30 diverse samples: technical / scientific / humanities / math / casual / multilingual / long and short
    # Note: FP8_DYNAMIC does not depend on data distribution, but diverse samples help oneshot
    # traverse activation paths across all Linear layers (even though scale is still computed dynamically per-token)
    sample_pairs = [
        # Math / Arithmetic
        ("What is 2+2?", "The answer is 4."),
        ("Calculate 15% of 200.", "30."),
        ("Solve x^2 = 25.", "x = 5 or x = -5."),
        # Science / Nature
        ("Explain photosynthesis briefly.", "Plants convert sunlight to energy via chlorophyll."),
        ("What is the boiling point of water?", "100 degrees Celsius at sea level."),
        ("Why is the sky blue?", "Rayleigh scattering of sunlight by air molecules."),
        ("Name 3 noble gases.", "Helium, neon, argon."),
        # Humanities / History / Literature
        ("Who wrote Hamlet?", "William Shakespeare."),
        ("When did World War II end?", "September 2, 1945."),
        ("Capital of France?", "Paris."),
        ("Translate 'hello' to Spanish.", "Hola."),
        # Programming / Technical
        ("What is recursion in programming?", "A function calling itself with smaller inputs."),
        ("Difference between TCP and UDP?", "TCP is connection-oriented and reliable; UDP is connectionless."),
        ("What does HTTP stand for?", "HyperText Transfer Protocol."),
        # Daily / Lists
        ("List 3 colors.", "Red, blue, green."),
        ("Suggest a quick breakfast.", "Toast with peanut butter and a banana."),
        ("Name 2 musical instruments.", "Piano and guitar."),
        # Longer instructions
        ("Write a short poem about autumn.",
         "Leaves of amber dance in fading light, "
         "the crisp air whispers summer's end."),
        ("Explain the theory of relativity in one sentence.",
         "Space and time are interwoven, and both bend under gravity and motion."),
        ("Summarize the plot of Romeo and Juliet.",
         "Two young lovers from feuding families secretly marry, "
         "then die by misunderstanding."),
        # Reasoning / Multi-step
        ("If a train travels 60 mph for 2 hours, how far did it go?", "120 miles."),
        ("Is 17 prime?", "Yes, 17 has only 1 and 17 as divisors."),
        # Negation / Bias check
        ("What's the capital of the moon?", "The moon has no capital city; it's not a country."),
        # Mixed Chinese-English / Multilingual
        ("What is the capital of China?", "Beijing is the capital of China."),
        ("How do you say 'thank you' in English?", "Thank you."),
        # Code
        ("Python print 'hello world'.", "print('hello world')"),
        ("What does this Python do: x = [1,2,3]; print(sum(x))?", "It prints 6, the sum of the list."),
        # Philosophy / Open-ended
        ("Is free will real?", "Philosophers debate this; both compatibilist and libertarian views exist."),
        ("Define consciousness.", "Subjective awareness of one's surroundings and inner experience."),
        # Compute-intensive / Long input
        ("List 10 fruits.",
         "Apple, banana, orange, grape, mango, strawberry, watermelon, pineapple, kiwi, peach."),
    ]
    random.seed(42)
    # Use all 30 diverse samples; rotate as needed if more are required
    samples = (sample_pairs * (n_samples // len(sample_pairs) + 1))[:n_samples]

    texts = []
    for user, assistant in samples:
        convs = [
            {"role": "user",      "content": user},
            {"role": "assistant", "content": assistant},
        ]
        text = tok.apply_chat_template(convs, tokenize=False)
        bos = tok.bos_token
        if bos and text.startswith(bos):
            text = text[len(bos):]
        texts.append(text)

    encodings = tok(
        texts, truncation=True, max_length=max_len, padding=False,
        return_tensors=None,
        return_attention_mask=True,   # explicitly request attention_mask; default behavior differs across transformers versions
    )
    # must include attention_mask; llm-compressor's data collator requires both fields by default
    return Dataset.from_dict({
        "input_ids":      encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm")
    p.add_argument("--dst", default="/workspace/SIA/models/VM-Qwen3-4B-merged-fp8-dynamic")
    p.add_argument("--n_samples", type=int, default=50)
    args = p.parse_args()

    print(f"[quantize] Loading model from {args.src}", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.src,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto",   # load model onto GPU so oneshot forward can compute activations on GPU
    )
    tok = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)

    print(f"[quantize] Building calibration dataset ({args.n_samples} samples)", flush=True)
    ds = build_calibration_dataset(tok, n_samples=args.n_samples)

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["score", "lm_head", "embed_tokens"],
    )

    print(f"[quantize] Running oneshot → {args.dst}", flush=True)
    # Intentionally not passing num_calibration_samples / max_seq_length kwargs:
    #   - Different llm-compressor versions accept these names inconsistently; passing wrong names causes TypeError
    #   - Number of calibration samples is determined by the dataset length
    #   - Sequence length is already truncated to max_len=1024 in build_calibration_dataset
    # Pass only the 4 necessary kwargs for best forward/backward version compatibility
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        output_dir=args.dst,
    )
    tok.save_pretrained(args.dst)
    print("[quantize] Done", flush=True)


if __name__ == "__main__":
    main()
```

#### Step 1.2 — Run the quantization

```bash
# Library for the quantization side (produces the checkpoint)
pip install llmcompressor==0.5.1            # or the current stable version

# Library also required at runtime on the RM server side (must install for vLLM to load compressed-tensors checkpoints)
# ⚠️ Version must match llmcompressor; a mismatch may cause load failures:
#    llmcompressor 0.5.1 corresponds to compressed-tensors 0.7.x
#    llmcompressor 0.4.x corresponds to compressed-tensors 0.6.x
# The actual version is whatever compressed-tensors llmcompressor pulls in during install;
# when crossing process boundaries (quantization side vs vLLM side), the versions must match
pip install compressed-tensors==0.7.1       # version paired with llmcompressor 0.5.1

python scripts/quantize_rm_fp8_dynamic.py \
    --src /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --dst /workspace/SIA/models/VM-Qwen3-4B-merged-fp8-dynamic
```

Before running, use `pip show llmcompressor compressed-tensors` to confirm both versions are aligned.

#### Step 1.3 — Verify the output

```bash
# 1. Weight size should drop from ~7.5 GB to ~4 GB
du -sh /workspace/SIA/models/VM-Qwen3-4B-merged-fp8-dynamic/

# 2. config.json should contain quantization_config with the correct ignore list
python -c "
import json
c = json.load(open('/workspace/SIA/models/VM-Qwen3-4B-merged-fp8-dynamic/config.json'))
qc = c.get('quantization_config', {})
print('format:', qc.get('format'))
print('scheme:', qc.get('config_groups', {}))
print('ignore:', qc.get('ignore'))
"
# Expected:
#   format: 'float-quantized'
#   ignore: ['score', 'lm_head', 'embed_tokens']
#   config_groups: weights.num_bits=8 type='float' strategy='channel'
#                  input_activations.num_bits=8 type='float' strategy='token' dynamic=True

# 3. Check that the score head is truly not quantized (should be BF16)
python -c "
import safetensors.torch as st
import glob
for f in sorted(glob.glob('/workspace/SIA/models/VM-Qwen3-4B-merged-fp8-dynamic/*.safetensors')):
    keys = st.load_file(f).keys()
    for k in keys:
        if 'score' in k:
            t = st.load_file(f)[k]
            print(f'{k}: dtype={t.dtype} shape={t.shape}')
"
# Expected: score.weight dtype=torch.bfloat16
```

### Phase 2: Start RM + Smoke Test

**Prerequisite**: Ensure ports 8001 / 8000 are not occupied by stale processes:
```bash
pkill -9 -f "vllm serve|sia_vllm_server|VLLM::EngineCore|vllm_serve_with_token_ids" 2>/dev/null
sleep 2
ss -tlnp 2>/dev/null | grep -E ":800[01]" && echo "Port still occupied, handle manually" || echo "(Ports are free)"
```

```bash
# Note: do NOT add --quantization fp8 flag (that is runtime quant, which conflicts with compressed-tensors)
# Preferred: vLLM 0.10.1.1 auto-detects from quantization_config in config.json
nohup python scripts/vllm_serve_with_token_ids.py serve \
      /workspace/SIA/models/VM-Qwen3-4B-merged-fp8-dynamic \
      --runner pooling \
      --convert classify \
      --enable-prefix-caching \
      --no-enable-chunked-prefill \
      --gpu-memory-utilization 0.3 \
      --max-model-len 2048 \
      --port 8001 \
      > log_vllm_rm_fp8d_$(date +%Y%m%d%H%M).txt 2>&1 &
```

**Fallback**: If the startup log reports "unsupported quantization config" or "no quantization detected", explicitly add the flag when auto-detection fails:

```bash
# Add this to the command above
--quantization compressed-tensors \
```

If `--quantization compressed-tensors` still errors, it means vLLM 0.10.1.1 does not support the current compressed-tensors format version. Proceed to risk branch R3.

**Key lines to check in startup log**:

| Pattern | Expected |
|---|---|
| `Model loading took X.X GiB` | ~**4.X GiB** (not 7.5) |
| `quantization=...` field | should recognize compressed-tensors / fp8 |
| `Using KV cache scaling factor 1.0` | **should NOT appear** (we did not touch KV) |
| `q_scale / k_scale / prob_scale` warnings | **should NOT appear** |
| `[vllm-rm-patch]` | should appear (Pydantic patch) |

**Immediate sanity check after startup** (wait for vLLM to be ready before checking reward values):

```bash
# 0) Wait for vLLM to be fully ready (startup + CUDA graph capture typically takes 30-60s)
echo "Waiting for vLLM ready..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
    echo "vLLM ready after ${i}s"
    break
  fi
  sleep 2
done
# Fallback: check that log ends with "Application startup complete"
grep "Application startup complete" log_vllm_rm_fp8d_*.txt | tail -1 || {
  echo "vLLM still not ready, abort sanity"; exit 1;
}

# 1) Basic reward value sanity check (includes NaN/Inf detection)
python - <<'EOF'
import math, requests
resp = requests.post("http://localhost:8001/classify", json={
    "model": "/workspace/SIA/models/VM-Qwen3-4B-merged-fp8-dynamic",
    "input": [
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>",
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n5<|im_end|>",
    ],
    "activation": False,
})
resp.raise_for_status()
data = sorted(resp.json()["data"], key=lambda x: x.get("index", 0))
scores = [d["probs"][0] for d in data]
print("scores:", scores)

# Must assert (stop immediately if any assertion fails)
assert all(not math.isnan(s) and not math.isinf(s) for s in scores), \
    f"FP8 RM returned NaN/Inf: {scores}"
assert scores[0] > scores[1], \
    f"FP8 RM score for '4' should not be lower than '5': scores={scores}"
print("Sanity pass.")
EOF
```

If sanity fails (NaN/Inf, inverted reward values, HTTP 500, etc.), **immediately fall back to BF16 and follow the §4.1 rollback tree**. Abandon the A2 route.

### Phase 3: Quality A/B Validation (non-skippable gate, **run sequentially**)

**Do not run BF16 + FP8 as two vLLM instances simultaneously**. Running two vLLM servers on the same GPU: (a) each only gets 0.25 GPU memory, (b) they compete for GPU context switches, **distorting latency measurements** and introducing extra noise in the reward comparison. **Run sequentially instead**: first run BF16 and save results to disk, kill it, then start FP8 with the same inputs, then compare the two saved files.

Create 2 scripts:

**`scripts/collect_rm_scores.py`** (same script run twice, querying BF16/FP8 server respectively, outputting to different JSONs):

```python
"""
Usage (run sequentially twice):
    # First start BF16 RM on 8001
    python scripts/collect_rm_scores.py --label bf16 --port 8001 \
        --out results/ab_bf16.json
    # Kill the BF16 RM
    # Start FP8 RM on 8001
    python scripts/collect_rm_scores.py --label fp8 --port 8001 \
        --out results/ab_fp8.json

Input: 100 MMLU + 50 AlpacaEval prompts sampled with a fixed seed,
       each prompt with LLM's top-5 candidates (chat-formatted text, 5 per group).
"""
import argparse, json, math, requests


def query_rm(port, model, formatted_texts):
    resp = requests.post(
        f"http://localhost:{port}/classify",
        json={"model": model, "input": formatted_texts, "activation": False},
        timeout=30,
    )
    resp.raise_for_status()
    data = sorted(resp.json()["data"], key=lambda x: x.get("index", 0))
    return [d["probs"][0] for d in data]


def main():
    import os
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, choices=["bf16", "fp8"])
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--model", required=True, help="RM model path, must match the path used when starting the server")
    p.add_argument("--prompts_file", default="results/ab_prompts.json",
                   help="Pre-generated fixed prompts + 5-candidate set")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if not os.path.exists(args.prompts_file):
        raise FileNotFoundError(
            f"{args.prompts_file} does not exist. Run scripts/gen_ab_prompts.py first "
            f"to generate the fixed prompt set for A/B testing."
        )
    prompts = json.load(open(args.prompts_file))  # list[ {prompt_id, candidates: [5 chat-formatted texts]} ]
    out = []
    for item in prompts:
        scores = query_rm(args.port, args.model, item["candidates"])
        assert all(not math.isnan(s) and not math.isinf(s) for s in scores), \
            f"{args.label} RM returned NaN/Inf at prompt {item['prompt_id']}: {scores}"
        out.append({"prompt_id": item["prompt_id"], "scores": scores})

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Wrote {len(out)} entries to {args.out}")


if __name__ == "__main__":
    main()
```

**`scripts/compare_rm_ab.py`** (reads two JSONs, outputs statistics):

```python
"""Analyze BF16 vs FP8 reward comparison collected sequentially."""
import argparse, json
import numpy as np
from scipy.stats import pearsonr, spearmanr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bf16", required=True)
    p.add_argument("--fp8",  required=True)
    args = p.parse_args()

    bf16 = json.load(open(args.bf16))
    fp8  = json.load(open(args.fp8))
    # Align by prompt_id
    bf16_map = {x["prompt_id"]: x["scores"] for x in bf16}
    fp8_map  = {x["prompt_id"]: x["scores"] for x in fp8}
    common = sorted(set(bf16_map) & set(fp8_map))
    assert common, "No aligned prompt_ids found"

    import math
    all_bf16, all_fp8 = [], []
    top1_agree = 0
    spearman_pass = 0
    for pid in common:
        b, f = bf16_map[pid], fp8_map[pid]
        all_bf16.extend(b)
        all_fp8.extend(f)
        if int(np.argmax(b)) == int(np.argmax(f)):
            top1_agree += 1
        sp = spearmanr(b, f).correlation
        # NaN occurs when all 5 scores are identical (constant input); treat as perfect agreement and count as pass
        if math.isnan(sp) or sp > 0.8:
            spearman_pass += 1

    pearson = pearsonr(all_bf16, all_fp8).statistic
    n = len(common)
    print(f"=== A/B Report (n={n} prompts × 5 candidates = {len(all_bf16)} data points) ===")
    print(f"Pearson (absolute reward):              {pearson:.4f}      expected ≥ 0.99")
    print(f"Top-1 candidate agreement:              {top1_agree}/{n} ({top1_agree/n*100:.1f}%)   expected ≥ 95%")
    print(f"Per-group Spearman > 0.8 fraction:      {spearman_pass}/{n} ({spearman_pass/n*100:.1f}%)   expected ≥ 90%")
    print(f"BF16 reward median/max: {np.median(all_bf16):.4f} / {np.max(all_bf16):.4f}")
    print(f"FP8  reward median/max: {np.median(all_fp8):.4f} / {np.max(all_fp8):.4f}")
    drift = abs(np.median(all_fp8) - np.median(all_bf16)) / (abs(np.median(all_bf16)) + 1e-6)
    print(f"Median absolute drift: {drift*100:.1f}%   expected < 10%")


if __name__ == "__main__":
    main()
```

**`scripts/gen_ab_prompts.py`** (one-time generation of the fixed prompt set for A/B testing):

```python
"""
Sample 150 prompts from MMLU + AlpacaEval, each with chat-formatted text for
top-5 LLM candidates (5 candidates = top-5 tokens from LLM logits, each appended
to response_so_far + chat template).

Output JSON structure:
[
  {"prompt_id": "mmlu_anatomy_0", "candidates": [text1, text2, text3, text4, text5]},
  ...
]

Note: candidates should be the 5 candidates SIA would send to the RM at the first step
(i.e., LLM step 0's top-5 tokens appended to an empty response_so_far), ensuring
calibration data matches the deployment distribution.
Simplified version (for A/B reward value consistency only; does not require real top-5):
use 5 fixed candidate tokens (e.g., ' A', ' B', ' C', ' D', ' E') appended to the chat template.
"""
import argparse, json, random
from datasets import load_dataset
from transformers import AutoTokenizer

LLM = "/workspace/SIA/models/Qwen3-14B"
FIXED_CANDIDATES = [" A", " B", " C", " D", " E"]  # simplified version

def chat_format(tok, user, assistant_text):
    convs = [
        {"role": "user",      "content": user},
        {"role": "assistant", "content": assistant_text},
    ]
    text = tok.apply_chat_template(convs, tokenize=False)
    bos = tok.bos_token
    if bos and text.startswith(bos):
        text = text[len(bos):]
    return text

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/ab_prompts.json")
    p.add_argument("--n_mmlu", type=int, default=100)
    p.add_argument("--n_alpaca", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)
    out = []

    # 100 MMLU samples, 20 per subject across 5 subjects, more representative of the real 30-subject distribution
    # ⚠️ load_dataset requires internet access (downloaded to ~/.cache/huggingface on first run).
    # For offline environments, pre-download: `python -c "from datasets import load_dataset; load_dataset('cais/mmlu', 'all')"`
    # or prepare the JSON manually.
    MMLU_SUBJECTS = ["anatomy", "clinical_knowledge", "astronomy",
                     "high_school_mathematics", "philosophy"]
    per_subject = args.n_mmlu // len(MMLU_SUBJECTS)
    mmlu_samples = []
    try:
        for subj in MMLU_SUBJECTS:
            ds = load_dataset("cais/mmlu", subj, split="test")
            mmlu_samples += [(subj, ex) for ex in random.sample(list(ds), min(per_subject, len(ds)))]
    except Exception as e:
        print(f"⚠️ MMLU load failed (possibly a network/cache issue): {e}")
        print(f"   Collected {len(mmlu_samples)} samples so far, continuing with subsequent steps")
    for i, (subj, ex) in enumerate(mmlu_samples):
        q = ex["question"] + "\n" + "\n".join([f"{c}: {v}" for c, v in zip("ABCD", ex["choices"])])
        # Simplified: 5 candidates = same prompt + 5 fixed suffixes
        cands = [chat_format(tok, q, "Answer:" + s) for s in FIXED_CANDIDATES]
        out.append({"prompt_id": f"mmlu_{subj}_{i}", "candidates": cands})

    # 50 AlpacaEval samples
    try:
        alpaca = load_dataset("tatsu-lab/alpaca_eval", split="eval")
        alpaca_sub = random.sample(list(alpaca), min(args.n_alpaca, len(alpaca)))
        for i, ex in enumerate(alpaca_sub):
            q = ex["instruction"]
            cands = [chat_format(tok, q, "Sure" + s + ".") for s in FIXED_CANDIDATES]
            out.append({"prompt_id": f"alpaca_{i}", "candidates": cands})
    except Exception as e:
        print(f"AlpacaEval load failed, skipping: {e}")

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"Wrote {len(out)} prompts → {args.out}")

if __name__ == "__main__":
    main()
```

**Sequential run procedure**:
```bash
# 0. One-time generation of fixed prompts set (results/ab_prompts.json)
python scripts/gen_ab_prompts.py   # see draft above; outputs ~150 prompts × 5 candidates

# 1. Start BF16 RM
nohup vllm serve /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --runner pooling --convert classify ... --port 8001 &
python scripts/collect_rm_scores.py --label bf16 --port 8001 \
    --model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --out results/ab_bf16.json

# 2. Kill the BF16 RM (including EngineCore subprocesses; previous experience showed that
#     pkill matching only "vllm serve" misses the spawned VLLM::EngineCore subprocess)
pkill -9 -f "VM-Qwen3-4B-merged-for-vllm|VLLM::EngineCore" 2>/dev/null
sleep 5
# Verify GPU memory is released (H200 at idle should drop to ~a few hundred MB;
# if GB-level usage remains, a subprocess is still alive)
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# 3. Start FP8 RM (same port 8001)
nohup vllm serve /workspace/SIA/models/VM-Qwen3-4B-merged-fp8-dynamic ... --port 8001 &
python scripts/collect_rm_scores.py --label fp8 --port 8001 \
    --model /workspace/SIA/models/VM-Qwen3-4B-merged-fp8-dynamic \
    --out results/ab_fp8.json

# 4. Compare
python scripts/compare_rm_ab.py --bf16 results/ab_bf16.json --fp8 results/ab_fp8.json
```

**Pass criteria (all must pass before proceeding to Phase 4)**:

| Metric | Threshold | Notes |
|---|---|---|
| Pearson correlation (absolute reward) | **≥ 0.99** | numerically near-equivalent |
| Top-1 candidate agreement | **≥ 95%** | argmax unchanged; SIA intervention decisions are consistent |
| Per-group Spearman > 0.8 fraction | **≥ 90%** | within-group ranking among 5 candidates is consistent. Note: with n=5, Spearman > 0.8 corresponds to p ≈ 0.1, which is statistically weak; but 5 candidates is too few to enforce stricter criteria — treat this as a weak signal; the primary signals are Pearson and Top-1 |
| Reward median absolute drift | **\|Δ\| < 30%** | scale is roughly stable. SIA internally applies `rm_scores - mean()` which cancels absolute offset, so 30% is already conservative. **If this fails but Pearson + Top-1 + Spearman all pass, it can still be approved** (note: Pearson already captures scale consistency more strictly; median drift is just a simple indicator) |
| End-to-end MMLU 100Q accuracy difference | **\|Δ\| < 1%pt** | overall quality does not drop |

Any failure → see §6 rollback plan.

### Phase 4: Performance Benchmark

Run a complete 600-question MMLU evaluation. **Reuse the RM instance started in Phase 2** (already running on 8001 and sanity-checked; no need to restart). **Only need to start LLM + eval**. Both use nohup + log redirect to `/tmp/` — so even if the interactive terminal (e.g., Claude session) exits, background processes continue running:

```bash
TS=$(date +%Y%m%d%H%M)

# === Prerequisite: Confirm Phase 2 RM is still alive ===
# RM log path is the one used when started in Phase 2 (in this run: /tmp/vllm_rm_fp8d_phase2.txt)
ps aux | grep -E "vllm_serve_with_token_ids|VM-Qwen3-4B-merged-fp8-dynamic" | grep -v grep
curl -sf http://localhost:8001/health && echo "RM 8001 OK" || { echo "RM not running; go back to Phase 2 and start RM first"; exit 1; }

# === 1) LLM server (C1 auto-enabled + C2 enabled via flag) ===
nohup python src/sia_vllm_server.py \
      --llm /workspace/SIA/models/Qwen3-14B \
      --rm_url http://localhost:8001 \
      --rm_backend vllm \
      --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-fp8-dynamic \
      --llm_gpu_mem 0.55 \
      --topk 5 --weight 1.0 --entropy_threshold 1.0 \
      --use_token_ids \
      --host 0.0.0.0 --port 8000 \
      > /tmp/sia_llm_fp8d_${TS}.log 2>&1 &
echo "LLM PID: $!"

# Wait for LLM ready (startup takes ~90-120s: weight load 60s + torch.compile 70s + CUDA graph 5s)
LLM_LOG=/tmp/sia_llm_fp8d_${TS}.log
until grep -q "Application startup complete\|Uvicorn running" $LLM_LOG 2>/dev/null \
   || grep -qE "Traceback|Error in|raise " $LLM_LOG 2>/dev/null
do sleep 5; done

# === 2) MMLU eval ===
mkdir -p results
nohup python eval/mmlu_eval.py \
      --base_url http://localhost:8000/v1 \
      --model /workspace/SIA/models/Qwen3-14B \
      --output results/test_vllmrm_fp8d_${TS}.json \
      --limit 20 \
      > /tmp/sia_eval_fp8d_${TS}.log 2>&1 &
echo "EVAL PID: $!"
echo "TS=$TS"
echo "LLM + eval running in background; safe to exit current session"
```

Check results after completion (note RM log path differs from LLM/eval because RM was started in Phase 2):
```bash
# Check final throughput + accuracy
tail -20 /tmp/sia_eval_fp8d_${TS}.log
# Check profiling aggregate (see if http_post dropped)
grep "SIA-pf-summary" /tmp/sia_llm_fp8d_${TS}.log | tail -1
# Check RM-side prefix cache hit rate (RM log is from Phase 2)
grep "Prefix cache hit rate" /tmp/vllm_rm_fp8d_phase2.txt | tail -3
```

**Cold-start scenario** (Phase 2 RM was killed or machine was rebooted): go back to §Phase 2 to start RM first, then return here. Or start all 3 services at once:

```bash
# === Full cold start: start all 3 processes together (backup option) ===
TS=$(date +%Y%m%d%H%M)

# 1) RM
nohup python scripts/vllm_serve_with_token_ids.py serve \
      /workspace/SIA/models/VM-Qwen3-4B-merged-fp8-dynamic \
      --runner pooling --convert classify \
      --enable-prefix-caching --no-enable-chunked-prefill \
      --gpu-memory-utilization 0.3 \
      --max-model-len 2048 --port 8001 \
      > /tmp/sia_vllm_rm_fp8d_${TS}.log 2>&1 &
RM_LOG=/tmp/sia_vllm_rm_fp8d_${TS}.log
until grep -q "Application startup complete" $RM_LOG || \
      grep -qE "Traceback|Error in|raise " $RM_LOG; do sleep 5; done

# 2) LLM (same as above), 3) eval (same as above)
# ……
```

**Expected readings**:

| Metric | C1+C2 baseline | A2 (this plan) expected | Improvement |
|---|---:|---:|---:|
| `[SIA-pf-summary] http_post` p50 | 53.89ms | **~38-42ms** | -25% |
| `[SIA-pf-summary] total` p50 | 59.84ms | **~45-50ms** | -20% |
| Overall throughput | 29.43 tok/s | **~33-36 tok/s** | **+12-22%** |
| Accuracy | 72.0% | 71-73% | within noise |

If http_post does not drop significantly (< 5%), FP8 GEMM acceleration is not actually taking effect and diagnosis is needed (see §6.3).

---

## 4. Risk Matrix + Rollback Plan

| # | Risk | Probability | Detection Point | Mitigation |
|---|---|---|---|---|
| R1 | `llm-compressor` fails to load `Qwen3ForSequenceClassification` | Medium | Step 1.2 oneshot errors | Switch to loading with `AutoModelForCausalLM` + manually add score head; or fork a compatible version |
| R1b | `QuantizationModifier(targets=...)` API changed (str vs list or requires regex) | Low-Medium | oneshot startup raises TypeError or ValueError | We already use list form `targets=["Linear"]`; if still errors, try regex form `targets="re:.*\\.proj$"` or adjust per the new API docs |
| R2 | `ignore=["score"]` has no effect; score head gets quantized | Low | Step 1.3 verification script shows `score.weight` is FP8 | Change ignore syntax (e.g., `["model.score"]`, `["^score$"]`); or manually overwrite score weights to BF16 after quantization |
| R3 | vLLM reports "unsupported quantization" when loading FP8_DYNAMIC checkpoint | Medium | Phase 2 startup fails | Try explicit `--quantization compressed-tensors` flag first; if that fails, upgrade vLLM 0.10.x → 0.11+, **but vLLM upgrade is a major change** — `src/sia_vllm_RM.py` uses internal APIs from `vllm.v1.sample.logits_processor.interface` (`BatchUpdate / MoveDirectionality`), so compatibility must be validated; as a last resort, fall back to static FP8 scheme |
| R4 | Pearson < 0.95 | Medium-Low | Phase 3 fail | Expand ignore list (add last 2-3 transformer blocks); or try `FP8` static scheme |
| R5 | CUDA graph capture fails on FP8 path | Low | Startup log shows `capture_size`-related error | Add `--enforce-eager` to disable graph (5-10% speed loss but functional) |
| R6 | Reward scale drifts significantly (\|Δ\| > 20%) | Low | Phase 3 shows BF16 vs FP8 reward median difference > 20% | Adjust `--weight` to adapt to new scale; or expand ignore list |
| R7 | http_post p50 does not drop; FP8 acceleration ineffective | Medium | Phase 4 shows http_post ≈ 54ms | Check whether vLLM is actually forwarding on the FP8 path (use nsys profile or `torch.cuda.is_current_stream_capturing`); vLLM upgrade may be needed |
| R8 | Phase 1 oneshot OOM (4B model + activation storage) | Low | OOM during oneshot | Reduce `n_samples`, reduce `max_seq_length`; or run on multiple GPUs |
| R9 | Quantization output partially written then interrupted | Low | dst directory incomplete | Delete dst and rerun |
| R10 | Incompatible with wrapper `vllm_serve_with_token_ids.py` | Very Low | Phase 2 startup errors | The wrapper only patches Pydantic schema, unrelated to FP8 loading logic; no theoretical conflict |

### 4.1 Overall Rollback Sequence

```
Phase 3 fail (Pearson < 0.95)
  ↓
  Try: scheme="FP8" (static, with calibration)
  ↓ fail
  Try: expand ignore list (add last 4 layers)
  ↓ fail
  Abandon A2, revert to BF16 G3 configuration (29.65 tok/s is the current achievable optimum)

Phase 4 fail (http_post does not drop)
  ↓
  Diagnose: is vLLM actually using FP8 kernels?
  ↓ No
  Upgrade vLLM 0.10.1.1 → 0.11+
  ↓ Still failing
  Abandon A2
```

### 4.2 How to Confirm "FP8 Is Actually Running on GPU"

After vLLM starts, trigger a forward pass with a long prompt, then in a separate terminal:

```bash
nvidia-smi dmon -s u  # watch SM utilization
# Or more precisely:
nsys profile -t cuda --output=fp8_check.qdrep python -c "import requests; requests.post(...)"
nsys stats fp8_check.qdrep | grep -i "fp8\|e4m3"
```

FP8 GEMM kernels (e.g., `cutlass_fp8_e4m3_gemm`) should be visible in the output.

---

## 5. Precise Comparison with G4's Failure

| Dimension | G4 (failed, -7%) | A2 (this plan) |
|---|---|---|
| Quantization path | vLLM runtime quant: `--quantization fp8` | Offline `llm-compressor` producing compressed-tensors checkpoint |
| Activation scale | **Uncalibrated**, defaults to 1.0 → triggers degraded path | **Dynamic per-token**, computed at each forward, always accurate |
| Weight scale | Computed at runtime, per-tensor | Computed offline, per-channel (finer granularity, more accurate) |
| KV cache | `--kv-cache-dtype fp8`, 3 WARNINGs | **Kept as BF16**, deliberately bypassed |
| score head | Quantized together (reward precision degraded) | **Explicitly ignored** (critical!) |
| Startup WARNINGs | 3 "uncalibrated scale" warnings | **0 warnings** (ideal state) |
| Measured vs BF16 | -7% (slower) | Expected +12-22% (pending measurement) |

---

## 6. Timeline and Milestones

| Milestone | Work | Duration | Pass Criteria |
|---|---|---|---|
| M1 | Write + run `scripts/quantize_rm_fp8_dynamic.py` | 30min | dst directory produced, config correct |
| M2 | Phase 2 startup + sanity check | 10min | service running, curl test shows reasonable reward values |
| M3 | Phase 3 quality A/B (150 prompts) | 30min | Pearson ≥ 0.99, Top-1 ≥ 95% |
| M4 | Phase 4 full 600-question MMLU | 4-5h | overall tok/s ≥ 32 (**+8% over G3**) |
| M5 | Documentation: append new row "G5: + A2 FP8_DYNAMIC" to the **§5 Full Experiment Review** section of `doc/vllm-rm-followup-optimizations.md`; also append G5 row to §3 results table (with tok/s / acc / intervention rate); update the §6 conclusion from "FP8 has no benefit on shared GPU" to "Uncalibrated FP8 (G4) has no benefit; calibrated FP8_DYNAMIC (G5) yields +XX%". Also add a "2026-XX-XX — G5: FP8_DYNAMIC" section to `exp/README.md` with log file links. | 30min | data + conclusions clear, PR-ready |

**Go/No-go checkpoint**: If M3 fails → activate rollback plan, do not run M4.

---

## 7. References

- The core user question and static vs dynamic explanation: this document §0
- Bottleneck analysis and A2 route proposal: [`doc/profiling-bottleneck-analysis.md`](profiling-bottleneck-analysis.md) §5.1
- Detailed root cause of G4's failure: [`doc/vllm-rm-followup-optimizations.md`](vllm-rm-followup-optimizations.md) §4.4
- Analysis of C1+C2 near-zero measured net gains: [`doc/profiling-bottleneck-analysis.md`](profiling-bottleneck-analysis.md) §5.3
- `llm-compressor` official documentation: https://github.com/vllm-project/llm-compressor
- vLLM compressed-tensors integration: https://docs.vllm.ai/en/latest/quantization/auto_awq.html (FP8 uses the same compressed-tensors path)

---

## 8. What Is Out of Scope

Things explicitly **not done** to avoid confusion:

- Do not quantize the LLM (Qwen3-14B): this plan only touches the RM
- Do not touch KV cache quantization: avoid G4's degraded path
- Do not use static FP8 calibration: use dynamic to bypass distribution dependency
- Do not switch to a smaller RM (Qwen3-1.7B-VM): that changes quality, violating the constraint
- Do not modify entropy_threshold / topk: that changes quality
- Do not implement a stateful RM: reserved for a future 2-GPU environment

---

## 9. Measured Results (2026-05-27): **A2 Route Validated as Ineffective**

After completing A2 Phases 1-4, **FP8_DYNAMIC did not yield end-to-end speedup under SIA's current workload**. This section documents the measured data and root cause analysis.

### 9.1 Measured Data (A2 vs Historical Baseline)

Full 600-question MMLU:

| Experiment | tok/s overall | tok/s median | Accuracy | Notes |
|---|---:|---:|---:|---|
| G3 BF16 baseline | 29.65 | 33.84 | 73.7% | C0 baseline |
| G4 old FP8 (runtime + FP8 KV) | 28.08 | 31.82 | 73.2% | Failed: uncalibrated KV scale degrades |
| C1+C2 (BF16 + client-side optimizations) | 29.43 | n/a | 72.0% | Protocol-layer optimization |
| **A2 FP8_DYNAMIC (this run)** | **29.81** | **34.53** | **71.50%** | Nearly identical to G3/C1+C2 |

A2 vs C1+C2 (BF16): **+1.3%** (within noise)
A2 vs G3 (BF16): **+0.5%** (within noise)
Accuracy -2.2%pt is within 95% CI of ±3.5%pt for 600 questions

**SIA processor profiling** (`[SIA-pf-summary]` @39800 calls):

| Phase | C1+C2 (BF16) | A2 (FP8_DYNAMIC) | Delta |
|---|---:|---:|---:|
| format_chat p50 | 0.01ms | 0.02ms | ≈ 0 |
| tokenize_client p50 | 6.46ms | 12.60ms | **+6.14ms** |
| http_post p50 | 53.89ms | 55.62ms | +1.73ms |
| parse_response p50 | 0.12ms | 0.17ms | ≈ 0 |
| **total p50** | **59.84ms** | **67.80ms** | +7.96ms |

The tokenize_client regression is a side effect of the environment upgrade (installing llmcompressor upgraded transformers from an older version to 4.55.2, with accelerate and other packages pulled in transitively), unrelated to FP8. Even discounting these 6ms, http_post shows no meaningful reduction.

### 9.2 Root Cause (Confirmed from Source Code)

The initial suspicion was that vLLM 0.10.1.1 was not actually dispatching FP8 kernels (falling back to BF16 dequant path). **This was ruled out after reading the source code**:

```
vLLM code path (vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8.py):

CompressedTensorsW8A8Fp8.apply_weights()
  → self.fp8_linear.apply()                              # Fp8LinearOp
    → dispatch_w8a8_scaled_mm(cutlass_fp8_supported=True)  # returns True for sm_89+
      → cutlass_w8a8_scaled_mm                            # ★ real cutlass FP8 kernel
        → ops.cutlass_scaled_mm(qinput, weight, ...)      # vLLM C++ extension
```

H200 sm_90 **confirmed** to use the cutlass FP8 GEMM path. FP8 is genuinely running on GPU.

### 9.3 The Real Bottleneck: Workload Too Small; GEMM Compute Is Not the Bottleneck

Actual GPU workload per RM call in SIA:

```
5 candidates × ~3 token suffix (the differing tokens after 99% prefix cache hit)
= Linear input shape per call ≈ (batch=15, hidden=2560)
= one q_proj GEMM: M=15, K=2560, N=4096
= FLOPs per GEMM ≈ 250M
```

For H200 FP8 tensor cores (peak ~3000 TFLOPS), this **"tall and thin" matrix** (M=15 << K=2560):

1. **Is entirely in the memory-bound regime**: compute units are underutilized; the bottleneck is data movement (KV, weights, activations)
2. **FP8 vs BF16 is 1.5-2x faster in the compute-bound region; but in the memory-bound region the difference is nearly zero** — because the constraint is memory bandwidth, not compute
3. **Additionally**, FP8_DYNAMIC requires `quant_fp8(input)` before each Linear at runtime, **adding one kernel launch + memory pass per layer**, which negates the minor GEMM speedup

Estimated kernel count per RM call:

```
36 layers × 7 Linear/layer = 252 GEMMs
+ 252 quant_fp8 calls (FP8_DYNAMIC activation quantization overhead)
+ attention kernels + softmax + other
≈ ~500 kernel launches / RM call

Overhead per kernel launch ~10μs → ~5ms purely from launches
Actual compute per GEMM is < 0.1ms due to memory-bound regime
→ launch + memory dominate; the "compute density" part that FP8 optimizes is not the bottleneck
```

### 9.4 Corrected Conclusion: The Fundamental Problem with the A2 Route

**It is not a vLLM implementation issue, nor a FP8 quantization failure — the SIA workload's GEMM shape is simply not in the sweet spot for FP8 tensor cores.**

| Optimization Direction | Effective on SIA's Current Workload | Reason |
|---|---|---|
| **A2 FP8_DYNAMIC** (this plan) | **Validated as ineffective** | Workload is not in compute-bound region; FP8 compute gains cannot materialize |
| INT4 / INT8 / AWQ | Also ineffective | Also relies on GEMM compute-bound assumption |
| Marlin / other tensor-core quantization | Also ineffective | Same |
| **B1 GPU split deployment** | Potentially effective | Eliminates GPU resource contention and cache pollution; not a compute optimization |
| **B2 Stateful RM** (async fix + KV reuse) | Genuinely effective | Eliminates prefix re-forward, **changes the workload shape** (reduces to computing only 5×1 token suffix) |
| **C3 In-process RM** (eliminate HTTP) | Genuinely effective | Removes fixed per-call protocol overhead; saves ~5-10ms per RM call |
| Increase `entropy_threshold` (fewer interventions) | Effective but sacrifices quality | Outside current constraints |
| Switch to smaller RM (4B → 1.7B) | Effective but sacrifices quality | Outside current constraints |

### 9.5 Hard Recommendations for Next Steps

**The ceiling of the A2 route (and any "quantization-accelerated GEMM" route) has been reached — hardware-level acceleration for SIA's current workload is essentially maxed out.** Remaining directions must operate at the **architecture level**:

1. **B1 multi-GPU deployment** (doc/parallel-decoding-design.md §Tier 1) — lowest hardware cost; potential +15-25%
2. **B2 Stateful RM** (same doc §2B) — high engineering cost (~1300 lines), but can **genuinely change the GEMM shape**; potential +50%+
3. **C3 In-process RM** (same doc §C3) — run RM and LLM in the same process, bypassing HTTP/JSON; ~10-15ms saved per call

The 4.2GB FP8 checkpoint produced by A2 is preserved at `/workspace/SIA/models/VM-Qwen3-4B-merged-fp8-dynamic/` — **save it for when SIA's workload becomes compute-bound (e.g., B2 increases batch size); FP8 gains will materialize then**.

### 9.6 Documented Anti-Pattern

Recorded here for future reference:

> If you want to "accelerate SIA by quantizing the RM" — **save yourself the time, don't go down this path**. SIA's INTERVENE workload processes very few tokens per call (5 candidates × ~3 tokens), so GEMM is memory-bound. Every approach that depends on compute-bound GEMM acceleration (FP8/INT8/INT4/AWQ/Marlin) is **ineffective** here.
>
> The only ways to accelerate "through RM-side optimization" are: **change the RM call's workload shape** (B2 makes each forward compute fewer tokens), or **eliminate fixed per-call overhead** (C3 in-process).
