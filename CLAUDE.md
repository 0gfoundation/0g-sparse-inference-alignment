# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Communication Language

**Use English only.** Technical terms (commands, parameter names, model names, etc.) are already in English. Japanese, Korean, and other non-English languages are not permitted.

## Project Overview

**SIA (Sparse Inference-time Alignment)** — a per-token intervention system for LLM inference. It wraps vLLM with a custom `LogitsProcessor` that scores candidate tokens using a Reward Model (RM) at each generation step, biasing token selection toward higher-reward outputs.

## Running the Project

**Dependencies — pick the venv that matches your main LLM:**

| Main LLM | Recommended venv | vllm | SIA path | Notes |
|----------|-----------------|------|----------|-------|
| Qwen3-14B | qwen14b | 0.10.1.1 | `--rm_backend b2` (inproc) | Original baseline |
| Qwen3-VL-30B-A3B-Instruct | **vl30b-fast** | **0.17.1** | `--rm_backend b2` (inproc) | Fastest, 1.46× ([doc](doc/vl30b-b2-inproc-speedup-20260605.md)) |
| Qwen3-VL-30B-A3B-Instruct | 0gm35b-http | 0.19.0 | `--rm_backend vllm` (HTTP) | Fallback |
| 0GM-1.0-35B-A3B | **0gm35b-b2** | **0.18.0** | `--rm_backend b2` (inproc) | Fastest, ~1.5× vs HTTP ([doc](doc/0gm-35b-b2-inproc-speedup-20260609.md)); requires `SIA_RM_CUDAGRAPH=none` |
| 0GM-1.0-35B-A3B | 0gm35b-http | 0.19.0 | `--rm_backend vllm` (HTTP) | Fallback (b2 fails entirely on 0.19) |

```bash
# Setup (one-shot)
scripts/setup_venv_qwen14b.sh        /path/to/venv-qwen14b
scripts/setup_venv_vl30b_fast.sh     /path/to/venv-vl30b-fast
scripts/setup_venv_0gm35b_b2.sh      /path/to/venv-0gm35b-b2
scripts/setup_venv_0gm35b_http.sh    /path/to/venv-0gm35b   # HTTP fallback
```

Each script: creates a fresh venv → `pip install -r requirements/<config>.txt` → `pip install -e .` (registers the `sia_rm` vllm plugin via `pyproject.toml`).

Requirements files live in `requirements/`:
- `base.txt` — shared HTTP / eval / scoring deps
- `qwen14b-b2-inproc.txt` — vllm 0.10.1.1 + torch 2.7.1 + transformers 4.55.2
- `vl30b-b2-inproc.txt` — vllm 0.17.1 + transformers 4.57.6
- `0gm35b-b2-inproc.txt` — vllm 0.18.0 + transformers 4.57.6 (b2 inproc for 0GM-35B)
- `0gm35b-or-vl30b-http.txt` — vllm 0.19.0 + transformers 4.57.6

> **Do not mix** — vllm 0.10 / 0.17 / 0.18 / 0.19 pin different torch versions; each must use its own isolated venv. The root `requirements.txt` is for the qwen14b default configuration only.

**CLI mode (no HTTP server):**
```bash
python src/sia_vllm_RM.py \
  --llm /path/to/llm \
  --rm /path/to/rm \
  --rm_device cuda:0 \
  --llm_gpu_mem 0.3 \
  --topk 5 --weight 1.0 \
  --max_tokens 128 --temperature 0.7
```

**HTTP server (OpenAI-compatible API):**
```bash
# Without LoRA
python src/sia_vllm_server.py \
  --llm /path/to/llm --rm /path/to/rm \
  --host 0.0.0.0 --port 8000 \
  --llm_gpu_mem 0.3 --topk 5 --weight 1.0

# With LoRA/ValueModel
python src/sia_vllm_server.py \
  --llm /path/to/llm --rm /path/to/rm \
  --rm_lora /path/to/lora/checkpoint \
  --host 0.0.0.0 --port 8000 \
  --llm_gpu_mem 0.3 --topk 5 --weight 1.0
```

**vLLM-backed RM (recommended — 35% faster than pytorch RM and more stable):**

First, perform a one-time conversion of the RM to a vLLM-compatible format:
```bash
python scripts/convert_rm_for_vllm.py \
  --rm /path/to/qwen3-base \
  --rm_lora /path/to/lora_and_token_reward_head \
  --output /path/to/merged-for-vllm
```

Start the vLLM RM server + SIA LLM server (see `doc/vllm-rm-backend.md` for details):
```bash
# RM
vllm serve /path/to/merged-for-vllm --runner pooling --convert classify \
  --enable-prefix-caching --gpu-memory-utilization 0.3 --port 8001 &

# LLM with --rm_backend vllm
python src/sia_vllm_server.py --llm /path/to/llm \
  --rm_url http://localhost:8001 \
  --rm_backend vllm \
  --rm_model /path/to/merged-for-vllm \
  --topk 5 --weight 1.0 --entropy_threshold 1.0 &
```

**0GM-35B b2 inproc (recommended — ~1.5× faster than HTTP):**

```bash
SIA_RM_CUDAGRAPH=none \
SIA_RM_MULTIPROCESS=0 \
python src/sia_vllm_server.py \
  --llm /path/to/0GM-1.0-35B-A3B \
  --rm_backend b2 \
  --rm_model /path/to/VM-Qwen3-4B-merged-for-vllm \
  --rm_b2_gpu_mem 0.13 --llm_gpu_mem 0.75 \
  --topk 10 --weight 1.0 --entropy_threshold 1.0 \
  --max_model_len 32768 --mamba_cache_mode align --port 8000
```

**Testing the server:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'

# Streaming
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50, "stream": true}'
```

There is no test suite.

## References

### Paper

**"Inference-time Alignment via Sparse Junction Steering"**
Runyi Hu et al. — [https://arxiv.org/pdf/2602.21215](https://arxiv.org/pdf/2602.21215)

Core idea: rather than intervening at every decoding step, **only intervene at high-entropy critical decision points (junctions)**. An intervention rate of 20–80% of tokens achieves or exceeds the effect of full dense steering while reducing computational overhead by up to 6×. The `--entropy_threshold` parameter directly corresponds to this paper's sparse intervention strategy.

### Experimental Code

Original experimental code (Value Model training, evaluation scripts, etc.):
[https://github.com/hurunyi/SIA](https://github.com/hurunyi/SIA)

### Pretrained Value Models

Pre-trained Value Model checkpoints (LoRA format, for use with `--rm_lora`):
[https://huggingface.co/Runyi-Hu/SIA/tree/main](https://huggingface.co/Runyi-Hu/SIA/tree/main)

| Checkpoint | Base Model |
|------------|------------|
| `VM-Qwen3-0.6B-Base` | Qwen3-0.6B-Base |
| `VM-Qwen3-1.7B-Base` | Qwen3-1.7B-Base |
| `VM-Qwen3-4B-Base` | Qwen3-4B-Base |
| `VM-Skywork-Reward-V2-Llama-3.2-1B` | Llama-3.2-1B |
| `VM-Skywork-Reward-V2-Llama-3.2-3B` | Llama-3.2-3B |

Pass the corresponding checkpoint directory path to `--rm_lora`, and point `--rm` at the corresponding base model.

## Architecture

### Files

| File | Purpose |
|------|---------|
| `src/sia_vllm_RM.py` | Core SIA logic: `SIALogitsProcessor`, `_ValueModelWrapper`, CLI entry point |
| `src/sia_vllm_server.py` | FastAPI HTTP server wrapping `sia_vllm_RM.py`, OpenAI-compatible API |

### Per-Token Intervention Flow

1. vLLM calls `SIALogitsProcessor.apply()` after computing raw logits for each token position
2. Processor extracts top-k candidate token IDs
3. Optionally skips intervention if logit entropy is below `--entropy_threshold` (model is confident)
4. Decodes each candidate token appended to the current generation so far
5. Formats the full conversation (user prompt + partial response + candidate) using RM tokenizer
6. RM model scores all candidates in a single batch
7. Scores are mean-normalized then multiplied by `--weight` and added to the corresponding top-k logits
8. vLLM samples from the modified distribution

### State Management

`SIALogitsProcessor` maintains per-request state (user content, accumulated output token IDs) via `update_state()`. This method receives batch delta updates from vLLM v1 engine — added/removed/reordered requests — and must be kept in sync with vLLM's internal request ordering.

### Reward Model Loading

- **Base RM only** (`--rm` without `--rm_lora`): loads model directly via `AutoModel.from_pretrained`
- **RM + LoRA** (`--rm` + `--rm_lora`): wraps in `_ValueModelWrapper` which combines base model + PEFT LoRA adapter + `token_reward_head` linear layer; exposes a `.logits` attribute expected by the scoring code

### HTTP Server

The server runs vLLM **synchronously** in a FastAPI async endpoint using `asyncio.get_event_loop().run_in_executor()`. Streaming is **simulated**: the full response is generated first, then chunked into SSE events. Both `/v1/chat/completions` and `/chat/completions` routes are exposed for broker compatibility.

### Key CLI Parameters

| Parameter | Description |
|-----------|-------------|
| `--topk` | Number of candidate tokens to score per step |
| `--weight` | RM score multiplier added to logits |
| `--entropy_threshold` | Skip intervention when entropy below this value (0 = always intervene) |
| `--llm_gpu_mem` | vLLM GPU memory fraction for LLM (leave remainder for RM on same device) |
| `--rm_device` | Device for RM model (e.g., `cuda:0`, `cuda:1`) |
