# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SIA (Sparse Inference-time Alignment)** — a per-token intervention system for LLM inference. It wraps vLLM with a custom `LogitsProcessor` that scores candidate tokens using a Reward Model (RM) at each generation step, biasing token selection toward higher-reward outputs.

## Running the Project

**Dependencies (install manually — no requirements.txt):**
```bash
pip install fastapi uvicorn "peft<0.15.0" vllm transformers torch
```

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
