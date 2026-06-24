# SIA vLLM Server

SIA (Sparse Inference-time Alignment) per-token intervention inference service, compatible with the OpenAI Chat API and the 0g-serving-broker.

Paper: [Inference-time Alignment via Sparse Junction Steering](https://arxiv.org/pdf/2602.21215) (Runyi Hu et al.)

---

## Work Beyond the Paper

The paper provides the core algorithm (SIA LogitsProcessor + entropy gate) and Value Model checkpoints, but the inference code is a hand-written transformers-based forward loop and does not cover production deployment. This project adds:

**1. Production-grade vLLM integration**
Rewritten as a vLLM v1 `LogitsProcessor`, integrated with the async batching engine, supporting concurrent requests and true token-level streaming (rather than generating the full response first and then simulating streaming).

**2. OpenAI-compatible HTTP service**
Full `/v1/chat/completions` implementation with streaming SSE, directly compatible with OpenAI SDK clients and the 0g broker dual-route setup.

**3. b2 inproc RM backend (core speedup)**
Moves the RM from a standalone HTTP server to an in-process nested vLLM `LLM()`, using prefix caching for incremental decoding (prefix KV reuse). Designed the `RMClient` state machine (`new_session / fix_a_token / score_candidates`). Measured results:
- VL-30B: RM call p50 71ms → 11ms (**6.4×**), end-to-end **1.46× speedup**, AlpacaEval Skywork reward statistically equivalent (p=0.73)
- 0GM-35B: end-to-end **~1.5× speedup** vs HTTP backend

**4. Heterogeneous tokenizer support**
The paper assumes LLM and RM share the same vocabulary; this project implements a cross-tokenizer bridge (decode→text→re-encode) to support mismatched vocabularies (e.g. 0GM-35B 248k vocab + VM-Qwen3-4B 152k vocab).

**5. Per-request parameter override**
`sia_weight / sia_topk / sia_entropy_threshold` can be individually overridden in each HTTP request without restarting the service.

**6. Critical bug fix: `repetition_penalty` bias**
Discovered that `repetition_penalty=1.3` enabled by default stacked with the vLLM sampler order, causing AlpacaEval Skywork score +3.09 vs paper's +10.30. After fixing (`repetition_penalty=1.0`), three configurations (Qwen3-14B / VL-30B / 14B reproduction) show SIA gain statistically equivalent to the paper (p=0.43).

**7. Multi-model validation**
The paper only validates on Qwen3-14B; this project completes AlpacaEval / MMLU end-to-end evaluation on Qwen3-VL-30B (7.5× weak-to-strong) and 0GM-1.0-35B-A3B (in-house model), confirming SIA gain transfers.

**8. vLLM version compatibility engineering**
Systematically investigated differences in PIECEWISE cudagraph behavior across vllm 0.17 / 0.18 / 0.19, determined usable versions per model (VL-30B → 0.17.1, 0GM-35B → 0.18.0), and provided independent venv setup scripts and requirements files for each.

---

## File Overview

| File | Description |
|------|-------------|
| `src/sia_vllm_RM.py` | Core SIA logic: per-token logits intervention, calls RM server for scoring via HTTP |
| `src/sia_vllm_server.py` | FastAPI HTTP service wrapping `sia_vllm_RM.py` |
| `src/sia_rm_server.py` | Standalone RM scoring service, supports hot-swapping RM / LoRA |

## Dependency Installation

```bash
# 1. Install runtime dependencies (vllm / torch / transformers / fastapi etc., 14 packages total,
#    pinned to the validated working set; see comments at the top of requirements.txt)
pip install -r requirements.txt

# 2. Install sia_rm as an editable package
#    Required — the b2 backend (in-process RM) uses the pyproject.toml
#    `vllm.general_plugins` entry-point to auto-register Qwen3WithScoreForCausalLM
#    in the vLLM EngineCore subprocess.
pip install -e .
```

**Notes**:
- Validated environment: Python 3.12 + CUDA 12.x, vLLM 0.10.1.1. For different CUDA versions, first install the matching torch wheel from [pytorch.org](https://pytorch.org/get-started/locally/), then run `pip install -r requirements.txt`.
- `peft` must be `>=0.14.0,<0.15.0` (required by the VM-Qwen3-4B LoRA loading path).
- `requirements.txt` includes `scipy` for validation scripts; comment it out if not needed.

## Starting the Service

SIA consists of two independent processes: the **RM server** for scoring, and the **LLM server** for generation that calls the RM server.

### Step 1: Start the RM server

**Without LoRA:**

```bash
python src/sia_rm_server.py \
  --rm /workspace/SIA/models/Qwen3-1.7B-Base \
  --rm_device cuda:0 --port 8001
```

**With LoRA (ValueModel checkpoint):**

```bash
python src/sia_rm_server.py \
  --rm      /workspace/SIA/models/Qwen3-1.7B-Base \
  --rm_lora /workspace/SIA/models/SIA-checkpoints/VM-Qwen3-1.7B-Base \
  --rm_device cuda:0 --port 8001
```

### Step 2: Start the LLM server

```bash
python src/sia_vllm_server.py \
  --llm    /workspace/SIA/models/Qwen3-1.7B-Base \
  --rm_url http://localhost:8001 \
  --host 0.0.0.0 --port 8000 \
  --llm_gpu_mem 0.3 --topk 5 --weight 1.0
```

## CLI Parameters

### LLM server (`sia_vllm_server.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--llm` | required | LLM model path |
| `--rm_url` | `http://localhost:8001` | RM server address |
| `--llm_gpu_mem` | `0.5` | vLLM GPU memory utilization fraction |
| `--topk` | `10` | Number of top-k candidate tokens to score per step |
| `--weight` | `1.0` | RM score additive weight |
| `--entropy_threshold` | None | Entropy threshold below which intervention is skipped (None = always intervene) |
| `--max_model_len` | `4096` | vLLM maximum sequence length |
| `--host` | `0.0.0.0` | HTTP listen address |
| `--port` | `8000` | HTTP listen port |
| `--model_id` | LLM basename | Model name exposed externally |

### RM server (`sia_rm_server.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rm` | required | RM base model path |
| `--rm_lora` | None | RM LoRA (ValueModel) checkpoint path |
| `--rm_device` | `cuda:0` | RM device |
| `--host` | `0.0.0.0` | HTTP listen address |
| `--port` | `8001` | HTTP listen port |

## API Endpoints

### Health check

```bash
curl http://localhost:8000/health
```

### List models

```bash
curl http://localhost:8000/v1/models
```

### Non-streaming chat

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

### Per-request SIA weight

`sia_weight` is an SIA extension field (does not conflict with the OpenAI API) for overriding the server-startup `--weight` default on a per-request basis. If omitted, the default value is used.

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50,
    "sia_weight": 2.0
  }'
```

### Streaming chat (SSE)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50,
    "stream": true
  }'
```

### Broker-compatible routes

The broker uses both of the following routes (both are supported):

- `POST /v1/chat/completions`
- `POST /chat/completions`

## RM Hot-Swap

Reload the RM without restarting any service by calling the RM server's `/reload` endpoint:

```bash
curl -X POST http://localhost:8001/reload \
  -H "Content-Type: application/json" \
  -d '{"rm": "/path/to/new/rm", "rm_lora": "/path/to/new/lora"}'
```

The endpoint is synchronous; once it returns the RM server has loaded the new model and all subsequent scoring requests will use it.

Query the current RM server status:

```bash
curl http://localhost:8001/status
```
