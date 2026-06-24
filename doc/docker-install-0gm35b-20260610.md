# 0GM-35B SIA Running in Docker — Complete Deployment Guide

**Last updated**: 2026-06-10  
**Target docker image**: `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel`  
**Target configuration**: 0GM-1.0-35B-A3B-0427 SIA b2 inproc accelerated path (vllm 0.18.0, [`requirements/0gm35b-b2-inproc.txt`](../requirements/0gm35b-b2-inproc.txt))

> This doc covers 3 deployment paths (quick run / Dockerfile build / **production compose**), each including: **build image** → **start server** → **smoke test** → **stop / restart / cleanup**. For production deployment, go directly to [Path C](#path-c--docker-compose--production-recommended).

---

## Key Differences from VL-30B Deployment

| Dimension | VL-30B ([doc](docker-install-vl30b-20260606.md)) | **0GM-35B (this doc)** |
|---|---|---|
| vllm version | 0.17.1 | **0.18.0** |
| requirements | `vl30b-b2-inproc.txt` | **`0gm35b-b2-inproc.txt`** |
| image tag | `sia-vl30b:0.17.1` | **`sia-0gm35b:0.18.0`** |
| Dockerfile | `Dockerfile` | **`Dockerfile.0gm35b`** |
| compose file | `docker-compose.yml` | **`docker-compose.0gm35b.yml`** |
| venv path | `/opt/venv-vl30b` | **`/opt/venv-0gm35b`** |
| Main LLM arch | `Qwen3VLMoe` | **`Qwen3_5MoeForConditionalGeneration`** |
| LLM CUDA graph | `SIA_LLM_CUDAGRAPH=piecewise` | **not set** (default FULL_AND_PIECEWISE, ~2× faster) |
| RM CUDA graph | `SIA_RM_CUDAGRAPH=piecewise` | **`SIA_RM_CUDAGRAPH=none`** (must be eager; otherwise RuntimeError) |
| `--max_model_len` | 4096 | **2048** |
| Minimum GPU | 80 GB (H100/H200) | **140 GB+ (H200)** — 35B weights ~70 GB; 80 GB is insufficient |

> **Why must vllm be 0.18.0?** 0.17.x does not support the `Qwen3_5MoeForConditionalGeneration` arch (0GM-35B cannot load); 0.19.x changed PIECEWISE to runtime capture, which conflicts 100% with b2 inproc (all configurations exhausted and failed). 0.18.0 is the only viable version; see [`doc/0gm-35b-b2-inproc-speedup-20260609.md`](0gm-35b-b2-inproc-speedup-20260609.md) for details.

---

## Prerequisites

### 1. Host Hardware + NVIDIA Driver

- **NVIDIA H200 (141 GB) or larger GPU** (0GM-35B weights ~70 GB; A100/H100 80 GB single card is insufficient)
- NVIDIA driver installed: `nvidia-smi` should run without errors

### 2. nvidia-container-toolkit

Allows Docker containers to access the GPU. First check if already installed:

```bash
docker run --rm --gpus all ubuntu:22.04 nvidia-smi
```

If you get an error (`could not select device driver "nvidia"`), it is not installed yet. Follow these steps to install (Ubuntu/Debian):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
# Verify
docker run --rm --gpus all ubuntu:22.04 nvidia-smi
```

### 3. Docker + Compose Version

Docker Compose V2+ is sufficient (`docker compose`, not `docker-compose`). Verify:

```bash
docker compose version
```

### 4. Directory Structure

Path B/C mounts the entire `/dstack/persistent/SIA` into the container at `/workspace`. Create this structure in advance:

```
/dstack/persistent/SIA/
├── models/
│   ├── 0GM-1.0-35B-A3B-0427/                    ← main LLM, ~70 GB  (§5.2)
│   ├── Qwen3-4B-Base/                       ← RM convert source, ~8 GB   (§5.3, option B only)
│   ├── SIA-checkpoints/                     ← LoRA checkpoints, ~3 GB  (§5.3, option B only)
│   └── VM-Qwen3-4B-merged-for-vllm/        ← final RM checkpoint, ~11 GB (§5.4)
└── sia-repo/
    └── 0g-sparse-inference-alignment/       ← this repo (§5.1)
```

```bash
mkdir -p /dstack/persistent/SIA/models
mkdir -p /dstack/persistent/SIA/sia-repo
```

### 5. Repo + Model Weights (~81 GB minimum; ~92 GB including convert source)

> Requires `huggingface-cli`: `pip install huggingface-hub`

#### 5.1 Clone the Repo

```bash
git clone <your-repo-url> \
  /dstack/persistent/SIA/sia-repo/0g-sparse-inference-alignment
```

#### 5.2 Main LLM — 0GM-1.0-35B-A3B-0427 (~70 GB)

0GM is an internal model from 0G Foundation. Contact the model admin for the specific path or HuggingFace repo name. Example:

```bash
# Replace <ORG>/<REPO> with the actual repo name
huggingface-cli download <ORG>/0GM-1.0-35B-A3B-0427 \
  --local-dir /dstack/persistent/SIA/models/0GM-1.0-35B-A3B-0427
```

Or copy directly from existing storage via rsync:

```bash
rsync -av /path/to/existing/0GM-1.0-35B-A3B-0427/ \
  /dstack/persistent/SIA/models/0GM-1.0-35B-A3B-0427/
```

Verify:
```bash
ls /dstack/persistent/SIA/models/0GM-1.0-35B-A3B-0427/
# Expected: config.json  model-*.safetensors  tokenizer*.json  ...
```

#### 5.3 RM Source — Qwen3-4B-Base + VM-Qwen3-4B-Base LoRA (~11 GB) (only needed for §5.4 option B)

```bash
# Qwen3-4B-Base: base model for RM (~8 GB)
huggingface-cli download Qwen/Qwen3-4B-Base \
  --local-dir /dstack/persistent/SIA/models/Qwen3-4B-Base

# VM-Qwen3-4B-Base: LoRA checkpoint from the paper authors (~3 GB)
huggingface-cli download Runyi-Hu/SIA \
  --local-dir /dstack/persistent/SIA/models/SIA-checkpoints
```

After download, the LoRA path is `/dstack/persistent/SIA/models/SIA-checkpoints/VM-Qwen3-4B-Base/`.

#### 5.4 Obtain VM-Qwen3-4B-merged-for-vllm

**Option A: Direct download (recommended — pre-merged version)**

```bash
huggingface-cli download TengGao/VM-Qwen3-4B-merged-for-vllm_public \
  --local-dir /dstack/persistent/SIA/models/VM-Qwen3-4B-merged-for-vllm
```

**Option B: Self-convert (use this when switching to a different RM / LoRA)**

```bash
# Run in a host Python environment (does not depend on the vllm venv; only needs transformers + peft)
pip install transformers peft safetensors

python /dstack/persistent/SIA/sia-repo/0g-sparse-inference-alignment/scripts/convert_rm_for_vllm.py \
  --rm      /dstack/persistent/SIA/models/Qwen3-4B-Base \
  --rm_lora /dstack/persistent/SIA/models/SIA-checkpoints/VM-Qwen3-4B-Base \
  --output  /dstack/persistent/SIA/models/VM-Qwen3-4B-merged-for-vllm
```

Estimated time: 2-5 minutes, output ~11 GB.

Verify (regardless of option A or B):

```bash
ls /dstack/persistent/SIA/models/VM-Qwen3-4B-merged-for-vllm/
# Expected: config.json  model.safetensors  tokenizer*.json  ...
```

Confirm the required models are in place:

```bash
ls /dstack/persistent/SIA/models/
# Option A: 0GM-1.0-35B-A3B-0427  VM-Qwen3-4B-merged-for-vllm
# Option B: 0GM-1.0-35B-A3B-0427  Qwen3-4B-Base  SIA-checkpoints  VM-Qwen3-4B-merged-for-vllm
```

**Once all prerequisites above are satisfied, proceed to one of the three paths below.**

---

## Choose One of Three Paths

| Path | Best for | Steps |
|---|---|---|
| **Path A: Command-line install** | One-time quick run, no persistent image | Step 1 → Step 2 (run install script) → Step 3 → Step 4 |
| **Path B: Dockerfile build** ⚡ | Multiple runs / sharing with others; no dep install on startup but server must be started manually | Step 1B (build image) → Step 1B' (docker run) → Step 3 → Step 4 |
| **Path C: docker compose** ⭐ **Production recommended** | Production deployment / auto-restart / health checks; **server starts automatically** | Step 1C (`docker compose -f docker-compose.0gm35b.yml up -d`) → Step 4 |

---

## Path A — Command-line Install

### Step 1 — Start docker container

```bash
docker run -it --rm \
  --gpus all \
  --shm-size=16g \
  --ipc=host \
  -p 8000:8000 \
  -v /dstack/persistent/SIA:/workspace \
  pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel \
  bash
```

If the repo has not been cloned yet, clone it **inside the container** (the pytorch image does not include git by default; install it first):
```bash
apt-get update && apt-get install -y --no-install-recommends git ca-certificates
mkdir -p /workspace/sia-repo
cd /workspace/sia-repo && git clone <your-repo-url> 0g-sparse-inference-alignment
```

> Recommended: clone the repo on the **host** following prerequisite §5, then mount with `-v /dstack/persistent/SIA:/workspace` — no need to install git inside the container.

### Step 2 — Run install script (inside container)

```bash
cd /workspace/sia-repo/0g-sparse-inference-alignment
bash scripts/docker_install_0gm35b.sh
```

What the script does in order:
1. **apt dependencies** — check + auto-install missing: `git ca-certificates curl` + `python${ver}-venv`
2. `python3 -m venv /opt/venv-0gm35b` (isolated from the conda torch 2.11 pre-installed in docker)
3. `pip install -r requirements/0gm35b-b2-inproc.txt` (installs vllm 0.18.0 + torch 2.10.0 + all dependencies)
4. `pip install -e .` (registers the `sia_rm` vllm plugin)
5. **Self-check**: import vllm/torch, verify CUDA available, verify `Qwen3_5MoeForConditionalGeneration` arch registered, verify `unlock_workspace` API, verify sia_rm plugin discoverable

Estimated time: **5-10 minutes** (mainly the vllm 0.18.0 wheel download, ~250 MB).

On success, output will resemble:
```
python: 3.12.x  (/opt/venv-0gm35b/bin/python3)
vllm  : 0.18.0
torch : 2.10.0+cu128
CUDA  : available=True  device_count=1
GPU 0 : NVIDIA H200
unlock_workspace API: ✓ available
Qwen3_5MoE arch     : ✓ registered
sia_rm plugin       : ✓ (['sia_rm'])
```

---

## Path B — Dockerfile Build (recommended for repeated runs or sharing)

Repo + models still use bind-mount — not baked into the image (image stays small ~10 GB).

### Step 1B — Build image (one-time, ~5-10 min)

On the host, `cd` to the repo root (containing `Dockerfile.0gm35b`):
```bash
cd /dstack/persistent/SIA/sia-repo/0g-sparse-inference-alignment
docker build -f Dockerfile.0gm35b -t sia-0gm35b:0.18.0 .
```

What the build does (mirrors the Path A install script):
1. `apt-get install python3.12-venv ca-certificates curl`
2. `python3 -m venv /opt/venv-0gm35b`
3. `pip install -r requirements/0gm35b-b2-inproc.txt` (vllm 0.18.0 + all Python dependencies)
4. COPY entrypoint script into the image; on container startup it auto-runs `pip install -e <mounted-repo>` to register the sia_rm plugin (1-2 seconds)

### Step 1B' — Start container

```bash
docker run -it --rm \
  --gpus all \
  --shm-size=16g \
  --ipc=host \
  -p 8000:8000 \
  -v /dstack/persistent/SIA:/workspace \
  sia-0gm35b:0.18.0 \
  bash
```

Inside the container you will see:
```
[entrypoint] registering sia_rm vllm plugin (pip install -e /workspace/sia-repo/0g-sparse-inference-alignment)
root@xxxx:/workspace/sia-repo/0g-sparse-inference-alignment#
```

The venv is already in `$PATH`; run `python --version` directly to confirm the venv Python. **Skip Step 2 entirely and proceed to Step 3 to start the server**.

### (Optional) Verify all dependencies are present in the image

```bash
python -c "
import vllm, torch
from importlib.metadata import entry_points
print('vllm :', vllm.__version__)
print('torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())
print('sia_rm plugin:', [e.name for e in entry_points(group='vllm.general_plugins') if 'sia' in e.name])
"
```
Expected: `vllm: 0.18.0`, `torch: 2.10.0+cu128`, `sia_rm plugin: ['sia_rm']`.

---

## Path C — docker compose (⭐ Production recommended)

All GPU resources / mounts / restart policies / health checks / complete server startup command are specified in [`docker-compose.0gm35b.yml`](../docker-compose.0gm35b.yml). **One command brings everything up** with the server starting automatically.

### Step 1C — Start services

On the host, cd to the repo root (containing `docker-compose.0gm35b.yml`):
```bash
cd /dstack/persistent/SIA/sia-repo/0g-sparse-inference-alignment
docker compose -f docker-compose.0gm35b.yml up -d --build   # First time: build image (~5-10 min) + start container in background
```

Day-to-day operations:
```bash
docker compose -f docker-compose.0gm35b.yml logs -f sia-0gm35b   # tail server log
docker compose -f docker-compose.0gm35b.yml ps                    # check health status
docker compose -f docker-compose.0gm35b.yml restart sia-0gm35b   # restart server
docker compose -f docker-compose.0gm35b.yml down                  # stop + remove container
```

### Skip Step 2 / Step 3, go directly to Step 4

Under Path C, **the server starts automatically after `docker compose up -d`** — Step 2 (install deps) and Step 3 (manual server start) are not needed. Wait until `docker compose -f docker-compose.0gm35b.yml ps` shows `(healthy)` status (approximately 20-25 minutes on first startup, including 35B weight loading + cudagraph capture; subsequent restarts ~5-8 minutes), then skip to Step 4 smoke test.

---

## Step 3 — Start SIA server (main LLM + Value Model in same process) — Path A / B manual start

> **Who needs this step**: Path A users (after Step 2 installs dependencies), Path B users (after entering container via Step 1B'). **Path C users skip** — the server is already started automatically by compose.

> ℹ️ **b2 inproc is a single-process topology**: the main LLM (0GM-35B) + Value Model (Qwen3-4B) run **nested in the same Python process**, RM calls are direct Python function calls rather than HTTP — **no separate RM server needs to be started**.

```bash
source /opt/venv-0gm35b/bin/activate

mkdir -p /workspace/sia-logs
LOG="/workspace/sia-logs/sia_server_$(date +%Y%m%d_%H%M%S).log"
echo "[launch] log → $LOG"

nohup env \
  SIA_RM_CUDAGRAPH=none \
  SIA_RM_MULTIPROCESS=0 \
  python src/sia_vllm_server.py \
    --llm        /workspace/models/0GM-1.0-35B-A3B-0427 \
    --rm_backend b2 \
    --rm_model   /workspace/models/VM-Qwen3-4B-merged-for-vllm \
    --llm_gpu_mem    0.55 \
    --rm_b2_gpu_mem  0.15 \
    --topk 10 --weight 1.0 --entropy_threshold 1.0 \
    --max_model_len 2048 --mamba_cache_mode align --port 8000 \
    > "$LOG" 2>&1 &

echo "[launch] pid=$!  (process detached from this shell, log: $LOG)"
```

Watch loading progress in real-time (`Ctrl+C` exits tail without affecting the server):
```bash
tail -f "$LOG"
```

Parameter reference:

| Parameter / Env var | Purpose |
|---|---|
| `--llm` | **Main inference LLM** (0GM-35B), uses 55% of GPU memory |
| `--rm_backend b2` | Nested in-process RM backend (vllm 0.18.0 sweet-spot path) |
| `--rm_model` | **Value Model** (Qwen3-4B), nested inside the main process, uses 15% of GPU memory |
| `--llm_gpu_mem 0.55 + --rm_b2_gpu_mem 0.15` | Total 70%, leaving 30% for cudagraph + KV cache headroom (tested on H200 141GB) |
| `--topk 10 --weight 1.0 --entropy_threshold 1.0` | SIA parameters: 10 candidates, intervention weight 1.0, intervene only when entropy > 1.0 (≈ 20% intervention rate) |
| `--max_model_len 2048` | Limit max sequence length to 2048 tokens to control KV cache usage |
| `--mamba_cache_mode align` | 0GM-35B is a hybrid model (full_attention + GatedDeltaNet linear_attention). In default `none` mode, mamba_block_size=max_model_len=2048, causing APC lcm_block_size to be very large with cached_tokens=0 for all requests. `align` aligns mamba_block_size to the attention block_size (~1056 tokens), allowing requests with prompt ≥ 1056 tokens to report cached_tokens > 0 on second request, meeting V7 provider-admission requirements |
| `SIA_RM_CUDAGRAPH=none` | RM uses eager mode (**required**; otherwise vllm 0.18.0 throws RuntimeError during inference) |
| `SIA_RM_MULTIPROCESS=0` | RM runs in the same process as the main LLM, not as a subprocess |
| `SIA_LLM_CUDAGRAPH` | **Not set** — main LLM defaults to FULL_AND_PIECEWISE (~2× faster than forcing piecewise) |

Wait for startup (35B weight loading ~15 min on H200, cudagraph capture ~3-5 min, RM loading ~1 min) until the log shows:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Check if the server is still alive:
```bash
pgrep -af sia_vllm_server
```

Stop the server:
```bash
pkill -f sia_vllm_server.py
```

---

## Step 4 — Smoke test (verify server is truly running)

Once the server is ready (Path A/B sees "Uvicorn running on http://0.0.0.0:8000", Path C shows `(healthy)` status), run a chat completion to verify end-to-end functionality.

### 4.1 Using curl (simplest)

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/0GM-1.0-35B-A3B-0427",
    "messages": [{"role":"user","content":"What are 3 colors of fruit?"}],
    "max_tokens": 100,
    "temperature": 0.7
  }' | python3 -m json.tool
```

### 4.2 Using OpenAI Python SDK

```bash
pip install openai
```
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
resp = client.chat.completions.create(
    model="/workspace/models/0GM-1.0-35B-A3B-0427",
    messages=[{"role": "user", "content": "What are 3 colors of fruit?"}],
    max_tokens=100, temperature=0.7,
)
print(resp.choices[0].message.content)
print("usage:", resp.usage)
```

### 4.3 Path C users testing from inside the container

```bash
docker compose -f docker-compose.0gm35b.yml exec sia-0gm35b bash -c \
  'curl -s -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d "{\"model\":\"/workspace/models/0GM-1.0-35B-A3B-0427\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":50}"'
```

### Verify SIA is truly intervening

Expected response is normal JSON, and server logs will print:
```
[SIA] req=0 DONE  intervened=N/M  ratio=X%  top1_flip=K/N (Y%)
```

Health indicators:

| Metric | Healthy range | Red flag |
|---|---|---|
| intervention ratio | 15–30% (approximately 20% with entropy_threshold=1.0) | **0%** → RM is dead; SIA degraded to no-op |
| top1 flip rate | 50–80% | 0% / 100% → bug |
| RM error log entries | 0 | >0 → RM forward failed (check whether `SIA_RM_CUDAGRAPH=none` is in effect) |

How to view logs:
- **Path A/B**: `tail -f "$LOG"`
- **Path C**: `docker compose -f docker-compose.0gm35b.yml logs -f sia-0gm35b`

### 4.4 Per-request SIA parameter testing

SIA supports overriding global parameters per request (`sia_weight` / `sia_topk` / `sia_entropy_threshold`) without restarting the server.

#### Disable SIA intervention (`sia_weight=0`)

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/0GM-1.0-35B-A3B-0427",
    "messages": [{"role":"user","content":"What are 3 colors of fruit?"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "sia_weight": 0
  }' | python3 -m json.tool
```

Expected server log: `ratio=0.0%` — RM is never called; degrades to pure vLLM inference.

#### Force intervention on every token (`sia_entropy_threshold=0`)

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/0GM-1.0-35B-A3B-0427",
    "messages": [{"role":"user","content":"What are 3 colors of fruit?"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "sia_entropy_threshold": 0
  }' | python3 -m json.tool
```

Expected server log: `ratio=100%` — all tokens are scored by RM.

### 4.5 Multimodal input (image + text) smoke test

0GM-1.0-35B is a Vision-Language model that supports image input. Qwen3-4B is a text-only model that cannot score image context, so **the server automatically forces `sia_weight` to `0.0` when an image is detected**, bypassing the VM and using the base model directly.

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "max_tokens": 200,
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAKklEQVR4nO3NsQ0AAAjDsF7L/x/AE3SzlDnOTqp17wAAAAAAAAAAAOCxA0O5lEwRQVLnAAAAAElFTkSuQmCC"
            }
          },
          {"type": "text", "text": "What color is in this image?"}
        ]
      }
    ]
  }' | python3 -m json.tool
```

Expected server log (`docker compose -f docker-compose.0gm35b.yml logs -f sia-0gm35b`):
```
[SIA] multimodal request detected — SIA bypassed (VM is text-only)
(EngineCore pid=...) [SIA] req=0 DONE  intervened=0/N  ratio=0.0%  top1_flip=0/0 (0.0%)
```

`ratio=0.0%` confirms the VM is fully bypassed. The model should correctly describe the image content (the example image is an orange square).

---

## Step 5 — Stop / Restart / Cleanup (per path)

### Path A / B (manual docker run)

| Operation | Command |
|---|---|
| Stop server (process level) | `pkill -f sia_vllm_server.py` |
| Check if server is still running | `pgrep -af sia_vllm_server` |
| Restart server | Re-run the nohup command from Step 3 |
| Exit container (server will also be killed) | `exit` inside the container |

### Path C (docker compose) — Production recommended

| Operation | Command |
|---|---|
| Stop server + remove container (preserve named volumes) | `docker compose -f docker-compose.0gm35b.yml down` |
| Stop server + remove container + clean up volumes | `docker compose -f docker-compose.0gm35b.yml down -v` |
| Restart server only, no rebuild | `docker compose -f docker-compose.0gm35b.yml restart sia-0gm35b` |
| Restart after source code change (bind-mount; restart is sufficient) | `docker compose -f docker-compose.0gm35b.yml restart sia-0gm35b` |
| Rebuild after changing `requirements/0gm35b-b2-inproc.txt` | `docker compose -f docker-compose.0gm35b.yml up -d --build` |
| View resource usage | `docker stats sia-0gm35b-server` |
| Enter container for debugging (server still running) | `docker compose -f docker-compose.0gm35b.yml exec sia-0gm35b bash` |

> **GPU memory release**: After stopping the container, GPU memory is typically released within a few seconds. If `nvidia-smi` still shows usage, force-kill with `pkill -9 -f sia_vllm_server.py`.

---

## Key Pitfalls / Notes

| Pitfall | Solution |
|---|---|
| Using vllm 0.17.x | 0.17.x does not support `Qwen3_5MoeForConditionalGeneration`; 0GM-35B cannot load. **Must use vllm 0.18.0** |
| Using vllm 0.19.x | 0.19.x PIECEWISE changed to runtime capture, which conflicts 100% with b2 inproc; all configurations fail. **Must use vllm 0.18.0** |
| Setting `SIA_RM_CUDAGRAPH=piecewise` (or `full`) | vllm 0.18.0 RM encounters a new batch_descriptor during inference and triggers runtime capture, throwing `RuntimeError: CUDA graph capturing detected at an inappropriate time`. **Must set `SIA_RM_CUDAGRAPH=none`** |
| Setting `SIA_LLM_CUDAGRAPH=piecewise` | Main LLM is locked to PIECEWISE-only mode; noSIA drops from ~107 tok/s to ~57 tok/s (~2× slower). **Do not set this** — let vllm default to FULL_AND_PIECEWISE |
| Using A100 / H100 (80 GB) single card | 0GM-35B weights are ~70 GB; an 80 GB GPU cannot fit LLM (55%) + RM (15%) + KV cache simultaneously. **Requires H200 (141 GB) or larger** |
| Docker image ships with PyTorch 2.11 (in conda), conflicting with torch 2.10.0 required by vllm 0.18.0 | Use `python3 -m venv` to create an isolated venv (the script handles this); **do not pip install vllm directly into the conda env** |
| Missing `--shm-size=16g --ipc=host` | vllm uses large amounts of shared memory for KV cache; the default docker 64 MB is insufficient |
| `--max_model_len` not limited | Without this, vllm allocates KV cache based on the model's max supported sequence length, which easily causes OOM. **Recommended: `--max_model_len 2048`** |
| Path C shows `(unhealthy)` on first startup | `start_period=25m` provides buffer for 35B weight loading + cudagraph capture. Unhealthy within 25 min is normal; unhealthy after that is a real problem |
| Production long-run logs filling up the disk | Add under the service in `docker-compose.0gm35b.yml`: `logging: {driver: "json-file", options: {max-size: "500m", max-file: "5"}}` |

---

## GPU Memory Budget — How to Adjust When Other Processes Occupy the GPU

If other processes are using the GPU, run `nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader` to compute free memory, then reduce `--llm_gpu_mem` and `--rm_b2_gpu_mem` (vllm treats these as percentages of **total** GPU memory, not free memory).

Also recommended: add `--max_model_len 1024` (halved from 2048) to reduce KV cache requirements.

For Path C, change the `command:` field in `docker-compose.0gm35b.yml` — no image rebuild needed.

---

## AlpacaEval Evaluation — SIA vs noSIA (Skywork scoring)

Reuse the SIA server running inside the docker container to run AlpacaEval quality evaluation. Both arms share the same running server, distinguished by per-request `sia_weight=0` — no service switching required.

**Reference results** (2026-06-10, H200, AlpacaEval 200Q, topk=10, entropy_threshold=1.0):

| Config | Throughput | SkyWork mean | Intervention rate | flip rate |
|---|---|---|---|---|
| noSIA | ~107 tok/s | 24.01 | — | — |
| **SIA (b2 inproc + stable prefix)** | **~65 tok/s** | **29.36 (+22.3%)** | **20.1%** | **65.1%** |

### Prerequisites

```bash
# Confirm Skywork scoring model (adjust path to actual location)
ls /dstack/persistent/SIA/models/Skywork-Reward-V2-Llama-3.1-8B/
# If not yet downloaded:
# huggingface-cli download Skywork/Skywork-Reward-V2-Llama-3.1-8B \
#   --local-dir /dstack/persistent/SIA/models/Skywork-Reward-V2-Llama-3.1-8B

# Confirm AlpacaEval dataset (included in repo; confirm file exists)
ls /dstack/persistent/SIA/sia-repo/0g-sparse-inference-alignment/data/alpaca_eval/alpaca_eval.json
```

### Phase 1 — Generation (server running normally, no downtime needed)

Enter the container:
```bash
docker compose -f docker-compose.0gm35b.yml exec sia-0gm35b bash
cd /workspace/sia-repo/0g-sparse-inference-alignment
MODEL=/workspace/models/0GM-1.0-35B-A3B-0427
mkdir -p /workspace/exp
```

**SIA arm** (server default parameters: topk=10, weight=1.0, entropy_threshold=1.0):
```bash
nohup python -u eval/alpaca_eval.py \
  --base_url http://localhost:8000/v1 \
  --model "$MODEL" \
  --limit 200 --max_tokens 2048 \
  --temperature 1.0 --top_p 0.95 --top_k 20 --repetition_penalty 1.0 \
  --output /workspace/exp/alpaca_0gm35b_sia_$(date +%Y%m%d_%H%M%S).json \
  > /workspace/exp/alpaca_0gm35b_sia_gen.log 2>&1 &
echo "SIA arm PID=$!"
```

**noSIA arm** (`--sia_weight 0` disables RM intervention):
```bash
nohup python -u eval/alpaca_eval.py \
  --base_url http://localhost:8000/v1 \
  --model "$MODEL" \
  --limit 200 --max_tokens 2048 \
  --temperature 1.0 --top_p 0.95 --top_k 20 --repetition_penalty 1.0 \
  --sia_weight 0 \
  --output /workspace/exp/alpaca_0gm35b_nosia_$(date +%Y%m%d_%H%M%S).json \
  > /workspace/exp/alpaca_0gm35b_nosia_gen.log 2>&1 &
echo "noSIA arm PID=$!"
```

> SIA arm approximately 50-60 min (~65 tok/s); noSIA arm approximately 15 min (~107 tok/s).

### Phase 2 — Skywork scoring (no server downtime needed)

On H200 (141GB), the default configuration (`llm_gpu_mem=0.55 + rm_b2_gpu_mem=0.15`) uses approximately 99GB, leaving ~42GB — enough to load Skywork 8B BF16 (~16GB). **No need to stop the SIA server** — run scoring directly inside the container.

```bash
# Enter the running container
docker compose -f docker-compose.0gm35b.yml exec sia-0gm35b bash
cd /workspace/sia-repo/0g-sparse-inference-alignment
RM=/workspace/models/Skywork-Reward-V2-Llama-3.1-8B

python scripts/measure_alpaca_reward.py \
  --input_file  /workspace/exp/alpaca_0gm35b_sia_*.json \
  --output_file /workspace/exp/alpaca_0gm35b_sia_scored.json \
  --rm "$RM" --device cuda:0 --strip_think

python scripts/measure_alpaca_reward.py \
  --input_file  /workspace/exp/alpaca_0gm35b_nosia_*.json \
  --output_file /workspace/exp/alpaca_0gm35b_nosia_scored.json \
  --rm "$RM" --device cuda:0 --strip_think
```

Each takes approximately 3-5 min (Skywork 8B BF16, ~2.5 questions/s).

> If remaining GPU memory is less than 20GB (e.g., other processes are occupying extra memory), stop the server first: `docker compose -f docker-compose.0gm35b.yml stop sia-0gm35b`, then `start` after scoring.

### Phase 3 — Compare results

```bash
python - <<'EOF'
import json, statistics

def stats(p):
    d = json.load(open(p))
    rs = [r["reward"] for r in d if r.get("reward") is not None]
    return statistics.mean(rs), statistics.median(rs), len(rs), len(d)

nosia_mean, nosia_p50, n1, t1 = stats("/workspace/exp/alpaca_0gm35b_nosia_scored.json")
sia_mean,   sia_p50,   n2, t2 = stats("/workspace/exp/alpaca_0gm35b_sia_scored.json")
delta_abs = sia_mean - nosia_mean
delta_rel = delta_abs / abs(nosia_mean) * 100

print(f"noSIA  {n1}/{t1} scored  mean={nosia_mean:+.4f}  p50={nosia_p50:+.4f}")
print(f"SIA    {n2}/{t2} scored  mean={sia_mean:+.4f}  p50={sia_p50:+.4f}")
print(f"Δ      {delta_abs:+.4f}  ({delta_rel:+.2f}%)")
EOF
```

After scoring, exit the container:
```bash
exit
```

---

## Related Docs

- [`docker-install-vl30b-20260606.md`](docker-install-vl30b-20260606.md) — VL-30B equivalent deployment doc (same structure as this one)
- [`0gm-35b-b2-inproc-speedup-20260609.md`](0gm-35b-b2-inproc-speedup-20260609.md) — Why vllm 0.18.0 is the sweet spot (exhaustive failed configurations + root cause analysis)
- [`0gm-35b-sia-perf-breakdown-20260609.md`](0gm-35b-sia-perf-breakdown-20260609.md) — Complete performance optimization history (including stable prefix optimization + AlpacaEval 200Q results)
- [`sia-speedup-summary-20260610.md`](sia-speedup-summary-20260610.md) — Optimization summary for both models (principles + data comparison)
- [`../docker-compose.0gm35b.yml`](../docker-compose.0gm35b.yml) — Path C production compose file
- [`../Dockerfile.0gm35b`](../Dockerfile.0gm35b) — Dockerfile shared by Path B / C
- [`../scripts/docker_entrypoint_0gm35b.sh`](../scripts/docker_entrypoint_0gm35b.sh) — Entrypoint shared by Path B / C
- [`../scripts/docker_install_0gm35b.sh`](../scripts/docker_install_0gm35b.sh) — Path A install script
- [`../requirements/0gm35b-b2-inproc.txt`](../requirements/0gm35b-b2-inproc.txt) — pip dependency list
- [`../CLAUDE.md`](../CLAUDE.md) — Overall venv matrix
