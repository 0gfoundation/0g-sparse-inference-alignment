# VL-30B SIA Running in Docker — Complete Deployment Guide

**Last updated**: 2026-06-07
**Target docker image**: `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel`
**Target configuration**: VL-30B (Qwen3-VL-30B-A3B-Instruct) SIA b2 inproc accelerated path (vllm 0.17.1, [`requirements/vl30b-b2-inproc.txt`](../requirements/vl30b-b2-inproc.txt))

> This doc covers 3 deployment paths (quick run / Dockerfile build / **production compose**), each including: **build image** → **start server** → **smoke test** → **stop / restart / cleanup**. For production deployment, go directly to [Path C](#path-c--docker-compose--production-recommended).

## Prerequisites

### 1. Host Hardware + NVIDIA Driver

- NVIDIA GPU (H100 / H200 / A100, **80GB+ VRAM**)
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
docker compose version    # any output is fine; no minimum version requirement
```

> `docker-compose.yml` uses `deploy: resources: reservations: devices:` GPU syntax, compatible with all versions.

### 4. Directory Structure

Path B/C mounts the entire `/dstack/persistent/SIA` into the container at `/workspace`. Create this structure in advance:

```
/dstack/persistent/SIA/
├── models/
│   ├── Qwen3-VL-30B-A3B-Instruct/     ← main LLM, ~60 GB  (§5.2)
│   ├── Qwen3-4B-Base/                  ← RM convert source, ~8 GB   (§5.3)
│   ├── SIA-checkpoints/                ← LoRA checkpoints, ~3 GB  (§5.3)
│   └── VM-Qwen3-4B-merged-for-vllm/   ← final RM checkpoint, ~11 GB (§5.4 download or convert)
└── sia-repo/
    └── 0g-sparse-inference-alignment/  ← this repo (§5.1)
```

```bash
mkdir -p /dstack/persistent/SIA/models
mkdir -p /dstack/persistent/SIA/sia-repo
```

### 5. Repo + Model Weights (~71 GB minimum; ~82 GB including convert source)

> Requires `huggingface-cli`: `pip install huggingface-hub`

#### 5.1 Clone the Repo (the convert script lives in the repo; clone first)

```bash
git clone <your-repo-url> \
  /dstack/persistent/SIA/sia-repo/0g-sparse-inference-alignment
```

#### 5.2 Main LLM — Qwen3-VL-30B-A3B-Instruct (~60 GB)

```bash
huggingface-cli download Qwen/Qwen3-VL-30B-A3B-Instruct \
  --local-dir /dstack/persistent/SIA/models/Qwen3-VL-30B-A3B-Instruct
```

#### 5.3 RM Source — Qwen3-4B-Base + VM-Qwen3-4B-Base LoRA (~11 GB) (only needed for §5.4 option B)

```bash
# Qwen3-4B-Base: base model for RM (~8 GB)
huggingface-cli download Qwen/Qwen3-4B-Base \
  --local-dir /dstack/persistent/SIA/models/Qwen3-4B-Base

# VM-Qwen3-4B-Base: LoRA checkpoint from the paper authors (~3 GB, includes all VM variants)
huggingface-cli download Runyi-Hu/SIA \
  --local-dir /dstack/persistent/SIA/models/SIA-checkpoints
```

After download, the LoRA path is `/dstack/persistent/SIA/models/SIA-checkpoints/VM-Qwen3-4B-Base/`.

#### 5.4 Obtain VM-Qwen3-4B-merged-for-vllm

**Option A: Direct download (recommended — pre-merged version)**

The merged checkpoint has been uploaded to HuggingFace and can be downloaded directly, skipping the merge step:

```bash
huggingface-cli download TengGao/VM-Qwen3-4B-merged-for-vllm_public \
  --local-dir /dstack/persistent/SIA/models/VM-Qwen3-4B-merged-for-vllm
```

**Option B: Self-convert (use this when switching to a different RM / LoRA)**

If you need to use a different base model or LoRA, run the convert script to re-merge:

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
# Option A: Qwen3-VL-30B-A3B-Instruct  VM-Qwen3-4B-merged-for-vllm
# Option B: Qwen3-VL-30B-A3B-Instruct  Qwen3-4B-Base  SIA-checkpoints  VM-Qwen3-4B-merged-for-vllm
```

**Once all prerequisites above are satisfied, proceed to one of the three paths below.**

---

## Choose One of Three Paths

| Path | Best for | Steps |
|---|---|---|
| **Path A: Command-line install** | One-time quick run, no persistent image | Step 1 → Step 2 (run install script) → Step 3 → Step 4 |
| **Path B: Dockerfile build** ⚡ | Multiple runs / sharing with others / CI; no dep install on startup but server must be started manually | Step 1B (build image) → Step 1B' (docker run) → Step 3 → Step 4 |
| **Path C: docker compose** ⭐ **Production recommended** | Production deployment / auto-restart / health checks / persistent compile cache; **server starts automatically** | Step 1C (`docker compose up -d`) → Step 4 (smoke test) |

---

## Path A — Command-line Install

### Step 1 — Start docker container

```bash
# If using the standard /dstack/persistent/SIA directory structure (see prerequisites §4), mount the whole directory:
docker run -it --rm \
  --gpus all \
  --shm-size=16g \
  --ipc=host \
  -p 8000:8000 \
  -v /dstack/persistent/SIA:/workspace \
  pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel \
  bash

# If repo and models are in different directories, you can mount them separately (replace paths as needed):
# docker run -it --rm --gpus all --shm-size=16g --ipc=host -p 8000:8000 \
#   -v /path/to/sia-repo/0g-sparse-inference-alignment:/workspace/sia-repo \
#   -v /path/to/models:/workspace/models \
#   pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel bash
```

If the repo has not been cloned yet, clone it **inside the container** — the pytorch image does not include git by default; install it first:
```bash
apt-get update && apt-get install -y --no-install-recommends git ca-certificates
mkdir -p /workspace/sia-repo
cd /workspace/sia-repo && git clone <your-repo-url> 0g-sparse-inference-alignment
```

> Recommended: clone the repo on the **host** following prerequisite §5, then mount with `-v /dstack/persistent/SIA:/workspace` — no need to install git inside the container.

### Step 2 — Run install script (inside container)

```bash
cd /workspace/sia-repo/0g-sparse-inference-alignment
bash scripts/docker_install_vl30b.sh
```

What the script does in order:
1. **apt dependencies** — check + auto-install missing: `git ca-certificates curl` + `python${ver}-venv` (probe `python3 -c "import ensurepip"`; if it fails, detect python version and install the matching package, e.g., `python3.12-venv`. The pytorch image's system python does not include the venv module by default; this step is required)
2. `python3 -m venv /opt/venv-vl30b` (using system python, isolated from the conda env torch 2.11 pre-installed in docker)
3. `pip install -r requirements/vl30b-b2-inproc.txt` (installs vllm 0.17.1 + torch 2.10.0 + all dependencies)
4. `pip install -e .` (registers the `sia_rm` vllm plugin, enabling the b2 inproc backend)
5. **Self-check**: import vllm/torch, verify CUDA available, verify Qwen3VLMoe arch registered, verify `unlock_workspace` API exists, verify sia_rm plugin discoverable

Estimated time: **5-10 minutes** (mainly the vllm 0.17.1 wheel download, ~250 MB).

On success, output will resemble:
```
python: 3.12.x  (/opt/venv-vl30b/bin/python3)
vllm  : 0.17.1
torch : 2.10.0+cu124
CUDA  : available=True  device_count=1
GPU 0 : NVIDIA H200
unlock_workspace API: ✓ available
Qwen3VLMoe arch     : ✓ registered
sia_rm plugin       : ✓ (['sia_rm'])
```

---

## Path B — Dockerfile Build (recommended for repeated runs or sharing)

Repo + models still use bind-mount — not baked into the image (image stays small ~10 GB).

### Step 1B — Build image (one-time, ~5-10 min)

On the host, `cd` to the repo root (containing `Dockerfile`):
```bash
cd /dstack/persistent/SIA/sia-repo/0g-sparse-inference-alignment
docker build -t sia-vl30b:0.17.1 .
```

What the build does (mirrors the Path A install script):
1. `apt-get install python3.12-venv ca-certificates curl` (curl is installed in the image for smoke tests)
2. `python3 -m venv /opt/venv-vl30b`
3. `pip install -r requirements/vl30b-b2-inproc.txt` (vllm 0.17.1 + all Python dependencies)
4. COPY an entrypoint script into the image; on container startup it auto-runs `pip install -e <mounted-repo>` to register the sia_rm plugin (1-2 seconds; required)

Note: **The Dockerfile does NOT COPY source code** — it only COPYs the `requirements/` subdirectory for pip install. Source code comes from bind-mount; modifying `src/` does not require rebuilding the image.

### Step 1B' — Start container

```bash
docker run -it --rm \
  --gpus all \
  --shm-size=16g \
  --ipc=host \
  -p 8000:8000 \
  -v /dstack/persistent/SIA:/workspace \
  sia-vl30b:0.17.1 \
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
Expected: `vllm: 0.17.1`, `torch: 2.10.0+cu124`, `sia_rm plugin: ['sia_rm']`.

---

## Path C — docker compose (⭐ Production recommended)

All GPU resources / mounts / restart policies / health checks / complete server startup command are specified in [`docker-compose.yml`](../docker-compose.yml) (in the repo root alongside Dockerfile). **One command brings everything up** with the server starting automatically.

### Step 1C — Start services

On the host, cd to the repo root (containing `docker-compose.yml`):
```bash
cd /dstack/persistent/SIA/sia-repo/0g-sparse-inference-alignment
docker compose up -d --build      # First time: build image (~5-10 min) + start container in background
```

Day-to-day operations:
```bash
docker compose logs -f sia-vl30b  # tail server log (Ctrl+C does not kill container)
docker compose ps                  # check health status
docker compose restart sia-vl30b   # restart server (no rebuild)
docker compose down                # stop + remove container (named volumes preserved)
```

### What the compose file does (differences from Path A/B)

| Dimension | Path A/B (docker run) | Path C (compose) |
|---|---|---|
| Starting server | Manually type `python src/sia_vllm_server.py ...` inside container | **Auto-starts** (`command:` field) |
| Background running | You write `nohup ... &` | **docker manages PID 1**, `up -d` runs in background |
| Logs | Redirect to file yourself | **`docker compose logs`** auto-collects stdout/stderr |
| Process crash | It dies and stays dead; manual restart needed | **`restart: unless-stopped`** auto-recovers |
| Health check | Curl manually | **healthcheck**: auto-curls `/health`; marks unhealthy on failure |
| Reuse compile cache across container restarts | None | **named volume `vllm-compile-cache`** persists; saves ~3-5 min torch.compile time on subsequent starts |
| GPU + shm + ipc | Many command-line flags | Written in yaml; managed in one place |
| Environment differences (dev/staging/prod) | Hard to manage with a single shell command | Separate `docker-compose.dev.yml` / `.prod.yml` files reusing the same image |

### Why the startup command goes in compose, not Dockerfile

| | Dockerfile `CMD` | docker-compose `command:` |
|---|---|---|
| Changing parameters | Requires rebuilding image | Edit yaml + `docker compose up` recreates container; **no rebuild** |
| Multiple scenarios with same image | Only one default | Separate compose files for dev/prod each override it |
| Debugging | `docker run sia-vl30b bash` is interfered by CMD | Dockerfile CMD stays as `bash`; compose uses `command:` to override; debug via `docker run` drops into bash; production uses compose |
| Industry standard | "image self-launching" style | **Microservice + k8s mainstream**: image is a generic artifact; startup config lives in the orchestration layer |

**The most standard approach** (as in this repo):
- `Dockerfile`'s `CMD ["bash"]` — default debug behavior, friendly with `docker run`
- `docker-compose.yml`'s `command:` — the actual production startup command with full parameters fixed

### Skip Step 2 / Step 3, go directly to Step 4

Under Path C, **the server starts automatically after `docker compose up -d`** — Step 2 (install deps) and Step 3 (manual server start) are not needed. Wait until `docker compose ps` shows `(healthy)` status (approximately 15-20 minutes on first startup, including first torch.compile; subsequent restarts ~3-5 minutes), then skip to Step 4 smoke test.

---

## Step 3 — Start SIA server (main LLM + Value Model in same process) — Path A / B manual start

> **Who needs this step**: Path A users (after Step 2 installs dependencies), Path B users (after entering container via Step 1B'). **Path C users skip** — the server is already started automatically by compose.

> ℹ️ **Important — b2 inproc is a single-process topology**: unlike the HTTP path which starts two servers (one main LLM, one RM on another port), b2 inproc has the **main LLM (VL-30B) + Value Model (Qwen3-4B) nested in the same Python process**, sharing a CUDA context, with RM calls as direct Python function calls rather than HTTP. So **this single command starts both the main LLM + Value Model** — no separate RM server is needed.

> For Path B containers, the venv is already in PATH; `source /opt/venv-vl30b/bin/activate` is optional. Path A users should remember to activate it.

```bash
source /opt/venv-vl30b/bin/activate

# Redirect output to a timestamped log file; use nohup to detach from the current shell
mkdir -p /workspace/sia-logs
LOG="/workspace/sia-logs/sia_server_$(date +%Y%m%d_%H%M%S).log"
echo "[launch] log → $LOG"

nohup env \
  SIA_LLM_CUDAGRAPH=piecewise \
  SIA_RM_CUDAGRAPH=piecewise \
  SIA_RM_MULTIPROCESS=0 \
  python src/sia_vllm_server.py \
    --llm        /workspace/models/Qwen3-VL-30B-A3B-Instruct \
    --rm_backend b2 \
    --rm_model   /workspace/models/VM-Qwen3-4B-merged-for-vllm \
    --llm_gpu_mem    0.55 \
    --rm_b2_gpu_mem  0.15 \
    --topk 10 --weight 1.0 --entropy_threshold 1.0 \
    --max_model_len 4096 --port 8000 \
    > "$LOG" 2>&1 &

echo "[launch] pid=$!  (process detached from this shell, log: $LOG)"
```

Watch loading progress in real-time (`Ctrl+C` exits tail without affecting the server, which continues running):
```bash
tail -f "$LOG"
```

Parameter reference:

| Parameter | Purpose |
|---|---|
| `--llm` | **Main inference LLM** (VL-30B), uses 55% of GPU memory |
| `--rm_backend b2` | Uses nested in-process backend (vllm 0.17.1 sweet-spot path) |
| `--rm_model` | **Value Model** (Qwen3-4B), nested inside the main process, uses 15% of GPU memory |
| `--llm_gpu_mem 0.55 + --rm_b2_gpu_mem 0.15` | Total 70%, leaving 30% for cudagraph + KV cache headroom |
| `--topk 10 --weight 1.0 --entropy_threshold 1.0` | SIA algorithm parameters: 10 candidates per token, intervention weight 1.0, intervene only when entropy > 1.0 |
| `SIA_LLM_CUDAGRAPH=piecewise` | Force main LLM to use PIECEWISE cudagraph (vllm 0.17.1 AOT mode), avoiding collision with nested RM |
| `SIA_RM_MULTIPROCESS=0` | RM runs in the same process as the main LLM (InprocClient), not as a subprocess |

Wait for startup (main LLM weights loading ~12 min on H200, then cudagraph capture + warmup ~3 min, then Value Model loading ~1 min) until `tail -f "$LOG"` shows:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Check at any time if the server is still alive:
```bash
ps -ef | grep sia_vllm_server | grep -v grep
# or:
pgrep -af sia_vllm_server
```

To stop the server:
```bash
pkill -f sia_vllm_server.py
```

## Step 4 — Smoke test (verify server is truly running)

Once the server is ready (Path A/B sees "Uvicorn running on http://0.0.0.0:8000", Path C shows `docker compose ps` displaying `(healthy)`), run a chat completion to verify end-to-end functionality.

The server exposes an **OpenAI-compatible** API; any of the three approaches below works:

### 4.1 Using curl (simplest)

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/Qwen3-VL-30B-A3B-Instruct",
    "messages": [{"role":"user","content":"What are 3 colors of fruit?"}],
    "max_tokens": 100,
    "temperature": 0.7
  }' | python3 -m json.tool
```

### 4.2 Using OpenAI Python SDK (for testing production code integration)

```bash
pip install openai
```
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")   # key not validated
resp = client.chat.completions.create(
    model="/workspace/models/Qwen3-VL-30B-A3B-Instruct",
    messages=[{"role": "user", "content": "What are 3 colors of fruit?"}],
    max_tokens=100, temperature=0.7,
)
print(resp.choices[0].message.content)
print("usage:", resp.usage)   # prompt_tokens / completion_tokens / total_tokens
```

### 4.3 Path C users can also test from inside the container (avoids port exposure)

```bash
docker compose exec sia-vl30b bash -c \
  'curl -s -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d "{\"model\":\"/workspace/models/Qwen3-VL-30B-A3B-Instruct\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":50}"'
```

### Verify SIA is truly intervening (not degraded to noSIA)

Expected response is normal JSON, and server logs will print:
```
[SIA] req=0 DONE  intervened=N/M  ratio=X%  top1_flip=K/N (Y%)
```

Health indicators (consistent with [`qwen3-vl-30b-sia-eval-20260605.md` §2.4](qwen3-vl-30b-sia-eval-20260605.md#24-sia-health-indicator-summary)):

| Metric | Healthy range | Red flag |
|---|---|---|
| intervention ratio | 10-40% (short output ~20%, long output ~25-30%) | **0%** → RM is dead; SIA degraded to no-op |
| top1 flip rate | 50-80% | 0% / 100% → bug |
| RM error log entries | 0 | >0 → RM forward failed |

How to view logs (per path):
- **Path A/B**: `tail -f "$LOG"` (the `/workspace/sia-logs/sia_server_*.log` you noted at startup)
- **Path C**: `docker compose logs -f sia-vl30b`

### 4.4 Per-request SIA parameter testing

SIA supports overriding global parameters per request (`sia_weight` / `sia_topk` / `sia_entropy_threshold`) without restarting the server. The three curl commands below each verify one parameter — **observe the ratio / top1_flip changes in the `DONE` line of server logs** to confirm they take effect.

#### Disable SIA intervention (`sia_weight=0`)

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/Qwen3-VL-30B-A3B-Instruct",
    "messages": [{"role":"user","content":"What are 3 colors of fruit?"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "sia_weight": 0
  }' | python3 -m json.tool
```

Expected server log: `ratio=0.0%` — RM is never called; degrades to pure vLLM inference.

#### Score only 1 candidate (`sia_topk=1`)

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/Qwen3-VL-30B-A3B-Instruct",
    "messages": [{"role":"user","content":"What are 3 colors of fruit?"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "sia_topk": 1
  }' | python3 -m json.tool
```

Expected server log: `ratio≈20-30%` (entropy gate working normally), `top1_flip=0%` (only 1 candidate; top-1 cannot be replaced).

#### Force intervention on every token (`sia_entropy_threshold=0`)

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/Qwen3-VL-30B-A3B-Instruct",
    "messages": [{"role":"user","content":"What are 3 colors of fruit?"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "sia_entropy_threshold": 0
  }' | python3 -m json.tool
```

Expected server log: `ratio=100%` — all tokens are scored by RM, including positions where the model is highly confident (entropy gate completely bypassed).

> The three parameters can be combined in the same request, e.g., `"sia_weight": 2.0, "sia_topk": 20, "sia_entropy_threshold": 0.5`. Parameters not provided use the global defaults set at server startup (`--weight` / `--topk` / `--entropy_threshold`).

### 4.5 Multimodal input (image + text) smoke test

Qwen3-VL-30B is a Vision-Language model that supports image input. Qwen3-4B is a text-only model that cannot score image context, so **the server automatically forces `sia_weight` to `0.0` when an image is detected**, bypassing the VM and using the base model directly.

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

Expected server log (`docker compose logs -f sia-vl30b`):
```
[SIA] multimodal request detected — SIA bypassed (VM is text-only)
(EngineCore_DP0 pid=...) [SIA] req=0 DONE  intervened=0/N  ratio=0.0%  top1_flip=0/0 (0.0%)
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
| Container started with `--rm`; container is automatically deleted after exit |

### Path C (docker compose) — Production recommended

| Operation | Command |
|---|---|
| Stop server + remove container (preserve named volumes, including compile cache) | `docker compose down` |
| Stop server + remove container + clean up volumes (reset compile cache) | `docker compose down -v` |
| Restart server only, no rebuild, no volume changes | `docker compose restart sia-vl30b` |
| Restart after source code change (bind-mount; restart is sufficient) | `docker compose restart sia-vl30b` |
| Rebuild after changing `requirements/vl30b-b2-inproc.txt` | `docker compose up -d --build` |
| Apply new configuration after changing `docker-compose.yml` | `docker compose up -d` (auto-recreates the container) |
| View resource usage | `docker stats sia-vl30b-server` |
| Enter container for debugging (server still running) | `docker compose exec sia-vl30b bash` |

> **GPU memory release**: After stopping the container (`docker compose down` or `pkill`), GPU memory is typically released **within a few seconds**. If `nvidia-smi` still shows memory occupied, zombie processes remain — force-kill with `pkill -9 -f sia_vllm_server.py`.

---

## Key Pitfalls / Notes

| Pitfall | Solution |
|---|---|
| Docker image ships with PyTorch 2.11 (in conda), conflicting with torch 2.10.0 required by vllm 0.17.1 | Use `python3 -m venv` to create an isolated venv (the script handles this); **do not pip install vllm directly into the conda env** |
| CUDA 12.8 vs vllm 0.17.1's cu124 wheel | Forward-compatible; OK (cu124 binary runs on 12.8 driver) |
| Must include `--shm-size=16g --ipc=host` | vllm uses large amounts of shared memory for KV cache; the default docker 64 MB is insufficient |
| Main LLM cudagraph mode must be PIECEWISE (`SIA_LLM_CUDAGRAPH=piecewise`) | 0.17.1's PIECEWISE is AOT compilation and will not collide with the nested RM's cudagraph flag. This is the critical env var for b2 inproc to work |
| Main process + nested RM share GPU memory on the same GPU | `--llm_gpu_mem 0.55 --rm_b2_gpu_mem 0.15` totals 0.70, leaving 30% for cudagraph; running a 30B main LLM + 4B RM on an 80 GB GPU is tight but sufficient. When other processes occupy the GPU, reduce to `0.48 + 0.08`; see [GPU Memory Budget](#gpu-memory-budget--how-to-adjust-when-other-processes-occupy-the-gpu) at the end of this doc |
| Path C shows `(unhealthy)` on first startup | start_period=20m provides buffer for main LLM weights loading + torch.compile. Docker will not kill the container during this period (`unless-stopped` + `start_period` work together). Still unhealthy after 20 min is a real problem |
| GPU access syntax version issue | `docker-compose.yml` already uses `deploy: resources: reservations: devices:` syntax, compatible with all versions. If you still get `Additional property gpus is not allowed`, your local file is still the old version; re-pull the latest code |
| Production long-run logs filling up the disk | `docker logs` has no limit by default; add under the `sia-vl30b` service in `docker-compose.yml`: `logging: {driver: "json-file", options: {max-size: "500m", max-file: "5"}}` |
| How to change server parameters in Path C (e.g., `--topk` / `--weight`) | Edit the `command:` field in `docker-compose.yml`, then `docker compose up -d` auto-recreates the container. **No image rebuild needed** |

## How to Verify Prefix Caching Is Enabled

vllm's **Automatic Prefix Caching (APC)** can give prefill **2–10× speedup** in shared-prefix scenarios (system prompt / RAG context / multi-turn conversation), and is a key caching mechanism for production deployments. This repository has it **enabled by default** — `sia_vllm_server.py`'s `--enable_prefix_caching` flag defaults to True, and the [`docker-compose.yml`](../docker-compose.yml) `command:` explicitly passes this flag.

> To disable for a comparison test: `--disable_prefix_caching` (not recommended for production).

### Verification Method 1: Check Server Startup Logs (Most Direct)

When the server starts, vllm dumps the engine config, which includes `enable_prefix_caching=True`:

```bash
# Path C
docker compose logs sia-vl30b 2>&1 | grep -m1 enable_prefix_caching

# Path A/B
grep -m1 enable_prefix_caching $LOG    # the log file you noted at Step 3 startup
```

Expected output (part of vllm's long internal config dump):
```
... seed=0, served_model_name=..., enable_prefix_caching=True, enable_chunked_prefill=True, ...
```

If you see `enable_prefix_caching=False` → APC is off; debug the startup flags.

### Verification Method 2: Check the Server's Own prefix_caching Flag

`sia_vllm_server.py` prints the flag it sees before loading the vllm engine (added by this repository's wrapper):

```bash
docker compose logs sia-vl30b 2>&1 | grep "main LLM prefix_caching"
```

Expected:
```
[SIA] main LLM prefix_caching = True
```

### Verification Method 3: Functional Test (Shared Prefix — Check if Second TTFT Drops Sharply)

Send two requests with **the same prefix but different endings**, and use `time` to measure wall time. When APC is active, the second request's prefill is nearly free (KV reused directly), so wall time should be noticeably shorter:

```bash
PROMPT='Imagine you are a senior software engineer reviewing a Python codebase. You should focus on correctness, readability, and idiomatic style. Provide concrete suggestions where appropriate. Now '

URL=http://localhost:8000/v1/chat/completions
MODEL=/workspace/models/Qwen3-VL-30B-A3B-Instruct

call() {
  curl -s -X POST "$URL" -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT $1\"}],\"max_tokens\":30,\"temperature\":0}" \
    > /dev/null
}

echo "First call (no cache hit):"
time call "review this function: def add(a, b): return a + b"

echo "Second call (same long prefix, different ending — should hit cache):"
time call "review this function: def sub(a, b): return a - b"
```

Expected: the second wall time is noticeably shorter than the first (mainly prefill time saved). With a ~50-token shared prefix on the 30B model, you should save ~50–100 ms TTFT.

> ⚠️ This test requires the prompt to have **exactly the same prefix length + content**. SIA intervention **does not affect** APC — APC only caches input prompt KV during the prefill phase, while SIA modifies logits during the decode phase; they operate at different stages.

### Disable for Comparison Test (Optional, for Future Tuning)

To measure APC's actual benefit for your workload:
```bash
# Edit docker-compose.yml command, change --enable_prefix_caching to --disable_prefix_caching
# Restart:
docker compose up -d
# Run the same benchmark and compare throughput / TTFT
```

Note: disabling APC does **not change SIA intervention effectiveness** (Skywork reward / accuracy metrics are statistically equivalent whether APC is on or off); only prefill is slower. This is the same as b2 inproc acceleration — engineering optimization is orthogonal to the SIA algorithm.

---

## GPU Memory Budget — How to Adjust When Other Processes Occupy the GPU

If other processes are running on the GPU, use `nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader` to calculate free memory, then reduce `--llm_gpu_mem` and `--rm_b2_gpu_mem` (because vllm treats them as percentages of **total** GPU memory, not free memory).

Example: 143 GB GPU, other processes using 57 GB, leaving 86 GB for vllm:

| Config | `--llm_gpu_mem` | `--rm_b2_gpu_mem` | Total vllm usage | After adding other processes | Headroom |
|---|---|---|---|---|---|
| Default (idle GPU) | 0.55 | 0.15 | 100,640 MiB | 158,002 MiB | ❌ **OOM** |
| Reduced | **0.48** | **0.08** | 80,512 MiB | 137,874 MiB | ✅ 5,897 MiB |

Adding `--max_model_len 2048` (halved from 4096) reduces KV cache demand and improves stability.

If you want to apply this to Path C as well, edit the corresponding line in the `command:` field of `docker-compose.yml`.

---

If Step 2 self-check shows any ✗, **do not proceed** — debug first. For any error at any step, read the specific error message before moving to Step 3.

---

## AlpacaEval Evaluation — SIA vs noSIA (Skywork Scoring)

Run AlpacaEval quality evaluation directly inside the docker container, reusing the running SIA server. Both arms share the same running server, distinguished by per-request `--sia_weight 0`, with no service switching required.

**Scripts**: `eval/alpaca_eval.py` (generation) + `scripts/measure_alpaca_reward.py` (scoring)
**Scoring model**: Skywork-Reward-V2-Llama-3.1-8B (third-party RM, independent of Qwen3-4B)
**Reference baseline**: Previous HTTP path VL-30B 200Q experiment: noSIA mean=29.26, SIA mean=28.67, Δ=-1.57% (p=0.29)

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
docker compose exec sia-vl30b bash
cd /workspace/sia-repo/0g-sparse-inference-alignment
MODEL=/workspace/models/Qwen3-VL-30B-A3B-Instruct
mkdir -p /workspace/exp
```

**SIA arm** (server default parameters: topk=10, weight=1.0, entropy_threshold=1.0):
```bash
nohup python -u eval/alpaca_eval.py \
  --base_url http://localhost:8000/v1 \
  --model "$MODEL" \
  --limit 200 --max_tokens 2048 \
  --temperature 1.0 --top_p 0.95 --top_k 20 --repetition_penalty 1.0 \
  --output /workspace/exp/alpaca_vl30b_b2_sia_$(date +%Y%m%d_%H%M%S).json \
  > /workspace/exp/alpaca_vl30b_b2_sia_gen.log 2>&1 &
echo "SIA arm PID=$!"
```

**noSIA arm** (`--sia_weight 0` disables RM intervention, speed ≈ raw vllm):
```bash
nohup python -u eval/alpaca_eval.py \
  --base_url http://localhost:8000/v1 \
  --model "$MODEL" \
  --limit 200 --max_tokens 2048 \
  --temperature 1.0 --top_p 0.95 --top_k 20 --repetition_penalty 1.0 \
  --sia_weight 0 \
  --output /workspace/exp/alpaca_vl30b_b2_nosia_$(date +%Y%m%d_%H%M%S).json \
  > /workspace/exp/alpaca_vl30b_b2_nosia_gen.log 2>&1 &
echo "noSIA arm PID=$!"
```

> SIA arm takes ~80–90 min (28 tok/s); noSIA arm takes ~20 min (RM skipped, speed returns to ~120 tok/s). Can run serially (wait for SIA to finish before running noSIA) or in parallel (the same server supports concurrent requests).

### Phase 2 — Skywork Scoring (requires stopping server to free GPU memory)

```bash
# Stop server on the host
docker compose stop sia-vl30b

# Start a temporary container for scoring (named volume preserved, compile cache intact)
docker compose run --rm sia-vl30b bash
cd /workspace/sia-repo/0g-sparse-inference-alignment
RM=/workspace/models/Skywork-Reward-V2-Llama-3.1-8B   # adjust to actual path

python scripts/measure_alpaca_reward.py \
  --input_file  /workspace/exp/alpaca_vl30b_b2_sia_*.json \
  --output_file /workspace/exp/alpaca_vl30b_b2_sia_scored.json \
  --rm "$RM" --device cuda:0 --strip_think

python scripts/measure_alpaca_reward.py \
  --input_file  /workspace/exp/alpaca_vl30b_b2_nosia_*.json \
  --output_file /workspace/exp/alpaca_vl30b_b2_nosia_scored.json \
  --rm "$RM" --device cuda:0 --strip_think
```

Each takes ~3–5 min (Skywork 8B BF16, ~2.5 questions/s).

### After Generation — Auxiliary Metrics

**tokens/s** (printed automatically at the end of `gen.log` after generation completes):
```bash
tail -5 /workspace/exp/alpaca_vl30b_b2_sia_gen.log
# expected output: throughput: XX.X tok/s
```

**Intervention rate + flip rate** (aggregated from server logs; run from repo root):
```bash
cd /dstack/persistent/SIA/sia-repo/0g-sparse-inference-alignment
docker compose logs sia-vl30b 2>&1 | grep "DONE" | tail -200 | \
python -u -c "
import sys, re
ratios, flips = [], []
for line in sys.stdin:
    m = re.search(r'ratio=(\d+\.?\d*)%', line)
    if m: ratios.append(float(m.group(1)))
    m = re.search(r'top1_flip=\d+/\d+ \((\d+\.?\d*)%\)', line)
    if m: flips.append(float(m.group(1)))
print(f'intervention rate  mean={sum(ratios)/len(ratios):.1f}%  n={len(ratios)}' if ratios else 'no ratio data')
print(f'flip rate  mean={sum(flips)/len(flips):.1f}%  n={len(flips)}' if flips else 'no flip data')
"
```

Healthy reference values (see [§4 smoke test health indicators](#verify-sia-is-truly-intervening-not-degraded-to-nosia)): intervention rate 10–40%, flip rate 50–80%.

### Phase 3 — Comparison Results

```bash
python - <<'EOF'
import json, statistics

def stats(p):
    d = json.load(open(p))
    rs = [r["reward"] for r in d if r.get("reward") is not None]
    return statistics.mean(rs), statistics.median(rs), len(rs), len(d)

nosia_mean, nosia_p50, n1, t1 = stats("/workspace/exp/alpaca_vl30b_b2_nosia_scored.json")
sia_mean,   sia_p50,   n2, t2 = stats("/workspace/exp/alpaca_vl30b_b2_sia_scored.json")
delta_abs = sia_mean - nosia_mean
delta_rel = delta_abs / abs(nosia_mean) * 100

print(f"noSIA  {n1}/{t1} scored  mean={nosia_mean:+.4f}  p50={nosia_p50:+.4f}")
print(f"SIA    {n2}/{t2} scored  mean={sia_mean:+.4f}  p50={sia_p50:+.4f}")
print(f"Δ      {delta_abs:+.4f}  ({delta_rel:+.2f}%)")
EOF
```

After scoring, restart the service:
```bash
exit   # exit the scoring container
docker compose start sia-vl30b
```

---

## Related Docs

- [`qwen3-vl-30b-sia-eval-20260605.md`](qwen3-vl-30b-sia-eval-20260605.md) — VL-30B SIA evaluation summary (quality + performance), with local experiment results
- [`vl30b-b2-inproc-speedup-20260605.md`](vl30b-b2-inproc-speedup-20260605.md) — Original experiment report for b2 inproc 1.46× speedup (includes vllm 0.15/0.17.1/0.19 three-way comparison explaining why 0.17.1 is the sweet spot)
- [`../docker-compose.yml`](../docker-compose.yml) — **Path C production compose file** (GPU + mounts + restart + healthcheck + full startup command)
- [`../Dockerfile`](../Dockerfile) — Dockerfile shared by Path B / C (vllm 0.17.1 baked-in image)
- [`../scripts/docker_entrypoint_vl30b.sh`](../scripts/docker_entrypoint_vl30b.sh) — Entrypoint shared by Path B / C; registers sia_rm plugin at startup
- [`../scripts/docker_install_vl30b.sh`](../scripts/docker_install_vl30b.sh) — Command-line install script for Path A
- [`../scripts/setup_venv_vl30b_fast.sh`](../scripts/setup_venv_vl30b_fast.sh) — Underlying venv creation script (called by docker_install)
- [`../requirements/vl30b-b2-inproc.txt`](../requirements/vl30b-b2-inproc.txt) — pip dependency list
- [`../CLAUDE.md`](../CLAUDE.md) — Full venv matrix (Qwen3-14B / VL-30B / 0GM-35B)
- [`vllm-rm-backend.md`](vllm-rm-backend.md) — Detailed explanation of convert_rm_for_vllm.py (RM checkpoint conversion principles)
