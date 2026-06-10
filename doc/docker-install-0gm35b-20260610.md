# 0GM-35B SIA 在 Docker 里跑通 — 完整部署指南

**最后更新**: 2026-06-10  
**适用 docker image**: `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel`  
**适用配置**: 0GM-1.0-35B-A3B SIA b2 inproc 加速路径 (vllm 0.18.0, [`requirements/0gm35b-b2-inproc.txt`](../requirements/0gm35b-b2-inproc.txt))

> 本 doc 覆盖 3 条部署路径 (临时跑 / Dockerfile build / **生产 compose**), 每条都包含: **构建 image** → **启动 server** → **smoke test** → **停止 / 重启 / 清理**。生产部署推荐直接看 [Path C](#path-c--docker-compose--生产推荐)。

---

## 与 VL-30B 部署的关键差异

| 维度 | VL-30B ([doc](docker-install-vl30b-20260606.md)) | **0GM-35B (本 doc)** |
|---|---|---|
| vllm 版本 | 0.17.1 | **0.18.0** |
| requirements | `vl30b-b2-inproc.txt` | **`0gm35b-b2-inproc.txt`** |
| image tag | `sia-vl30b:0.17.1` | **`sia-0gm35b:0.18.0`** |
| Dockerfile | `Dockerfile` | **`Dockerfile.0gm35b`** |
| compose 文件 | `docker-compose.yml` | **`docker-compose.0gm35b.yml`** |
| venv 路径 | `/opt/venv-vl30b` | **`/opt/venv-0gm35b`** |
| 主 LLM arch | `Qwen3VLMoe` | **`Qwen3_5MoeForConditionalGeneration`** |
| LLM CUDA graph | `SIA_LLM_CUDAGRAPH=piecewise` | **不设置**（默认 FULL_AND_PIECEWISE，快 ~2×） |
| RM CUDA graph | `SIA_RM_CUDAGRAPH=piecewise` | **`SIA_RM_CUDAGRAPH=none`**（必须 eager，否则 RuntimeError） |
| `--max_model_len` | 4096 | **2048** |
| GPU 最低要求 | 80 GB (H100/H200) | **140 GB+ (H200)** — 35B 权重 ~70 GB，80 GB 不够 |

> **为什么 vllm 必须是 0.18.0？** 0.17.x 不支持 `Qwen3_5MoeForConditionalGeneration` arch（0GM-35B 无法加载）；0.19.x 的 PIECEWISE 改为 runtime capture，与 b2 inproc 100% 冲突（穷举所有配置均失败）。0.18.0 是唯一可用版本，详见 [`doc/0gm-35b-b2-inproc-speedup-20260609.md`](0gm-35b-b2-inproc-speedup-20260609.md)。

---

## 前提

### 1. 主机硬件 + NVIDIA 驱动

- **NVIDIA H200（141 GB）或更大显存的 GPU**（0GM-35B 权重 ~70 GB，A100/H100 80 GB 单卡不足）
- NVIDIA 驱动已装：`nvidia-smi` 应当可以正常运行

### 2. nvidia-container-toolkit

让 Docker 容器能访问 GPU。先检查是否已装：

```bash
docker run --rm --gpus all ubuntu:22.04 nvidia-smi
```

如果报错 (`could not select device driver "nvidia"`) 说明还没装，按以下步骤装 (Ubuntu/Debian)：

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
# 验证
docker run --rm --gpus all ubuntu:22.04 nvidia-smi
```

### 3. Docker + Compose 版本

Docker Compose V2+ 即可（`docker compose` 而非 `docker-compose`）。验证：

```bash
docker compose version
```

### 4. 目录结构

Path B/C 把整个 `/dstack/persistent/SIA` 挂载到容器 `/workspace`，需要提前建好这个结构：

```
/dstack/persistent/SIA/
├── models/
│   ├── 0GM-1.0-35B-A3B/                    ← 主 LLM，~70 GB  (§5.2)
│   ├── Qwen3-4B-Base/                       ← RM convert 原料，~8 GB   (§5.3，仅选项 B)
│   ├── SIA-checkpoints/                     ← LoRA checkpoints，~3 GB  (§5.3，仅选项 B)
│   └── VM-Qwen3-4B-merged-for-vllm/        ← RM 最终 checkpoint，~11 GB (§5.4)
└── sia-repo/
    └── 0g-sparse-inference-alignment/       ← 本仓库 (§5.1)
```

```bash
mkdir -p /dstack/persistent/SIA/models
mkdir -p /dstack/persistent/SIA/sia-repo
```

### 5. 仓库 + 模型权重 (~81 GB 最少；含 convert 原料共 ~92 GB)

> 需要 `huggingface-cli`：`pip install huggingface-hub`

#### 5.1 Clone 仓库

```bash
git clone <your-repo-url> \
  /dstack/persistent/SIA/sia-repo/0g-sparse-inference-alignment
```

#### 5.2 主 LLM — 0GM-1.0-35B-A3B (~70 GB)

0GM 是 0G Foundation 的内部模型，下载方式请联系模型管理员获取具体路径或 HuggingFace 仓库名。示例：

```bash
# 替换 <ORG>/<REPO> 为实际仓库名
huggingface-cli download <ORG>/0GM-1.0-35B-A3B \
  --local-dir /dstack/persistent/SIA/models/0GM-1.0-35B-A3B
```

或从已有存储直接 rsync/copy：

```bash
rsync -av /path/to/existing/0GM-1.0-35B-A3B/ \
  /dstack/persistent/SIA/models/0GM-1.0-35B-A3B/
```

验证：
```bash
ls /dstack/persistent/SIA/models/0GM-1.0-35B-A3B/
# 预期看到: config.json  model-*.safetensors  tokenizer*.json  ...
```

#### 5.3 RM 原料 — Qwen3-4B-Base + VM-Qwen3-4B-Base LoRA (~11 GB)（仅 §5.4 选项 B 需要）

```bash
# Qwen3-4B-Base: RM 的基础模型 (~8 GB)
huggingface-cli download Qwen/Qwen3-4B-Base \
  --local-dir /dstack/persistent/SIA/models/Qwen3-4B-Base

# VM-Qwen3-4B-Base: 论文作者提供的 LoRA checkpoint (~3 GB)
huggingface-cli download Runyi-Hu/SIA \
  --local-dir /dstack/persistent/SIA/models/SIA-checkpoints
```

下载完后 LoRA 路径为 `/dstack/persistent/SIA/models/SIA-checkpoints/VM-Qwen3-4B-Base/`。

#### 5.4 获取 VM-Qwen3-4B-merged-for-vllm

**选项 A：直接下载（推荐，已融合好的版本）**

```bash
huggingface-cli download TengGao/VM-Qwen3-4B-merged-for-vllm_public \
  --local-dir /dstack/persistent/SIA/models/VM-Qwen3-4B-merged-for-vllm
```

**选项 B：自行 convert（换了别的 RM / LoRA 时用这个）**

```bash
# 在宿主机 Python 环境里跑（不依赖 vllm venv，只需要 transformers + peft）
pip install transformers peft safetensors

python /dstack/persistent/SIA/sia-repo/0g-sparse-inference-alignment/scripts/convert_rm_for_vllm.py \
  --rm      /dstack/persistent/SIA/models/Qwen3-4B-Base \
  --rm_lora /dstack/persistent/SIA/models/SIA-checkpoints/VM-Qwen3-4B-Base \
  --output  /dstack/persistent/SIA/models/VM-Qwen3-4B-merged-for-vllm
```

预计耗时 2-5 分钟，输出 ~11 GB。

验证（无论选项 A 还是 B）：

```bash
ls /dstack/persistent/SIA/models/VM-Qwen3-4B-merged-for-vllm/
# 预期看到: config.json  model.safetensors  tokenizer*.json  ...
```

完成后确认必要模型到位：

```bash
ls /dstack/persistent/SIA/models/
# 选项 A: 0GM-1.0-35B-A3B  VM-Qwen3-4B-merged-for-vllm
# 选项 B: 0GM-1.0-35B-A3B  Qwen3-4B-Base  SIA-checkpoints  VM-Qwen3-4B-merged-for-vllm
```

**以上前提全部满足后，再进入下面的三条路径。**

---

## 三条路径选一条

| 路径 | 适用场景 | 步骤 |
|---|---|---|
| **Path A: 命令行装** | 临时想跑一次, 不留持久 image | Step 1 → Step 2 (跑安装脚本) → Step 3 → Step 4 |
| **Path B: Dockerfile 构建** ⚡ | 跑多次 / 分发给别人; 启动后无依赖安装但仍需手动起 server | Step 1B (build image) → Step 1B' (docker run) → Step 3 → Step 4 |
| **Path C: docker compose** ⭐ **生产推荐** | 生产部署 / 自动重启 / 健康检查; **server 完全自动起** | Step 1C (`docker compose -f docker-compose.0gm35b.yml up -d`) → Step 4 |

---

## Path A — 命令行装

### Step 1 — 启动 docker container

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

如果还没 clone repo，在**容器里**再 clone（pytorch image 默认没装 git，必须先装）：
```bash
apt-get update && apt-get install -y --no-install-recommends git ca-certificates
mkdir -p /workspace/sia-repo
cd /workspace/sia-repo && git clone <your-repo-url> 0g-sparse-inference-alignment
```

> 推荐做法：在**主机上**先按前提 §5 clone 好再用 `-v /dstack/persistent/SIA:/workspace` 挂进来，容器内就不用装 git 了。

### Step 2 — 跑安装脚本 (容器内)

```bash
cd /workspace/sia-repo/0g-sparse-inference-alignment
bash scripts/docker_install_0gm35b.sh
```

脚本按顺序做的事:
1. **apt 依赖** — 检查 + 自动装缺失的：`git ca-certificates curl` + `python${ver}-venv`
2. `python3 -m venv /opt/venv-0gm35b`（与 docker 预装的 conda torch 2.11 隔离）
3. `pip install -r requirements/0gm35b-b2-inproc.txt`（装 vllm 0.18.0 + torch 2.10.0 + 全部依赖）
4. `pip install -e .`（注册 `sia_rm` vllm 插件）
5. **自检**：import vllm/torch，验证 CUDA 可用，验证 `Qwen3_5MoeForConditionalGeneration` arch 注册，验证 `unlock_workspace` API，验证 sia_rm 插件可发现

预计耗时：**5-10 分钟**（主要是 vllm 0.18.0 wheel 下载，~250 MB）。

成功后会打印类似：
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

## Path B — Dockerfile 构建 (推荐, 跑多次或分发用)

仓库 + 模型仍然走 bind-mount，不进 image（image 维持小巧 ~10 GB）。

### Step 1B — 构建 image (一次性, ~5-10 min)

在主机上，`cd` 到仓库根（含 `Dockerfile.0gm35b`）：
```bash
cd /dstack/persistent/SIA/sia-repo/0g-sparse-inference-alignment
docker build -f Dockerfile.0gm35b -t sia-0gm35b:0.18.0 .
```

构建过程做的事（跟 Path A 的 install 脚本对应）：
1. `apt-get install python3.12-venv ca-certificates curl`
2. `python3 -m venv /opt/venv-0gm35b`
3. `pip install -r requirements/0gm35b-b2-inproc.txt`（vllm 0.18.0 + 全部 Python 依赖）
4. COPY entrypoint 脚本进 image，容器启动时自动 `pip install -e <mounted-repo>` 注册 sia_rm 插件（1-2 秒）

### Step 1B' — 启动 container

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

进容器后看到：
```
[entrypoint] registering sia_rm vllm plugin (pip install -e /workspace/sia-repo/0g-sparse-inference-alignment)
root@xxxx:/workspace/sia-repo/0g-sparse-inference-alignment#
```

venv 已在 `$PATH`，直接 `python --version` 就能看到 venv 的 python。**跳过 Step 2 整段，直接走 Step 3 启 server**。

### (可选) 验证 image 内的依赖都齐了

```bash
python -c "
import vllm, torch
from importlib.metadata import entry_points
print('vllm :', vllm.__version__)
print('torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())
print('sia_rm plugin:', [e.name for e in entry_points(group='vllm.general_plugins') if 'sia' in e.name])
"
```
预期：`vllm: 0.18.0`，`torch: 2.10.0+cu128`，`sia_rm plugin: ['sia_rm']`。

---

## Path C — docker compose (⭐ 生产推荐)

把 GPU 资源 / 挂载 / 重启策略 / 健康检查 / 完整 server 启动命令全部写进 [`docker-compose.0gm35b.yml`](../docker-compose.0gm35b.yml)。**一条命令拉起全套**，server 自动起来跑。

### Step 1C — 起服务

在 host 上，cd 到 repo 根（含 `docker-compose.0gm35b.yml`）：
```bash
cd /dstack/persistent/SIA/sia-repo/0g-sparse-inference-alignment
docker compose -f docker-compose.0gm35b.yml up -d --build   # 首次: build image (~5-10 min) + 后台起 container
```

之后日常运维：
```bash
docker compose -f docker-compose.0gm35b.yml logs -f sia-0gm35b   # tail server log
docker compose -f docker-compose.0gm35b.yml ps                    # 看健康状态
docker compose -f docker-compose.0gm35b.yml restart sia-0gm35b   # 重启 server
docker compose -f docker-compose.0gm35b.yml down                  # 停 + 移除 container
```

### 跳过 Step 2 / Step 3，直接到 Step 4

Path C 之下，**server 在 `docker compose up -d` 后已经在自动启动**，不需要 Step 2（装 deps）和 Step 3（手动启 server）。等 `docker compose -f docker-compose.0gm35b.yml ps` 显示 `(healthy)` 状态（大约 20-25 分钟首次启动，包括 35B 权重加载 + cudagraph capture；后续重启 ~5-8 分钟）后，直接跳到 Step 4 smoke test。

---

## Step 3 — 启 SIA server (主 LLM + Value Model 同进程) — Path A / B 手动启动

> **谁需要看这一步**：Path A 用户（Step 2 装完依赖后），Path B 用户（Step 1B' 进容器后）。**Path C 用户跳过**——server 已经被 compose 自动启起来了。

> ℹ️ **b2 inproc 是单进程拓扑**：主 LLM（0GM-35B）+ Value Model（VM-Qwen3-4B）在**同一个 Python 进程里嵌套跑**，RM 调用是直接 Python 函数调用而非 HTTP，**不需要再单独开一个 RM server**。

```bash
source /opt/venv-0gm35b/bin/activate

mkdir -p /workspace/sia-logs
LOG="/workspace/sia-logs/sia_server_$(date +%Y%m%d_%H%M%S).log"
echo "[launch] log → $LOG"

nohup env \
  SIA_RM_CUDAGRAPH=none \
  SIA_RM_MULTIPROCESS=0 \
  python src/sia_vllm_server.py \
    --llm        /workspace/models/0GM-1.0-35B-A3B \
    --rm_backend b2 \
    --rm_model   /workspace/models/VM-Qwen3-4B-merged-for-vllm \
    --llm_gpu_mem    0.55 \
    --rm_b2_gpu_mem  0.15 \
    --topk 10 --weight 1.0 --entropy_threshold 1.0 \
    --max_model_len 2048 --port 8000 \
    > "$LOG" 2>&1 &

echo "[launch] pid=$!  (process detached from this shell, log: $LOG)"
```

加载过程实时观察（`Ctrl+C` 退出 tail 不影响 server）：
```bash
tail -f "$LOG"
```

参数说明：

| 参数 / 环境变量 | 作用 |
|---|---|
| `--llm` | **主推理 LLM**（0GM-35B），占 GPU 55% 显存 |
| `--rm_backend b2` | 嵌套同进程 RM backend（vllm 0.18.0 sweet-spot 路径） |
| `--rm_model` | **Value Model**（VM-Qwen3-4B），嵌套在主进程内，占 GPU 15% 显存 |
| `--llm_gpu_mem 0.55 + --rm_b2_gpu_mem 0.15` | 合计 70%，剩 30% 给 cudagraph + KV cache 头空间（H200 141GB 上测试通过） |
| `--topk 10 --weight 1.0 --entropy_threshold 1.0` | SIA 参数：10 个候选，干预权重 1.0，entropy > 1.0 才介入（≈ 20% 干预率） |
| `--max_model_len 2048` | 限制最长序列 2048 token，控制 KV cache 占用 |
| `SIA_RM_CUDAGRAPH=none` | RM 用 eager 模式（**必须**，否则 vllm 0.18.0 推理时抛 RuntimeError） |
| `SIA_RM_MULTIPROCESS=0` | RM 与主 LLM 同进程，不走 subprocess |
| `SIA_LLM_CUDAGRAPH` | **不设置**，主 LLM 默认走 FULL_AND_PIECEWISE（比强制 piecewise 快 ~2×） |

等待启动（35B 权重加载 ~15 min on H200，cudagraph capture ~3-5 min，RM 加载 ~1 min），直到日志出现：
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

检查 server 是否还活：
```bash
pgrep -af sia_vllm_server
```

停掉 server：
```bash
pkill -f sia_vllm_server.py
```

---

## Step 4 — Smoke test (验证 server 真活)

Server 已 ready（Path A/B 看到 "Uvicorn running on http://0.0.0.0:8000"，Path C 看到 `(healthy)` 状态）后，跑一个 chat completion 验证端到端工作。

### 4.1 用 curl (最简单)

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/0GM-1.0-35B-A3B",
    "messages": [{"role":"user","content":"What are 3 colors of fruit?"}],
    "max_tokens": 100,
    "temperature": 0.7
  }' | python3 -m json.tool
```

### 4.2 用 OpenAI Python SDK

```bash
pip install openai
```
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
resp = client.chat.completions.create(
    model="/workspace/models/0GM-1.0-35B-A3B",
    messages=[{"role": "user", "content": "What are 3 colors of fruit?"}],
    max_tokens=100, temperature=0.7,
)
print(resp.choices[0].message.content)
print("usage:", resp.usage)
```

### 4.3 Path C 用户从容器内测

```bash
docker compose -f docker-compose.0gm35b.yml exec sia-0gm35b bash -c \
  'curl -s -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d "{\"model\":\"/workspace/models/0GM-1.0-35B-A3B\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":50}"'
```

### 验证 SIA 真的在干预

预期 response 是正常 JSON，同时 server 日志会打印：
```
[SIA] req=0 DONE  intervened=N/M  ratio=X%  top1_flip=K/N (Y%)
```

健康指标：

| 指标 | 健康范围 | 红灯 |
|---|---|---|
| intervention ratio | 15–30%（entropy_threshold=1.0 下约 20%） | **0%** → RM 死了，SIA 退化为 no-op |
| top1 flip rate | 50–80% | 0% / 100% → bug |
| RM error 日志 | 0 | >0 → RM forward 失败（检查 `SIA_RM_CUDAGRAPH=none` 是否生效） |

看日志方法：
- **Path A/B**：`tail -f "$LOG"`
- **Path C**：`docker compose -f docker-compose.0gm35b.yml logs -f sia-0gm35b`

### 4.4 Per-request SIA 参数测试

SIA 支持在单次 request 里覆盖全局参数（`sia_weight` / `sia_topk` / `sia_entropy_threshold`），无需重启 server。

#### 关闭 SIA 干预 (`sia_weight=0`)

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/0GM-1.0-35B-A3B",
    "messages": [{"role":"user","content":"What are 3 colors of fruit?"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "sia_weight": 0
  }' | python3 -m json.tool
```

预期 server 日志：`ratio=0.0%`——RM 完全不调用，退化为纯 vLLM 推理。

#### 强制每 token 都干预 (`sia_entropy_threshold=0`)

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/0GM-1.0-35B-A3B",
    "messages": [{"role":"user","content":"What are 3 colors of fruit?"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "sia_entropy_threshold": 0
  }' | python3 -m json.tool
```

预期 server 日志：`ratio=100%`——所有 token 都经过 RM 评分。

---

## Step 5 — 停止 / 重启 / 清理 (per path)

### Path A / B (手动起 docker run)

| 操作 | 命令 |
|---|---|
| 停 server（进程级） | `pkill -f sia_vllm_server.py` |
| 看 server 是否还在跑 | `pgrep -af sia_vllm_server` |
| 重启 server | 重新跑 Step 3 的 nohup 命令 |
| 退出 container（server 也会被 kill） | 在容器里 `exit` |

### Path C (docker compose) — 生产推荐

| 操作 | 命令 |
|---|---|
| 停 server + 移除 container（保留 named volumes） | `docker compose -f docker-compose.0gm35b.yml down` |
| 停 server + 移除 container + 清理 volumes | `docker compose -f docker-compose.0gm35b.yml down -v` |
| 仅重启 server，不重 build | `docker compose -f docker-compose.0gm35b.yml restart sia-0gm35b` |
| 改了源码后重启（源码是 bind-mount，直接 restart 即可） | `docker compose -f docker-compose.0gm35b.yml restart sia-0gm35b` |
| 改了 `requirements/0gm35b-b2-inproc.txt` 后重 build | `docker compose -f docker-compose.0gm35b.yml up -d --build` |
| 查看资源占用 | `docker stats sia-0gm35b-server` |
| 直接进容器 debug（server 仍在跑） | `docker compose -f docker-compose.0gm35b.yml exec sia-0gm35b bash` |

> **GPU 显存释放**：container 停掉后 GPU 显存通常几秒内自动释放。若 `nvidia-smi` 还显示占用，用 `pkill -9 -f sia_vllm_server.py` 强杀。

---

## 关键陷阱 / 注意事项

| 陷阱 | 解决 |
|---|---|
| 用了 vllm 0.17.x | 0.17.x 不支持 `Qwen3_5MoeForConditionalGeneration`，0GM-35B 无法加载。**必须 vllm 0.18.0** |
| 用了 vllm 0.19.x | 0.19.x PIECEWISE 改为 runtime capture，与 b2 inproc 100% 冲突，所有配置均失败。**必须 vllm 0.18.0** |
| 设置了 `SIA_RM_CUDAGRAPH=piecewise`（或 `full`） | vllm 0.18.0 RM 在推理时遇到新 batch_descriptor 触发 runtime capture，抛 `RuntimeError: CUDA graph capturing detected at an inappropriate time`。**必须 `SIA_RM_CUDAGRAPH=none`** |
| 设置了 `SIA_LLM_CUDAGRAPH=piecewise` | 主 LLM 被锁定在 PIECEWISE-only 模式，noSIA 从 ~107 tok/s 降至 ~57 tok/s（慢 ~2×）。**不要设置**，让 vllm 默认走 FULL_AND_PIECEWISE |
| 用 A100 / H100（80 GB）单卡 | 0GM-35B 权重约 70 GB，80 GB GPU 无法同时容纳 LLM（55%）+ RM（15%）+ KV cache。**需要 H200（141 GB）或更大显存** |
| Docker image 自带 PyTorch 2.11（conda 里），跟 vllm 0.18.0 需要的 torch 2.10.0 冲突 | 用 `python3 -m venv` 创建隔离 venv（脚本已处理），**不要直接 pip install vllm 到 conda env** |
| `--shm-size=16g --ipc=host` 未加 | vllm 用大量共享内存做 KV cache，默认 docker 的 64 MB 不够 |
| `--max_model_len` 未限制 | 不设置时 vllm 会按模型最大支持序列长度分配 KV cache，容易 OOM。**建议 `--max_model_len 2048`** |
| Path C 首次启动看到 `(unhealthy)` | `start_period=25m`，给 35B 权重加载 + cudagraph capture 留余地。25 min 内的 unhealthy 是正常的，之后还 unhealthy 才是真问题 |
| 生产长跑日志撑满磁盘 | 在 `docker-compose.0gm35b.yml` 的 service 下加：`logging: {driver: "json-file", options: {max-size: "500m", max-file: "5"}}` |

---

## 显存预算 — 其他进程占用 GPU 时如何调

如果 GPU 上还有别的进程，用 `nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader` 算出 free 显存，然后调小 `--llm_gpu_mem` 和 `--rm_b2_gpu_mem`（vllm 把它们当作**总显存**的百分比，不是 free 显存）。

同时建议加 `--max_model_len 1024`（从 2048 降一半）减少 KV cache 需求。

如果 Path C 也想适配，改 `docker-compose.0gm35b.yml` 的 `command:` 字段即可，不用重 build image。

---

## AlpacaEval 评测 — SIA vs noSIA (Skywork 打分)

在 docker 容器内复用 SIA server 跑 AlpacaEval 效果评测。两个 arm 共用同一个运行中的 server，通过 per-request `sia_weight=0` 区分，无需切换服务。

**参考实测结果**（2026-06-10，H200，AlpacaEval 200Q，topk=10，entropy_threshold=1.0）：

| 配置 | 吞吐 | SkyWork 均分 | 干预率 | flip rate |
|---|---|---|---|---|
| noSIA | ~107 tok/s | 24.01 | — | — |
| **SIA（b2 inproc + stable prefix）** | **~65 tok/s** | **29.36（+22.3%）** | **20.1%** | **65.1%** |

### 前提

```bash
# 确认 Skywork 评分模型
ls /dstack/persistent/SIA/models/Skywork-Reward-V2-Llama-3.1-8B/
# 若未下载:
# huggingface-cli download Skywork/Skywork-Reward-V2-Llama-3.1-8B \
#   --local-dir /dstack/persistent/SIA/models/Skywork-Reward-V2-Llama-3.1-8B

# 确认 AlpacaEval 数据集
ls /dstack/persistent/SIA/sia-repo/0g-sparse-inference-alignment/data/alpaca_eval/alpaca_eval.json
```

### Phase 1 — Generation（server 正常运行，无需停机）

进入容器：
```bash
docker compose -f docker-compose.0gm35b.yml exec sia-0gm35b bash
cd /workspace/sia-repo/0g-sparse-inference-alignment
MODEL=/workspace/models/0GM-1.0-35B-A3B
mkdir -p /workspace/exp
```

**SIA arm**（server 默认参数：topk=10, weight=1.0, entropy_threshold=1.0）：
```bash
nohup python -u eval/alpaca_eval.py \
  --base_url http://localhost:8000/v1 \
  --model "$MODEL" \
  --limit 200 --max_tokens 1800 \
  --temperature 1.0 --top_p 0.95 --top_k 20 --repetition_penalty 1.0 \
  --output /workspace/exp/alpaca_0gm35b_sia_$(date +%Y%m%d_%H%M%S).json \
  > /workspace/exp/alpaca_0gm35b_sia_gen.log 2>&1 &
echo "SIA arm PID=$!"
```

**noSIA arm**（`--sia_weight 0` 关掉 RM 干预）：
```bash
nohup python -u eval/alpaca_eval.py \
  --base_url http://localhost:8000/v1 \
  --model "$MODEL" \
  --limit 200 --max_tokens 1800 \
  --temperature 1.0 --top_p 0.95 --top_k 20 --repetition_penalty 1.0 \
  --sia_weight 0 \
  --output /workspace/exp/alpaca_0gm35b_nosia_$(date +%Y%m%d_%H%M%S).json \
  > /workspace/exp/alpaca_0gm35b_nosia_gen.log 2>&1 &
echo "noSIA arm PID=$!"
```

> SIA arm 约 50-60 min（~65 tok/s）；noSIA arm 约 15 min（~107 tok/s）。

### Phase 2 — Skywork 打分（需要停 server 释放显存）

```bash
# 在宿主机停 server
docker compose -f docker-compose.0gm35b.yml stop sia-0gm35b

# 起临时容器跑打分
docker compose -f docker-compose.0gm35b.yml run --rm sia-0gm35b bash
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

各约 3-5 min（Skywork 8B BF16，~2.5 题/s）。

### Phase 3 — 对比结果

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

打分完重启服务：
```bash
exit   # 退出打分容器
docker compose -f docker-compose.0gm35b.yml start sia-0gm35b
```

---

## 相关 doc

- [`docker-install-vl30b-20260606.md`](docker-install-vl30b-20260606.md) — VL-30B 同类型部署文档（结构与本文一致）
- [`0gm-35b-b2-inproc-speedup-20260609.md`](0gm-35b-b2-inproc-speedup-20260609.md) — 为什么 vllm 0.18.0 是 sweet spot（穷举失败配置 + 根因分析）
- [`0gm-35b-sia-perf-breakdown-20260609.md`](0gm-35b-sia-perf-breakdown-20260609.md) — 完整性能优化历程（含 stable prefix 优化 + AlpacaEval 200Q 结果）
- [`sia-speedup-summary-20260610.md`](sia-speedup-summary-20260610.md) — 两个模型优化汇总（原理 + 数据对比）
- [`../docker-compose.0gm35b.yml`](../docker-compose.0gm35b.yml) — Path C 生产 compose 文件
- [`../Dockerfile.0gm35b`](../Dockerfile.0gm35b) — Path B / C 共用的 Dockerfile
- [`../scripts/docker_entrypoint_0gm35b.sh`](../scripts/docker_entrypoint_0gm35b.sh) — Path B / C 共用 entrypoint
- [`../scripts/docker_install_0gm35b.sh`](../scripts/docker_install_0gm35b.sh) — Path A 安装脚本
- [`../requirements/0gm35b-b2-inproc.txt`](../requirements/0gm35b-b2-inproc.txt) — pip 依赖清单
- [`../CLAUDE.md`](../CLAUDE.md) — 整体 venv 矩阵
