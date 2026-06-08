# VL-30B SIA 在 Docker 里跑通 — 完整部署指南

**最后更新**: 2026-06-07
**适用 docker image**: `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel`
**适用配置**: VL-30B (Qwen3-VL-30B-A3B-Instruct) SIA b2 inproc 加速路径 (vllm 0.17.1, [`requirements/vl30b-b2-inproc.txt`](../requirements/vl30b-b2-inproc.txt))

> 本 doc 覆盖 3 条部署路径 (临时跑 / Dockerfile build / **生产 compose**), 每条都包含: **构建 image** → **启动 server** → **smoke test** → **停止 / 重启 / 清理**。新读者直接跳到 [三条路径选一条](#三条路径选一条).

## 快速参考 (TL;DR)

**生产部署** (复制粘贴即用, 假设前提全部满足, 模型已在 `/dstack/persistent/SIA/models/`):

> **新机器?** 先完整走完 [前提](#前提) — 需要安装 nvidia-container-toolkit、下载模型权重 (~71-82 GB)（RM checkpoint 可直接下载，无需 convert），再回来执行下面的命令。

```bash
cd /dstack/persistent/SIA/sia-repo/0g-sparse-inference-alignment
docker compose up -d --build         # 起服务 (首次 ~15-20 min 含 build + warmup)
docker compose logs -f sia-vl30b     # 看启动 + 运行日志
docker compose ps                    # 等 STATUS = (healthy)
curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"/workspace/models/Qwen3-VL-30B-A3B-Instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":50}'
docker compose down                  # 停服务 (named volume 保留, 下次启动复用 compile cache)
```

## 前提

### 1. 主机硬件 + NVIDIA 驱动

- NVIDIA GPU (H100 / H200 / A100, **80GB+ 显存**)
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
docker compose version    # 有输出即可，无版本下限要求
```

> `docker-compose.yml` 使用 `deploy: resources: reservations: devices:` GPU 语法，全版本兼容。

### 4. 目录结构

Path B/C 把整个 `/dstack/persistent/SIA` 挂载到容器 `/workspace`，需要提前建好这个结构：

```
/dstack/persistent/SIA/
├── models/
│   ├── Qwen3-VL-30B-A3B-Instruct/     ← 主 LLM，~60 GB  (Step 5.1)
│   ├── Qwen3-4B-Base/                  ← RM convert 原料，~8 GB   (Step 5.2)
│   ├── SIA-checkpoints/                ← LoRA checkpoints，~3 GB  (Step 5.2)
│   └── VM-Qwen3-4B-merged-for-vllm/   ← RM 最终 checkpoint，~11 GB (Step 5.4 下载或 convert)
└── sia-repo/
    └── 0g-sparse-inference-alignment/  ← 本仓库 (Step 5.1)
```

```bash
mkdir -p /dstack/persistent/SIA/models
mkdir -p /dstack/persistent/SIA/sia-repo
```

### 5. 仓库 + 模型权重 (~71 GB 最少；含 convert 原料共 ~82 GB)

> 需要 `huggingface-cli`：`pip install huggingface-hub`

#### 5.1 Clone 仓库（convert 脚本在仓库里，先 clone）

```bash
git clone <your-repo-url> \
  /dstack/persistent/SIA/sia-repo/0g-sparse-inference-alignment
```

#### 5.2 主 LLM — Qwen3-VL-30B-A3B-Instruct (~60 GB)

```bash
huggingface-cli download Qwen/Qwen3-VL-30B-A3B-Instruct \
  --local-dir /dstack/persistent/SIA/models/Qwen3-VL-30B-A3B-Instruct
```

#### 5.3 RM 原料 — Qwen3-4B-Base + VM-Qwen3-4B-Base LoRA (~11 GB)（仅 §5.4 选项 B 需要）

```bash
# Qwen3-4B-Base: RM 的基础模型 (~8 GB)
huggingface-cli download Qwen/Qwen3-4B-Base \
  --local-dir /dstack/persistent/SIA/models/Qwen3-4B-Base

# VM-Qwen3-4B-Base: 论文作者提供的 LoRA checkpoint (~3 GB，含全部 VM 变种)
huggingface-cli download Runyi-Hu/SIA \
  --local-dir /dstack/persistent/SIA/models/SIA-checkpoints
```

下载完后 LoRA 路径为 `/dstack/persistent/SIA/models/SIA-checkpoints/VM-Qwen3-4B-Base/`。

#### 5.4 获取 VM-Qwen3-4B-merged-for-vllm

**选项 A：直接下载（推荐，已融合好的版本）**

融合后的 checkpoint 已上传至 HuggingFace，可以直接下载，跳过融合步骤：

```bash
huggingface-cli download TengGao/VM-Qwen3-4B-merged-for-vllm_public \
  --local-dir /dstack/persistent/SIA/models/VM-Qwen3-4B-merged-for-vllm
```

**选项 B：自行 convert（换了别的 RM / LoRA 时用这个）**

如果需要用不同的 base model 或 LoRA，运行 convert 脚本重新融合：

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
# 选项 A: Qwen3-VL-30B-A3B-Instruct  VM-Qwen3-4B-merged-for-vllm
# 选项 B: Qwen3-VL-30B-A3B-Instruct  Qwen3-4B-Base  SIA-checkpoints  VM-Qwen3-4B-merged-for-vllm
```

**以上前提全部满足后，再进入下面的三条路径。**

---

## 三条路径选一条

| 路径 | 适用场景 | 步骤 |
|---|---|---|
| **Path A: 命令行装** | 临时想跑一次, 不留持久 image | Step 1 → Step 2 (跑安装脚本) → Step 3 → Step 4 |
| **Path B: Dockerfile 构建** ⚡ | 跑多次 / 分发给别人 / CI; 启动后无依赖安装但仍需手动起 server | Step 1B (build image) → Step 1B' (docker run) → Step 3 → Step 4 |
| **Path C: docker compose** ⭐ **生产推荐** | 生产部署 / 自动重启 / 健康检查 / 持久 compile cache; **server 完全自动起** | Step 1C (`docker compose up -d`) → Step 4 (smoke test) |

Path A 每次进新容器都要重装 deps (3-5 min); Path B 一次性 `docker build` 后, 后续 `docker run` 进去就能直接跑 server, 但要手动敲启动命令; Path C 把启动命令 + 重启策略 + 健康检查 + GPU + 挂载全部固化到 `docker-compose.yml`, `docker compose up` 一条命令搞定。

---

## Path A — 命令行装

### Step 1 — 启动 docker container

```bash
# 如果使用标准 /dstack/persistent/SIA 目录结构 (见前提 §4), 推荐整体挂载:
docker run -it --rm \
  --gpus all \
  --shm-size=16g \
  --ipc=host \
  -p 8000:8000 \
  -v /dstack/persistent/SIA:/workspace \
  pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel \
  bash

# 如果 repo 和 models 在不同目录, 也可以分开挂载 (路径按实际情况替换):
# docker run -it --rm --gpus all --shm-size=16g --ipc=host -p 8000:8000 \
#   -v /path/to/sia-repo/0g-sparse-inference-alignment:/workspace/sia-repo \
#   -v /path/to/models:/workspace/models \
#   pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel bash
```

如果还没 clone repo, 在**容器里**再 clone — pytorch image 默认没装 git, 必须先装:
```bash
apt-get update && apt-get install -y --no-install-recommends git ca-certificates
mkdir -p /workspace/sia-repo
cd /workspace/sia-repo && git clone <your-repo-url> 0g-sparse-inference-alignment
```

> 推荐做法: 在**主机上**先按前提 §5 clone 好再用 `-v /dstack/persistent/SIA:/workspace` 挂进来, 容器内就不用装 git 了。

## Step 2 — 跑安装脚本 (容器内)

```bash
cd /workspace/sia-repo/0g-sparse-inference-alignment
bash scripts/docker_install_vl30b.sh
```

脚本按顺序做的事:
1. **apt 依赖** — 检查 + 自动装缺失的: `git ca-certificates curl` + `python${ver}-venv` (probe `python3 -c "import ensurepip"`; 失败就查 python 版本装匹配包, 例 `python3.12-venv`。pytorch image 系统 python 默认不带 venv 模块, 这一步必须有)
2. `python3 -m venv /opt/venv-vl30b` (用系统 python, 跟 docker 预装的 conda env torch 2.11 隔离)
3. `pip install -r requirements/vl30b-b2-inproc.txt` (装 vllm 0.17.1 + torch 2.10.0 + 全部依赖)
4. `pip install -e .` (注册 `sia_rm` 这个 vllm 插件, 让 b2 inproc backend 能工作)
5. **自检**: import vllm/torch, 验证 CUDA 可用, 验证 Qwen3VLMoe arch 注册, 验证 `unlock_workspace` API 存在, 验证 sia_rm 插件可发现

预计耗时: **5-10 分钟** (主要是 vllm 0.17.1 wheel 下载, ~250 MB)。

成功后会打印类似:
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

## Path B — Dockerfile 构建 (推荐, 跑多次或分发用)

Path A 每次起新容器都要等 3-5 分钟装 deps。Path B 一次 `docker build` 把所有 Python 依赖 baked 进 image (vllm 0.17.1 + torch 2.10.0 + transformers ...), 之后每次 `docker run` 进去**直接可以跑 server, 零等待**。

仓库 + 模型仍然走 bind-mount, 不进 image (image 维持小巧 ~10 GB)。

### Step 1B — 构建 image (一次性, ~5-10 min)

在主机上, `cd` 到仓库根 (含 `Dockerfile`):
```bash
cd /dstack/persistent/SIA/sia-repo/0g-sparse-inference-alignment
docker build -t sia-vl30b:0.17.1 .
```

构建过程做的事 (跟 Path A 的 install 脚本对应):
1. `apt-get install python3.12-venv ca-certificates curl` (curl 装在 image 里以便 smoke test)
2. `python3 -m venv /opt/venv-vl30b`
3. `pip install -r requirements/vl30b-b2-inproc.txt` (vllm 0.17.1 + 全部 Python 依赖)
4. COPY 一个 entrypoint 脚本进 image, 容器启动时自动 `pip install -e <mounted-repo>` 注册 sia_rm 插件 (1-2 秒, 必要)

注意: **Dockerfile 不 COPY 源码**, 只 COPY `requirements/` 子目录用于 pip install。源码靠 bind-mount, 你改了 src/ 不用重 build image。

### Step 1B' — 启动 container

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

进容器后看到:
```
[entrypoint] registering sia_rm vllm plugin (pip install -e /workspace/sia-repo/0g-sparse-inference-alignment)
root@xxxx:/workspace/sia-repo/0g-sparse-inference-alignment#
```

venv 已经在 `$PATH` 里, 进容器直接 `python --version` 就能看到 venv 的 python。**跳过 Step 2 整段, 直接走 Step 3 启 server**。

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
预期: `vllm: 0.17.1`, `torch: 2.10.0+cu124`, `sia_rm plugin: ['sia_rm']`.

---

## Path C — docker compose (⭐ 生产推荐)

把 GPU 资源 / 挂载 / 重启策略 / 健康检查 / 完整 server 启动命令全部写进 [`docker-compose.yml`](../docker-compose.yml) (跟 Dockerfile 同在 repo 根)。**一条命令拉起全套**, server 自动起来跑。

### Step 1C — 起服务

在 host 上, cd 到 repo 根 (含 `docker-compose.yml`):
```bash
cd /dstack/persistent/SIA/sia-repo/0g-sparse-inference-alignment
docker compose up -d --build      # 首次: build image (~5-10 min) + 后台起 container
```

之后日常运维:
```bash
docker compose logs -f sia-vl30b  # tail server log (Ctrl+C 不杀 container)
docker compose ps                  # 看健康状态
docker compose restart sia-vl30b   # 重启 server (不重 build)
docker compose down                # 停 + 移除 container (named volume 保留)
```

### compose 文件做了什么 (跟 Path A/B 相比的差异)

| 维度 | Path A/B (docker run) | Path C (compose) |
|---|---|---|
| 启动 server | 进容器后手动敲 `python src/sia_vllm_server.py ...` | **自动启** (`command:` 字段) |
| 后台运行 | 你要写 `nohup ... &` | **docker 管 PID 1**, `up -d` 后台跑 |
| 日志 | 自己 redirect 到文件 | **`docker compose logs`** 自动收集 stdout/stderr |
| 进程崩了 | 死了就死了, 要手动重启 | **`restart: unless-stopped`** 自动拉起 |
| 健康检查 | 自己 curl 测 | **healthcheck**: 自动 curl `/health`, 失败标 unhealthy |
| 跨容器重启复用 compile cache | 无 | **named volume `vllm-compile-cache`** 持久化, 二次启动省 ~3-5 min torch.compile 时间 |
| GPU + shm + ipc | 命令行 flag 一堆 | 写在 yaml 里, 一处管理 |
| 环境差异 (dev/staging/prod) | 一份 shell 命令很难管 | `docker-compose.dev.yml` / `.prod.yml` 各一份, 重用 image |

### 启动命令: 为什么放 compose 不放 Dockerfile

| | Dockerfile `CMD` | docker-compose `command:` |
|---|---|---|
| 改参数 | 要重 build image | 改 yaml + `docker compose up` 重新 create container, **不重 build** |
| 同一 image 多场景 | 只能一份 default | dev/prod 各一份 compose 文件覆盖 |
| 调试 | `docker run sia-vl30b bash` 会被 CMD 干扰 | Dockerfile CMD 留 `bash`, compose 用 `command:` 覆盖; debug 时 `docker run` 直接进 bash, 生产用 compose | 
| 业界标准 | "image self-launching" 风格 | **微服务 + k8s 主流**: image 是通用 artifact, 启动配置在编排层 |

**最规范的做法** (本仓库就是这样):
- `Dockerfile` 的 `CMD ["bash"]` — debug 默认行为, 跟 `docker run` 友好
- `docker-compose.yml` 的 `command:` — 生产实际启动的命令, 完整参数固化

### 跳过 Step 2 / Step 3, 直接到 Step 4

Path C 之下, **server 在 `docker compose up -d` 后已经在自动启动**, 不需要 Step 2 (装 deps) 和 Step 3 (手动启 server)。等 `docker compose ps` 显示 `(healthy)` 状态 (大约 15-20 分钟首次启动, 包括首次 torch.compile; 后续重启 ~3-5 分钟) 后, 直接跳到 Step 4 smoke test。

---

## Step 3 — 启 SIA server (主 LLM + Value Model 同进程) — Path A / B 手动启动

> **谁需要看这一步**: Path A 用户 (Step 2 装完依赖后), Path B 用户 (Step 1B' 进容器后)。**Path C 用户跳过** — server 已经被 compose 自动启起来了。

> ℹ️ **重要 — b2 inproc 是单进程拓扑**: 跟 HTTP path 起两个 server (一个主 LLM, 一个 RM 在另一个端口) 不同, b2 inproc 让 **主 LLM (VL-30B) + Value Model (VM-Qwen3-4B) 在同一个 Python 进程里 nested 跑**, 共享一个 CUDA context, RM 调用是直接 Python 函数调用而非 HTTP。所以下面**这一条命令就同时启动了主 LLM + Value Model**, 不需要再开一个 RM server。

> Path B 容器里 venv 已在 PATH, `source /opt/venv-vl30b/bin/activate` 可省。Path A 用户记得激活。

```bash
source /opt/venv-vl30b/bin/activate

# 输出重定向到带时间戳的 log 文件, 用 nohup 让进程独立于当前 shell
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

加载过程实时观察 (`Ctrl+C` 退出 tail 不影响 server, server 仍在跑):
```bash
tail -f "$LOG"
```

参数对应:

| 参数 | 作用 |
|---|---|
| `--llm` | **主推理 LLM** (VL-30B), 占 GPU 55% 显存 |
| `--rm_backend b2` | 用 nested in-process backend (vllm 0.17.1 sweet-spot 路径) |
| `--rm_model` | **Value Model** (VM-Qwen3-4B), nested 在主进程内, 占 GPU 15% 显存 |
| `--llm_gpu_mem 0.55 + --rm_b2_gpu_mem 0.15` | 加起来 70%, 剩 30% 给 cudagraph + KV cache 头空间 |
| `--topk 10 --weight 1.0 --entropy_threshold 1.0` | SIA 算法参数: 每个 token 候选 10 个, 干预权重 1.0, entropy > 1.0 才介入 |
| `SIA_LLM_CUDAGRAPH=piecewise` | 强制主 LLM 用 PIECEWISE cudagraph (vllm 0.17.1 AOT 模式), 避免跟 nested RM 撞 |
| `SIA_RM_MULTIPROCESS=0` | RM 跟主 LLM 同进程 (InprocClient), 不走 subprocess |

等待启动 (主 LLM weights 加载 ~12 min on H200, 然后 cudagraph capture + warmup ~3 min, 再 Value Model 加载 ~1 min), 直到 `tail -f "$LOG"` 输出里看到:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

随时检查 server 是否还活:
```bash
ps -ef | grep sia_vllm_server | grep -v grep
# 或:
pgrep -af sia_vllm_server
```

要停掉 server:
```bash
pkill -f sia_vllm_server.py
```

## Step 4 — Smoke test (验证 server 真活)

Server 已 ready (Path A/B 看到 "Uvicorn running on http://0.0.0.0:8000", Path C 看到 `docker compose ps` 显示 `(healthy)`) 后, 跑一个 chat completion 验证端到端工作。

Server 暴露的是**OpenAI 兼容** API, 三种方式都行:

### 4.1 用 curl (最简单)

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

### 4.2 用 OpenAI Python SDK (生产代码集成可这么测)

```bash
pip install openai
```
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")   # key 不校验
resp = client.chat.completions.create(
    model="/workspace/models/Qwen3-VL-30B-A3B-Instruct",
    messages=[{"role": "user", "content": "What are 3 colors of fruit?"}],
    max_tokens=100, temperature=0.7,
)
print(resp.choices[0].message.content)
print("usage:", resp.usage)   # prompt_tokens / completion_tokens / total_tokens
```

### 4.3 Path C 用户也可以从容器内测 (省去暴露端口的麻烦)

```bash
docker compose exec sia-vl30b bash -c \
  'curl -s -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d "{\"model\":\"/workspace/models/Qwen3-VL-30B-A3B-Instruct\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":50}"'
```

### 验证 SIA 真的在干预 (不是退化成 noSIA)

预期 response 是正常 JSON, 同时 server 日志会打印:
```
[SIA] req=0 DONE  intervened=N/M  ratio=X%  top1_flip=K/N (Y%)
```

健康指标 (跟 doc [`qwen3-vl-30b-sia-eval-20260605.md` §2.4](qwen3-vl-30b-sia-eval-20260605.md#24-sia-健康指标汇总) 一致):

| 指标 | 健康范围 | 红灯 |
|---|---|---|
| intervention ratio | 10-40% (短输出 ~20%, 长输出 ~25-30%) | **0%** → RM 死了, SIA 退化为 no-op |
| top1 flip rate | 50-80% | 0% / 100% → bug |
| RM error 日志 | 0 | >0 → RM forward 失败 |

看日志方法 (per path):
- **Path A/B**: `tail -f "$LOG"` (你启动时记的那个 `/workspace/sia-logs/sia_server_*.log`)
- **Path C**: `docker compose logs -f sia-vl30b`

### 4.4 Per-request SIA 参数测试

SIA 支持在单次 request 里覆盖全局参数（`sia_weight` / `sia_topk` / `sia_entropy_threshold`），无需重启 server。以下三条 curl 各验证一个参数，**观察 server 日志里 `DONE` 行的 ratio / top1_flip 变化**确认生效。

#### 关闭 SIA 干预 (`sia_weight=0`)

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

预期 server 日志：`ratio=0.0%` — RM 完全不调用，退化为纯 vLLM 推理。

#### 只评分 1 个候选 (`sia_topk=1`)

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

预期 server 日志：`ratio≈20-30%`（entropy gate 正常工作）、`top1_flip=0%`（只有 1 个候选，top-1 不可能被替换）。

#### 强制每 token 都干预 (`sia_entropy_threshold=0`)

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

预期 server 日志：`ratio=100%` — 所有 token 都经过 RM 评分，包括模型置信度很高的位置（entropy gate 完全旁路）。

> 三个参数可以在同一个 request 里组合使用，例如 `"sia_weight": 2.0, "sia_topk": 20, "sia_entropy_threshold": 0.5`。未传的参数沿用 server 启动时的全局默认值（`--weight` / `--topk` / `--entropy_threshold`）。

---

## Step 5 — 停止 / 重启 / 清理 (per path)

### Path A / B (手动起 docker run)

| 操作 | 命令 |
|---|---|
| 停 server (进程级) | `pkill -f sia_vllm_server.py` |
| 看 server 是否还在跑 | `pgrep -af sia_vllm_server` |
| 重启 server | 重新跑 Step 3 的 nohup 命令 |
| 退出 container (server 也会被 kill) | 在容器里 `exit` |
| 容器 docker run 时加了 `--rm`, exit 后容器自动删除 |

### Path C (docker compose) — 生产推荐

| 操作 | 命令 |
|---|---|
| 停 server + 移除 container (保留 named volumes, 含 compile cache) | `docker compose down` |
| 停 server + 移除 container + 清理 volumes (重置 compile cache) | `docker compose down -v` |
| 仅重启 server, 不重 build, 不动 volume | `docker compose restart sia-vl30b` |
| 改了源码后重启 (源码是 bind-mount, 直接 restart 即可) | `docker compose restart sia-vl30b` |
| 改了 `requirements/vl30b-b2-inproc.txt` 后重 build | `docker compose up -d --build` |
| 改了 `docker-compose.yml` 后应用新配置 | `docker compose up -d` (会自动 recreate container) |
| 查看资源占用 | `docker stats sia-vl30b-server` |
| 直接进容器 debug (server 仍在跑) | `docker compose exec sia-vl30b bash` |

> **GPU 显存释放**: container 停掉后 (`docker compose down` 或 `pkill`) GPU 显存通常**几秒内自动释放**。如果 `nvidia-smi` 还显示显存被占, 说明有僵尸进程, 用 `pkill -9 -f sia_vllm_server.py` 强杀。

---

## 关键陷阱 / 注意事项

| 陷阱 | 解决 |
|---|---|
| Docker image 自带 PyTorch 2.11 (conda 里), 跟 vllm 0.17.1 需要的 torch 2.10.0 冲突 | 用 `python3 -m venv` 创建隔离 venv (脚本已处理), **不要直接 pip install vllm 到 conda env** |
| CUDA 12.8 vs vllm 0.17.1 的 cu124 wheel | 向前兼容, OK (cu124 binary 能跑在 12.8 驱动上) |
| `--shm-size=16g --ipc=host` 必须加 | vllm 用大量共享内存做 KV cache, 默认 docker 的 64MB 不够 |
| 主 LLM cudagraph mode 必须 PIECEWISE (`SIA_LLM_CUDAGRAPH=piecewise`) | 0.17.1 的 PIECEWISE 是 AOT 编译, 不会撞 nested RM 的 cudagraph flag。这是 b2 inproc 跑通的关键 env var |
| 主进程 + nested RM 在同一 GPU 共享显存 | `--llm_gpu_mem 0.55 --rm_b2_gpu_mem 0.15` 加起来 0.70, 给 cudagraph 留 30%; 80 GB GPU 上跑 30B 主 LLM + 4B RM 是紧但够。GPU 上有其他进程时, 减到 `0.48 + 0.08`, 详见 doc 末尾 [显存预算](#显存预算其他进程占用-gpu-时如何调) |
| Path C 首次启动看 `(unhealthy)` 状态 | start_period=20m, 给主 LLM weights 加载 + torch.compile 留余地。容器不会在这段时间被 docker kill (`unless-stopped` + `start_period` 配合)。20 min 后还 unhealthy 才是真问题 |
| GPU 访问语法版本问题 | `docker-compose.yml` 已改用 `deploy: resources: reservations: devices:` 写法，兼容全版本。若仍报 `Additional property gpus is not allowed`，说明你本地文件还是旧版，重新 pull 最新代码即可 |
| Path A 的 `-v /data/...` 只是示例路径 | Path A 的 docker run 示例用了泛化路径 `/data/...`；如果你按前提 §4 建了 `/dstack/persistent/SIA` 目录, 用整体挂载 `-v /dstack/persistent/SIA:/workspace` 即可（跟 Path B/C 一致） |
| 生产长跑日志撑满磁盘 | `docker logs` 默认无上限, 在 `docker-compose.yml` 的 `sia-vl30b` service 下加: `logging: {driver: "json-file", options: {max-size: "500m", max-file: "5"}}` |
| Path C 怎么改 server 参数 (e.g. `--topk` / `--weight`) | 改 `docker-compose.yml` 的 `command:` 字段, 然后 `docker compose up -d` 自动 recreate container。**不用重 build image** |

## 如何验证 Prefix Caching 是否启用

vllm 的 **Automatic Prefix Caching (APC)** 在共享前缀场景 (system prompt / RAG context / 多轮对话) 能给 prefill **2-10× 加速**, 是生产部署的关键 cache 机制。本仓库已经把它**默认开启** — `sia_vllm_server.py` 的 `--enable_prefix_caching` flag 默认 True, 而且 [`docker-compose.yml`](../docker-compose.yml) `command:` 里显式传了这个 flag。

> 想关掉测对照: `--disable_prefix_caching` (生产**不推荐**)。

### 验证方法 1: 看 server 启动日志 (最直接)

server 启动时 vllm 会 dump engine 配置, 里面有 `enable_prefix_caching=True`:

```bash
# Path C
docker compose logs sia-vl30b 2>&1 | grep -m1 enable_prefix_caching

# Path A/B
grep -m1 enable_prefix_caching $LOG    # 你 Step 3 启动时记的那个 log 文件
```

预期输出 (vllm 内部一长串配置 dump 的一部分):
```
... seed=0, served_model_name=..., enable_prefix_caching=True, enable_chunked_prefill=True, ...
```

如果看到 `enable_prefix_caching=False` → APC 关了, 要 debug 启动 flag。

### 验证方法 2: 看 server 自己的 prefix_caching 标记

`sia_vllm_server.py` 在加载 vllm engine 之前打印自己看到的 flag (本仓库的 wrapper 加的):

```bash
docker compose logs sia-vl30b 2>&1 | grep "main LLM prefix_caching"
```

预期:
```
[SIA] main LLM prefix_caching = True
```

### 验证方法 3: 功能性测试 (共享前缀, 看第二次 TTFT 是否暴跌)

发两次 prompt **相同前缀, 不同结尾** 的 request, 用 `time` 量 wall time。APC 生效时, 第二个 request 的 prefill 几乎免费 (KV 直接复用), wall time 应该明显短:

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

预期: 第二次 wall time 明显小于第一次 (主要是 prefill 时间省掉)。30B 模型上 ~50 token 共享 prefix 应该能省 ~50-100 ms TTFT。

> ⚠️ 此测试需要 prompt **完全相同的前缀长度 + content**。SIA 干预**不影响** APC — APC 只对 prefill 阶段的 input prompt KV 做缓存, SIA 介入是 decode 阶段改 logits, 两者作用不同阶段。

### 关掉对照测速 (可选, 留作日后调优)

想看 APC 对自己业务的实际收益:
```bash
# 改 docker-compose.yml 的 command, 把 --enable_prefix_caching 换成 --disable_prefix_caching
# 重启:
docker compose up -d
# 用相同 benchmark 测一遍, 对比 throughput / TTFT
```

注意: 关 APC 后, **SIA 干预效果不变** (Skywork reward / accuracy 等指标跟开 APC 时统计等价), 只是 prefill 慢。这一点跟 b2 inproc 加速一样 — 工程优化跟 SIA 算法层正交。

---

## 显存预算 — 其他进程占用 GPU 时如何调

如果 GPU 上还有别的进程, 用 `nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader` 算出 free 显存, 然后调小 `--llm_gpu_mem` 和 `--rm_b2_gpu_mem` (因为 vllm 把它们当作**总显存**的百分比, 不是 free 显存)。

例: 143 GB GPU, 别的进程占 57 GB, 剩 86 GB 给 vllm:

| 配置 | `--llm_gpu_mem` | `--rm_b2_gpu_mem` | 总 vllm 占 | 加其他进程后 | 留 headroom |
|---|---|---|---|---|---|
| 默认 (空闲 GPU) | 0.55 | 0.15 | 100,640 MiB | 158,002 MiB | ❌ **OOM** |
| 调小后 | **0.48** | **0.08** | 80,512 MiB | 137,874 MiB | ✅ 5,897 MiB |

加上 `--max_model_len 2048` (从 4096 降一半) 减少 KV cache 需求, 更稳。

如果 Path C 也想适配, 改 `docker-compose.yml` 的 `command:` 字段对应行即可。

---

如果 Step 2 自检有任何 ✗, **不要继续**, 先 debug。任何一步报错先看具体错误信息, 不要硬上 Step 3。

---

## AlpacaEval 评测 — SIA vs noSIA (Skywork 打分)

在 docker 容器内直接复用 SIA server 跑 AlpacaEval 效果评测。两个 arm 共用同一个运行中的 server，通过 per-request `--sia_weight 0` 区分，无需切换服务。

**脚本**: `eval/alpaca_eval.py` (generation) + `scripts/measure_alpaca_reward.py` (scoring)
**评分模型**: Skywork-Reward-V2-Llama-3.1-8B (第三方 RM，独立于 VM-Qwen3-4B)
**参考基线**: 旧实验 HTTP path VL-30B 200Q: noSIA mean=29.26, SIA mean=28.67, Δ=-1.57% (p=0.29)

### 前提

```bash
# 确认 Skywork 评分模型（路径按实际位置调整）
ls /dstack/persistent/SIA/models/Skywork-Reward-V2-Llama-3.1-8B/
# 若未下载:
# huggingface-cli download Skywork/Skywork-Reward-V2-Llama-3.1-8B \
#   --local-dir /dstack/persistent/SIA/models/Skywork-Reward-V2-Llama-3.1-8B

# 确认 AlpacaEval 数据集（仓库内已包含，确认文件存在）
ls /dstack/persistent/SIA/sia-repo/0g-sparse-inference-alignment/data/alpaca_eval/alpaca_eval.json
```

### Phase 1 — Generation（server 正常运行，无需停机）

进入容器：
```bash
docker compose exec sia-vl30b bash
cd /workspace/sia-repo/0g-sparse-inference-alignment
MODEL=/workspace/models/Qwen3-VL-30B-A3B-Instruct
mkdir -p /workspace/exp
```

**SIA arm**（server 默认参数：topk=10, weight=1.0, entropy_threshold=1.0）：
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

**noSIA arm**（`--sia_weight 0` 关掉 RM 干预，speed ≈ raw vllm）：
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

> SIA arm 约 80-90 min (28 tok/s)；noSIA arm 约 20 min (RM 跳过，速度回到 ~120 tok/s)。可以串行跑（等 SIA 完再跑 noSIA），也可以并行跑（同一 server 支持并发请求）。

### Phase 2 — Skywork 打分（需要停 server 释放显存）

```bash
# 在宿主机停 server
docker compose stop sia-vl30b

# 起临时容器跑打分（named volume 保留，compile cache 不丢）
docker compose run --rm sia-vl30b bash
cd /workspace/sia-repo/0g-sparse-inference-alignment
RM=/workspace/models/Skywork-Reward-V2-Llama-3.1-8B   # 按实际路径改

python scripts/measure_alpaca_reward.py \
  --input_file  /workspace/exp/alpaca_vl30b_b2_sia_*.json \
  --output_file /workspace/exp/alpaca_vl30b_b2_sia_scored.json \
  --rm "$RM" --device cuda:0 --strip_think

python scripts/measure_alpaca_reward.py \
  --input_file  /workspace/exp/alpaca_vl30b_b2_nosia_*.json \
  --output_file /workspace/exp/alpaca_vl30b_b2_nosia_scored.json \
  --rm "$RM" --device cuda:0 --strip_think
```

各约 3-5 min（Skywork 8B BF16, ~2.5 题/s）。

### Generation 结束后 — 辅助指标查看

**tokens/s**（generation 结束后 `gen.log` 末尾自动打印）：
```bash
tail -5 /workspace/exp/alpaca_vl30b_b2_sia_gen.log
# 预期看到: throughput: XX.X tok/s
```

**干预率 + flip rate**（从 server 日志聚合，需在仓库根目录执行）：
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
print(f'干预率  mean={sum(ratios)/len(ratios):.1f}%  n={len(ratios)}' if ratios else 'no ratio data')
print(f'flip率  mean={sum(flips)/len(flips):.1f}%  n={len(flips)}' if flips else 'no flip data')
"
```

健康参考值（参见 [§4 smoke test 健康指标](doc/docker-install-vl30b-20260606.md#验证-sia-真的在干预)）：干预率 10–40%，flip rate 50–80%。

### Phase 3 — 对比结果

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

打分完重启服务：
```bash
exit   # 退出打分容器
docker compose start sia-vl30b
```

---

## 相关 doc

- [`qwen3-vl-30b-sia-eval-20260605.md`](qwen3-vl-30b-sia-eval-20260605.md) — VL-30B SIA 评测汇总 (效果 + 性能), 含本地实验结果
- [`vl30b-b2-inproc-speedup-20260605.md`](vl30b-b2-inproc-speedup-20260605.md) — b2 inproc 加速 1.46× 的原始实验报告 (含 vllm 0.15/0.17.1/0.19 三档对比为什么 0.17.1 是 sweet spot)
- [`../docker-compose.yml`](../docker-compose.yml) — **Path C 生产 compose 文件** (GPU + 挂载 + restart + healthcheck + 完整启动命令)
- [`../Dockerfile`](../Dockerfile) — Path B / C 共用的 Dockerfile (vllm 0.17.1 baked-in 镜像)
- [`../scripts/docker_entrypoint_vl30b.sh`](../scripts/docker_entrypoint_vl30b.sh) — Path B / C 共用 entrypoint, 启动时注册 sia_rm 插件
- [`../scripts/docker_install_vl30b.sh`](../scripts/docker_install_vl30b.sh) — Path A 用的命令行安装脚本
- [`../scripts/setup_venv_vl30b_fast.sh`](../scripts/setup_venv_vl30b_fast.sh) — venv 创建底层脚本 (docker_install 会调用它)
- [`../requirements/vl30b-b2-inproc.txt`](../requirements/vl30b-b2-inproc.txt) — pip 依赖清单
- [`../CLAUDE.md`](../CLAUDE.md) — 整体 venv 矩阵 (Qwen3-14B / VL-30B / 0GM-35B)
- [`vllm-rm-backend.md`](vllm-rm-backend.md) — convert_rm_for_vllm.py 详细说明 (RM checkpoint 转换原理)
