# VL-30B SIA 在 Docker 里跑通 — 完整安装步骤

**最后更新**: 2026-06-06
**适用 docker image**: `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel`
**适用配置**: VL-30B (Qwen3-VL-30B-A3B-Instruct) SIA b2 inproc 加速路径 (vllm 0.17.1, [`requirements/vl30b-b2-inproc.txt`](../requirements/vl30b-b2-inproc.txt))

## 前提

- 主机有 NVIDIA GPU (H100 / H200 / A100 **80GB+ 显存**), nvidia-container-toolkit 装好
- 主机上有模型权重 (~71 GB 总计):
  - `Qwen3-VL-30B-A3B-Instruct/` (主 LLM, ~60 GB)
  - `VM-Qwen3-4B-merged-for-vllm/` (RM, ~11 GB)

## 两条路径选一条

| 路径 | 适用场景 | 步骤 |
|---|---|---|
| **Path A: 命令行装** | 临时想跑一次, 不留持久 image | Step 1 → Step 2 (跑安装脚本) → Step 3 → Step 4 |
| **Path B: Dockerfile 构建** ⚡ | 跑多次 / 分发给别人 / CI; 启动后无任何依赖安装 | Step 1B (build image) → Step 1B' (docker run) → Step 3 → Step 4 |

Path A 每次进新容器都要重装 deps (3-5 min); Path B 一次性 `docker build` 后, 后续 `docker run` 进去就能直接跑 server, 没有安装等待。

---

## Path A — 命令行装

### Step 1 — 启动 docker container

```bash
# 假设主机模型在 /data/models/, 仓库 clone 到 /data/sia-repo/
docker run -it --rm \
  --gpus all \
  --shm-size=16g \
  --ipc=host \
  -p 8000:8000 \
  -v /data/sia-repo:/workspace/sia-repo \
  -v /data/models:/workspace/models \
  pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel \
  bash
```

如果还没 clone repo, 在**容器里**再 clone — pytorch image 默认没装 git, 必须先装:
```bash
apt-get update && apt-get install -y --no-install-recommends git ca-certificates
cd /workspace && git clone <your-repo-url> sia-repo
```

> 推荐做法: 在**主机上**先 clone 好再用 `-v /data/sia-repo:/workspace/sia-repo` 挂进来, 容器内就不用装 git 了。

## Step 2 — 跑安装脚本 (容器内)

```bash
cd /workspace/sia-repo
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
python: 3.11.x  (/opt/venv-vl30b/bin/python3)
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

## Step 3 — 激活 venv + 启 SIA server (主 LLM + Value Model 同进程)

> Path B 容器里 venv 已在 PATH, `source /opt/venv-vl30b/bin/activate` 可省。Path A 用户记得激活。

> ℹ️ **重要 — b2 inproc 是单进程拓扑**: 跟 HTTP path 起两个 server (一个主 LLM, 一个 RM 在另一个端口) 不同, b2 inproc 让 **主 LLM (VL-30B) + Value Model (VM-Qwen3-4B) 在同一个 Python 进程里 nested 跑**, 共享一个 CUDA context, RM 调用是直接 Python 函数调用而非 HTTP。所以下面**这一条命令就同时启动了主 LLM + Value Model**, 不需要再开一个 RM server。

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

## Step 4 — Smoke test (容器外或新 shell)

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

预期: 收到正常 JSON response。同时 server stderr 会打印 `[SIA] req=0 DONE intervened=N/M ratio=X%`, 确认 SIA 真的在干预 (intv 率应该在 15-25% 区间)。

## 关键陷阱 / 注意事项

| 陷阱 | 解决 |
|---|---|
| Docker image 自带 PyTorch 2.11 (conda 里), 跟 vllm 0.17.1 需要的 torch 2.10.0 冲突 | 用 `python3 -m venv` 创建隔离 venv (脚本已处理), **不要直接 pip install vllm 到 conda env** |
| CUDA 12.8 vs vllm 0.17.1 的 cu124 wheel | 向前兼容, OK (cu124 binary 能跑在 12.8 驱动上) |
| `--shm-size=16g --ipc=host` 必须加 | vllm 用大量共享内存做 KV cache, 默认 docker 的 64MB 不够 |
| 主 LLM cudagraph mode 必须 PIECEWISE (`SIA_LLM_CUDAGRAPH=piecewise`) | 0.17.1 的 PIECEWISE 是 AOT 编译, 不会撞 nested RM 的 cudagraph flag。这是 b2 inproc 跑通的关键 env var |
| 主进程 + nested RM 在同一 GPU 共享显存 | `--llm_gpu_mem 0.55 --rm_b2_gpu_mem 0.15` 加起来 0.70, 给 cudagraph 留 30%; 80 GB GPU 上跑 30B 主 LLM + 4B RM 是紧但够 |

如果 step 2 自检有任何 ✗, **不要继续**, 先 debug。任何一步报错先看具体错误信息, 不要硬上 step 3。

## 相关 doc

- [`qwen3-vl-30b-sia-eval-20260605.md`](qwen3-vl-30b-sia-eval-20260605.md) — VL-30B SIA 评测汇总 (效果 + 性能), 含本地实验结果
- [`vl30b-b2-inproc-speedup-20260605.md`](vl30b-b2-inproc-speedup-20260605.md) — b2 inproc 加速 1.46× 的原始实验报告 (含 vllm 0.15/0.17.1/0.19 三档对比为什么 0.17.1 是 sweet spot)
- [`../Dockerfile`](../Dockerfile) — Path B 用的 Dockerfile (vllm 0.17.1 baked-in 镜像)
- [`../scripts/docker_entrypoint_vl30b.sh`](../scripts/docker_entrypoint_vl30b.sh) — Path B image 的 entrypoint, 启动时注册 sia_rm 插件
- [`../scripts/docker_install_vl30b.sh`](../scripts/docker_install_vl30b.sh) — Path A 用的命令行安装脚本
- [`../scripts/setup_venv_vl30b_fast.sh`](../scripts/setup_venv_vl30b_fast.sh) — venv 创建底层脚本 (docker_install 会调用它)
- [`../requirements/vl30b-b2-inproc.txt`](../requirements/vl30b-b2-inproc.txt) — pip 依赖清单
- [`../CLAUDE.md`](../CLAUDE.md) — 整体 venv 矩阵 (Qwen3-14B / VL-30B / 0GM-35B)
