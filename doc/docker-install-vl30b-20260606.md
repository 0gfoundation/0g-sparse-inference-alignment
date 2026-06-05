# VL-30B SIA 在 Docker 里跑通 — 完整安装步骤

**最后更新**: 2026-06-06
**适用 docker image**: `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel`
**适用配置**: VL-30B (Qwen3-VL-30B-A3B-Instruct) SIA b2 inproc 加速路径 (vllm 0.17.1, [`requirements/vl30b-b2-inproc.txt`](../requirements/vl30b-b2-inproc.txt))

## 前提

- 主机有 NVIDIA GPU (H100 / H200 / A100 **80GB+ 显存**), nvidia-container-toolkit 装好
- 主机上有模型权重 (~71 GB 总计):
  - `Qwen3-VL-30B-A3B-Instruct/` (主 LLM, ~60 GB)
  - `VM-Qwen3-4B-merged-for-vllm/` (RM, ~11 GB)

## Step 1 — 启动 docker container

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

如果还没 clone repo, 在容器里再 clone:
```bash
cd /workspace && git clone <your-repo-url> sia-repo
```

## Step 2 — 跑安装脚本 (容器内)

```bash
cd /workspace/sia-repo
bash scripts/docker_install_vl30b.sh
```

脚本按顺序做的事:
1. `apt-get install -y git ca-certificates curl` (如果缺)
2. `python3 -m venv /opt/venv-vl30b` (用 docker 的 conda python, 隔离 torch 2.11)
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

## Step 3 — 激活 venv + 启 SIA server

```bash
source /opt/venv-vl30b/bin/activate

SIA_LLM_CUDAGRAPH=piecewise \
SIA_RM_CUDAGRAPH=piecewise \
SIA_RM_MULTIPROCESS=0 \
python src/sia_vllm_server.py \
  --llm /workspace/models/Qwen3-VL-30B-A3B-Instruct \
  --rm_backend b2 \
  --rm_model /workspace/models/VM-Qwen3-4B-merged-for-vllm \
  --rm_b2_gpu_mem 0.15 --llm_gpu_mem 0.55 \
  --topk 10 --weight 1.0 --entropy_threshold 1.0 \
  --max_model_len 4096 --port 8000
```

等待启动 (主 LLM weights 加载 ~12 min on H200, 然后 cudagraph capture + warmup ~3 min), 直到看到:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
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
- [`../scripts/docker_install_vl30b.sh`](../scripts/docker_install_vl30b.sh) — 本 doc 引用的安装脚本
- [`../scripts/setup_venv_vl30b_fast.sh`](../scripts/setup_venv_vl30b_fast.sh) — venv 创建底层脚本 (docker_install 会调用它)
- [`../requirements/vl30b-b2-inproc.txt`](../requirements/vl30b-b2-inproc.txt) — pip 依赖清单
- [`../CLAUDE.md`](../CLAUDE.md) — 整体 venv 矩阵 (Qwen3-14B / VL-30B / 0GM-35B)
