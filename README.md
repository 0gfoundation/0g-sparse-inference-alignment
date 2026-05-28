# SIA vLLM Server

SIA（Sparse Inference-time Alignment）per-token 干预推理服务，兼容 OpenAI Chat API，可与 0g-serving-broker 对接。

## 文件说明

| 文件 | 说明 |
|------|------|
| `src/sia_vllm_RM.py` | SIA 核心逻辑：per-token logits 干预，通过 HTTP 调用 RM server 打分 |
| `src/sia_vllm_server.py` | FastAPI HTTP 服务，包装 `sia_vllm_RM.py` |
| `src/sia_rm_server.py` | 独立 RM 打分服务，支持热切换 RM / LoRA |

## 依赖安装

```bash
# 1. 安装运行时依赖（vllm / torch / transformers / fastapi 等共 14 包，
#    版本钉死在已验证的工作集，详见 requirements.txt 顶部注释）
pip install -r requirements.txt

# 2. 把 sia_rm 安装为 editable package
#    必需 —— b2 backend (in-process RM) 通过 pyproject.toml 的
#    `vllm.general_plugins` entry-point 让 vLLM EngineCore subprocess
#    自动 register Qwen3WithScoreForCausalLM。
pip install -e .
```

**注意**:
- 验证环境：Python 3.12 + CUDA 12.x，vLLM 0.10.1.1。不同 CUDA 版本机器需先按 [pytorch.org](https://pytorch.org/get-started/locally/) 装匹配的 torch wheel，再跑 `pip install -r requirements.txt`。
- `peft` 必须 `>=0.14.0,<0.15.0`（VM-Qwen3-4B LoRA 加载路径要求）。
- `requirements.txt` 含验证脚本用的 `scipy`，不需要可以注释掉。

## 启动服务

SIA 由两个独立进程组成：**RM server** 负责打分，**LLM server** 负责生成并调用 RM server。

### 第一步：启动 RM server

**不带 LoRA：**

```bash
python src/sia_rm_server.py \
  --rm /workspace/SIA/models/Qwen3-1.7B-Base \
  --rm_device cuda:0 --port 8001
```

**带 LoRA（ValueModel checkpoint）：**

```bash
python src/sia_rm_server.py \
  --rm      /workspace/SIA/models/Qwen3-1.7B-Base \
  --rm_lora /workspace/SIA/models/SIA-checkpoints/VM-Qwen3-1.7B-Base \
  --rm_device cuda:0 --port 8001
```

### 第二步：启动 LLM server

```bash
python src/sia_vllm_server.py \
  --llm    /workspace/SIA/models/Qwen3-1.7B-Base \
  --rm_url http://localhost:8001 \
  --host 0.0.0.0 --port 8000 \
  --llm_gpu_mem 0.3 --topk 5 --weight 1.0
```

## CLI 参数

### LLM server（`sia_vllm_server.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--llm` | 必填 | LLM 模型路径 |
| `--rm_url` | `http://localhost:8001` | RM server 地址 |
| `--llm_gpu_mem` | `0.5` | vLLM GPU 显存占用比例 |
| `--topk` | `10` | 每步干预 top-k 候选 token 数 |
| `--weight` | `1.0` | RM 分数叠加权重 |
| `--entropy_threshold` | None | 跳过干预的熵阈值（None=始终干预） |
| `--max_model_len` | `4096` | vLLM 最大序列长度 |
| `--host` | `0.0.0.0` | HTTP 监听地址 |
| `--port` | `8000` | HTTP 监听端口 |
| `--model_id` | LLM basename | 对外暴露的 model 名称 |

### RM server（`sia_rm_server.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--rm` | 必填 | RM 基础模型路径 |
| `--rm_lora` | None | RM LoRA（ValueModel）checkpoint 路径 |
| `--rm_device` | `cuda:0` | RM 运行设备 |
| `--host` | `0.0.0.0` | HTTP 监听地址 |
| `--port` | `8001` | HTTP 监听端口 |

## API 接口

### 健康检查

```bash
curl http://localhost:8000/health
```

### 查询模型列表

```bash
curl http://localhost:8000/v1/models
```

### 非流式对话

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

### 指定 per-request SIA weight

`sia_weight` 是 SIA 扩展字段（不与 OpenAI API 冲突），用于在单次请求中覆盖启动时的 `--weight` 默认值。不传则沿用默认值。

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50,
    "sia_weight": 2.0
  }'
```

### 流式对话（SSE）

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50,
    "stream": true
  }'
```

### broker 兼容路由

broker 同时使用以下两个路由（均支持）：

- `POST /v1/chat/completions`
- `POST /chat/completions`

## RM 热切换

无需重启任何服务，直接调用 RM server 的 `/reload` 接口：

```bash
curl -X POST http://localhost:8001/reload \
  -H "Content-Type: application/json" \
  -d '{"rm": "/path/to/new/rm", "rm_lora": "/path/to/new/lora"}'
```

接口同步执行，返回后 RM server 已加载新模型，后续打分请求自动使用新模型。

查询 RM server 当前状态：

```bash
curl http://localhost:8001/status
```
