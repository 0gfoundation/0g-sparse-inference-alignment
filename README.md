# SIA vLLM Server

SIA（Sparse Inference-time Alignment）per-token 干预推理服务，兼容 OpenAI Chat API，可与 0g-serving-broker 对接。

## 文件说明

| 文件 | 说明 |
|------|------|
| `sia_vllm_RM.py` | SIA 核心逻辑：per-token logits 干预 + RM 评分 |
| `sia_vllm_server.py` | FastAPI HTTP 服务，包装 `sia_vllm_RM.py` |

## 依赖安装

```bash
pip install fastapi uvicorn
pip install "peft<0.15.0"   # transformers 4.57.x 兼容版本
```

## 启动服务

### 不使用 LoRA

```bash
python sia_vllm_server.py \
  --llm /workspace/SIA/models/Qwen3-1.7B-Base \
  --rm  /workspace/SIA/models/Qwen3-1.7B-Base \
  --host 0.0.0.0 --port 8000 \
  --llm_gpu_mem 0.3 --topk 5 --weight 1.0
```

### 使用 LoRA（ValueModel checkpoint）

```bash
python sia_vllm_server.py \
  --llm     /workspace/SIA/models/Qwen3-1.7B-Base \
  --rm      /workspace/SIA/models/Qwen3-1.7B-Base \
  --rm_lora /workspace/SIA/models/SIA-checkpoints/VM-Qwen3-1.7B-Base \
  --host 0.0.0.0 --port 8000 \
  --llm_gpu_mem 0.3 --topk 5 --weight 1.0
```

## CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--llm` | 必填 | LLM 模型路径 |
| `--rm` | 必填 | RM 模型路径 |
| `--rm_lora` | None | RM LoRA（ValueModel）checkpoint 路径 |
| `--rm_device` | `cuda:0` | RM 运行设备 |
| `--llm_gpu_mem` | `0.5` | vLLM GPU 显存占用比例 |
| `--topk` | `10` | 每步干预 top-k 候选 token 数 |
| `--weight` | `1.0` | RM 分数叠加权重 |
| `--entropy_threshold` | None | 跳过干预的熵阈值（None=始终干预） |
| `--max_model_len` | `4096` | vLLM 最大序列长度 |
| `--host` | `0.0.0.0` | HTTP 监听地址 |
| `--port` | `8000` | HTTP 监听端口 |
| `--model_id` | LLM basename | 对外暴露的 model 名称 |

## API 接口

服务启动后，可通过以下接口验证：

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
