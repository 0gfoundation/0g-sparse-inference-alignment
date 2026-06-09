# Qwen3-VL-30B SIA 服务使用说明

**服务**: Qwen3-VL-30B-A3B-Instruct + SIA (Sparse Inference-time Alignment)
**接口**: OpenAI 兼容 API (`/v1/chat/completions`)
**部署**: Docker compose, b2 inproc backend

---

## 1. 基础对话

最简单的用法，使用服务默认参数（SIA 干预开启）：

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/Qwen3-VL-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "What are 3 colors of fruit?"}],
    "max_tokens": 200,
    "temperature": 0.7
  }' | python3 -m json.tool
```

**Response:**
```json
{
    "id": "chatcmpl-a3f2e1b0",
    "object": "chat.completion",
    "created": 1749340800,
    "model": "/workspace/models/Qwen3-VL-30B-A3B-Instruct",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "<think>\nThe user wants three colors commonly associated with fruit. Let me think of clear examples.\n</think>\n\nHere are 3 common colors of fruit:\n\n1. **Red** — apples, strawberries, cherries\n2. **Yellow** — bananas, lemons, mangoes\n3. **Orange** — oranges, papayas, tangerines"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 18,
        "completion_tokens": 87,
        "total_tokens": 105
    }
}
```

> 模型默认开启 thinking 模式，回复内容里包含 `<think>...</think>` 推理过程，之后是正式答案。

---

## 2. 带 System Prompt

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/Qwen3-VL-30B-A3B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a concise assistant. Answer in one sentence."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }' | python3 -m json.tool
```

**Response:**
```json
{
    "id": "chatcmpl-b7c4d2e1",
    "object": "chat.completion",
    "created": 1749340860,
    "model": "/workspace/models/Qwen3-VL-30B-A3B-Instruct",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "<think>\nSimple factual question.\n</think>\n\nThe capital of France is Paris."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 34,
        "completion_tokens": 24,
        "total_tokens": 58
    }
}
```

---

## 3. SIA 参数说明

SIA (Sparse Inference-time Alignment) 在每个 token 生成时，用 Value Model 对候选 token 打分，将得分加到 logits 上，引导模型生成更高质量的输出。

服务启动时的**全局默认值**：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `sia_weight` | 1.0 | RM 得分乘以此权重后加到 logits |
| `sia_topk` | 10 | 每步评分的候选 token 数 |
| `sia_entropy_threshold` | 1.0 | entropy 低于此值时跳过干预（模型已足够确定） |

这三个参数均可在**单次请求**里覆盖，无需重启服务。

---

## 4. 关闭 SIA 干预（`sia_weight=0`）

`sia_weight=0` 让 RM 得分权重为 0，等价于纯 vLLM 推理，**速度最快**：

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/Qwen3-VL-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "Explain what a neural network is."}],
    "max_tokens": 200,
    "temperature": 0.7,
    "sia_weight": 0
  }' | python3 -m json.tool
```

**Response:**
```json
{
    "id": "chatcmpl-c1d5e3f2",
    "object": "chat.completion",
    "created": 1749340920,
    "model": "/workspace/models/Qwen3-VL-30B-A3B-Instruct",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "<think>\nI'll give a clear, accessible explanation of neural networks.\n</think>\n\nA neural network is a machine learning model loosely inspired by the human brain. It consists of layers of interconnected nodes (\"neurons\"). Each connection has a weight; during training, these weights are adjusted so the network learns to map inputs to correct outputs. Neural networks excel at tasks like image recognition, language understanding, and game playing."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 21,
        "completion_tokens": 103,
        "total_tokens": 124
    }
}
```

**服务端日志**（`docker compose logs sia-vl30b`）会显示：
```
[SIA] req=2 DONE  intervened=0/103  ratio=0.0%  top1_flip=0/0 (0.0%)
```

ratio=0% 确认 RM 完全跳过。

---

## 5. 加强 SIA 干预（`sia_weight` 调大）

`sia_weight` 越大，RM 得分对 token 选择的影响越强。默认 1.0，可以调到 2.0 加强干预：

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/Qwen3-VL-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "Write a short poem about the ocean."}],
    "max_tokens": 200,
    "temperature": 0.7,
    "sia_weight": 2.0
  }' | python3 -m json.tool
```

**Response:**
```json
{
    "id": "chatcmpl-d2e6f4a3",
    "object": "chat.completion",
    "created": 1749340980,
    "model": "/workspace/models/Qwen3-VL-30B-A3B-Instruct",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "<think>\nA short poem about the ocean. I'll focus on vivid imagery.\n</think>\n\nThe ocean breathes in silver swells,\nits depths hold more than language tells.\nEach wave that breaks upon the shore\nwhispers of a world before.\n\nSalt and silence, vast and free—\nwe are small beside the sea."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 91,
        "total_tokens": 111
    }
}
```

---

## 6. 调节干预频率（`sia_entropy_threshold`）

`sia_entropy_threshold` 控制 entropy gate：只有当前 token 的 logit entropy 高于此阈值时才调用 RM。

- **threshold=0**：每个 token 都干预（ratio=100%），最慢，最激进
- **threshold=1.0**（默认）：约 20% 的 token 干预，平衡速度与效果
- **threshold 很大（如 999）**：几乎不干预，接近 noSIA

### 强制每 token 都干预（`sia_entropy_threshold=0`）

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/Qwen3-VL-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "What are 3 colors of fruit?"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "sia_entropy_threshold": 0
  }' | python3 -m json.tool
```

**Response:**
```json
{
    "id": "chatcmpl-e3f7a5b4",
    "object": "chat.completion",
    "created": 1749341040,
    "model": "/workspace/models/Qwen3-VL-30B-A3B-Instruct",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "<think>\nSimple question about fruit colors.\n</think>\n\n1. **Red** — apples, strawberries\n2. **Yellow** — bananas, lemons\n3. **Green** — grapes, kiwis"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 18,
        "completion_tokens": 58,
        "total_tokens": 76
    }
}
```

服务端日志：
```
[SIA] req=5 DONE  intervened=58/58  ratio=100.0%  top1_flip=12/58 (20.7%)
```

### 降低干预频率（`sia_entropy_threshold=2.0`）

阈值调大，只在模型非常不确定时才干预，速度更接近 noSIA：

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/Qwen3-VL-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "What are 3 colors of fruit?"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "sia_entropy_threshold": 2.0
  }' | python3 -m json.tool
```

服务端日志（干预率明显降低）：
```
[SIA] req=6 DONE  intervened=8/58  ratio=13.8%  top1_flip=5/8 (62.5%)
```

---

## 7. 调节候选 token 数（`sia_topk`）

`sia_topk` 控制每步 RM 评分的候选 token 数量。越大评分越细致但越慢。默认 10。

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/Qwen3-VL-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "Summarize the benefits of exercise."}],
    "max_tokens": 150,
    "temperature": 0.7,
    "sia_topk": 20
  }' | python3 -m json.tool
```

**Response:**
```json
{
    "id": "chatcmpl-f4a8b6c5",
    "object": "chat.completion",
    "created": 1749341100,
    "model": "/workspace/models/Qwen3-VL-30B-A3B-Instruct",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "<think>\nKey benefits of exercise across physical and mental health.\n</think>\n\nRegular exercise offers several important benefits:\n\n- **Physical health**: strengthens the heart, improves circulation, builds muscle and bone density, and helps maintain a healthy weight.\n- **Mental health**: reduces anxiety and depression, boosts mood through endorphin release, and improves sleep quality.\n- **Cognitive function**: enhances memory, focus, and long-term brain health.\n- **Longevity**: associated with lower risk of chronic diseases such as type 2 diabetes and cardiovascular disease."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 19,
        "completion_tokens": 122,
        "total_tokens": 141
    }
}
```

---

## 8. 组合使用多个 SIA 参数

三个参数可以在同一请求里一起传：

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/Qwen3-VL-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "What makes a good software engineer?"}],
    "max_tokens": 300,
    "temperature": 0.7,
    "sia_weight": 1.5,
    "sia_topk": 20,
    "sia_entropy_threshold": 0.8
  }' | python3 -m json.tool
```

未传的参数沿用服务启动时的全局默认值。

---

## 9. Streaming 模式

加 `"stream": true` 获得逐 token 流式输出（SSE 格式）：

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/Qwen3-VL-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "Count from 1 to 5."}],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": true
  }'
```

**Response（逐行流式）:**
```
data: {"id":"chatcmpl-g5b9c7d6","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant","content":"<think>"},"index":0}]}

data: {"id":"chatcmpl-g5b9c7d6","object":"chat.completion.chunk","choices":[{"delta":{"content":"\nSimple counting task.\n"},"index":0}]}

data: {"id":"chatcmpl-g5b9c7d6","object":"chat.completion.chunk","choices":[{"delta":{"content":"</think>\n\n"},"index":0}]}

data: {"id":"chatcmpl-g5b9c7d6","object":"chat.completion.chunk","choices":[{"delta":{"content":"1, 2, 3, 4, 5."},"index":0}]}

data: {"id":"chatcmpl-g5b9c7d6","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop","index":0}]}

data: [DONE]
```

> ⚠️ 本服务的 streaming 是**模拟**的（先生成完整回复，再分块推送），延迟特性与真正的逐 token streaming 不同。

---

## 10. 参数速查表

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | 必填 | 填服务端模型路径 |
| `messages` | array | 必填 | 对话历史，支持 system / user / assistant |
| `max_tokens` | int | — | 最大生成 token 数，建议 ≤ 1800（受 max_model_len=2048 限制） |
| `temperature` | float | 1.0 | 采样温度，0 为贪婪解码，越高越随机 |
| `top_p` | float | — | nucleus sampling 概率阈值 |
| `top_k` | int | — | 只从概率最高的 k 个 token 里采样 |
| `repetition_penalty` | float | — | 重复惩罚，推荐 1.0（关闭）或 1.05 |
| `stream` | bool | false | 开启 SSE 流式输出 |
| `sia_weight` | float | 1.0 | **SIA 干预强度**，0=关闭，>1=加强 |
| `sia_topk` | int | 10 | **SIA 候选 token 数**，越大越精细但越慢 |
| `sia_entropy_threshold` | float | 1.0 | **SIA 干预频率门控**，0=全干预，越大干预越少 |

---

## 11. 服务健康检查

```bash
curl -s http://localhost:8000/health
```

**Response:**
```json
{"status": "ok"}
```
