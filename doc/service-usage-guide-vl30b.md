# Qwen3-VL-30B SIA 服务使用说明

**服务**: Qwen3-VL-30B-A3B-Instruct + SIA (Sparse Inference-time Alignment)
**接口**: OpenAI 兼容 API (`/v1/chat/completions`)
**部署**: Docker compose, b2 inproc backend

## 接口规范合规确认

| 要求 | 状态 | 说明 |
|---|---|---|
| OpenAI 规范 | ✅ | 完全遵循 OpenAI `/v1/chat/completions` 接口规范。请求字段（`model`、`messages`、`max_tokens`、`temperature`、`stream` 等）和响应结构（`choices[].message.content`、`finish_reason`、`usage` 等）均与 OpenAI API 一致，可直接使用 OpenAI Python SDK 或任何兼容客户端接入。 |
| Input Cache（Prefix Caching）| ✅ | 服务启用了 vLLM Automatic Prefix Caching (APC)（docker-compose.yml 的 `--enable_prefix_caching`）。System prompt、对话历史等共享前缀的 KV 会被自动缓存，后续请求命中缓存时 prefill 几乎免费，显著降低 TTFT。注：同事所说的 "input cache" 即此 prefix caching 机制。 |
| Response 支持 usage | ✅ | 每个响应均包含 `usage` 字段，报告本次请求的 `prompt_tokens`、`completion_tokens`、`total_tokens`（见下方 §1 示例）。 |

---

> **性能备注**: 当前部署机器已有约 50+ GB 显存被其他任务占用，留给 30B 模型的可用显存偏紧（`--llm_gpu_mem 0.48`，空闲 GPU 时正常为 0.55），KV Cache 空间不充裕，推理速度低于理论水平。实测吞吐：SIA ≈ 59 tok/s，noSIA ≈ 81 tok/s；空闲 GPU 下参考值约为 70 / 120 tok/s。

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
    "id": "chatcmpl-0fdb87be20e3",
    "object": "chat.completion",
    "created": 1780971415,
    "model": "/workspace/models/Qwen3-VL-30B-A3B-Instruct",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Three common colors of fruit are:\n\n1. **Red** – Example: Apple, strawberry, cherry  \n2. **Yellow** – Example: Banana, peach, mango  \n3. **Green** – Example: Green apple, kiwi, green grape  \n\nThese colors represent just a few of the many naturally occurring fruit colors!"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 16,
        "completion_tokens": 67,
        "total_tokens": 83
    }
}
```

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

**服务端日志**（`docker compose logs sia-vl30b`）会显示：
```
[SIA] req=2 DONE  intervened=0/83  ratio=0.0%  top1_flip=0/0 (0.0%)
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

服务端日志：
```
[SIA] req=5 DONE  intervened=40/40  ratio=100.0%  top1_flip=8/40 (20.0%)
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
data: {"id":"chatcmpl-g5b9c7d6","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant","content":""},"index":0}]}

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
