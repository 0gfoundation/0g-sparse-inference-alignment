# 0GM-1.0-35B SIA 服务使用说明

**服务**: 0GM-1.0-35B-A3B + SIA (Sparse Inference-time Alignment)  
**接口**: OpenAI 兼容 API (`/v1/chat/completions`)  
**部署**: Docker compose, b2 inproc backend (`docker-compose.0gm35b.yml`)

## 接口规范合规确认

| 要求 | 状态 | 说明 |
|---|---|---|
| OpenAI 规范 | ✅ | 完全遵循 OpenAI `/v1/chat/completions` 接口规范。请求字段（`model`、`messages`、`max_tokens`、`temperature`、`stream` 等）和响应结构（`choices[].message.content`、`finish_reason`、`usage` 等）均与 OpenAI API 一致，可直接使用 OpenAI Python SDK 或任何兼容客户端接入。服务同时暴露 `/chat/completions`（无 `/v1/` 前缀）作为 broker billing 路由，两者行为完全一致。 |
| Input Cache（Prefix Caching）| ✅ | 服务启用了 vLLM APC（`--enable_prefix_caching --mamba_cache_mode align`）。0GM-35B 是 hybrid 模型（full_attention + GatedDeltaNet），需要 `align` 模式才能实际生效：默认 `none` 会把 mamba_block_size 设为 max_model_len=2048，导致 APC lcm_block_size 超大，事实上 cached_tokens 始终为 0。`align` 将两者 block_size 对齐（约 1056 tokens），**prompt ≥ 1056 tokens 的请求**在命中缓存时 `usage.prompt_tokens_details.cached_tokens > 0`。 |
| Response 支持 usage | ✅ | 每个响应均包含 `usage` 字段，报告本次请求的 `prompt_tokens`、`completion_tokens`、`total_tokens`（见下方 §1 示例）。 |

---

> **性能参考**: 标准 Docker 部署下（`llm_gpu_mem=0.55`，`rm_b2_gpu_mem=0.15`），AlpacaEval 200Q 实测：SIA ≈ 66 tok/s，noSIA（纯 vLLM）≈ 114 tok/s。如机器有其他任务占用显存，实际速度会下降。

---

## 1. 基础对话

最简单的用法，使用服务默认参数（SIA 干预开启）：

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/0GM-1.0-35B-A3B-0427",
    "messages": [{"role": "user", "content": "What are 3 colors of fruit?"}],
    "max_tokens": 200,
    "temperature": 0.7
  }' | python3 -m json.tool
```

**Response:**
```json
{
    "id": "chatcmpl-3a7f1b92e4d0",
    "object": "chat.completion",
    "created": 1749600000,
    "model": "/workspace/models/0GM-1.0-35B-A3B-0427",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Three common colors of fruit are:\n\n1. **Red** – Examples: apple, strawberry, cherry\n2. **Yellow** – Examples: banana, lemon, mango\n3. **Green** – Examples: kiwi, green grape, lime"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 16,
        "completion_tokens": 52,
        "total_tokens": 68
    }
}
```

---

## 2. 带 System Prompt

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/0GM-1.0-35B-A3B-0427",
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
| `sia_weight` | 1.0 | VM 得分乘以此权重后加到 logits |
| `sia_topk` | 10 | 每步评分的候选 token 数 |
| `sia_entropy_threshold` | 1.0 | entropy 低于此值时跳过干预（模型已足够确定） |

这三个参数均可在**单次请求**里覆盖，无需重启服务。

---

## 4. 关闭 SIA 干预（`sia_weight=0`）

`sia_weight=0` 让 VM 得分权重为 0，等价于纯 vLLM 推理，**速度最快**：

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/0GM-1.0-35B-A3B-0427",
    "messages": [{"role": "user", "content": "Explain what a neural network is."}],
    "max_tokens": 200,
    "temperature": 0.7,
    "sia_weight": 0
  }' | python3 -m json.tool
```

**服务端日志**（`docker compose -f docker-compose.0gm35b.yml logs sia-0gm35b`）会显示：
```
[SIA] req=2 DONE  intervened=0/83  ratio=0.0%  top1_flip=0/0 (0.0%)
```

ratio=0% 确认 VM 完全跳过。

---

## 5. 加强 SIA 干预（`sia_weight` 调大）

`sia_weight` 越大，VM 得分对 token 选择的影响越强。默认 1.0，可以调到 2.0 加强干预：

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/0GM-1.0-35B-A3B-0427",
    "messages": [{"role": "user", "content": "Write a short poem about the ocean."}],
    "max_tokens": 200,
    "temperature": 0.7,
    "sia_weight": 2.0
  }' | python3 -m json.tool
```

---

## 6. 调节干预频率（`sia_entropy_threshold`）

`sia_entropy_threshold` 控制 entropy gate：只有当前 token 的 logit entropy 高于此阈值时才调用 VM。

- **threshold=0**：每个 token 都干预（ratio=100%），最慢，最激进
- **threshold=1.0**（默认）：约 20% 的 token 干预，平衡速度与效果
- **threshold 很大（如 999）**：几乎不干预，接近 noSIA

### 强制每 token 都干预（`sia_entropy_threshold=0`）

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/0GM-1.0-35B-A3B-0427",
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
    "model": "/workspace/models/0GM-1.0-35B-A3B-0427",
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

`sia_topk` 控制每步 VM 评分的候选 token 数量。越大评分越细致但越慢。默认 10。

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/0GM-1.0-35B-A3B-0427",
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
    "model": "/workspace/models/0GM-1.0-35B-A3B-0427",
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

加 `"stream": true` 获得逐 token 流式输出（SSE 格式）。

### 9.1 带 usage 的流式（推荐，计费要求）

加 `"stream_options": {"include_usage": true}`，结尾 chunk 会携带 `usage` 字段（0G router 计费强制要求，生产环境必须带此参数）：

```bash
curl -sN http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/workspace/models/0GM-1.0-35B-A3B-0427","messages":[{"role":"user","content":"hi"}],"max_tokens":2048,"stream":true,"stream_options":{"include_usage":true}}'
```

**Response（逐行流式，省略中间 token chunk，展示末尾关键部分）:**
```
...
data: {"id": "chatcmpl-a9d6f09e3d3a", "object": "chat.completion.chunk", "created": 1781530216, "model": "/workspace/models/0GM-1.0-35B-A3B-0427", "choices": [{"index": 0, "delta": {"content": " 😊"}, "finish_reason": null}]}

data: {"id": "chatcmpl-a9d6f09e3d3a", "object": "chat.completion.chunk", "created": 1781530216, "model": "/workspace/models/0GM-1.0-35B-A3B-0427", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: {"id": "chatcmpl-a9d6f09e3d3a", "object": "chat.completion.chunk", "created": 1781530216, "model": "/workspace/models/0GM-1.0-35B-A3B-0427", "choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 157, "total_tokens": 168}}

data: [DONE]
```

### 9.2 不带 usage 的流式

不传 `stream_options` 时，结尾不包含 usage chunk（直连调试用，不经过 router 计费时可用）：

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/0GM-1.0-35B-A3B-0427",
    "messages": [{"role": "user", "content": "Count from 1 to 5."}],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": true
  }'
```

> 本服务实现的是**真·token-level streaming**，每生成一个 token 立即推送一个 SSE chunk，TTFT 与首 token 到达时间一致。

---

## 10. 参数速查表

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | 必填 | 填服务端模型路径 |
| `messages` | array | 必填 | 对话历史，支持 system / user / assistant |
| `max_tokens` | int | 512 | 最大生成 token 数，建议 ≤ 1800（受 max_model_len=2048 限制；长 system prompt 会进一步压缩上限） |
| `temperature` | float | 0.7 | 采样温度，0 为贪婪解码，越高越随机 |
| `top_p` | float | — | nucleus sampling 概率阈值 |
| `top_k` | int | — | 只从概率最高的 k 个 token 里采样 |
| `repetition_penalty` | float | 1.0 | 重复惩罚，1.0=关闭，1.05 轻度抑制重复 |
| `stream` | bool | false | 开启 SSE 流式输出 |
| `sia_weight` | float | 1.0 | **SIA 干预强度**，0=关闭，>1=加强 |
| `sia_topk` | int | 10 | **SIA 候选 token 数**，越大越精细但越慢 |
| `sia_entropy_threshold` | float | 1.0 | **SIA 干预频率门控**，0=全干预，越大干预越少 |

---

## 11. 多模态输入（图像 + 文本）

0GM-1.0-35B 是 Vision-Language 模型，支持 OpenAI `image_url` 格式的图像输入。

**重要**：Value Model（Qwen3-4B）是纯文本模型，无法对含图像上下文的候选 token 打分。因此，**当 request 包含图像时，server 会自动将 `sia_weight` 强制设为 `0.0`，跳过 VM 评分，直接用原始模型推理**。无需在 request 里手动设置 `sia_weight`——server 自动处理。

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
          {"type": "text", "text": "这张图片里是什么颜色？"}
        ]
      }
    ]
  }' | python3 -m json.tool
```

**服务端日志**（`docker compose -f docker-compose.0gm35b.yml logs sia-0gm35b`）会打印：
```
[SIA] multimodal request detected — SIA bypassed (VM is text-only)
(EngineCore pid=...) [SIA] req=0 DONE  intervened=0/N  ratio=0.0%  top1_flip=0/0 (0.0%)
```

`ratio=0.0%` 确认 VM 完全跳过，推理由原始模型独立完成。

**Response 示例：**
```json
{
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "图片是橙色的。"
            },
            "finish_reason": "stop"
        }
    ]
}
```

---

## 12. 服务健康检查

```bash
curl -s http://localhost:8000/health
```

**Response:**
```json
{"status": "ok"}
```

---

## 13. 验证模型名称校验（OpenAI 错误格式）

服务对 `model` 字段做 OpenAI 兼容校验：传入不存在的模型名时返回标准 404 错误，而非 FastAPI 默认的 `{"detail": ...}` 格式。

### 13.1 不存在的模型名 → 404

```bash
curl -s -w "\nHTTP %{http_code}\n" \
  -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "does-not-exist", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}' \
  | python3 -m json.tool
```

**预期输出：**
```json
{
    "error": {
        "message": "The model `does-not-exist` does not exist or is not loaded.",
        "type": "invalid_request_error",
        "param": null,
        "code": "model_not_found"
    }
}
HTTP 404
```

### 13.2 确认合法模型名

模型名从服务日志获取（`docker compose logs sia-0gm35b | grep "Model ID"`），例如 `0GM-1.0-35B-A3B-0427`：

```bash
# 传正确的模型名
curl -s -o /dev/null -w "HTTP %{http_code}\n" \
  -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "0GM-1.0-35B-A3B-0427", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}'

# 不传 model 字段（服务自动使用已加载模型）
curl -s -o /dev/null -w "HTTP %{http_code}\n" \
  -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}'
```

两条均应返回 `HTTP 200`。

> **说明**：合法值有三种——`null`/不传、`_model_id`（basename，如 `0GM-1.0-35B-A3B-0427`）、完整路径（如 `/workspace/models/0GM-1.0-35B-A3B-0427`）。

### 13.3 空 messages 数组 → 400

```bash
curl -s -w "\nHTTP %{http_code}\n" \
  -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [], "max_tokens": 5}' \
  | python3 -m json.tool
```

**预期输出：**
```json
{
    "error": {
        "message": "[] is too short - 'messages'",
        "type": "invalid_request_error",
        "param": null,
        "code": null
    }
}
HTTP 400
```

### 13.4 传入 tools → 400

0GM-1.0-35B 模型本身支持 tool call，但当前服务端尚未实现解析层（Value Model Qwen3-4B-Base 未经 tool call 对齐训练，评分不可靠），因此拒绝含 `tools` 的请求：

```bash
curl -s -w "\nHTTP %{http_code}\n" \
  -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}], "tool_choice": "auto", "max_tokens": 5}' \
  | python3 -m json.tool
```

**预期输出：**
```json
{
    "error": {
        "message": "Tool calls are not yet implemented in this server.",
        "type": "invalid_request_error",
        "param": null,
        "code": null
    }
}
HTTP 400
```

---

## 14. 验证 Prefix Cache 命中（`cached_tokens`）

运行项目自带的测试脚本，向服务连发两次相同请求，验证第二次响应中 `cached_tokens > 0`：

```bash
python tests/test_cache_hit.py [--url http://localhost:8000]
```

**预期输出：**
```
目标: http://localhost:8000
system prompt 长度: 5800 chars
发送两次相同请求，验证第二次 cached_tokens > 0 ...

请求 1: prompt_tokens=1218, cached_tokens=0    ✅
请求 2: prompt_tokens=1218, cached_tokens=1056 ✅

预期: 请求1 cached_tokens=0，请求2 cached_tokens=1056
```

**说明：**
- `cached_tokens` 以 block 为单位，block_size ≈ 1056 tokens（由 0GM-35B 的 GatedDeltaNet 状态大小决定）
- prompt < 1056 tokens 时永远不会命中缓存（不足一个完整 block）
- 测试脚本构造了约 1218 tokens 的 prompt，确保超过阈值
