# SIA vLLM Server

SIA（Sparse Inference-time Alignment）per-token 干预推理服务，兼容 OpenAI Chat API，可与 0g-serving-broker 对接。

论文：[Inference-time Alignment via Sparse Junction Steering](https://arxiv.org/pdf/2602.21215)（Runyi Hu et al.）

---

## 相比论文的附加工作

论文提供了核心算法（SIA LogitsProcessor + entropy gate）和 Value Model checkpoints，但推理代码是基于 transformers 的手写 forward loop，未覆盖生产部署。本项目在此基础上完成了以下工作：

**1. 生产级 vLLM 集成**
重写为 vLLM v1 `LogitsProcessor` 接口，接入异步批处理引擎，支持并发请求和真·token-level streaming（而非先生成完再模拟）。

**2. OpenAI 兼容 HTTP 服务**
`/v1/chat/completions` 完整实现，支持 streaming SSE，可直接接入 OpenAI SDK 客户端和 0g broker 双路由。

**3. b2 inproc RM backend（核心加速）**
把 RM 从独立 HTTP server 改为同进程嵌套 vLLM `LLM()`，利用 prefix caching 实现增量 decode（前缀 KV 复用）。设计了 `RMClient` 状态机（`new_session / fix_a_token / score_candidates`）。实测结果：
- VL-30B：RM call p50 71ms → 11ms（**6.4×**），端到端 **1.46× 加速**，AlpacaEval Skywork reward 统计等价（p=0.73）
- 0GM-35B：端到端 **~1.5× 加速** vs HTTP backend

**4. 异构 tokenizer 支持**
论文假设 LLM 和 RM 共享词表；项目实现了 cross-tokenizer bridge（decode→text→re-encode），支持 LLM 和 RM 词表不同的场景（如 0GM-35B 248k vocab + VM-Qwen3-4B 152k vocab）。

**5. 逐请求参数覆盖**
`sia_weight / sia_topk / sia_entropy_threshold` 支持在每次 HTTP 请求中独立覆盖，无需重启服务。

**6. 关键 bug 修复：`repetition_penalty` 偏差**
发现 `repetition_penalty=1.3` 默认开启叠加 vLLM sampler 顺序，造成 AlpacaEval Skywork score +3.09 vs 论文 +10.30。修复后（`repetition_penalty=1.0`）三组配置（Qwen3-14B / VL-30B / 14B 复现）SIA gain 与论文统计等价（p=0.43）。

**7. 多模型验证**
论文只验证 Qwen3-14B；项目在 Qwen3-VL-30B（7.5× weak-to-strong）和 0GM-1.0-35B-A3B（自研模型）上完成 AlpacaEval / MMLU 端到端评测，确认 SIA gain 可迁移。

**8. vLLM 版本适配工程**
系统排查 vllm 0.17 / 0.18 / 0.19 三版本在 PIECEWISE cudagraph 上的差异，确定各模型的可用版本（VL-30B → 0.17.1，0GM-35B → 0.18.0），提供各自独立的 venv setup 脚本和 requirements 文件。

---

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
