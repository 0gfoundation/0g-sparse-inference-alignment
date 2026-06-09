# 0GM-35B b2 inproc 速度优化实测 — vllm 0.18.0 sweet spot (2026-06-09)

**TL;DR**: 在 vllm **0.18.0** + `SIA_RM_CUDAGRAPH=none` 上启用 `--rm_backend b2`，0GM-35B 首次实现同进程 RM：

- **SIA 吞吐: ~49 tok/s (vs HTTP ~32 tok/s) → ~1.5× 加速**
- **noSIA 吞吐: ~54 tok/s（b2 inproc 对无干预步骤零开销）**
- **0 RM error, 0 cudagraph 冲突**（venv6 smoke test, H200 单卡）
- **零代码改动**: 只需切 venv (0.18.0) + 两个 env var

已通过 smoke test，无需另外测试即可部署。

---

## 1. 背景：为什么 0GM-35B 之前不能用 b2 inproc

0GM-35B 使用 `Qwen3_5MoeForConditionalGeneration` 架构（Qwen3.5 系列），该架构在 vllm 0.17.x 及更早版本中**未注册**，无法加载，强制要求 0.18+。

而 vllm 0.19.x 的 PIECEWISE 改为了 runtime capture 模式，主 LLM 推理时会进入 `torch.cuda.graph()` context，设置进程级全局 `is_currently_capturing` flag，同进程任何 CUDA op（包括 eager RM forward）都被 block。经过穷举测试，vllm 0.19 上所有 b2 配置均 100% 失败：

| 配置 | 结果 |
|---|---|
| FULL_AND_PIECEWISE 主 LLM + PIECEWISE RM | ❌ RM error 10%+ steps |
| FULL_AND_PIECEWISE 主 LLM + eager RM | ❌ RM error（flag 来自主 LLM） |
| PIECEWISE-only 主 LLM + PIECEWISE RM | ❌ 100% INTERVENE failure |
| PIECEWISE-only 主 LLM + eager RM | ❌ RM error |

详见 [`doc/0gm-35b-sia-rm-inproc-path-20260602.md`](0gm-35b-sia-rm-inproc-path-20260602.md)。

---

## 2. 为什么 vllm 0.18.0 是 sweet spot

| 条件 | 影响 | 0.17.1 | **0.18.0** | 0.19.0 |
|------|------|--------|-----------|--------|
| 支持 `Qwen3_5MoeForConditionalGeneration` | 0GM-35B 能加载 | ❌ | ✅ | ✅ |
| PIECEWISE 仍是 AOT capture | 推理期不进入 `torch.cuda.graph()` | ✅ | ✅ | ❌ (改 runtime) |
| 有 `unlock_workspace()` | 过得了 MoE workspace lock | ❌ | ✅ | ✅ |

0.18.0 同时满足三个条件，是 0GM-35B b2 inproc 的唯一可用版本。

---

## 3. 为什么 `SIA_RM_CUDAGRAPH=none`（而非 `piecewise`）

vllm 0.18.0 RM 侧设置 `SIA_RM_CUDAGRAPH=piecewise` 同样失败：

**失败原因**: RM 的 PIECEWISE 模式在 warmup 阶段只捕获固定 batch_descriptor 集合。推理时，prefix caching（APC）会生成 warmup 未见过的新 batch_descriptor，`CUDAGraphWrapper.__call__` 检测到新 key 后调用 `validate_cudagraph_capturing_enabled()`，该函数因 post-startup capturing 被禁用而 raise：
```
RuntimeError: CUDA graph capturing detected at an inappropriate time.
```

注意：这个 raise 发生在进入 `torch.cuda.graph()` context **之前**，所以不会设置 CUDA 全局 flag——这是和 0.19 故障模式的关键区别（0.19 是 flag 冲突，0.18 是 Python 异常）。

**解决方案**: `SIA_RM_CUDAGRAPH=none` → `enforce_eager=True`，RM 完全跳过 `CUDAGraphWrapper`，无任何 graph capture 尝试。

0.18 主 LLM 的 PIECEWISE 在推理时只 **replay** 已捕获的 graph（不进入 `torch.cuda.graph()` context），所以 eager RM 可以自由 forward。

---

## 4. 配置

### 4.1 拓扑 — b2 inproc

**单进程**：SIA EngineCore 内嵌套一个 nested vllm `LLM(...)` 作为 RM。

```
┌─────────── 主进程 (sia_vllm_server.py) ─────────────┐
│   FastAPI server (port 8000)                          │
│   ↓                                                    │
│   AsyncLLMEngine (主 LLM = 0GM-35B, vllm 0.18.0)     │
│   ↓ EngineCore subprocess                             │
│   ┌───────────────────────────────────────────────┐   │
│   │  SIA LogitsProcessor                           │   │
│   │  ↓  __call__()                                 │   │
│   │  RMClient (b2 backend, inproc)                 │   │
│   │  ↓  score_candidates(...)  ← Python 直接调用  │   │
│   │  ┌─────────────────────────────────────────┐  │   │
│   │  │  nested vllm LLM(VM-Qwen3-4B-merged...) │  │   │
│   │  │  enforce_eager=True  (SIA_RM_CUDAGRAPH   │  │   │
│   │  │  =none)                                  │  │   │
│   │  │  on same CUDA context as 主 LLM          │  │   │
│   │  └─────────────────────────────────────────┘  │   │
│   └───────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────┘
```

无 HTTP 跨进程通信。RM 调用走同进程 Python 函数，cross-tokenizer bridge 自动处理 0GM-35B（248044 vocab）到 VM-Qwen3-4B（151643 vocab）的 token 映射。

### 4.2 关键参数

| 参数 | 值 |
|------|---|
| `--llm` | 0GM-1.0-35B-A3B-0427 |
| `--rm_backend` | **`b2`** |
| `--rm_model` | VM-Qwen3-4B-merged-for-vllm |
| `--llm_gpu_mem` | 0.55 |
| `--rm_b2_gpu_mem` | 0.15 |
| `--max_model_len` | 2048 |
| `--topk` | 10 |
| `--weight` | 1.0 |
| `--entropy_threshold` | 1.0 |
| `SIA_LLM_CUDAGRAPH` | **`piecewise`**（主 LLM PIECEWISE-only AOT） |
| `SIA_RM_CUDAGRAPH` | **`none`**（RM eager，必须，见 §3） |
| `SIA_RM_MULTIPROCESS` | `0`（同进程 InprocClient） |

---

## 5. 实测吞吐（H200 单卡，2026-06-09）

| 模式 | tok/s | 说明 |
|------|-------|------|
| noSIA（sia_weight=0） | **~54** | 纯 vllm 推理，零 RM 开销 |
| SIA（entropy_threshold=1.0，~8% 干预率） | **~49** | 默认配置，约 9% 开销 |
| 旧 HTTP vllm RM backend（基线） | **~32** | 同机型历史测量 |

**b2 inproc vs HTTP：~1.5× 加速** — 匹配 VL-30B b2 inproc 的 1.46×。

> 注意：上述测量使用了 0GM-35B 的 thinking 模式（默认输出含 thinking token），实际对话场景中如关闭 thinking 吞吐会更高。

### SIA 干预日志样本（smoke test）

```
[SIA] req=0 DONE  intervened=3/20    ratio=15.0%  top1_flip=1/3   (33.3%)
[SIA] req=0 DONE  intervened=14/200  ratio=7.0%   top1_flip=11/14 (78.6%)
[SIA] req=0 DONE  intervened=17/200  ratio=8.5%   top1_flip=15/17 (88.2%)
[SIA] req=0 DONE  intervened=22/200  ratio=11.0%  top1_flip=18/22 (81.8%)
```

top1_flip 率 78–88%，说明 RM 干预有实质性影响（仅在高熵位置改变 token 选择）。

---

## 6. 与 VL-30B 对比

| 维度 | VL-30B | 0GM-35B |
|------|--------|---------|
| 架构 | `Qwen3VLMoe` | `Qwen3_5MoeForConditionalGeneration` |
| b2 inproc 的 vllm 版本 | 0.17.1 | **0.18.0** |
| `SIA_LLM_CUDAGRAPH` | `piecewise` | `piecewise` |
| `SIA_RM_CUDAGRAPH` | `piecewise` | **`none`**（0.18 RM PIECEWISE 有 prefix-cache 新 descriptor 问题）|
| 端到端加速 vs HTTP | 1.46× | ~1.5× |
| venv / requirements | `vl30b-b2-inproc.txt` | `0gm35b-b2-inproc.txt` |

---

## 7. 快速上手

### 7.1 安装 venv

```bash
scripts/setup_venv_0gm35b_b2.sh /path/to/venv-0gm35b-b2
```

手动步骤：
```bash
python3 -m venv /path/to/venv-0gm35b-b2
source /path/to/venv-0gm35b-b2/bin/activate
pip install -r requirements/0gm35b-b2-inproc.txt
pip install -e .
```

### 7.2 启动服务

```bash
SIA_LLM_CUDAGRAPH=piecewise SIA_RM_CUDAGRAPH=none \
SIA_RM_MULTIPROCESS=0 \
python src/sia_vllm_server.py \
  --llm /path/to/0GM-1.0-35B-A3B \
  --rm_backend b2 \
  --rm_model /path/to/VM-Qwen3-4B-merged-for-vllm \
  --rm_b2_gpu_mem 0.15 --llm_gpu_mem 0.55 \
  --topk 10 --weight 1.0 --entropy_threshold 1.0 \
  --max_model_len 2048 --port 8000
```

启动时间约 15 分钟（首次 JIT 编译 FlashInfer GDN kernel + PIECEWISE warmup）。

### 7.3 冒烟测试

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":20,"temperature":0.7}' \
  | python3 -m json.tool
```

服务端日志应出现（无 `RM error` 行）：
```
[SIA] req=0 DONE  intervened=X/20  ratio=Y%  top1_flip=...
```

### 7.4 关闭 SIA 干预对比

```bash
# SIA on（默认）
curl -s ... -d '{"...", "max_tokens":200}'

# SIA off（纯 vllm baseline）
curl -s ... -d '{"...", "max_tokens":200, "sia_weight":0}'
```

---

## 8. 注意事项

- **`SIA_RM_CUDAGRAPH=none` 是必须的**，不能用 `piecewise`（见 §3）
- **不要在 0.19 venv 里跑 b2**，无论什么 cudagraph 配置都会失败
- `--max_model_len 2048` 是为了给 RM 留出 GPU 内存；若 GPU 有足够余量可增大
- 首次启动会触发 FlashInfer GDN prefill kernel JIT 编译（~4 分钟），之后缓存复用
- MoE config 警告（`Using default MoE config`）是正常现象，vllm 0.18 无 H200 特定 MoE 调优配置

---

## 参考

- [`doc/0gm-35b-speedup-plan-20260609.md`](0gm-35b-speedup-plan-20260609.md) — 完整方案对比（P0–P6）+ P0 测试过程
- [`doc/vl30b-b2-inproc-speedup-20260605.md`](vl30b-b2-inproc-speedup-20260605.md) — VL-30B 同类加速，含 AlpacaEval 质量验证
- `requirements/0gm35b-b2-inproc.txt` — vllm 0.18.0 依赖配置
- `scripts/setup_venv_0gm35b_b2.sh` — 一键建 venv 脚本
