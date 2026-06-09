# 0GM-35B SIA 推理加速方案 (2026-06-09)

**模型**: 0GM-1.0-35B-A3B-Instruct (Qwen3.5/3.6 MoE arch)
**当前基线**: SIA ~32 tok/s vs noSIA ~98 tok/s → **3× 慢**
**per-INTERVENE**: ~94 ms (HTTP /classify)，干预率 ~10%
**核心约束**: 只有 vllm 0.19 支持 `Qwen3_5MoeForConditionalGeneration`，无法降级

---

## 已确认失败，不要再试

**b2 inproc（同进程 RM）在 vllm 0.19 上** — 测了 4 种配置全部失败：

| 配置 | 结果 |
|---|---|
| FULL_AND_PIECEWISE 主 LLM + PIECEWISE RM，同进程 | ❌ RM error 10%+ steps |
| FULL_AND_PIECEWISE 主 LLM + eager RM，同进程 | ❌ RM error（flag 来自主 LLM 侧） |
| PIECEWISE-only 主 LLM + PIECEWISE RM，同进程 | ❌ 100% INTERVENE failure |
| PIECEWISE-only 主 LLM + eager RM，同进程 | ❌ RM error |

根因：vllm 0.19 的 PIECEWISE 改成了 runtime capture。主 LLM 推理时随时触发 `torch.cuda.graph()`，设置进程级全局 `is_currently_capturing` flag，同进程任何 CUDA op 都被 block。改 RM 侧配置无效。

**降级 vllm 版本** — 0GM-35B 用的 `Qwen3_5MoeForConditionalGeneration` 架构在 vllm 0.10/0.15/0.17.1 里均未注册，无法加载模型，死路。

---

## 可行方案（按可行性 × 收益排序）

### P0 — vllm 0.18.x 试探

**工作量**: 30 分钟 | **潜在收益**: 若成功 → 端到端 ~1.4–1.5×

PyPI 上存在 0.18.0 / 0.18.1，但从未测试过。需同时满足：

1. 支持 `Qwen3_5MoeForConditionalGeneration` 架构
2. PIECEWISE 仍是 AOT capture（如 0.17.1），而非 runtime trigger（如 0.19）

若两者都满足，0GM-35B 就能走和 VL-30B 完全相同的 b2 inproc 路径，预计获得 VL-30B 同等加速（1.46×）。

成本极低，30 分钟内出结论。失败了也只是在失败表里加一行。

```bash
python3 -m venv /opt/venv-0gm-018
source /opt/venv-0gm-018/bin/activate
pip install vllm==0.18.0
python -c "from vllm.model_executor.models import ModelRegistry; print([k for k in ModelRegistry.get_supported_archs() if 'Qwen3' in k or 'Moe' in k])"
```

---

### P1 — `SIA_RM_MULTIPROCESS=1`（b2 backend 子进程模式）

**工作量**: 0.5 天 | **潜在收益**: per-call 94ms → ~24ms，端到端 ~32 → **~70–80 tok/s**

b2 backend 支持两种模式：
- `SIA_RM_MULTIPROCESS=0`：同进程 InprocClient → 被 cudagraph flag 杀死
- `SIA_RM_MULTIPROCESS=1`：独立子进程，通信走 vllm 内部 ZMQ + /dev/shm

子进程有独立 CUDA context，不受主 LLM 的 cudagraph flag 影响，同时比 HTTP 快（无 JSON 序列化 + TCP 开销）。0GM-35B 和 VL-30B 都没有测试过这条路径。

启动命令（在现有 `0gm35b-http` venv 里测试）：
```bash
SIA_RM_MULTIPROCESS=1 \
SIA_LLM_CUDAGRAPH=piecewise \
python src/sia_vllm_server.py \
  --llm /path/to/0GM-35B \
  --rm_backend b2 \
  --rm_model /path/to/VM-Qwen3-4B-merged-for-vllm \
  --topk 10 --weight 1.0 --entropy_threshold 1.0
```

验证：server log 里应出现正常 `[SIA] req=N DONE intervened=M/T ratio=X%`，无 RM error。

**风险**：vllm 0.19 内部 `unlock_workspace()` 调用是为 inproc 模式设计的，MP 模式下行为需验证。

---

### P2 — Option A Step 1 致命 PoC

**工作量**: 1 小时 | **作用**: 决定 Option A（P5）生死，避免 3–5 天无效投入

在同一进程里加载 vllm 0.19 主 LLM（0GM-35B，FULL_AND_PIECEWISE）和 `transformers` Qwen3-4B，同时跑推理，观察是否出现 `CUDA graph capturing detected at an inappropriate time`。

```python
import torch
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM

# 1. 加载主 LLM
llm = LLM("/path/to/0GM-35B", gpu_memory_utilization=0.6)

# 2. 同进程加载 transformers RM
rm = AutoModelForCausalLM.from_pretrained(
    "/path/to/VM-Qwen3-4B-merged-for-vllm",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)

# 3. 主 LLM 生成时同步调用 RM forward，看是否报错
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=20))
with torch.no_grad():
    rm(input_ids=torch.tensor([[1,2,3]]).cuda())
print("PASS — no cudagraph conflict")
```

- 不出现错误 → Option A 可行，继续 P5
- 出现 `CUDA graph capturing detected` → Option A 死路，省去 3–5 天

---

### P3 — Unix Socket 替换 TCP

**工作量**: 0.5 天 | **潜在收益**: per-INTERVENE 94ms → ~75ms，端到端 +10%

把 RM HTTP server 的监听地址从 `127.0.0.1:8001` 改成 Unix domain socket，消除 TCP 协议栈开销。

风险极低，与其他所有方案正交，任何时候都可以顺手做。

```bash
# RM server 侧
vllm serve /path/to/RM --runner pooling --convert classify \
  --uds-path /tmp/sia_rm.sock ...   # 或等效参数

# LLM server 侧
python src/sia_vllm_server.py \
  --rm_url http+unix://%2Ftmp%2Fsia_rm.sock ...
```

---

### P4 — Dual CUDA Stream 流水线

**工作量**: 2–3 天 | **潜在收益**: 端到端 **+20–40%**（单项工程最大收益）

当前 `apply(logits)` 是同步阻塞：主 LLM decode → 等 RM 返回 → 继续。

改成异步流水线：主 LLM 在 stream A 跑 decode 的同时，RM 在 stream B 评分**上一步**的 token，两者重叠执行。

```
Step t:   [主LLM decode] ──────────────────────
Step t:                    [RM score step t-1]  ←── overlap
Step t+1: [主LLM decode] ──────────────────────
```

代价：需要改 SIA LogitsProcessor 的同步语义（当前 `apply()` 是阻塞式），引入 1-step 延迟的 RM 评分结果。有一定架构侵入性。

---

### P5 — Option A 完整实现（条件：P2 PoC 通过）

**工作量**: 3–5 天 | **潜在收益**: per-INTERVENE 94ms → 25–35ms，端到端 **+15–25%**

用 `transformers` + `DynamicCache` 替换 vllm RM：
- 不调用 `torch.cuda.graph()`，绕过 flag 冲突
- 共享前缀 KV cache：1 次 prefix forward + 5 次 1-token candidate forward（等价于 b2 InprocClient 的 1+4 优化）
- 接口保持与现有 b2 backend 一致：`new_session` / `fix_a_token` / `score_candidates`

注意：DynamicCache 共享前缀批量 forward 需要手动 expand KV cache（prefix KV × 5 copies），实现复杂度中等。

---

### P6 — gRPC / msgpack 替换 HTTP/JSON

**工作量**: 1–2 天 | **潜在收益**: per-INTERVENE 94ms → ~50ms，端到端 **+18%**

把 RM server 接口从 HTTP + JSON 改成 gRPC 或 msgpack-rpc，减少序列化和协议栈开销。比 Unix Socket（P3）效果更好，但工作量也更大。

---

## 推荐执行顺序

```
P0 (30min)  → 测 vllm 0.18.x，成功则直接 b2 inproc，任务完成
    ↓ 失败
P1 (0.5天)  → 测 SIA_RM_MULTIPROCESS=1，预期最大单项收益
P2 (1h)     → Option A PoC，决定 P5 是否值得做
P3 (0.5天)  → Unix socket，低风险稳拿，任何时候可并行
    ↓
P4 (2-3天)  → 双流流水线，最大工程收益但侵入性强
P5 (3-5天)  → 仅 P2 PoC 通过后做
P6          → 若 P3 效果不够再加
```

---

## 参考文档

- [`0gm-35b-sia-rm-inproc-path-20260602.md`](0gm-35b-sia-rm-inproc-path-20260602.md) — 所有 inproc 配置失败记录 + 根因分析
- [`0gm-35b-speedup-options-20260605.md`](0gm-35b-speedup-options-20260605.md) — 各方案详细分析（含性能估算）
- [`vl30b-b2-inproc-speedup-20260605.md`](vl30b-b2-inproc-speedup-20260605.md) — VL-30B 1.46× 加速参考（b2 inproc sweet spot）
- `/tmp/vl30b-inproc-vm-feasibility-analysis-20260604.md` — vllm 版本对比表 + 0.18.x 候选分析
