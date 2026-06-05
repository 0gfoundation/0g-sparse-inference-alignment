# SIA per-request 开关 — 设计方案 (2026-06-05)

**目标**: 让 OpenAI-compatible API client 在每个 request 上指定是否开 SIA 干预, 不用重启 server / 不用起两个进程。

**现状**:
- `src/sia_vllm_server.py:78-93` 的 `ChatCompletionRequest` 没有 SIA 相关字段, client 传 `enable_sia=False` 被 Pydantic 静默丢弃 (`extra="ignore"` 是 v2 默认)
- `SIALogitsProcessor` 在 engine init 时全局注册, 一份 instance 跑所有 request
- 已有半成品: `src/sia_vllm_RM.py:1087` 的 `effective_weight = self._weight_per_req.get(i, self._WEIGHT)` 支持 per-req weight override, 但 `_weight_per_req[i] = ...` **从未在请求入口被写入** (只在 batch reorder 时复制)

**测试约束 (must-have)**:
- 不开 SIA 的 request 的推理速度跟原生 vllm 接近 (期望 <1% 偏离, 单 request 场景)
- 不破坏现有命令的 byte 兼容性: 老的 `--weight 1.0 --entropy_threshold 1.0` 命令行启动 + 客户端不传新字段, 行为跟现在 byte-identical

---

## 一. 三处改动 (基本方案 — per-request toggle)

### 改动 1 — `src/sia_vllm_server.py`: 请求 schema 加 SIA 字段

```python
class ChatCompletionRequest(BaseModel):
    ...  # 现有字段保持不动
    chat_template_kwargs: Optional[dict] = None

    # === 新增 ===
    # per-request SIA 控制 (None = 用 server 启动时的默认值)
    sia_weight: Optional[float] = None              # 0.0 = 关闭干预
    sia_entropy_threshold: Optional[float] = None   # ↑ 设为很大也能关 (但 topk/entropy 仍计算, 不省 SKIP 成本)
```

**为什么用 `sia_weight` 而不是 `enable_sia: bool`**:
- 跟现有 server 端 `--weight` 含义对齐
- 允许 client 做局部强度调整 (e.g. `sia_weight=0.5` 减半干预)
- `0.0 = 完全关` 是自然的, 跟现有 noSIA arm (`--weight 0.0`) 一致

`CompletionRequest` 同样加这两个字段 (一处改动两个 class)。

### 改动 2 — `src/sia_vllm_server.py`: 通过 `SamplingParams.extra_args` 把字段传到 engine

vllm v1 的 `SamplingParams` 支持 `extra_args: dict` 透传给 LogitsProcessor (vllm 0.10+ 已有, 0.17/0.19 都可用)。

```python
def _build_sampling_params(req: ChatCompletionRequest) -> SamplingParams:
    kwargs = dict(...)  # 现有逻辑

    # === 新增: 透传 SIA 字段 ===
    sia_extras = {}
    if req.sia_weight is not None:
        sia_extras["sia_weight"] = req.sia_weight
    if req.sia_entropy_threshold is not None:
        sia_extras["sia_entropy_threshold"] = req.sia_entropy_threshold
    if sia_extras:
        kwargs["extra_args"] = sia_extras

    return SamplingParams(**kwargs)
```

**`/v1/completions` 路径** (`CompletionRequest`) 也要同样改 (server 同一处函数, 复用 OK)。

### 改动 3 — `src/sia_vllm_RM.py`: `SIALogitsProcessor.update_state` 读 extra_args

vllm v1 在 `BatchUpdate.added` 里给 LogitsProcessor 传**每个新加入 request 的 `SamplingParams`** (见 vllm 0.17/0.19 的 `LogitsProcessor.update_state` API)。

```python
def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
    if batch_update is None:
        return

    # === 现有 removed / moved 逻辑保持不动 ===
    for idx in batch_update.removed:
        ...
    if batch_update.moved:
        ...

    # === 新增: added 时读 extra_args 写入 _weight_per_req ===
    if batch_update.added:
        for idx, sampling_params, _output_tok_ids in batch_update.added:
            extras = getattr(sampling_params, "extra_args", None) or {}
            sia_w = extras.get("sia_weight")
            if sia_w is not None:
                # 显式覆盖, 包含 0.0 (关 SIA)
                self._weight_per_req[idx] = float(sia_w)
            # else: 不写, apply() fallback 到全局 self._WEIGHT
```

> ⚠️ `batch_update.added` 的 schema 在 vllm 0.17 vs 0.19 可能略有差异 (`tuple[idx, sampling_params, output_tok_ids]` vs `tuple[idx, sampling_params]`), 实施时先 print(batch_update.added[0]) 确认 schema。

#### 关于 `sia_entropy_threshold`
- 内部状态加 `self._entropy_threshold_per_req: dict[int, float] = {}`, 跟 `_weight_per_req` 同样的生命周期管理 (added/moved/removed 处理)
- `apply()` 里, decision logic 改成 `eff_thr = self._entropy_threshold_per_req.get(i, self._ENTROPY_THRESHOLD)`, 然后用 `eff_thr` 替换原本的 `self._ENTROPY_THRESHOLD`

---

## 二. 第 4 处改动 (可选, 推荐做 — 真"零开销")

**问题**: 上面 3 处改动后, **SIA-disabled 的 request 仍然 pay 整个 SKIP step 成本** (VL-30B ~3.7ms / 35B ~0.7ms per token), 因为 topk + entropy + cpu_sync 在 `apply()` 入口就跑了 (`src/sia_vllm_RM.py:950-973`), 还没轮到 `_weight_per_req[i]` 判断。

**第 4 处改动** — 在 `apply()` 最入口加 全批 early-return 判断:

```python
def apply(self, logits: torch.Tensor) -> torch.Tensor:
    batch_size = logits.shape[0]

    # === 新增: 整批全 noSIA 时直接 return, 跟原生 vllm 等价开销 ===
    all_disabled = all(
        self._weight_per_req.get(i, self._WEIGHT) == 0.0
        for i in range(batch_size)
    )
    if all_disabled:
        # 维护 _total_steps 计数仍然要做 (不写也行, 但 batch reorder 复制会乱)
        for i in range(batch_size):
            self._total_steps[i] = self._total_steps.get(i, 0) + 1
        return logits

    # === 以下保持原状: topk + entropy + 决策 + intervention ===
    ...
```

**效果**:

| 场景 | apply() 开销 |
|---|---|
| 单 request 全 SIA | 现状 (~3.7ms VL-30B SKIP / ~95ms INTERVENE) |
| 单 request 全 noSIA (sia_weight=0) | **~0.01 ms** (Python dict lookup × batch_size) ← 跟原生 vllm 等价 ✅ |
| 混合 batch (>=1 SIA + noSIA) | 全批仍跑 topk+entropy (GPU op 不支持子集 mask 省时), noSIA 那条只是不 modify logits |

**生产场景判断**:
- 全公司 traffic 大多数走 noSIA, 少数 traffic 走 SIA → **第 4 处必加**, 否则全公司 traffic 都 pay SIA tax
- 全公司 traffic 都走 SIA → 第 4 处可省 (现状 = 期望)
- 评测 (单 request 串行) → **第 4 处也加**, 否则 noSIA arm 跟 SIA arm 共用 server 时 noSIA 那条仍 pay SKIP cost

---

## 三. 实施工作量

| 改动 | 文件 | 行数 | 测试 |
|---|---|---|---|
| 1. 请求 schema | `src/sia_vllm_server.py` | ~6 (`ChatCompletionRequest`) + ~4 (`CompletionRequest`) | curl 单元 |
| 2. SamplingParams 传 extra_args | `src/sia_vllm_server.py:_build_sampling_params` (复用给两个 endpoint) | ~10 | curl + server log 看 extra_args 出现 |
| 3. update_state added 读 extras | `src/sia_vllm_RM.py:update_state` + 加 `_entropy_threshold_per_req` 字段 + apply 里用 eff_thr | ~25 (含 added 处理 + moved/removed 给 entropy threshold 同步) | smoke 跑 2 request, 一开一关, server log 看 `effective_weight` |
| 4. (可选) 全批 early-return | `src/sia_vllm_RM.py:apply` 入口 | ~6 | A/B benchmark: noSIA via per-req flag vs 原生 vllm, throughput < 1% 差 |

**总计**: 3 处 ~45 行 + 第 4 处 ~6 行 ≈ **~50 行 + 配套单元测试**

---

## 四. 测试计划

### 4.1 向后兼容
- 老命令 `mmlu_eval.py --base_url ... --model ... --output ...` (不传 sia_weight) → server log 显示无 `extra_args`, `_weight_per_req` 为空, apply 走全局 `_WEIGHT` 默认路径
- 复跑一个已知 acc 的 eval (e.g. `eval/results/vl30b_SIA_mmlu_150q.json`), accuracy 在 ±1pp 内吻合

### 4.2 per-request 关 SIA
```bash
curl -X POST http://localhost:8000/v1/chat/completions -d '{
  "model":"...","messages":[...],"max_tokens":200,
  "sia_weight": 0.0
}'
```
预期 server log: `[SIA] req=0 DONE  intervened=0/N  ratio=0.0%`

### 4.3 per-request 开 SIA (server 默认 noSIA)
启动 server 时 `--weight 0.0 --entropy_threshold 999999`, client 传:
```json
{"sia_weight": 1.0, "sia_entropy_threshold": 1.0}
```
预期 server log: `intervened=M/N  ratio=10-40%`, M > 0

### 4.4 速度回归 (第 4 处改动后)
- noSIA via per-req flag, 单 request 跑 200Q AlpacaEval max=256
- 跟 raw `vllm serve` (无 sia processor) 同 prompt 同 sampling 比 tok/s
- 期望: 差距 < 1% (单 request scenarios)

---

## 五. 跟现有 noSIA arm 的对比

| 路径 | 启动 | client 调用 | apply 开销 (per step) |
|---|---|---|---|
| 老 noSIA arm (跟 SIA byte-identical 仅 2 处 diff) | `--weight 0.0 --entropy_threshold 999999` | 不传 SIA 字段 | ~3.7ms (VL-30B) / ~0.7ms (35B) ← topk+entropy 仍跑 |
| 新 per-req (本方案 3 处) | `--weight 1.0 --entropy_threshold 1.0` (server 默认开 SIA) | `sia_weight: 0.0` | 同上 (3.7/0.7 ms) ← 跟老 noSIA arm 同开销 |
| **新 per-req + 第 4 处 early-return** | 同上 | `sia_weight: 0.0` | **<0.01 ms** ← 跟 raw vllm 等价 ✅ |

**第 4 处是核心优化, 没它的话 per-req SIA off 没省时间, 只是"代码共享"省了 1 个 server 进程**。

---

## 六. 风险 + 防护

| 风险 | 防护 |
|---|---|
| vllm 0.17 vs 0.19 `BatchUpdate.added` schema 差异 | 实施时 print(batch_update.added) 抓 schema; 写防御性 `if len(item) >= 2: idx, sp = item[0], item[1]` |
| `SamplingParams.extra_args` 在 b2 inproc backend 不传 | b2 RMClient 内部 sampling 是另一个 vllm, 不接收 client extra_args (没影响, b2 的 sampling 跟 client 隔离) |
| Client 传 `sia_weight=1.0` 但 server 启动时没起 RM (因为预期 noSIA only) | server 启动时检查: 如果有 sia_weight 字段被任何 request 设, 但 `_rm is None`, 返回 400 给该 request (graceful) |
| Per-req 字段被中间 proxy/CDN 剥掉 | 推荐放在 OpenAI 协议允许的 `extra_body` 透传通道, 或者用 header `X-SIA-Weight` 作为 fallback (本方案先不做) |

---

## 七. 排序建议

实施顺序: **3 处先做 + 单元测试** → 跑一个 mixed-batch 实验确认 _weight_per_req 路径正确 → 再加 **第 4 处** + benchmark vs raw vllm。

第 1 步 + 第 4 步是核心, 第 2-3 步是 plumbing, 没有跳过空间。
