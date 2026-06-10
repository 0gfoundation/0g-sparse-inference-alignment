# 0GM-35B SIA 性能瓶颈分析与优化路线图 (2026-06-09)

**TL;DR**：去掉 `SIA_LLM_CUDAGRAPH=piecewise` 后（FULL_AND_PIECEWISE 模式），整体 tok/s 从 37.8 → **54.1（+43%）**。
`apply_cpu_sync` 表面增大 64×，但因与 LLM forward 重叠，净影响仅 0.34ms/token。
真正的瓶颈是 **b2_score_call（55ms，顺序执行）**，启用 RM CUDA graph 可再提升 **+58%（→ ~85 tok/s）**。

---

## 一. 实验背景

基于 [200Q AlpacaEval 实验](0gm-35b-alpacaeval-200q-b2-inproc-thinking-20260609.md) §三，取 PIECEWISE（旧）vs FULL_AND_PIECEWISE（FULL fix）两组 SIA 数据。

| 项目 | 值 |
|------|---|
| 主 LLM | 0GM-1.0-35B-A3B-0427 |
| Value Model | VM-Qwen3-4B-merged-for-vllm（b2 inproc） |
| venv / vllm | venv6 / 0.18.0 |
| AlpacaEval | 200Q，natural thinking，max_tokens=2048 |
| 干预率 | 16.9%（两组相同）|

---

## 二. per-step 耗时对比

pf-summary 数据来自各自 server log 最终快照（~53200 token steps）。

| 环节 | PIECEWISE p50 | FULL fix p50 | 变化 |
|------|:---:|:---:|:---:|
| **skip_step**（非干预 step 全部开销） | 0.50ms | 4.31ms | ↑ 8.6× |
| apply_topk_ent（GPU top-K + entropy） | 0.43ms | 0.44ms | 持平 ✅ |
| **apply_cpu_sync**（`tensor.item()` CPU-GPU sync） | **0.06ms** | **3.85ms** | **↑ 64×** |
| **b2_score_call**（inproc RM forward） | **53ms** | **55ms** | ↑ 4% |
| intv_prepare（prefix advance） | 0.07ms | 0.08ms | 持平 ✅ |
| intv_apply_logits（logit 修改） | 0.41ms | 0.41ms | 持平 ✅ |
| **整体 tok/s** | **37.8** | **54.1** | **+43% ✅** |

---

## 三. apply_cpu_sync 变慢的原因与真实影响

### 3.1 为什么 apply_cpu_sync 从 0.06ms 涨到 3.85ms

- **PIECEWISE 模式**：model forward 分多段 CUDA graph 提交，Python 在各段之间交替调度，GPU stream 积压相对少；`tensor.item()` 触发 CPU-GPU sync 时，大部分 GPU kernel 已经排空，等待时间极短（0.06ms）。
- **FULL 模式**：整个 model forward 作为一个大 CUDA graph 一次性 replay。`replay()` 立即返回 Python，但 GPU 上仍有大量 pending kernel。SIA 的 `tensor.item()`（用于 entropy 阈值判断）需等所有 pending kernel 排空 → 3.85ms 等待。

### 3.2 net 影响：比看起来小得多

表面上 skip_step 涨了 8.6×，但大部分时间与 LLM forward 重叠：

```
cpu_sync 等待期间，GPU 正在执行 LLM forward kernel。
等 LLM forward 完成，cpu_sync 返回，两者同步结束。
→ cpu_sync 本质是"等 LLM 跑完"，不是额外浪费。
```

从实测数据反推净开销：

```
t_skip (实测) = total_time / ((1 - intv_rate) + intv_rate × t_intv/t_skip)
             = 5822s 反推 → 9.11ms

纯 vLLM baseline = 1/114.1 tok/s = 8.76ms

SIA skip 净额外开销 = 9.11 - 8.76 = 0.34ms / token
```

**3.85ms 的 apply_cpu_sync，真正额外消耗仅约 0.34ms**，其余 3.5ms 隐藏在 LLM forward 等待时间里。

### 3.3 消除 apply_cpu_sync 的收益上限

```
如 apply_cpu_sync → 0ms：
  t_skip → 8.76ms（= 纯 vLLM）
  预期 tok/s ≈ 55.1（当前 54.1 → +1.9%）
```

收益很小，不值得作为优先项。

---

## 四. 真正的主瓶颈：b2_score_call（55ms）

每个干预 step 的时间构成：

```
t_intv = t_skip(9.11ms) + b2_score_call(55ms) + intv_prepare(0.08ms) + intv_apply_logits(0.41ms)
       = 64.60ms
```

RM forward 完全顺序执行，与 LLM forward 无任何重叠，构成 **干预 step 的 85% 耗时**。

b2_score_call 从 PIECEWISE（53ms）到 FULL fix（55ms）还轻微增加了 4%，可能是 FULL graph 占用 GPU 时序资源更多，给 RM 留的带宽略少。

---

## 五. 优化路线图

### 5.1 理论 tok/s 计算模型

```
avg_step_time = (1 - intv_rate) × t_skip + intv_rate × t_intv
tok/s = 1 / avg_step_time
```

参数：`intv_rate=0.169`，`t_skip=9.11ms`，`t_intv=64.60ms`，当前 tok/s=54.1。

### 5.2 各优化项收益估算

| 优化项 | 原值 | 目标值 | 预期 tok/s | 提升幅度 | 优先级 |
|-------|------|--------|-----------|---------|--------|
| 当前基线 | — | — | **54.1** | — | — |
| **① RM CUDA graph** (`SIA_RM_CUDAGRAPH=full`) | 55ms | ~15ms | **~85** | **+57%** | ⭐⭐⭐ 高 |
| ② 消除 apply_cpu_sync（entropy 判断移至 GPU） | 3.85ms | ~0ms | ~55 | +2% | 低 |
| ③ ①+② 同时 | — | — | **~88** | **+63%** | — |

### 5.3 RM CUDA graph 详解（最高优先级）

**目标**：将 `SIA_RM_CUDAGRAPH` 从 `none`（enforce_eager）改为 `full`，让 RM forward 也走 CUDA graph，预计 55ms → ~15ms。

**为何目前用 `none`**：在 vllm 0.18.0 PIECEWISE 模式下，RM 的 prefix caching 在推理期会生成新的 `batch_descriptor`，触发 runtime CUDA graph capture，而此时 SIA 主进程的 `_state` 已有真实 request，导致 `RuntimeError`。详见 [b2 inproc speedup doc](0gm-35b-b2-inproc-speedup-20260609.md)。

**为何 FULL fix 后值得重新验证**：
- FULL fix 已切到 FULL_AND_PIECEWISE 模式（主 LLM）
- RM 是独立的 `vllm.LLM` 实例，AOT capture 独立进行
- 主 LLM 模式变化不影响 RM capture 路径
- **需要验证**：RM 在 `full` 或 `piecewise` 模式下，prefix caching 是否仍会在推理期触发新 batch_descriptor

**验证方法**（小实验，不影响当前 noSIA eval）：
```bash
# 启动前先确认 noSIA eval 结束
# 在 venv6 环境下，仅测试 RM CUDA graph 启动是否报错
SIA_LLM_CUDAGRAPH=none SIA_RM_CUDAGRAPH=full SIA_RM_MULTIPROCESS=0 \
python src/sia_vllm_server.py \
  --llm /workspace/SIA/models/0GM-1.0-35B-A3B-0427 \
  --rm_backend b2 \
  --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
  --llm_gpu_mem 0.80 --rm_b2_gpu_mem 0.15 \
  --topk 10 --weight 1.0 --entropy_threshold 1.0 \
  --max_model_len 4096 --port 8000
# 发几条请求，观察 b2_score_call p50 是否降至 ~15ms，以及是否有 RuntimeError
```

**若成功**：`SIA_RM_CUDAGRAPH=full` 加入推荐配置，tok/s 从 54 → ~85。

### 5.4 ② 消除 apply_cpu_sync（低优先级，备查）

将 entropy 阈值判断从 CPU（`tensor.item()`）改为 GPU 侧：

```python
# 当前（触发 CPU-GPU sync）
entropy = -(probs * probs.log()).sum().item()  # .item() 是 sync 点
if entropy < self.entropy_threshold:
    return logits

# 改进（GPU 侧判断，无 sync）
entropy_tensor = -(probs * probs.log()).sum()         # 保持在 GPU
should_skip = entropy_tensor < self.entropy_threshold  # GPU bool tensor
# 利用 torch.cond 或在 apply() 末尾统一 sync 一次
```

代价：逻辑改动较大（需要 GPU-conditional 分支），净收益仅 +2%，暂不建议优先投入。

---

## 六. VL-30B 同类问题分析

### 6.1 相同的瓶颈

`vl30b-b2-inproc-speedup-20260605.md` §5b.1（MMLU 150Q）有一个直接对照：

| 运行 | `SIA_LLM_CUDAGRAPH` | tok/s |
|------|:---:|:---:|
| noSIA | 未设置（FULL_AND_PIECEWISE 默认）| **120.1** |
| SIA b2 inproc | `piecewise` | **73.2** |

VL-30B 的 SIA b2 inproc 同样设置了 `SIA_LLM_CUDAGRAPH=piecewise`，导致主 LLM 从 120.1 降至 73.2 tok/s（1.64×），与 0GM-35B 的 2× 同性质。

### 6.2 对 VL-30B 同样安全

| 安全条件 | 0GM-35B (vllm 0.18.0) | VL-30B (vllm 0.17.1) |
|---------|:---:|:---:|
| LLM PIECEWISE 是 AOT capture | ✅ | ✅ |
| RM PIECEWISE 是 AOT capture | ✅（但 prefix caching 触发新 descriptor → 改用 `none`）| ✅ **且已验证 0 RM error（200Q × 48,236 steps）** |
| `SIA_RM_CUDAGRAPH` 推荐值 | `none` | `piecewise`（保持不变）|

vllm 0.17.1 的 PIECEWISE 是 AOT capture，是选它作为 VL-30B "sweet spot" 的关键条件之一（详见 [vl30b-b2-inproc-speedup doc §1](vl30b-b2-inproc-speedup-20260605.md)）。VL-30B 的 `SIA_RM_CUDAGRAPH=piecewise` 已在完整实验中零错误运行，不需要改为 `none`。

**需要修改的只有一处**：去掉 `SIA_LLM_CUDAGRAPH=piecewise`，`SIA_RM_CUDAGRAPH=piecewise` 保留。

### 6.3 预期收益估算

VL-30B 的 b2_score_call p50 仅 **10.83ms**（对比 0GM-35B 的 55ms），RM 本身已不是瓶颈。去掉 PIECEWISE 限制后主 LLM 可达 ~120 tok/s 基线。

```
intv_rate ≈ 15%，RM overhead = 10.83ms，t_skip_net ≈ 8.3ms（纯 vLLM ~120 tok/s）

avg_step = (1 - 0.15) × 8.3 + 0.15 × (8.3 + 10.83 + 0.5)
         ≈ 7.06 + 2.90 = 9.96ms → ~100 tok/s

当前 PIECEWISE SIA: 73.2 tok/s → 预期提升至 ~95–100 tok/s（+30%）
```

---

## 七. 0GM-35B RM 加速：对标 VL-30B 的路径

### 7.1 两组 RM 耗时对比

两个模型用的是**完全相同的 RM**（VM-Qwen3-4B-merged-for-vllm），但 b2_score_call 差了 5×：

| 模型 | vllm | `SIA_RM_CUDAGRAPH` | b2_score_call p50 |
|------|:---:|:---:|:---:|
| VL-30B b2 inproc | 0.17.1 | `piecewise`（PIECEWISE AOT）| **11ms** |
| **0GM-35B b2 inproc** | **0.18.0** | **`none`（enforce_eager）** | **55ms** |

差距 100% 来自 RM 有没有 CUDA graph，与模型本身无关。

### 7.2 0GM-35B RM 为何不能直接用 PIECEWISE

详见 [`0gm-35b-b2-inproc-speedup-20260609.md §3`](0gm-35b-b2-inproc-speedup-20260609.md)。

根本原因：prefix caching（APC）在推理时生成 warmup 未见过的新 `batch_descriptor`，`CUDAGraphWrapper` 检测到新 key 后调用 `validate_cudagraph_capturing_enabled()` → `RuntimeError`。**根因是 prefix caching，不是 PIECEWISE 本身。**

### 7.3 两条修复路径

**方案 A（推荐）：对 RM 关闭 prefix caching（改一行代码）**

RM 实例化时加 `enable_prefix_caching=False`，推理时不再产生新 batch_descriptor，PIECEWISE 即可正常使用。

```python
# src/sia_vllm_RM.py，RM 实例化处
llm = LLM(
    model=rm_model,
    enable_prefix_caching=False,   # 新增：防止 APC 触发新 batch_descriptor
    ...
)
```

预期：b2_score_call 55ms → **~11ms**（对标 VL-30B），整体 SIA **54 → ~85 tok/s（+57%）**。

**方案 B（无代码改动，先试）：`SIA_RM_CUDAGRAPH=full`**

FULL 模式 batch_descriptor 比 PIECEWISE 更简单（主要依赖 batch size，不依赖 attention boundary split），有可能绕过 prefix caching 触发的问题，无需修改代码，改一个 env var 验证：

```bash
SIA_RM_CUDAGRAPH=full SIA_RM_MULTIPROCESS=0 \  # none → full
python src/sia_vllm_server.py ...
```

若启动后跑几条请求无 RM error，则生效；b2_score_call 预期降至 ~15ms（FULL graph 开销比 PIECEWISE 略高）。

### 7.4 综合优化路线图（更新）

| 优化项 | 当前值 | 目标值 | 预期 tok/s | 提升 | 优先级 |
|-------|:---:|:---:|:---:|:---:|:---:|
| 当前基线（FULL fix）| — | — | **54.1** | — | — |
| **① RM CUDA graph**（方案 B 先试，方案 A 保底）| 55ms | ~11–15ms | **~83–85** | **+54–57%** | ⭐⭐⭐ 高 |
| ② 消除 apply_cpu_sync | 3.85ms | ~0ms | ~55 | +2% | 低 |
| ③ ①+② 同时 | — | — | **~87–90** | **+61–66%** | — |

---

## 八. 总结

| 问题 | 根因 | 实际影响 | 解法 |
|------|------|---------|------|
| skip_step 涨 8.6× | FULL graph 积压 GPU kernel，cpu_sync 等待 LLM drain | 净仅 +0.34ms/token，大部分与 LLM forward 重叠 | 可接受，或改 GPU entropy 判断（+2%）|
| **b2_score_call 55ms** | **RM enforce_eager，无 CUDA graph** | **主瓶颈，顺序阻塞，5× 慢于 VL-30B** | **方案 B（`SIA_RM_CUDAGRAPH=full`）先试；方案 A（RM 关 prefix caching）保底** |
| b2_score_call 比 PIECEWISE 多 2ms | FULL graph GPU 资源占用略多 | 微小 | RM CUDA graph 后自然缓解 |
