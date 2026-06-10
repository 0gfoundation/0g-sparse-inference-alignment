# 0GM-35B SIA 性能瓶颈分析与优化路线图 (2026-06-09, 更新 2026-06-10)

**TL;DR（最终结论）**：经过三轮优化，0GM-35B SIA b2 inproc 从 37.8 tok/s 最终提升至 **66–68 tok/s（≈ noSIA 的 59%）**。
- **阶段一**：去掉 `SIA_LLM_CUDAGRAPH=piecewise` → 37.8 → **54.1 tok/s（+43%）**
- **阶段二**：RM CUDA graph（piecewise/full）尝试 → ❌ 对 prefill-heavy RM 无效，反而更慢
- **阶段三**：stable prefix 跨分词器优化 → 54 → **66–68 tok/s（+22%）**；b2_score_call: 55ms → **~30ms（稳定，不随长度增长）**

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
| 0GM-35B b2 inproc | 0.18.0 | `piecewise`（三层修复后）| **115–140ms（❌ 更慢）** |

VL-30B 上 piecewise 有效（11ms）；0GM-35B 上 piecewise 反而比 eager **慢 2-3×**，原因见 §7.3 最终结论。

### 7.2 0GM-35B RM 为何不能直接用 PIECEWISE

详见 [`0gm-35b-b2-inproc-speedup-20260609.md §3`](0gm-35b-b2-inproc-speedup-20260609.md)。

根本原因：prefix caching（APC）在推理时生成 warmup 未见过的新 `batch_descriptor`，`CUDAGraphWrapper` 检测到新 key 后调用 `validate_cudagraph_capturing_enabled()` → `RuntimeError`。**根因是 prefix caching，不是 PIECEWISE 本身。**

### 7.3 修复历程（2026-06-10）

**第一次尝试（失败）**：根因定位为 `enable_prefix_caching=True` 硬编码，新增 `SIA_RM_PREFIX_CACHING` env var，35B 设 `0` 禁用 APC。实测仍失败：

```
[SIA] step=1663 RM error: CUDA graph capturing detected at an inappropriate time.
```

错误从 step 1663 开始（`total_num_scheduled_tokens=116`，padded 到 120，在预捕获范围内）。APC 并不是真正根因。

---

**真正根因（通过 vllm 0.18.0 源码分析）**：

`gpu_model_runner.py:5669`：

```python
set_cudagraph_capturing_enabled(False)
CUDAGraphWrapper.clear_all_graphs()   # ← 关键！
```

主 LLM 初始化的 **profiling 阶段结束时**，代码通过类级 `WeakSet` `CUDAGraphWrapper._all_instances` 清空进程内 **所有** `CUDAGraphWrapper` 实例的 `concrete_cudagraph_entries`——包括已经完成 capture 的 RM。

执行序列：
1. RM init → RM wrappers 注册到 `_all_instances` → 捕获 51 个 size 的图 → `concrete_cudagraph_entries` 填满
2. 主 LLM init 的 profiling 阶段 → 调用 `clear_all_graphs()` → **RM 的 concrete_cudagraph_entries 被清空**
3. 主 LLM 正式 capture → 主 LLM wrappers 图填满，RM wrappers 为空
4. 推理时 RM 接到 batch（如 116→120 tokens）→ dispatcher 找到 key `(120, None, False, False, 0)` in cudagraph_keys → CUDAGraphWrapper 查 concrete_cudagraph_entries 为空 → 创建新 entry（cudagraph=None）→ 调用 `validate_cudagraph_capturing_enabled()` → flag=False → RuntimeError

vllm 0.17.1 不存在此问题是因为它不调用 `clear_all_graphs()`（AOT 路径不同）。

---

**第二次修复（已确认，2026-06-10）**：

RM init 完成后，立即把 `_all_instances` 替换为新的空 WeakSet。主 LLM 的 wrappers 注册到新 WeakSet；`clear_all_graphs()` 只清新 WeakSet（不含 RM）。RM 的 `concrete_cudagraph_entries` 在推理时完好。

```
[RMClient] Isolated 37 RM CUDAGraphWrapper instances from global registry
```

"CUDA graph capturing detected at an inappropriate time" 错误**已消除** ✅。

---

**第三层问题（2026-06-10 发现并修复）**：

第二次修复后，出现新错误：

```
[SIA] step=1663 RM error: reward channel (inproc buffer) returned 11 values, expected 10.
```

**根因**：当序列长度 > 1638 tokens（10 × L > 16384 = vllm 默认 `max_num_batched_tokens`），vllm scheduler 把 10 条序列的 prefill 分成 2 个 step 处理（第一步 9 条完整 + 1 条 partial，第二步处理剩余）。vllm 代码（`gpu_model_runner.py:1839`）对每步 batch 内的**所有**序列（包括 partial prefill）调用 `compute_logits`：

```python
# NOTE: partial requests also included, "we do so for simplicity, will ignore"
logits_indices = query_start_loc[1:] - 1
```

→ step1 写 10 个 reward（9 correct + 1 wrong），step2 写 1 个 reward（correct）
→ 总计 11 个，expected 10 → RuntimeError。

**第三次修复**（`src/sia_rm/client.py`）：

```python
elif not rm_apc and not multiprocessing:
    safe_mbt = max_model_len * 32  # covers topk ≤ 32
    llm_kwargs["max_num_batched_tokens"] = safe_mbt
```

把 RM 的 `max_num_batched_tokens` 设为 `max_model_len × 32 = 65536`，确保 topk=10 条序列始终在同一步内完成 prefill，不触发 chunking。

验证（server.log）：

```
[RMClient] APC disabled: max_num_batched_tokens=65536 (prevents chunked-prefill reward count mismatch)
Chunked prefill is enabled with max_num_batched_tokens=65536.
```

---

**三次修复联合的启动命令**：
```bash
SIA_RM_CUDAGRAPH=piecewise SIA_RM_PREFIX_CACHING=0 \
SIA_RM_MULTIPROCESS=0 \
python src/sia_vllm_server.py \
  --llm /path/to/0GM-1.0-35B-A3B \
  --rm_backend b2 \
  --rm_model /path/to/VM-Qwen3-4B-merged-for-vllm \
  --rm_b2_gpu_mem 0.15 --llm_gpu_mem 0.55 \
  --topk 10 --weight 1.0 --entropy_threshold 1.0 \
  --max_model_len 2048 --port 8000
```

exp `alpaca-0gm35b-piecewise-fix3-20260610` 已完成运行，三个修复全部激活，结果如下：

**实测结果（❌ 失败）**：RM b2_score_call 在 piecewise 模式下 **更慢**，不是更快：

| 步骤段 | piecewise p50 | none(eager) p50 | 差异 |
|--------|:---:|:---:|:---:|
| @100 | 46ms | 33ms | +39% 更慢 |
| @500 | 139ms | 40ms | +248% 更慢 |
| @800 | 130ms | 49ms | +165% 更慢 |

**piecewise 对 RM 无效的根本原因**：

1. **RM 是 prefill-heavy workload**：每步处理 `topk=10` 个完整序列（无 decode 阶段），序列长度随步骤线性增长（50 → 2048 tokens）
2. **vllm PIECEWISE 的 CUDA graph 只优化 decode 步骤**（固定 token batch 大小），对 prefill 无效
3. **vllm 0.18.0 的 torch.compile 范围为 1–65536**（`compile_ranges_endpoints=[65536]`），远大于 0.17.1；编译 overhead 抵消了任何潜在收益
4. **VL-30B 上 piecewise 11ms 的原因**：vllm 0.17.1 走 AOT capture 路径，RM batch 特征（固定 topk batch 大小）在捕获图内，命中率高

**结论**：`SIA_RM_CUDAGRAPH=none` 仍是 0GM-35B b2 inproc 的最佳设置，无需 piecewise。

### 7.4 综合优化路线图（更新）

| 优化项 | 当前值 | 目标值 | 预期 tok/s | 提升 | 状态 |
|-------|:---:|:---:|:---:|:---:|:---:|
| 当前基线（FULL fix）| — | — | **54.1** | — | ✅ 已完成 |
| **① RM CUDA graph**（`SIA_RM_CUDAGRAPH=piecewise`）| 55ms | ~11ms | **~85** | **+57%** | ❌ 失败：piecewise 比 none 慢 2-3×，RM 为 prefill-heavy，vllm 0.18.0 不适用 |
| **② stable prefix 跨分词器**（cross-tokenizer APC 修复）| 55ms | ~30ms | **~67** | **+24%** | ✅ 已实现并验证 |
| ③ 消除 apply_cpu_sync | 3.85ms | ~0ms | ~55 | +2% | 低优先级，目前已由 stable prefix 绕过 |
| ④ 训练与 0GM-35B 相同词表的 VM | 30ms | ~11ms | **~80–85** | **+20%↑** | 长期，根本消除跨分词器开销 |

---

## 八. Stable Prefix 跨分词器优化（2026-06-10）

### 8.1 背景：为何 0GM-35B 的 b2_score_call 比 VL-30B 慢且递增

VL-30B 和 0GM-35B 用的是同一个 RM（VM-Qwen3-4B），但 b2_score_call 差距显著：

| 模型 | 分词器 | b2_score_call p50 | 随长度变化 |
|------|-------|:---:|:---:|
| VL-30B | 与 RM **相同**（Qwen3 151K）| 11ms | **稳定** |
| 0GM-35B（旧代码）| **跨分词器**（Qwen3.5 248K → Qwen3 151K）| 34ms @100步 → 43ms @300步 | **递增 ↑** |

旧代码的跨分词器处理方式（`score_candidates` 中）：

```python
# 旧代码：每个 candidate 各自 encode 完整 prefix + candidate
for c in candidate_token_ids:
    cand_text = llm_tok.decode([c])
    full_text = prefix_text + cand_text           # 不同 cand → 不同末尾 BPE
    rm_ids = rm_tok.encode(full_text)             # ← 10 个 prompt 有不同前缀！
    prompts.append(TP(prompt_token_ids=rm_ids))
```

问题：`prefix_text + cand_text` 的 BPE merge 在边界处因 `cand_text` 不同而产生不同的末尾 token，导致 10 个 candidate prompt 的前缀 token ID 序列各不相同（最后 1–2 token 不同）。vllm APC 按 hash block 缓存，一旦 prefix 不完全相同就 miss，等效于 APC 失效，随序列增长需重算越来越多的 token。

### 8.2 Stable Prefix 方案

将 prefix 部分的编码与 candidate 部分**分离**：

```python
# 新代码：stable prefix 编码一次，所有 candidate 共享
stable_rm_prefix_ids = rm_tok.encode(prefix_text, add_special_tokens=False)  # 编码一次

for c in candidate_token_ids:
    cand_text = llm_tok.decode([c], skip_special_tokens=False)
    cand_rm_ids = rm_tok.encode(cand_text, add_special_tokens=False)          # 仅候选 token
    prompts.append(TP(prompt_token_ids=stable_rm_prefix_ids + cand_rm_ids))
```

效果：
- **同一步内 10 个 candidate**：共享完全相同的 `stable_rm_prefix_ids` → APC 命中率接近 100%，仅需 forward 1–3 个 suffix token
- **跨步（step N → N+1）**：prefix 增长 1 个 LLM token，`rm_tok.encode(prefix_text_{N+1})` 与 `rm_tok.encode(prefix_text_N)` 共享所有完整 block → 仅最后 1 个 block（16 token）需 partial 重算

### 8.3 实验结果（exp: `alpaca-0gm35b-stable-prefix-20260610`）

配置：`SIA_RM_CUDAGRAPH=none SIA_RM_MULTIPROCESS=0`（APC=ON 默认）

| 指标 | 旧代码（FULL fix）| stable prefix | 改进 |
|------|:---:|:---:|:---:|
| b2_score_call p50 @100步 | 34ms | 30ms | -12% |
| b2_score_call p50 @300步 | 43ms | 30ms | -30% |
| b2_score_call p50 @1000–4000步 | ~50ms（继续增长）| **~30ms（稳定）** | **稳定 ✅** |
| **SIA tok/s** | **54.1** | **66–68** | **+22–26%** |
| **SIA / noSIA** | **~48%** | **~59%** | **+11pp** |

pf-summary（稳态，@2000–4400 步）：

```
b2_score_call: p50=29.7ms  p95=45ms  max=78ms
apply_cpu_sync: p50=3.85ms（与旧代码相同，不受影响）
```

### 8.4 与 VL-30B 的剩余差距分析

| 来源 | 说明 | 估算 |
|------|------|:---:|
| 跨分词器 CPU 编码 | `rm_tok.encode(prefix_text)` + `rm_tok.encode(cand_text) × 10` 每步执行一次 | ~2ms |
| eager 模式 kernel dispatch overhead | VL-30B 用 piecewise CUDA graph；0GM-35B 用 eager | ~5ms |
| 跨步 BPE 边界偶发 cache miss | ~30% 几率末尾 BPE merge 改变最后 block hash | ~2ms（概率）|
| 合计 | 30ms - 11ms | ~19ms |

**进一步降低到 ~11ms 的路径**：训练使用 Qwen3.5 248K 词表的 VM（长期投入，消除所有跨分词器开销）。

---

## 九. 总结

### 9.1 全优化路径回顾

| 阶段 | 操作 | SIA tok/s | SIA/noSIA | b2_score_call |
|------|------|:---:|:---:|:---:|
| 0. 原始 PIECEWISE LLM | `SIA_LLM_CUDAGRAPH=piecewise` | 37.8 | ~33% | 53ms |
| 1. FULL fix（去掉 LLM piecewise）| `SIA_LLM_CUDAGRAPH` 不设置 | 54.1 | ~48% | 55ms |
| 2. RM piecewise 尝试 | `SIA_RM_CUDAGRAPH=piecewise` | ~40 | ~35% | 115–140ms ❌ |
| **3. stable prefix（当前最优）** | 跨分词器 APC 修复 | **66–68** | **~59%** | **~30ms ✅** |

对比 VL-30B：74–78 tok/s（64% of noSIA），0GM-35B 稳定在 59%，主要差距来自跨分词器 overhead（2ms CPU + 5ms eager dispatch）。

### 9.2 已修复 Bug 汇总

| 问题 | 根因 | 解法 |
|------|------|------|
| skip_step 涨 8.6× | FULL graph GPU 积压，cpu_sync 等待 LLM drain | 净 0.34ms/token，可接受 |
| "CUDA graph capturing" RuntimeError | 主 LLM profiling 阶段 `clear_all_graphs()` 通过 WeakSet 清空 RM 捕获图 | RM init 后重置 `CUDAGraphWrapper._all_instances = WeakSet()` |
| "reward 11≠10" 错误 | topk×length > max_num_batched_tokens → chunked prefill → partial 序列也写 reward | `max_num_batched_tokens = max_model_len × 32 = 65536` |
| b2_score_call 随长度递增 | 跨分词器：每 candidate 各自 encode 全文，BPE 边界不同 → APC miss | stable prefix：prefix 编码一次，candidates 只 encode suffix |

### 9.3 推荐启动命令（当前最优）

```bash
SIA_RM_CUDAGRAPH=none SIA_RM_MULTIPROCESS=0 \
python src/sia_vllm_server.py \
  --llm /path/to/0GM-1.0-35B-A3B \
  --rm_backend b2 \
  --rm_model /path/to/VM-Qwen3-4B-merged-for-vllm \
  --rm_b2_gpu_mem 0.15 --llm_gpu_mem 0.55 \
  --topk 10 --weight 1.0 --entropy_threshold 1.0 \
  --max_model_len 2048 --port 8000
```
