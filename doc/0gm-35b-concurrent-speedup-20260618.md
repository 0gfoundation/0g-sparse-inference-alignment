# 0GM-35B 高并发吞吐优化记录 (2026-06-18)

**模型**：0GM-1.0-35B-A3B-0427  
**Value Model**：VM-Qwen3-4B-merged-for-vllm（b2 inproc）  
**venv / vllm**：0gm35b-b2 / 0.18.0  
**Branch**：`speedup-concurrent`

---

## 一、优化前基线

数据来源：`exp/bench-0gm35b-bench-v2-20260617.md`（优化前，见 [`doc/sia-high-concurrency-throughput-analysis-20260618.md`](sia-high-concurrency-throughput-analysis-20260618.md)）。

Concurrency Sweep（input≈480 tokens, max_out=304）：

| Conc | SIA ITL | noSIA ITL | ITL 倍数 | SIA tok/s | noSIA tok/s |
|------|---------|-----------|----------|-----------|-------------|
| 1    | 10.9ms  | 9.0ms     | 1.2×     | 88.5      | 105.4       |
| 2    | 19.2ms  | 9.6ms     | 2.0×     | 100.7     | 191.5       |
| 4    | 24.8ms  | 10.5ms    | 2.4×     | 156.6     | 352.7       |
| 8    | 35.8ms  | 11.8ms    | 3.0×     | 217.3     | 617.4       |
| **16**   | **65.4ms**  | **13.7ms**    | **4.8×**     | **239.4**     | **1040.3**      |

**核心问题**：conc=16 时，SIA ITL 是 noSIA 的 4.8×，VM 串行调用（N 次顺序调用 `score_candidates`）是主要瓶颈。

---

## 二、本次优化项

### 优化一：VM 批量打分（Batch VM Scoring）

**Commit**：`6b6f239`  
**改动文件**：`src/sia_vllm_RM.py`、`src/sia_rm/client.py`

**背景**

原始 `apply()` 实现中，对当前 decode step 里所有 INTERVENE 的请求，逐一串行调用 `RMClient.score_candidates()`，每次独立走一次 `vm_llm.generate()`：

```
apply(N=16):
  for i in range(N):
      if INTERVENE:
          VM.score_candidates(sid_i, candidates_i)  # 每个请求各自一次 GPU forward
```

**改动**

新增 `RMClient.score_candidates_batch()` 方法，把 N 个 INTERVENE 请求的 N×K 个 prompt 合并成一次 `vm_llm.generate()` 提交：

```
apply(N=16):
  # Phase 1：纯 Python，为所有 INTERVENE 请求推进 session 前缀（fix_a_token）
  batch = [(sid_i, candidates_i) for i in INTERVENE_requests]

  # Phase 2：1 次 GPU forward 替代原来的 N 次
  all_scores = VM.score_candidates_batch(batch)  # N×K prompts 合并提交
```

`score_candidates_batch` 内部保证 FIFO 顺序提交（vLLM InprocClient 单步同步添加所有请求），取回 flat tensor 后按 `(start, end)` offset 切片还原各请求的 K 个分数。

**效果**

| 指标 | 优化前 | 优化后 |
|------|:---:|:---:|
| SIA ITL conc=16 | 65.4ms | **43.8ms（−33%）** |
| SIA ITL conc=8 | 35.8ms | **30.0ms（−16%）** |
| SIA tok/s conc=16 | 239.4 | **~350**（估算）|

---

### 优化二：增大 VM KV Cache（rm_b2_gpu_mem 0.13 → 0.20）

**Commit**：`abd29c0`  
**改动文件**：`docker-compose.0gm35b.yml`

**背景**

VM 的 `rm_b2_gpu_mem=0.13` 对应约 1200 个 APC block（max_model_len=32768 时）。高并发（conc=16）且序列较长时，16 个会话的 KV 总量可能超出 VM 可用 block 数，触发 block eviction，导致下次访问该会话前缀时需要完整重新 prefill（等效于 APC 全 miss）。

**改动**

将 `--rm_b2_gpu_mem` 从 `0.13` 提升至 `0.20`，可用 block 数从 ~1200 提升至 ~1850，总 GPU 利用率从 88% 提升至 95%（`llm_gpu_mem=0.75` 不变）。

**效果**

对中短序列（512–4096 tokens）场景下的并发 KV eviction 有防护作用。独立基准测试未测量；与批量打分优化合并部署。

---

### 优化三：增量跨分词器前缀缓存（P-3 Incremental Prefix Cache）

**Commits**：`93b732e`（实现）、`e87ace1`（修复 delta>1 bug）  
**改动文件**：`src/sia_rm/client.py`

#### 3.1 背景：P-2（stable prefix）的遗留问题

[P-2 优化](0gm-35b-sia-perf-breakdown-20260609.md)（`src/sia_rm/client.py` 的 `score_candidates`）已将 stable prefix 的编码与 candidate 分离，消除了**同步内**（同一 INTERVENE step 内 10 个 candidate 之间）的 APC miss。

但 P-2 在**跨步**（step N → N+1 的 INTERVENE 到下一次 INTERVENE）仍存在 APC miss：

```python
# P-2（每次 INTERVENE 步均全量 re-encode）
prefix_text = llm_tok.decode(prefix, ...)
stable_rm_prefix_ids = rm_tok.encode(prefix_text, ...)
```

re-encode 时，BPE 在 prefix 末尾与新 token 的拼接处可能发生 merge，导致最后一个完整 APC block（16 tokens）的内容变化，hash 不匹配 → **cross-step block miss → 每 prompt 多 forward ~16 extra tokens**。

实测：35B b2_score_call p50 ≈ 30ms（P-2 稳态），VL-30B 约 11ms。

#### 3.2 P-3 实现原理

在 `RMClient` 中新增 `_rm_prefix_cache: dict[int, tuple[int, list[int]]]`，记录每个 session 上次 INTERVENE 时的 `(prefix_len_llm, stable_rm_prefix_ids)`。

每次 INTERVENE 时：
- **增量路径（delta > 0）**：只对自上次 INTERVENE 以来新增的 delta 个 LLM token，逐 token 各自 `decode([tid])` + `encode(text)` 后 append 到缓存前缀 ID 列表，不改动已有 ID
- **首次调用（cached=None）或负 delta（异常）**：全量 re-encode，重建缓存

```python
def _get_stable_rm_prefix(self, sid, prefix):
    cached = self._rm_prefix_cache.get(sid)
    cur_len = len(prefix)
    if cached is not None:
        cached_len, cached_ids = cached
        delta = cur_len - cached_len
        if delta == 0:
            return cached_ids
        if delta > 0:
            stable = cached_ids  # Python list，不 mutate
            for i in range(delta):
                new_text = llm_tok.decode([prefix[cached_len + i]], ...)
                new_rm_ids = rm_tok.encode(new_text, ...)
                stable = stable + new_rm_ids  # 追加，生成新列表
            self._rm_prefix_cache[sid] = (cur_len, stable)
            return stable
    # fallback: full re-encode
    ...
```

#### 3.3 关键 Bug 修复（commit e87ace1）

**初始实现（93b732e）的 bug**：只在 `delta == 1` 时走增量路径；delta > 1 退化为全量 re-encode。

**实际调用模式分析**：`_prepare_b2_session` 在每次 INTERVENE 时，把自上次以来所有 SKIP 步生成的 output token 批量追加（`for tid in output_ids[n_already:]: fix_a_token(sid, tid)`）。以 `entropy_threshold=1.0`、INTERVENE 率 ~20% 为例：

- 两次 INTERVENE 之间平均 ~4 个 SKIP 步
- `_get_stable_rm_prefix` 被调用时 `delta ≈ 4-5`，不等于 1
- 旧代码：`if delta == 1` 几乎永不命中，P-3 优化等同于 no-op

**修复**：将 `if delta == 1:` 改为 `if delta > 0:` 的通用循环，对任意正 delta 均走增量路径。

#### 3.4 效果分析

| 场景 | 每 prompt forward tokens（估算）|
|------|------|
| P-2（全量 re-encode，cross-step APC miss ~30% 概率）| ~20 tokens 期望（miss 时 ~31，无 miss 时 ~15）|
| P-3（增量 append，cross-step APC 100% 命中）| ~15 tokens（仅 partial tail + delta new tokens + candidate）|

附带收益：CPU 编码开销从 O(prefix_len) 降至 O(delta)，长上下文（>8K tokens）时尤为显著。

---

## 三、优化后基准测试结果

**测试时间**：2026-06-18  
**部署**：docker-compose.0gm35b.yml（三项优化全部生效）

### 3.1 Concurrency Sweep

（input≈480 tokens, max_out=304）

| Conc | SIA ITL | noSIA ITL（参考）| 变化（vs 基线）|
|------|---------|-------|------|
| 1    | 12.4ms  | ~9ms  | +1.5ms |
| 2    | 17.2ms  | —     | −2.0ms |
| 4    | 21.5ms  | —     | −3.3ms |
| 8    | 30.0ms  | —     | −5.8ms |
| **16**   | **41.3ms**  | **~13.6ms** | **−24.1ms（−37%）** |

> noSIA ITL 参考值来自之前实验（13.6ms at conc=16）。

### 3.2 全程优化路径汇总

| 阶段 | SIA ITL conc=16 | 相比上阶段 |
|------|:---:|:---:|
| 基线（2026-06-17，bench-v2）| 65.4ms | — |
| + 批量打分（commit 6b6f239）| 43.8ms | **−33%** |
| + P-3 增量前缀缓存（commits 93b732e + e87ace1）| **41.3ms** | **−5.7%** |
| 总改善 | — | **−37%**（65.4 → 41.3ms）|

> `rm_b2_gpu_mem` 提升未独立测量；P-3 的效果已包含 KV cache 增大的影响。

### 3.3 Input-Length Sweep

（max_out=128，conc=1 或 2）

| Input | Conc | SIA ITL |
|-------|------|---------|
| 512   | 2    | 14.4ms  |
| 2048  | 2    | 14.2ms  |
| 4096  | 2    | 15.8ms  |
| 16384 | 1    | 18.0ms  |
| 32768 | 1    | **34.6ms** |

短至中等 context（512–4096 tokens）ITL 稳定在 14–16ms，说明 P-3 增量缓存在常见长度下工作良好。32K context 时 ITL 跳升至 34.6ms，来源：
- 32K LLM prefix → ~30K RM tokens；partial tail ≈ (30K mod 16) + delta ≈ 13 tokens/prompt
- eager 模式下长序列 GPU compute 自然增长

---

## 四、当前瓶颈分析

批量打分解决了"串行 N 次调用"的主要问题后，conc=16 的 SIA/noSIA ITL 倍数从 4.8× 降至约 3.0×（41.3/13.6）。剩余 gap 的构成：

| 来源 | 估算 | 可消除？|
|------|------|---------|
| eager 模式 kernel dispatch overhead | ~5ms/VM call | 需 vllm 0.18.0 支持 VM CUDA graph（目前受限，见 [§Appendix B](sia-speedup-summary-20260610.md)）|
| 跨分词器 CPU 编码（delta tokens） | ~0.5ms/step（P-3 后已大幅降低）| 根本消除需训练同词表 VM |
| VM batch size 大时 memory-bound 效率 | — | 增大 topk 或使用更大 batch 可改善 GPU 利用率，但 ITL 可能上升 |
| partial tail compute（~15 tokens/prompt）| ~15ms/batch | 不可消除（APC 无法缓存 partial block）|

**进一步大幅提升的路径**：
1. 训练与 0GM-35B 相同词表（Qwen3.5 248K）的 VM → 消除跨分词器开销，且可能恢复 CUDA graph（b2_score_call 有望从 ~30ms 降至 ~11–17ms 水平）
2. Dual CUDA Stream 流水线（将 VM 打分与下一步主 LLM forward 重叠）→ 约 +20–40% 端到端吞吐，但需改造 LogitsProcessor 同步语义

---

## 参考文档

- [`sia-high-concurrency-throughput-analysis-20260618.md`](sia-high-concurrency-throughput-analysis-20260618.md) — 并发问题分析与批量打分方案设计
- [`0gm-35b-sia-perf-breakdown-20260609.md`](0gm-35b-sia-perf-breakdown-20260609.md) — P-2 stable prefix 优化详情及完整历史
- [`sia-speedup-summary-20260610.md`](sia-speedup-summary-20260610.md) — 全量优化总结（含 VL-30B）
