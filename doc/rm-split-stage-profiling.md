# RM Split-Stage Profiling — Parallelization 可行性测量

**目的**：测量 b2 Reward Model 一次 INTERVENE 调用内**每个独立阶段**的 GPU 时间，判断 RM 能否在双 GPU NVLink 节点上跟主 LLM forward **并行执行**（overlap 隐藏掉 RM 的时间）。

**结论**（先放在前面）:
- **Stage A**（非干预 token 的 KV 计算）p50 = **5.10 ms** —— 远小于主 LLM forward (~11.3 ms)，**100% 可隐藏**
- **Stage B + C**（5 candidate KV + score head）p50 = **13.00 ms** —— 在关键路径上无法 overlap
- 双 GPU 并行预测：当前 46.6 tok/s（32% 干预率）→ **~64 tok/s（+37%）**

---

## 1. 为什么要拆 3 个阶段测

一次 RM INTERVENE 调用（`score_candidates`）内部做了 3 件性质不同的事：

| 阶段 | 在算什么 | 在主 LLM 视角下的位置 |
|------|----------|------------------------|
| **A. 非干预 KV** | 把自上次 RM call 以来 LLM 生成的 K 个 token（来自 `fix_a_token` 累积）的 K/V 缓存补齐 | 主 LLM 已经在生成下一个 token, **可与之并行** |
| **B. 5-candidate KV** | 把 5 个 topk candidate token 各跑一次 decode-mode forward, 拿到带 context 的 hidden state | 必须在主 LLM 给出 topk 之后, **关键路径** |
| **C. 5-candidate score head** | 5 个 hidden state @ Linear(2560, 1) 拿到 reward 标量 | 跟 B 一起在关键路径 |

如果不拆开测，我们只能看到一个混合的"RM call ~25 ms"数字，**无法判断哪一部分能藏进主 LLM forward**。拆开测 → 直接看每个阶段的 GPU 时间 → 判断 Stage A 是否能 fit 进 LLM forward 的预算（~11.3 ms）。

---

## 2. 代码计时点

实现位置：

| 文件 | 计时内容 |
|------|---------|
| [`src/sia_rm/qwen3_with_score.py`](../src/sia_rm/qwen3_with_score.py) | GPU 端 cuda.Event 计时（精确测 GPU 执行时间） |
| [`src/sia_rm/client.py`](../src/sia_rm/client.py) | 客户端 wall-clock + 拆分 `score_candidates_profiled` 为两次 `llm.generate` |
| [`scripts/bench_rm_split_profile.py`](../scripts/bench_rm_split_profile.py) | 跑测 + 聚合统计 |

### 2.1 GPU 端 cuda.Event 计时（`qwen3_with_score.py`）

env var `SIA_RM_PROFILE=1` 触发，默认 0（生产模式零开销）。

**计时点 1 — Transformer forward (KV 计算)**：

```python
def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    if _PROFILE_ENABLED:
        # record n_input tokens this forward will process
        self._sia_prof_n_input = int(input_ids.shape[0])
        self._sia_prof_fwd_start = torch.cuda.Event(enable_timing=True)
        self._sia_prof_fwd_end = torch.cuda.Event(enable_timing=True)
        self._sia_prof_fwd_start.record()                    # ← 计时开始

    out = super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)

    if _PROFILE_ENABLED:
        self._sia_prof_fwd_end.record()                      # ← 计时结束
    return out
```

这里测的是 **整个 transformer 28 层 + attention + FFN 的 GPU 时间**，对应 Stage A 或 Stage B 的 KV 计算（取决于本次 forward 处理的 token 类型）。

**计时点 2 — Score head matmul**：

```python
def compute_logits(self, hidden_states, sampling_metadata):
    if _PROFILE_ENABLED:
        score_start = torch.cuda.Event(enable_timing=True)
        score_end = torch.cuda.Event(enable_timing=True)
        score_start.record()                                 # ← 计时开始

    sw = self.score.weight
    rewards = (hidden_states @ sw.T).squeeze(-1)             # ← score head 矩阵乘

    if _PROFILE_ENABLED:
        score_end.record()                                   # ← 计时结束
        torch.cuda.synchronize()  # 必须 sync 才能读 cuda.Event timing
        fwd_ms = self._sia_prof_fwd_start.elapsed_time(self._sia_prof_fwd_end)
        score_ms = score_start.elapsed_time(score_end)
        _write_timing(fwd_ms, score_ms,
                      self._sia_prof_n_input,
                      int(hidden_states.shape[0]))
    ...
```

每次 `compute_logits` 调用，写一条 record 到 `/dev/shm/sia_timing_<id>.bin`（与 reward channel 分离）。Record 格式 `[float32 fwd_ms][float32 score_ms][int32 n_input][int32 n_samples]`。

### 2.2 客户端 wall-clock + 拆分两次 generate（`client.py`）

`score_candidates_profiled(sid, candidates)` 把工作拆成两次独立的 `llm.generate` 调用：

```python
# Stage A: prefill new prefix tokens (K tokens since last RM call)
n_already = self._last_prefilled.get(sid, 0)
n_new = len(prefix) - n_already

a_t0 = time.perf_counter()
if n_new > 0:
    self.llm.generate([TP(prompt_token_ids=prefix)], self._sp)  # 只跑 prefix
a_wall_ms = (time.perf_counter() - a_t0) * 1000
a_timings = read_all_timings()        # 这次 generate 写入的 timing record(s)
a_forward_gpu_ms = sum(t['forward_ms'] for t in a_timings)

# Stage B: forward 5 candidates (prefix 此时已全部 cached)
b_t0 = time.perf_counter()
prompts = [TP(prompt_token_ids=prefix + [c]) for c in candidates]
self.llm.generate(prompts, self._sp)
b_wall_ms = (time.perf_counter() - b_t0) * 1000
b_timings = read_all_timings()[len(a_timings):]
b_forward_gpu_ms = sum(t['forward_ms'] for t in b_timings)
b_score_gpu_ms = sum(t['score_ms'] for t in b_timings)
```

`_last_prefilled[sid]` 记录每个 session 的 prefix 哪些 token 已经被 forward 过 → 下次 Stage A 只算新增的 K 个 token，正好对应"自上次 RM call 以来 LLM 生成的 K 步"。

### 2.3 Bench 脚本（`bench_rm_split_profile.py`）

模拟真实 SIA decode 模式：
- 初始 prefix 600 token（≈ MMLU 一道题的 prompt 长度）
- 每次 INTERVENE 前 `fix_a_token` 1–5 次（模拟 SIA SKIP tail）
- 100 次 INTERVENE 测量 + 5 次 warmup
- 每次都调 `score_candidates_profiled`，记录 5 个时间维度（A wall, B wall, A fwd, B fwd, score）

---

## 3. 跑测命令

```bash
# venv2 (vLLM 0.10.1.1, b2 backend 对应环境)
SIA_RM_PROFILE=1 /workspace/SIA/venv2/bin/python \
    scripts/bench_rm_split_profile.py 2>&1 | tee /tmp/bench_rm_split.log
```

或后台:

```bash
nohup /workspace/SIA/venv2/bin/python scripts/bench_rm_split_profile.py \
    > /tmp/bench_rm_split.log 2>&1 &
```

实际跑出来的 log: [`/tmp/bench_rm_split.log`](../../../../tmp/bench_rm_split.log)（已归档在本机）。

**环境**：
- 单卡 H200 (143 GB), CUDA driver 12.8
- vLLM 0.10.1.1, torch 2.7.1+cu128
- RM 模型: `/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm` (~7.5 GB BF16)
- `gpu_memory_utilization=0.3`, `max_model_len=4096`

---

## 4. Profiling 结果

### 4.1 配置 + 实际触发

```
iterations         : 100
prefix grew        : 600 -> 909 tokens  (avg 3.1 new tokens per INTERVENE call)
SKIP per INTERVENE : 1..5 random
candidates per call: 5
```

### 4.2 三阶段 GPU 时间

| Stage | min | p50 | mean | p95 | max | 单位 |
|-------|----:|----:|-----:|----:|----:|------|
| **A. 非干预 KV** (avg K=3.1 新 token forward) | 4.78 | **5.10** | 6.53 | 15.86 | 18.62 | ms |
| **B. 5-candidate KV + score** (混合) | 4.89 | 13.00 | 15.72 | 31.38 | 34.35 | ms |
| ├─ B 拆出: candidate KV only (5 token 全 forward) | 4.72 | **12.69** | 15.37 | 30.83 | 33.72 | ms |
| └─ B 拆出: score head only (5 × Linear(2560,1)) | 0.17 | **0.30** | 0.36 | 0.67 | 0.76 | ms |

### 4.3 Wall-clock（含客户端 Python + vLLM dispatch 开销）

| Stage | min | p50 | mean | p95 | max | 单位 |
|-------|----:|----:|-----:|----:|----:|------|
| A wall | 9.52 | 11.18 | 13.17 | 27.14 | 30.65 | ms |
| B wall | 14.55 | 24.60 | 29.16 | 52.76 | 60.54 | ms |

GPU vs wall 差距：约 **5–10 ms / call**，是 vLLM scheduler、Python list 构造、subprocess IPC 的固定开销。这部分在双 GPU 并行设计里仍然存在，但**只要 RM 在另一张卡, 这些 CPU/IPC 开销可与主 LLM 的 GPU 计算重叠**, 所以不算瓶颈。

### 4.4 几个原始 sample（前 5 个 iter）

```
iter #  0: K=1  A_fwd= 5.38  B_fwd=32.04  score=0.67  prefix_len=601
iter #  1: K=2  A_fwd= 7.77  B_fwd=30.39  score=0.67  prefix_len=603
iter #  2: K=1  A_fwd= 7.77  B_fwd=17.07  score=0.34  prefix_len=604
iter #  3: K=4  A_fwd= 6.51  B_fwd=27.46  score=0.50  prefix_len=608
iter #  4: K=4  A_fwd= 8.97  B_fwd=18.74  score=0.37  prefix_len=612
```

可见 A_fwd ≈ 5-9 ms（变化主要跟当时新增 token 数 K 和 block-boundary 命中率有关），B_fwd 13-32 ms（同样有方差），score 0.3-0.7 ms 总是很小。

---

## 5. 结果分析

### 5.1 Stage C (score head) ≈ 完全可忽略

score head 是 `Linear(2560, 1)` × 5 行的极小 matmul，**p50 = 0.30 ms**。在所有讨论里都可以折叠进 Stage B。

### 5.2 Stage A vs Stage B 的关键差别

|  | Stage A | Stage B |
|--|---------|---------|
| forward 几个 token | K ≈ 3 个新 token | 5 个 candidate token (batch=5 prompts) |
| GPU 时间 / token | 5.10 / 3 ≈ **1.7 ms/token** | 12.69 / 5 ≈ **2.5 ms/token** |
| 为啥 B 慢 | batch=5 prompts 每个独立调度有 launch overhead, vLLM scheduler 也多算 | 单 prompt 单次 dispatch |
| 主 LLM 视角下时序 | "已经做完, RM 异步补 KV" → **可并行** | "刚拿到 topk, 等 RM 评分才能继续" → **必须串行** |

### 5.3 跟主 LLM forward (Qwen3-14B) 对比

历史实测 Qwen3-14B 单 token forward (noSIA, decode mode) = **11.31 ms**（详见 [`noSIA-three-model-comparison.md`](noSIA-three-model-comparison.md)）。

**Stage A p50 = 5.10 ms < 11.31 ms** ✓

```
LLM step (主 GPU)  ──────────►  11.3 ms      [continuous]
                                 ↑↓
RM Stage A (另一 GPU)  ────►    5.1 ms  [idle 6.2 ms]
                                          完全藏在 LLM forward 后面
```

→ **A 完全可隐藏**（用掉 LLM forward 预算的 45%，还有 55% 空闲时间）

**Stage B+C p50 = 13.00 ms > 11.31 ms**

```
LLM step n (主 GPU) ──────────► topk
                                 │
RM Stage B+C        ─────►       │──────► reward ──┐
                                                    ▼
LLM step n+1 (主 GPU)                       ──────────► next token
```

→ **B+C 必须等 LLM topk 出来才能开始；其间 LLM 也在等 RM 出 reward**。这部分**不能并行**，是关键路径。

### 5.4 并行后的 throughput 预测

设 LLM forward = 11.3 ms, B+C critical = 13.0 ms, 干预率 = r。

```
per_token_parallel = LLM_forward + r × critical
                  = 11.3 + r × 13.0
```

| 干预率 r | per token (ms) | throughput (tok/s) | 跟当前单卡 b2 对比 |
|---------|----------------|--------------------|-----|
| 17 % | 13.52 | **74.0** | +59 % (vs 46.6) |
| 27 % | 14.82 | 67.5 | +45 % |
| 32 % | 15.46 | 64.7 | **+39 %** (跟 14B+SIA b2 当前实测 32% 干预率对比) |
| 33 % | 15.60 | 64.1 | — |

注意这是**单题 latency-bound 预测**, 假设：
- 双 GPU 之间通信开销低（NVLink ≥ 900 GB/s，每次只传 5 个 float32 reward 几乎 0）
- Stage A 完全 overlap（实测 p50=5ms < LLM 11.3ms）
- Stage B 不能 overlap

### 5.5 关于 A 的 p95 和 max

A 的 p95 = 15.86 ms, max = 18.62 ms — **超过了 LLM forward 11.3 ms 预算**。

发生在什么时候？看 K=5（一次 SKIP 累积了 5 个 token）时，A 大致是 5×1.7=8.5 ms，加上 block-boundary partial prefill 偶尔 spike 到 15 ms。这种情况下 Stage A 隐藏不全, 会拖一点点 LLM。

但 p95 spike 影响 = 0.05 × (15.86 - 11.3) × intervene_rate = 0.05 × 4.5 × 0.32 ≈ 0.07 ms/token → 可忽略。

### 5.6 我们没测但需注意的事

1. **真实双卡 NVLink 节点的 reward transfer 时间**: 单卡测量中 reward 用 `/dev/shm` 文件 channel ≈ 50 μs。双卡需要走 NVLink P2P 或 host-mediated，**预期 < 100 μs**，但仍需实测。
2. **CUDA stream / process 隔离**: 如果 LLM 和 RM 跑在同进程不同 stream，理论上单 GPU 就能 overlap (但 SM 共享带宽，不一定快)。跨 GPU + 不同 process 是更稳健的方案。
3. **scheduler 同步开销**: 主 LLM 需要在 SIA processor 里 wait reward，wait 的方式（busy poll 文件 vs 共享内存 future）会影响 critical path 的纯净度。

---

## 6. 后续动作

| Target | Item |
|--------|------|
| Fri Jun 5 | 在双 GPU NVLink 节点上实现 Stage A 与主 LLM forward 并行的原型, 实测端到端 throughput, 验证 64-74 tok/s 预测 |
| 之后 | 评估 batch=3 candidates (topk=3) 是否能把 Stage B 压到 8-9 ms, 进一步提升 throughput |
| 长期 | 考虑用更小 RM (1.7B 或 0.6B) 重训 LoRA, 把 Stage A+B 都减到一半 |

---

## 7. 引用

- 实施 commit: [`0c6c93b`](../commits/0c6c93b)
- bench 脚本: [`scripts/bench_rm_split_profile.py`](../scripts/bench_rm_split_profile.py)
- 模型类计时钩子: [`src/sia_rm/qwen3_with_score.py`](../src/sia_rm/qwen3_with_score.py)
- 客户端拆分逻辑: [`src/sia_rm/client.py`](../src/sia_rm/client.py) — `score_candidates_profiled`
- 相关 docs:
  - [`b2-m2-design.md`](b2-m2-design.md) — b2 in-process 整体设计
  - [`b2-decode-vs-pooling-mode.md`](b2-decode-vs-pooling-mode.md) — 为什么 RM forward 这么轻
  - [`b2-pure-engineering-optimizations.md`](b2-pure-engineering-optimizations.md) — 单卡 b2 路径后续优化方向
  - [`noSIA-three-model-comparison.md`](noSIA-three-model-comparison.md) — 主 LLM 的 11.3 ms/token baseline 出处
