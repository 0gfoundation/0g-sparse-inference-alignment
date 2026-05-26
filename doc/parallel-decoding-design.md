# 两张 GPU 上 SIA 并行解码三套方案设计文档

**日期**：2026-05-26
**适用场景**：LLM 和 RM 分卡部署（GPU 0 跑 LLM，GPU 1 跑 RM），希望在 Tier 1（两进程纯隔离）的基础上把 RM 工作和 LLM forward 真正在时间上重叠。
**前置背景**：当前 vLLM `/classify` 模式下 SIA 的同步阻塞模型把 LLM 锁死在 RM 调用上，每个 INTERVENE 步都付完整的 RM 调用代价。

---

## 0. 符号约定

| 符号 | 含义 | 当前实测值 |
|---|---|---|
| `L` | LLM 单步 forward 时间（Qwen3-14B, 2 GPUs） | ~11.3ms |
| `F` | RM `fix_a_token`：1 个 token 的 KV 增量 forward | ~3-5ms（设计目标，需实测） |
| `S` | RM `get_candidates_scores`：5 candidates × 1 token 批量 forward | ~8ms |
| `I` | INTERVENE 率（entropy_threshold=1.0 下实测） | ~28% |
| `P_flip` | INTERVENE 步内 top-1 token 翻转率（实测） | ~67% |
| `R_t1` | Tier 1 下 vLLM `/classify` 单次调用代价（同步阻塞） | ~30ms |

---

## 1. Tier 1 基线（已部署，用于对照）

LLM 和 RM 跑在两张独立 GPU 上，但 SIA 仍然走同步 HTTP：
- SKIP 步：LLM forward 完直接采样，不调 RM 。耗时 = `L`
- INTERVENE 步：LLM forward 完 → 阻塞调 `/classify` → 拿到 5 分数 → 加权采样。耗时 = `L + R_t1`

每 token 平均耗时：
```
per_token_tier1 = L + I × R_t1 = 11.3 + 0.28 × 30 ≈ 19.7ms → ~51 tok/s
```

**问题**：INTERVENE 步上 LLM 完全干等 `R_t1` 那段时间，没干活；RM 的 forward 包装在 vLLM 协议层有额外开销。

---

## 2. 方案 2B：有状态 RM Server（fix_a_token + get_candidates_scores）

### 2.1 架构

放弃 `vllm serve /classify` 无状态模式，自己写一个 RM 服务进程，**显式维护 per-request KV cache**，对外暴露 3 个 API：

| API | 协议 | 工作 |
|---|---|---|
| `start_new_sentence(req_id)` | sync | 清空该请求的 KV cache |
| `fix_a_token(req_id, token_id)` | **async fire-and-forget** | RM 把 token 追加到 KV cache，forward 1 步（`F`） |
| `get_candidates_scores(req_id, topk_ids)` | sync | 从当前 KV state 出发，并行 forward 5 candidates，返回 5 分数（`S`） |

通信建议用本机 IPC（Unix socket / shared memory）而不是 HTTP，省一层 JSON 序列化和 TCP 开销。

### 2.2 LLM 端工作流

```
for step N in 0..max_tokens:
    logits = LLM.forward()                        # L
    decide SKIP or INTERVENE by entropy

    if SKIP:
        token = sample(logits)
        RM.fix_a_token(token)                     # 异步，立即返回
        continue

    # INTERVENE
    scores = RM.get_candidates_scores(topk)       # 同步阻塞 S
    weighted = logits[topk] + scores * weight
    token = sample(weighted)
    RM.fix_a_token(token)                         # 异步
```

### 2.3 并行机制：SKIP 步上的隐式重叠

```
时间线（一个 SKIP 步）:
LLM (GPU 0):  ──< step N forward L=11ms >──< step N+1 forward 11ms >──
                 ↓采样出 token
                 │
                 │ async fire fix_a_token
                 ↓
RM  (GPU 1):     ──< 1-token KV 扩展 F=4ms >──── idle ─────────────────
```

- LLM 算 step N+1 和 RM 处理 step N 的 token，**真正并发跑在两张卡上**
- 80% 的 SKIP 步（如果干预率 = 20%）都是这个模式，**完全免费**
- 只要 `F ≤ L`，RM 在每个 SKIP 步上**永不落后**于 LLM 的位置

### 2.4 INTERVENE 步：RM 已经"热"

每次 INTERVENE 来临时，RM 已经被前序所有 `fix_a_token` 喂到了当前位置 N-1。所以 `get_candidates_scores` 只需要：
- 从位置 N-1 的 KV 出发
- 并行 forward 5 个候选 token（batch size 5，1 步深度）
- pooling + classifier head → 5 个分数

耗时 `S ≈ 8ms`，远小于 Tier 1 的 `/classify` 全量调用 `R_t1 ≈ 30ms`。

### 2.5 数学模型

```
per_token_2B = (1-I) × L  +  I × (L + S)
             = L + I × S
             = 11.3 + 0.28 × 8 = 13.54ms → ~74 tok/s
```

**前提条件**：`F < L`。这是关键工程约束，下面 §5 详细讨论。

### 2.6 收益

```
Tier 1:        ~51 tok/s
Tier 1 + 2B:   ~74 tok/s     (+45%)
```

### 2.7 实现成本

| 模块 | LOC 估算 |
|---|---|
| 自定义 RM server（FastAPI/aiohttp + 自管 KV） | ~600 行 |
| 异步队列、并发安全 | ~200 行 |
| LLM 端 SIA processor 改造（async client） | ~150 行 |
| 测试 + 故障恢复 + 监控 | ~150 行 |
| **总计** | **~1100 行** |

代价：失去 vLLM 自带的 prefix caching、CUDA graph、continuous batching —— 这些都得在自定义 server 里手工复刻或放弃。

---

## 3. 方案 2C：复用 scoring 时计算出的 KV cache

### 3.1 核心 insight

2B 的 INTERVENE 步里，`get_candidates_scores` 已经为 5 个候选 token 各 forward 了 1 步、各自得到了一份 position N 的 KV state（attention 算分时必然要算 KV）。**胜出 token 选定后，无需再调用 `fix_a_token` 重新 forward 这个 token**，直接把胜出分支的 KV "提交"为正式当前状态，丢弃其他 4 个分支即可。

### 3.2 API 改造

把 `get_candidates_scores` 拆成两阶段：

```
scores = RM.score_candidates(req_id, topk_ids)    # forward 5 个分支，保留 5 份 KV
# ... LLM 选出 winner ...
RM.commit_candidate(req_id, winner_index)         # 提升 winner 分支，丢弃其他 4 个
```

或者用更省往返的合并 API：

```
score_and_commit(req_id, topk_ids) → (scores, commit_token=None)   # 先返回分数
                                  → 等 LLM 回调 commit_index
                                  → 提升对应分支
```

底层数据结构：vLLM v1 的 block manager 已经有 `fork` 和 `select` 原语用于 speculative decoding，可以直接复用。

### 3.3 节省的工作量

| | 每 SKIP 步 RM 工作 | 每 INTERVENE 步 RM 工作 | 每 token 平均（I=0.2） |
|---|---|---|---|
| 2B | `F` | `F` (score) + `F` (fix winner) = **2F** | `F × (1+I)` |
| 2B + 2C | `F` | `F` (score, KV 直接挪用) = **F** | `F` |

每 INTERVENE 步省掉一次 1-token forward，即 `F`。

### 3.4 数学模型（严格版）

考虑 RM 是否成为瓶颈：

```
per_token_2B    = max(L, F × (1+I)) + I × S_extra
per_token_2B+2C = max(L, F)         + I × S_extra
```

其中 `S_extra` 是 INTERVENE 步上 LLM 必须等的 RM 同步部分（score 调用本身的延迟），不被 LLM 影子隐藏。

### 3.5 何时 2C 有意义

| F vs L | 2B 系统瓶颈 | 2C 收益 |
|---|---|---|
| **F < L**（理想） | LLM | **零吞吐提升**（RM 反正闲）。但 RM 更省电、给并发请求腾空间。 |
| **F ≈ L**（边缘） | 接近双瓶颈 | **20-40% 救命级**：把"刚好翻车"的 2B 救回稳定 LLM-bound 状态。 |
| **F > L**（差） | RM | **救一半**：把 RM 工作量从 `F × (1+I)` 砍到 `F`，吞吐从 `1/(F×(1+I))` 提到 `1/F`。 |

例：`F = 15ms, L = 11ms, I = 0.2, S_extra = 2ms`

```
2B:    max(11, 15×1.2) + 0.2×2 = 18 + 0.4 = 18.4ms → ~54 tok/s
2B+2C: max(11, 15)     + 0.2×2 = 15 + 0.4 = 15.4ms → ~65 tok/s  (+20%)
```

### 3.6 实现成本

| 模块 | LOC 估算 |
|---|---|
| RM 端 fork-5-branches + commit/discard | ~200 行 |
| API 协议 + LLM 客户端改造 | ~50 行 |
| **总计（在 2B 基础上）** | **~250 行** |

### 3.7 现实意义：扩大稳定性区间

```
2B   工作良好的 F 上限 ≈ L / (1+I)         (I=0.2 时约 9ms)
2C   工作良好的 F 上限 ≈ L                 (I=0.2 时约 11ms)
```

**`F` 在 9-11ms 这段灰色地带是 2C 救命的核心区**。工程上 `F` 实测往往比预期慢一些（IPC 抖动、launch overhead 偶发尖刺），2C 提供一层鲁棒性 buffer。

---

## 4. 方案 2D（Rewind）：LLM 推测式 forward + RM 并行打分

### 4.1 核心思路

在 INTERVENE 步上，LLM **不等 RM 结果**，先按 logits 的 top-1 token 推测性地继续往下 forward；RM 同时在另一张卡上算 top-k 分数。等 RM 返回：

- **若 winner == top-1（命中）**：LLM 的推测 forward 是有效的，继续即可
- **若 winner ≠ top-1（翻转）**：丢弃推测 forward 的结果，用真正的 winner token 重做一遍下一步

### 4.2 时间线

INTERVENE 命中（hit, 概率 `1 - P_flip`）：

```
LLM (GPU 0):  ──< step N L >──< step N+1 (speculative) L >─────►
                 ↓采样 top-1
                 │ fire async score
RM  (GPU 1):     ──< score S=8ms >──── idle ─── (winner == top-1, 提交)
                                  ↑ result back, no rewind needed
```

INTERVENE 翻转（miss, 概率 `P_flip`）：

```
LLM (GPU 0):  ──< step N L >──< step N+1 (spec) L >──< step N+1 (redo) L >─────►
                 ↓采样 top-1                            ↑ 用真正 winner 重做
                 │ fire async score
RM  (GPU 1):     ──< score S >── idle ─── (winner != top-1)
                              ↑ result back, rewind!
```

### 4.3 数学模型

对每个 INTERVENE 步，相对于 SKIP 的额外耗时：

```
2B 单独:        INTERVENE 多花 S
                额外 = I × S

2B + 2D:        命中(33%) 多花 0    （推测 step N+1 反正本来也要算）
                翻转(67%) 多花 L    （重做 step N+1）
                额外 = I × P_flip × L

per_token_2B+2D = L + I × P_flip × L
                = 11.3 + 0.28 × 0.67 × 11.3 = 13.42ms → ~74.5 tok/s
```

### 4.4 当前 flip 率下 2D 几乎没收益

```
2B(+2C):    per_token = L + I × S          = 11 + 0.28 × 8     = 13.24ms → 75.5 tok/s
2B+2D:      per_token = L + I × P_flip × L = 11 + 0.28×0.67×11 = 13.06ms → 76.6 tok/s
```

**仅 +1.5%**。原因：rewind 在 hit 时省掉 `S` (=8ms)，在 miss 时多付 `L` (=11ms)。当 `P_flip × L ≈ S` 时（67% × 11 ≈ 7.4 ≈ 8），两边几乎抵消。

### 4.5 2D 在哪种 regime 才显著

2D 净收益的临界条件：

```
2D 比 2B 强  ⟺  P_flip × L < S
              ⟺  P_flip < S / L
```

| `S / L` 阈值 | 当前 flip 67% 下 2D 是否值得 |
|---|---|
| S=8, L=11 → 阈值 73% | **勉强**（67% < 73%，+1.5%） |
| S=15, L=11 → 阈值 136% | **强烈推荐**（任何 flip 都救） |
| S=4, L=11 → 阈值 36% | **不要做**（67% > 36%，反而更慢） |

简单说：**RM 越慢、flip 越低，2D 越值得**。当前 RM 在 2B 下已经把 `S` 压到 ~8ms，2D 边际收益已经被挤干。

### 4.6 实现成本

需要在 vLLM v1 engine 里劫持 "采样 → 下一步 forward" 这条流水线，把它做成"speculative，可回滚"：

| 模块 | LOC 估算 |
|---|---|
| SIA processor 改异步、维护 pending RM future | ~150 行 |
| vLLM engine 内 hack：abort + rewind 当前 sequence 一步 | ~150 行（动 engine 源码） |
| 推测/正式 KV cache 状态切换 | ~100 行 |
| **总计** | **~400 行 + vLLM 源码侵入** |

工程量大于 2C，但收益小于 2C。**不建议优先做**。

---

## 5. 综合对比

### 5.1 吞吐预期（基于当前实测参数）

参数：`L=11.3ms, F=4ms（设计目标）, S=8ms, I=0.28, P_flip=0.67, R_t1=30ms`

| 方案 | per-token | tok/s | vs Tier 1 | 工程量 |
|---|---:|---:|---:|---|
| Tier 1（基线） | 19.7ms | ~51 | — | 零（已部署） |
| Tier 1 + 2B | 13.5ms | ~74 | +45% | ~1100 行 |
| Tier 1 + 2B + 2C | 13.5ms | ~74 | +45% | +250 行（在 F<L 时不增吞吐，但显著扩大鲁棒区间） |
| Tier 1 + 2B + 2D | 13.4ms | ~75 | +47% | +400 行 + vLLM hack |
| Tier 1 + 2B + 2C + 2D | 13.4ms | ~75 | +47% | +650 行 + vLLM hack |

### 5.2 在 RM 不够快 (F=15ms) 时的对比

| 方案 | per-token | tok/s | 注 |
|---|---:|---:|---|
| 2B 单独 | 18.4ms | ~54 | RM 瓶颈，**比 Tier 1 还慢 -6%** |
| 2B + 2C | 15.4ms | ~65 | **2C 救命，+27% vs Tier 1** |
| 2B + 2C + 2D | 15.4ms | ~65 | 2D 在 RM-bound 状态下无帮助 |

**关键观察**：F 实测偏慢时，**2C 是必须的**；2D 此时也救不了。

### 5.3 干预率的影响（2C 后）

| `I` (entropy_threshold 决定) | 2B+2C per-token | tok/s | 提升 vs Tier 1 |
|---|---:|---:|---:|
| 28%（当前） | 13.5ms | 74 | +45% |
| 20% | 12.9ms | 78 | +53% |
| 10% | 12.1ms | 83 | +63% |

**注意**：降低 `I` 是质量换速度（更多 SKIP → 弱化 SIA 干预），需要 MMLU A/B 验证 accuracy 是否可接受。

---

## 6. 落地优先级建议

1. **先做 Tier 1**：分卡部署、`--gpu-memory-utilization` 拉到 0.85。已知收益 +52%，零代码改动。**实测验证后再投入下面**。

2. **如果 Tier 1 已经够用**：止步。SIA 在 ~51 tok/s（相对 noSIA 88 tok/s 约 58%）已经能用于多数业务。

3. **如果还要再压**：实现 **2B + 2C**。两者一起做，理由：
   - 2C 单独不存在（依赖 2B 的有状态 RM）
   - 2B 单独有"F 偏慢时翻车"的风险，2C 是稳定性 buffer
   - 一起做总工程量 ~1350 行，吞吐 +45%

4. **不要做 2D**：在当前 flip 率 67% 下边际收益 < 2%，但要侵入 vLLM 源码。除非 flip 率能通过别的手段降到 30% 以下（比如换 RM 模型、调 weight 参数），否则不值得。

---

## 7. 风险与未决问题

### 7.1 `F` 实测是否真能做到 < `L`

这是 2B/2C 整套方案的命门。需要先做实测：

```bash
# 直接 PyTorch forward 测算力底
python scripts/probe_rm_forward_latency.py  # 待写
```

测：单 token forward（带 KV cache hit）的稳态 median + p95。

- 若 median < 8ms 且 p95 < 10ms：**安全做 2B**
- 若 median ~10ms：**必须配 2C 一起做**
- 若 median > 12ms：**重新考虑 RM 加速**（FP8、小模型），别做 2B

### 7.2 RM 端实现 CUDA graph 的难度

vLLM 内部的 CUDA graph 是为它自己的调度器设计的。自定义 server 想拿到同等性能，要么：

- 自己写 CUDA graph capture（hard，~300 行 RM 端代码）
- 用 `torch.compile`（中等，模型兼容性问题）
- 接受 PyTorch eager 模式的额外 launch overhead（~3-5ms/step，可能直接让 `F > L`）

**这是工程上最大的不确定性**。

### 7.3 多请求并发

文中分析都假设单请求（MMLU 串行 eval）。多请求并发场景下：

- 2B 的 per-request KV cache 占显存（4B 模型 × 2048 ctx × N 并发）
- RM batch 调度策略（同一 batch 内多个请求各 fork 5 候选 → 25 候选起步）
- 当前文档不覆盖，需另行设计

---

## 8. 引用

- 早期 PyTorch RM 分析：`doc/vllm-rm-experiment-report.md`
- 用户提案原文 + 客观分析：`/workspace/tmp/sia_rm_parallel_design_analysis.md`
- 优化路线图（Tier 1 来源）：`/workspace/tmp/sia_vllm_rm_optimization_roadmap.md`
