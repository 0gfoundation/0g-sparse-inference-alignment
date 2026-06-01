# 单卡纯工程优化路线图 — InprocClient 之后的剩余空间

**日期**: 2026-06-01
**前置背景**: InprocClient 优化已落地, MMLU 600q throughput 46.6 → ~65 tok/s ([inproc-client-optimization-20260531.md](inproc-client-optimization-20260531.md))
**约束**:
- ✅ 单 GPU（不考虑双卡 NVLink / 跨机并行）
- ✅ 不动 `entropy_threshold` / `topk`（这两个影响干预率和 alignment 效果）
- ✅ 纯工程优化（不动模型结构 / SIA 干预语义）
- ✅ 不改 vLLM 主干源码（只在 SIA 层和 RMClient 层动手）
**目标**: 把单卡 SIA throughput 从 65 推到尽量接近 noSIA 上限 (88 tok/s)

---

## 1. 当前状态拆解

```
65 tok/s, 17 ms / INTERVENE (mean)
├─ ~9 ms — b2_score_call (InprocClient 之后, 这块已经很紧)
│         ├─ ~7 ms GPU forward (Stage A+B+C)
│         └─ ~2 ms vLLM dispatch + /dev/shm IPC
│
└─ ~8 ms — SIA processor + apply() 框架 wrap
          ├─ apply() 内 GPU sync (entropy / topk indices)
          ├─ _intervene_with_b2 Python 逻辑
          ├─ verbose log + flip 检测的额外 sync
          └─ vLLM generate() Python dispatch (RequestOutput 构造 / detokenize / sampler)

noSIA 上限: 88 tok/s
剩余空间约 23 tok/s (35%)
```

InprocClient 把 RM 端 (9 ms) 已经压得很接近 GPU 物理下限。**剩余优化空间主要在外层 Python wrap 和 vLLM dispatch 层**, 以及未来的 in-process executor 改造。

---

## 2. 剩余优化候选总表（按 ROI 排序）

### 🥇 第一梯队 — 1 周内可全部完成

| # | 优化项 | 节省 / INTERVENE | 预期 throughput | 工时 | 风险 |
|---|---|---:|---:|---:|---|
| **A** | **关 verbose log + 多余 GPU sync 清理** | 0.5-1 ms | **+1-2 tok/s** | **半天** | 低 |
| **B** | **`/dev/shm` IPC 改成进程内 tensor 直接共享** | 1-2 ms | **+2-4 tok/s** | **2 天** | 低-中 |
| **C** | **skip detokenize + skip sampler** | 0.3-0.5 ms | **+0.5-1 tok/s** | **1 天** | 低 |
| **D** | **rm_scores 直接 GPU tensor 加 logits, 不绕 CPU** | 0.3-0.5 ms | **+0.5-1 tok/s** | **半天** | 低 |
| 小计 | | **2-4 ms** | **+4-8 tok/s → 69-73 tok/s** | **4 天** | |

### 🥈 第二梯队 — 1-2 周

| # | 优化项 | 节省 | 预期 throughput | 工时 | 风险 |
|---|---|---:|---:|---:|---|
| **E** | **In-process executor (B-1)** — 跳过 vLLM `LLM.generate()` 调用栈, 直接调 `GPUModelRunner.execute_model` | 3-5 ms | **+5-8 tok/s** | **1-2 周** | 中 |
| **F** | **RM FP8 量化** (Qwen3-4B + InprocClient 后值得重新评估) | 1-2 ms | **+1-3 tok/s** | **2-3 天** | 中-高 |
| 小计 | | **4-7 ms** | **+6-11 tok/s 叠加 → ~76-84 tok/s** | | |

### ❌ 不推荐（doc 已分析失败 / 收益太小 / 工程量极大）

| 项 | 不推荐原因 |
|---|---|
| **CUDA graph batch=5 显式补齐 (X-7)** | 实测无效, 因为 InprocClient 之后 vLLM 已经命中合适 graph (见 [rm-batch5-ab-test.md](rm-batch5-ab-test.md)) |
| **复用 score 时计算的 KV cache (2C)** | vLLM cache manager 难 hack, 收益 < 1 tok/s |
| **Score head GPU kernel fuse** | Stage C 才 0.3 ms, 摸不到天花板 |
| **Speculative decoding 思路 (2D)** | 工程量极大, 单卡下 LLM/RM 共享 SM, 并行收益不明确 |
| **VM 重训为 MoE 或更小尺寸** | 训练 + ablation 2-3 周, 风险大, 不属于"纯工程优化" |

---

## 3. 各项详细方案

### A. 关 verbose log + 多余 GPU sync 清理（半天，最容易）

**当前代码** (`src/sia_vllm_RM.py` apply() 内):
```python
entropy_values = entropies.cpu().tolist()                  # sync 1, 必须
topk_indices_lists = topk_result.indices.cpu().tolist()    # sync 2 (b2 path 必须, 要 decode token)
topk_values_lists = topk_result.values.cpu().tolist()      # sync 3, 仅 verbose log + flip 检测需要
```

**verbose 模式下还有的 sync**:
```python
print(f"rm=[{rm_scores.min():.3f}, {rm_scores.max():.3f}]")   # 若 rm_scores 是 GPU 则 2× sync
print(... f"flip={'Y' if flipped else 'N'} ...")              # flip 检测需要 topk_values_lists
```

**改动**:
1. 默认设 `SIA_LOG_LEVEL=quiet`, 跳过 `topk_values_lists` 那次 sync 和 verbose print
2. flip 检测如果保留, 移到 GPU 端 argmax (节省 50-100 μs)
3. 给 `_LOG_LEVEL` 加 env var 支持随时回到 verbose debug

**预期收益**: ~0.5 ms / INTERVENE → **+1-2 tok/s**

**风险**: 失去 per-step debug log, 但 env var 一键回切

---

### B. 消除 /dev/shm IPC（2 天，最大单项 win）

**当前问题**: 即使 InprocClient 模式下 RMClient 和 RM EngineCore 已经在同一个 Python 进程, reward 仍走历史多进程时代留下的 `/dev/shm/sia_reward_<id>.bin` 通道。每次 INTERVENE 这条链路:

```
qwen3_with_score.compute_logits()
  → rewards.cpu().numpy() (GPU → CPU sync + numpy copy)
  → /dev/shm 文件 write (tmpfs in-memory, 但仍有 syscall + 序列化)
  → RMClient.read_rewards()
  → np.frombuffer + torch.from_numpy
  → tensor 重建
```

**改动**: InprocClient 下让 `compute_logits` **直接把 reward tensor 存到 module 实例属性**, RMClient 同进程直接读:

```python
# src/sia_rm/qwen3_with_score.py
def compute_logits(self, hidden_states, ...):
    rewards = (hidden_states @ self.score.weight.T).squeeze(-1)
    if _INPROC_MODE:
        self._sia_last_rewards = rewards  # 保留 GPU tensor, 不写 /dev/shm
    else:
        _write_rewards(rewards.cpu().numpy())  # legacy path 保留

# src/sia_rm/client.py
def score_candidates(self, sid, cand_ids):
    _ = self.llm.generate([...], self._sp)
    if self._inproc:
        return self._get_model_attr("_sia_last_rewards")
    return read_rewards()  # legacy
```

**预期收益**: 节省每次 /dev/shm write + read + tensor reconstruct ≈ 1-2 ms → **+2-4 tok/s**

**风险**: 多个并发请求时 `_sia_last_rewards` 需要按 request_id 区分（用 dict 或 per-step 队列）。建议先单请求 path 实现, 后续按需扩展。

---

### C. Skip detokenize + skip sampler（1 天）

每次 `LLM.generate([5 prompts], SamplingParams(max_tokens=1))` 内部仍跑:
- TokenSampler (top-p, top-k, temperature) — 我们 temperature=0 + max_tokens=1, 但 sampler 还是计算一遍
- Detokenize 生成的 token 回字符串
- 构造 `RequestOutput` 对象

我们只要 score head 的 reward, **完全不需要这些**。

**改动方案 (轻量版, 不动 vLLM 源码)**:
```python
# 自定义 SamplingParams 跳过最贵的部分
self._sp = SamplingParams(
    temperature=0.0,
    max_tokens=1,
    min_tokens=1,
    ignore_eos=True,
    detokenize=False,         # vLLM 0.10.x 支持此选项, 跳过 detokenize
    logprobs=None,            # 不要 logprobs
)
```

**改动方案 (激进版, 配合 E 一起做)**:
直接调用底层 EngineCore.step() 一两次拿到 compute_logits 输出, 不走完整 LLM.generate() pipeline。

**预期收益**: 0.3-0.5 ms / call → **+0.5-1 tok/s**

**风险**: detokenize=False 在某些 vLLM 版本可能没生效, 要做实测验证

---

### D. rm_scores 直接 GPU tensor 加 logits（半天，配合 B）

跟 B 配合做：B 完成后 `_sia_last_rewards` 已经是 GPU tensor, apply() 端直接拿 GPU tensor 加:

**当前**:
```python
rewards = read_rewards()                              # tensor on CPU
rm_scores = rewards.float()
...
logits[i, topk_indices_gpu] += rm_scores.to(logits.device) * effective_weight
# 5 个 float32 走一次 host→device transfer (~10-50 μs, 但每次 INTERVENE 都付)
```

**改动**:
```python
rm_scores_gpu = self._rm.get_last_rewards()           # 直接 GPU tensor (来自 B)
logits[i, topk_indices_gpu] += rm_scores_gpu * effective_weight
```

**预期收益**: 0.3-0.5 ms → **+0.5-1 tok/s**

**风险**: 跟 B 强耦合, 需要先完成 B 再做 D

---

### E. In-process executor (B-1) — 单卡架构最大剩余收益（1-2 周）

绕过 vLLM 的 `LLM` / `LLMEngine` / `EngineCore` 全栈, 直接管 `GPUModelRunner.execute_model`:

**当前**:
```
RMClient.score_candidates(5 prompts)
  → LLM.generate(prompts, sp)
    → for prompt in prompts: add_request()  # 5 次串行
    → _run_engine() → EngineCore.step()
      → scheduler.schedule()
      → executor.execute_model() → GPUModelRunner.execute_model() ◄── 实际 GPU 计算
    → 构造 RequestOutput × 5
  → 解析 RequestOutput 提取 rewards
```

**改动**:
```
RMClient.score_candidates(5 prompts)
  → self._build_scheduler_output(prompts)
  → runner.execute_model(scheduler_output) ◄── 直接调用
  → reward 直接从 model._sia_last_rewards 拿
```

**跳过的开销**:
- Request 对象构造 + add_request 协议 (~1-2 ms)
- Scheduler 调度逻辑 (~1-2 ms)
- RequestOutput 构造 + 解析 (~0.5-1 ms)
- 多余的 KV cache manager 协议

**预期收益**: 3-5 ms / INTERVENE → **+5-8 tok/s**

**难点**:
- 要复制 `EngineCore.__init__` 的 KV cache 分配、CUDA graph capture、sampling state setup 逻辑
- 这些 vLLM v1 内部 API 没正式文档
- 自己处理 batched input 构造（vLLM `LLM.generate` 帮你做的部分）

**风险**:
- vLLM 升级时需要重新适配 → 钉死 `vllm==0.10.1.1` 直到 in-process executor 写完
- 工程量约 300 LOC, 风险中

**参考**:
- doc/b2-pure-engineering-optimizations.md §B-1 已经详细规划过

---

### F. FP8 量化 RM（重新评估，2-3 天）

**历史**: G4 实验 (sparse_logs / vllm-rm-followup-optimizations.md) 用 `--quantization fp8` 反而慢了 7%。但当时 RM 是 multiprocess subprocess, FP8 收益被 IPC + scheduler overhead 吃掉。

**现在**: InprocClient + batch=5 forward 已经把 RM GPU 时间压到 4.8 ms。这点 GPU 时间正是 FP8 该发力的地方 (H100/H200 FP8 GEMM 理论 1.5-2× 加速)。

**改动**:
```bash
# 离线量化 (一次性, 用 llmcompressor)
python scripts/quantize_rm_fp8_dynamic.py \
  --model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
  --output /workspace/SIA/models/VM-Qwen3-4B-fp8

# 启动改用 fp8 模型 + fp8 KV cache
self.llm = LLM(model="/workspace/SIA/models/VM-Qwen3-4B-fp8",
               quantization="fp8",
               kv_cache_dtype="fp8",
               ...)
```

**预期收益**: 1-2 ms / INTERVENE → **+1-3 tok/s**

**风险**:
- accuracy 退化 (FP8 在 value model 这种数值差异敏感场景可能掉点)
- 必须严格 ablate: 跑一次 MMLU 600q 看 accuracy 是否保持
- 已经有 `scripts/quantize_rm_fp8_dynamic.py`, 工程量主要在 ablation

---

## 4. 推荐一周排期

| 天 | 任务 | 预期累计 throughput |
|---|---|---|
| **周一上午** | **A** (log/sync 清理) | ~66-68 tok/s |
| **周一下午** | **C** (skip detokenize/sampler, 改 SamplingParams) | ~67-69 tok/s |
| **周二-周三** | **B** (消除 /dev/shm IPC) | ~70-72 tok/s |
| **周三下午** | **D** (rm_scores GPU 直传, 配合 B) | ~71-73 tok/s |
| **周四** | 集成测 + MMLU 600q 验证 + 调优 | ~71-73 tok/s 稳定数字 |
| **周五** | **启动 E (In-process executor) 原型工作** | 这周做不完, 周末写完主流程, 下周完整跑通 |

**一周后预期**: 65 → **~71-73 tok/s** (+10-12%)
**两周后预期 (含 E)**: → **~76-80 tok/s** (+17-23%)
**两周后含 F (FP8 ablation 通过)**: → **~78-82 tok/s** (距离 noSIA 上限 88 只剩 6-10 tok/s)

---

## 5. 风险与决策点

### 5.1 验证机制

每个优化项落地后必须跑 **MMLU 600q full eval** 确认 accuracy 不退化：
```bash
# 优化前 baseline: 74.83% accuracy, ~65 tok/s
# 每项优化后跑一次, accuracy 落点应该跟 baseline 在 ±1pp 内
```

throughput 数字以 `mmlu_eval.py` 输出的 `Throughput: X.X tokens/s` 为准。

### 5.2 中途暂停判断

如果发现某项优化 throughput **没达到预期 50%**, 立即记录原因并跳过下一项，不要硬上。可能的失败模式:
- B 的 inproc tensor 共享被 vLLM 内部 cache 机制干扰 → 退回 /dev/shm
- C 的 detokenize=False 在 vLLM 0.10.1.1 没有生效 → 跳过
- F 的 FP8 accuracy 掉超过 1pp → 立即回滚到 bf16

### 5.3 跟"减少干预率/topk"的对比

本路线图所有数字都假设 32% 干预率 + topk=5 不变。如果允许调这两个超参，**单独靠 entropy_threshold 从 1.0 提到 2.0 大约能直接拿 +30-50% throughput** (论文 §5 指出 20% 干预率即可媲美 dense)。但那属于 quality trade-off, 不在本路线图范围。

---

## 6. 引用

- 当前 baseline (InprocClient 后): [`inproc-client-optimization-20260531.md`](inproc-client-optimization-20260531.md)
- 历史优化项清单 (B-1 / D-1 / D-5 / X-5 / X-7): [`b2-pure-engineering-optimizations.md`](b2-pure-engineering-optimizations.md)
- 3 阶段 GPU profiling: [`rm-split-stage-profiling.md`](rm-split-stage-profiling.md)
- FP8 历史失败记录: [`vllm-rm-followup-optimizations.md`](vllm-rm-followup-optimizations.md)
- 并行/双卡 (本路线图不涉及): [`parallel-decoding-design.md`](parallel-decoding-design.md)
- 关键源码:
  - SIA processor apply(): `src/sia_vllm_RM.py:653-768`
  - b2 score path: `src/sia_vllm_RM.py:_score_candidates_b2`
  - RMClient.score_candidates: `src/sia_rm/client.py:139-174`
  - score head + /dev/shm 通道: `src/sia_rm/qwen3_with_score.py`
