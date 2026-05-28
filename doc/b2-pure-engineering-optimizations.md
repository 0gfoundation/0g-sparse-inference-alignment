# B2 路径纯工程优化候选（零质量影响）

**适用范围**：M2 step 5 完成后，b2 in-process RM 路径在 Qwen3-14B + VM-Qwen3-4B + 600 题 MMLU baseline-aligned 实测达 **45.5 tok/s**（vs HTTP /classify G1 历史最佳 33.5 tok/s, +36%；vs noSIA 上限 88 tok/s, 占 52%）。本文落档**纯工程优化候选**——即不改变 SIA 干预语义、不牺牲生成内容质量的优化方向，按预期 throughput 提升从高到低排序。

> **不在本文范围**（属于 quality trade-off，单独 ablation 评估）：
> - 提高 `entropy_threshold` 减少干预率
> - 降低 `topk` 减少 candidate 数量
> - 减小 `max_model_len` 缩短调度元数据（有截断风险）
>
> **不在本文范围**（属于架构级改动）：
> - 让 LLM 和 RM 并发使用 GPU（多 CUDA stream / 多 GPU）
> - Speculative decoding 思路把 RM 当 draft model
> - 用更小的 RM backbone

---

## 1. 当前性能数据（基线）

**配置**：Qwen3-14B (LLM) + VM-Qwen3-4B (RM)，topk=5, weight=1.0, entropy=1.0，MMLU-Redux 600 题 limit=20。

| 维度 | 当前值 |
|------|--------|
| 整体 throughput | **45.5 tok/s** |
| 每 token (mean) | 22.0 ms |
| SKIP step (LLM-only, mean) | 12.3 ms |
| INTERVENE step (mean) | 44.4 ms |
| `b2_score_call` (RM 内部, mean) | ~24 ms |
| INTERVENE Python wrap | ~8 ms |
| 干预率 | 32.6% |

**时间贡献拆解**:
- LLM forward (every token) = 12.3 ms/token = **54%**
- RM call + Python wrap (only INTERVENE) = 10.5 ms/token = **46%**

**SIA 引入的退化**: noSIA 88 tok/s → SIA 45.5 tok/s，**10.6 ms/token 退化 100% 由 RM call 造成**。

---

## 2. 优化候选总览（按预期收益从高到低）

| # | 优化项 | 预期收益 (tok/s) | 代码改动量 | 工作量 |
|---|--------|------------------|-----------|--------|
| 🥇 1 | **B-1: vLLM in-process executor for RM** | **+3.5-5** | ~300 LOC | 1-2 周 |
| 🥈 2 | **D-1: 关 INTERVENE log + GPU sync** | **+1-1.5** | ~30 LOC | 1 天 |
| 🥉 3 | **D-5: state dict + 减少 `.item()` sync** | **+0.5-1** | ~50 LOC | 1 天 |
| 4 | **X-7: CUDA graph capture sizes 补 batch=5** | **+0.5-1** | ~10 LOC | 半天 |
| 5 | **X-5: SIA processor 内部 tensor 拷贝优化** | **+0.2-0.5** | ~30 LOC | 半天 |

**理论总和**: 五项叠加 ≈ **+5.7-9 tok/s**（45.5 → ~51-54 tok/s）。距 noSIA 上限 88 tok/s 仍差 34 tok/s，要进一步突破需架构级改动。

---

## 3. 各项详细方案

### 🥇 1. B-1: vLLM in-process executor for RM

**目标**：消除当前嵌套 EngineCore subprocess 架构的 IPC 开销。

**当前问题诊断**:

架构链：
```
SIA main proc
  → [zmq] LLM EngineCore subprocess A
    → [zmq + /dev/shm] RM EngineCore subprocess B (RMClient 嵌套启动)
```

每次 `score_candidates` 调用涉及（实测 ~5-7 ms overhead）:

| 嵌套 EngineCore 的实际 overhead | ms |
|---------------------------------|----|
| zmq 跨进程 message dispatch | 1-2 |
| Python pickle 序列化 input/output | 1-2 |
| `/dev/shm` 文件 IO（写 reward + 读） | 0.5-1 |
| 进程切换 + scheduler 等待 | 1-2 |
| **合计** | **~5-7** |

**实施路径**：改写 `src/sia_rm/client.py::RMClient.__init__`，不再调 `vllm.LLM()`（它会 fork 一个 EngineCore subprocess），改成在 LLM EngineCore A 的同一线程内直接构造 `GPUModelRunner`：

```python
# 当前 (src/sia_rm/client.py)
from vllm import LLM
self.llm = LLM(model=VM_path, hf_overrides={...}, ...)  # ← fork EngineCore B

# 改成
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

engine_args = EngineArgs(model=VM_path, hf_overrides=..., ...)
config = engine_args.create_engine_config()
self.runner = GPUModelRunner(vllm_config=config, device=...)
self.runner.load_model()
self.runner.initialize_kv_cache(...)
self.runner.capture_model(...)   # CUDA graph capture

# score_candidates 改成直接调
def score_candidates(self, sid, cand_ids):
    prompts = [self._sessions[sid] + [c] for c in cand_ids]
    seq_data, slot_mapping = self._build_input(prompts)
    out = self.runner.execute_model(seq_data, positions, sampling_metadata)
    rewards = (out @ self.runner.model.score.weight.T).squeeze(-1)
    return rewards.tolist()
```

**难点**：
- 要复制 `EngineCoreProc.__init__` 内部的 KV cache 分配、CUDA graph capture、sampling state setup 逻辑
- 这些 vLLM v1 内部 API 没正式文档
- 自己处理 batched input 构造（vLLM `LLM.generate` 帮你做的部分）

**可行性**：vLLM 0.10.x `GPUModelRunner.execute_model` 接口相对稳定；setup 部分需要照搬 `EngineCore.__init__` 的逻辑（参考 `vllm/v1/engine/core.py`）。

**风险**：vLLM 升级时需要重新适配。建议钉死 `vllm==0.10.1.1` 直到 in-process executor 写完为止。

**预期收益**：消除 5-7 ms / INTERVENE step → 加权（× 32.6% 干预率）= 1.6-2.3 ms/token → **+3.5-5 tok/s**。

---

### 🥈 2. D-1: 关 INTERVENE log + GPU sync

**目标**：移除每 INTERVENE step 的 `print()` + 2 次 `.item()` GPU 同步（共 ~1-2 ms / INTERVENE step）。

**当前代码**（`src/sia_vllm_RM.py` 约 line 645-655）:
```python
print(
    f"[SIA] {time.strftime('%H:%M:%S')}.{int(time.time() % 1 * 1_000_000):06d} "
    f"step={req_step:3d} req={i} INTERVENE "
    f"entropy={entropy:.3f} "
    f"gen_len={len(output_ids)} "
    f"rm=[{rm_scores.min():.3f}, {rm_scores.max():.3f}] "  # ← 2× GPU→CPU sync
    f"flip={'Y' if flipped else 'N'} "
    f"pre_top1={pre_top1} post_top1={post_top1}",
    flush=True,
)
```

**改动方案**：

```python
class SIALogitsProcessor(LogitsProcessor):
    _LOG_LEVEL: str = os.environ.get("SIA_LOG_LEVEL", "verbose")  # quiet | verbose

    # 在 apply() 内
    if self._LOG_LEVEL == "verbose":
        print(
            f"[SIA] ... rm=[{rm_scores.min():.3f}, {rm_scores.max():.3f}] ..."
        )
    # else: 完全跳过 print + 2 次 GPU sync
```

**保留 debug 能力**：默认 `SIA_LOG_LEVEL=quiet`，需要时设 `SIA_LOG_LEVEL=verbose` 恢复完整 log。

**预期收益**：每 INTERVENE step 省 1-2 ms → 加权 × 32.6% = 0.5 ms/token → **+1-1.5 tok/s**。

**副作用**：失去 per-step debug log（可通过 env var 保留）。

---

### 🥉 3. D-5: state dict 优化 + 减少 `.item()` sync

**目标**：合并多个分散的 `.item()` / `.tolist()` GPU→CPU sync 到一次 batched copy（~0.5-1 ms / INTERVENE）。

**当前代码**（`src/sia_vllm_RM.py` apply 内）:
```python
topk_indices_lists = topk_result.indices.cpu().tolist()  # sync 1
topk_values_lists = topk_result.values.cpu().tolist()    # sync 2
entropy_values = entropy.tolist()                         # sync 3
intervene_flags = (entropy > self._ENTROPY_THRESHOLD).cpu().tolist()  # sync 4
```

**改动方案**：把所有需要 CPU 端读取的 GPU tensor 合并成一个 stack，单次 `.cpu()`：

```python
# stack 成 (4, batch_size) 一次 sync
batched_cpu = torch.stack([
    topk_result.indices.flatten().float(),  # 注意 dtype 兼容
    topk_result.values.flatten(),
    entropy.repeat(topk_k),
    (entropy > threshold).float().repeat(topk_k),
]).cpu()
# 再 unpack
```

**进一步优化**：`self._output_ids[i]`、`self._prompt_user[i]` 等 dict 改成 list（要求 req_idx 是 dense int 0..N-1）。**但要小心 update_state 的 add/remove 让 idx 稀疏**——需要先确认 vLLM v1 是否保证稠密。

**预期收益**：0.5-1 ms / INTERVENE → 加权 = **+0.5-1 tok/s**。

**副作用**：dict→list 假设 idx 稠密，需要在 update_state 处 detect 稀疏化并 fall back。

---

### 4. X-7: CUDA graph capture sizes 补 batch=5

**目标**：让 vLLM 给 batch=5 单独 capture 一个 CUDA graph，避免 fall back 到 batch=8 的 graph + padding 3 个 dummy token。

**当前问题诊断**：vLLM v1 默认 cuda graph capture sizes 是 `[1, 2, 4, 8, 16, ..., 512]`，**batch=5 不在列**。每次 `score_candidates` 用 batch=5 prompt 列表，vLLM 会 padding 到 batch=8 跑 captured graph，多算 3 个无用 token 的 forward。

**Server log 验证**:
```
cudagraph_capture_sizes:[512, 504, 496, 488, ..., 16, 8, 4, 2, 1]
```
（步长 8 / 4 / 2 / 1 组合，没 5）

**改动方案**：在 `src/sia_rm/client.py::RMClient.__init__` 给 vLLM 传 `cuda_graph_sizes=[1, 2, 4, 5, 8, 16, 32, 64, 128]`：

```python
self.llm = LLM(
    model=model_path,
    hf_overrides={"architectures": ["Qwen3WithScoreForCausalLM"]},
    cuda_graph_sizes=[1, 2, 4, 5, 8, 16, 32, 64, 128],  # 加入 5
    ...
)
```

**风险**：vllm-rm-experiment-report.md §3.1 提过 "`--cuda-graph-sizes 5` 单值语义是 `[1,2,4]+range(8,N+1,8)`，5 时只 capture [1,2,4]，batch=5 落空回退到 eager 反而慢"。**必须传 list 而不是单值**。

**预期收益**：每个 `score_candidates` 省 0.5-1 ms → 加权 = **+0.5-1 tok/s**。

**副作用**：稍微增加 vLLM startup 时间（多 capture 一个 size），运行时无副作用。

---

### 5. X-5: SIA processor 内部 tensor 拷贝优化

**目标**：消除 SIA processor 内的几处隐式 `.to(device)` 拷贝和 Python list concat 操作。

**当前代码**（`src/sia_vllm_RM.py`）:
```python
logits[i, topk_indices_gpu] = (
    logits[i, topk_indices_gpu]
    + rm_scores.to(logits.device) * effective_weight  # ← rm_scores 是 CPU tensor, 隐式 copy
)
```

**改动方案 A — rm_scores 直接生成在 GPU 上**:

修改 `RMClient.score_candidates` 返回 GPU tensor 而不是 list:
```python
def score_candidates(self, sid, cand_ids) -> torch.Tensor:  # 返回 GPU tensor
    ...
    rewards_gpu = read_rewards_to_gpu()  # 直接 read /dev/shm 到 cuda tensor
    return rewards_gpu
```

但要注意：当前 reward 是 /dev/shm 文件读出来的 CPU float32。改成 read 直接放 GPU 也得 host→device transfer，**净收益接近 0**。

**改动方案 B — prefix list concat 用 deque 或 array**:

```python
# 当前 (RMClient.score_candidates)
prompts = [TokensPrompt(prompt_token_ids=prefix + [c]) for c in candidate_token_ids]
# Python list concat × 5, 800 token prefix → ~0.25 ms 总和

# 优化: 用 array + slice
prefix_array = self._sessions_array[sid]
prompts = [TokensPrompt(prompt_token_ids=np.concatenate([prefix_array, [c]])) for c in cand]
```

**预期收益**：0.2-0.5 ms / INTERVENE → **+0.2-0.5 tok/s**。

**副作用**：无（纯重构）。

---

## 4. 分组实施方案（按风险维度分组 + 是否捆绑测）

分组核心判断标准：

| 判断 | 是 → 必须**单独**测 | 否 → 可**捆绑**测 |
|------|---------------------|--------------------|
| 改变 logits 计算（数值可能漂移） | B-1 | D-1, X-7, X-5, D-5 |
| 有 correctness assumption（数据结构假设） | D-5 | D-1, X-7, X-5, B-1 |
| 收益太小可能淹没在噪声 | X-5 必须跟其他绑 | D-1, X-7, D-5, B-1 |
| 失去 debug 能力 → 需要保留 env var 开关 | D-1（用 SIA_LOG_LEVEL 控制） | X-7, X-5, D-5, B-1 |

### 🅐 Group A — 安全捆绑（一起做一起测）

**特点**：三项都**不改 logits 计算**，accuracy 必然不变；都是 throughput-only 收益。三项独立 commit，万一回归可以单独 git revert + bisect。

| 项 | 是什么 | 预期效果 | 改动难度 | 风险 |
|----|--------|---------|----------|------|
| D-1 | 关 INTERVENE log + 2 次 GPU sync（`rm_scores.min()/.max().item()` + `print`），加 `SIA_LOG_LEVEL` env var 控制 | +1-1.5 tok/s | 低（~30 LOC, 1 天） | 失去 debug log（env var 可控） |
| X-7 | vLLM `cuda_graph_sizes` 显式包含 batch=5，避免 padding 到 batch=8 graph | +0.5-1 tok/s | 低（~10 LOC, 半天） | 启动时间略增（一次性） |
| X-5 | SIA processor 内 tensor copy / list concat 重构 | +0.2-0.5 tok/s | 低（~30 LOC, 半天） | 0 |
| **小计** | | **+1.7-3 tok/s** | **2 天** | 三项都不改 logits，accuracy 必然不变 |

**测试**：1× Tier 2 (300 题, ~45 min) 即可验证捆绑效果。

### 🅑 Group B — 单独测（correctness 风险）

| 项 | 是什么 | 预期效果 | 改动难度 | 风险 |
|----|--------|---------|----------|------|
| D-5 | SIA processor 内 `_output_ids` / `_prompt_user` 等 dict 改 list，合并多个 GPU sync 到一次 batched `.cpu()` | +0.5-1 tok/s | 中（~50 LOC, 1 天） | ⚠️ dict→list 假设 req_idx 是 dense int 0..N-1；若 vLLM `update_state` 的 add/remove 让 idx 稀疏会 KeyError |

**为什么单独测**: dict→list 改动有 correctness 风险，跟 Group A 捆绑会让 crash 时不易定位是哪个改动引入的。

**测试**：1× Tier 2 (300 题, ~45 min)。

### 🅒 Group C — 大改动，前后都要严格测

| 项 | 是什么 | 预期效果 | 改动难度 | 风险 |
|----|--------|---------|----------|------|
| B-1 | 把 RM 从嵌套 EngineCore subprocess 改成在 LLM EngineCore 同线程直接调 `GPUModelRunner.execute_model`，消除 zmq + pickle + /dev/shm IPC | +3.5-5 tok/s | 高（~300 LOC, 1-2 周） | ⚠️ 直接调 vLLM 内部 API，forward 数值可能 ε 级偏差，经 softmax+topk 可能让 sample 决策漂移 → accuracy 退化；vLLM 升级要重写 |

**为什么前后都要 Tier 3**:
- 直接调 `GPUModelRunner.execute_model` 跟 `LLM.generate` 在 CUDA graph capture 时机、KV cache 分配顺序等细节上可能不一致
- ε 级数值偏差经 softmax + topk 可能改变 sample 决策 → accuracy 漂移
- 必须严格 before-after 对比 accuracy

**测试**：2× Tier 3 (600 题, 各 ~90 min)，before/after 严格对比 accuracy 不退化 + throughput 提升。

### 累计预期路径

```
当前基线:                 45.5 tok/s, accuracy 83.98% (vs 历史 SIA 83.48%, side-by-side n=256)
+ Group A:                47-48.5 tok/s   (~2 天)        accuracy 不变
+ Group B:                48-49.5 tok/s   (~3 天累计)    accuracy 不变
+ Group C:                51-54 tok/s     (~3 周累计)    accuracy ±1pp（B-1 实施质量决定）
```

### 评测分层定义（前文 §3 配套）

| Tier | 题数 | 时间 | 灵敏度 | 用途 |
|------|------|------|--------|------|
| Tier 1 Smoke | 15（3 subj × 5） | ~3 min | ±5 tok/s | 启动 + 不 crash |
| Tier 2 Mini | 300（30 subj × 10） | ~45 min | ±1.5 tok/s, acc ±3% | accuracy 不退化 + throughput 趋势 |
| Tier 3 Full | 600（30 subj × 20） | ~90 min | ±0.5 tok/s, acc ±2% | 严格对齐 baseline |

---

## 5. 终极天花板分析

实施全部 5 项后预期到 **51-54 tok/s**，仍距 noSIA 88 tok/s 上限有 34 tok/s 差距。

**为什么不能再压**：当前 SIA 引入的退化 10.6 ms/token = 0.326 × (RM call 32 ms) 完全是 RM 调用时间。即使把 RM call 时间压到 0，throughput 上限也只是 noSIA 的 88 tok/s。

**真正能突破 noSIA 上限的方向**（不在本文范围，属架构改动）：
1. **多 GPU**: LLM 一张卡，RM 一张卡，并发 forward
2. **Speculative decoding**: 用 RM 当 draft，每 N token 验证一次
3. **多 CUDA stream + kernel concurrency**: 单卡上让 LLM 和 RM kernel overlap（实际效果存疑，SM 共享 + 带宽共享）
4. **更小的 RM backbone**: 用 Qwen3-1.7B 或 0.6B 替换 4B（需重新训练 VM checkpoint）

---

## 6. 引用

- B2 路径 M2 设计与实测：[`b2-m2-design.md`](b2-m2-design.md) §6.5, §7.X, §7
- vLLM RM 实验报告（HTTP /classify 路径历史最佳）：[`vllm-rm-experiment-report.md`](vllm-rm-experiment-report.md)
- vLLM RM 后续优化教训（FP8 / G2-G4 原地踏步原因）：[`vllm-rm-followup-optimizations.md`](vllm-rm-followup-optimizations.md)
- SIA 整体性能瓶颈分析：[`performance-report.md`](performance-report.md)
- vLLM 0.10.x EngineCore 启动逻辑参考：`vllm/v1/engine/core.py::EngineCoreProc.__init__`
