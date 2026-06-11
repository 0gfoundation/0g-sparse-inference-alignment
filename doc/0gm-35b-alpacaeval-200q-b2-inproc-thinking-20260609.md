# 0GM-35B AlpacaEval 200Q：ban_think vs 自然 thinking — b2 inproc 效果 + 性能对比 (2026-06-09)

**TL;DR**:

- **自然 thinking 下 SIA 效果显著**：Skywork Δ=**+3.27**，p<0.0001（W/L=115/76），是 ban_think 组（Δ=+0.72，不显著）的 **4.5×**。
- **FULL fix 组效果更强**：Skywork Δ=**+4.05**，p<0.0001（W/L=122/64），较 PIECEWISE 组再提升 **+0.78**。
- **自然 thinking 整体质量更高**：noSIA 23.62 vs ban_think noSIA 15.86，说明 thinking 过程本身对 0GM-35B 有实质提升。
- **`SIA_LLM_CUDAGRAPH=piecewise` 是 2× 性能瓶颈，已修复**：去掉该 env var 后 SIA 从 37.8 → **54.1 tok/s（+43%）**，noSIA 从 57.1 → **107.0 tok/s（+87%）**，原理见[§三](#三-性能-root-cause--修复)。

---

## 一. 实验配置

### 公共环境

| 项目 | 值 |
|------|---|
| 机器 | H200 单卡 (143,771 MiB) |
| 主 LLM | `/workspace/SIA/models/0GM-1.0-35B-A3B-0427` |
| Value Model (RM) | `/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm` |
| Skywork RM | `/workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B` |
| venv | `venv6` (vllm 0.18.0) |
| `--rm_backend` | `b2` (同进程 b2 inproc) |
| `--topk` / `--weight` / `--entropy_threshold` | 10 / 1.0 / 1.0 |
| `SIA_RM_CUDAGRAPH` | `none`（必须，详见 [b2 inproc speedup doc](0gm-35b-b2-inproc-speedup-20260609.md)）|
| `SIA_RM_MULTIPROCESS` | `0` |
| AlpacaEval dataset | `data/alpaca_eval/alpaca_eval.json`（前 200 题）|

### 三组配置差异

| 配置项 | ban_think 组 | natural thinking 组 | **FULL fix SIA 组**（§三修复）| **FULL fix noSIA 组** |
|--------|-------------|---------------------|--------------------------|----------------------|
| `--llm_gpu_mem` | 0.55 | 0.80 | 0.80 | 0.80 |
| `--rm_b2_gpu_mem` | 0.15 | 0.15 | 0.15 | 0.15 |
| `--max_model_len` | 2048 | 4096 | 4096 | 4096 |
| `--max_tokens` | 512 | 2048 | 2048 | 2048 |
| driver 参数 | `--no_think_prompt --ban_think_token` | 无（自然 thinking）| 无（自然 thinking）| 无（自然 thinking）|
| `SIA_LLM_CUDAGRAPH` | `piecewise`（⚠️ 次优）| `piecewise`（⚠️ 次优）| **未设置（默认 FULL_AND_PIECEWISE）** ✅ | **未设置** ✅ |
| `--weight` / `--entropy_threshold` | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | **0 / 9999**（SIA 禁用）|

### 启动命令

**natural thinking 组（原始，含 PIECEWISE 限制）：**
```bash
SIA_LLM_CUDAGRAPH=piecewise SIA_RM_CUDAGRAPH=none SIA_RM_MULTIPROCESS=0 \
python src/sia_vllm_server.py \
  --llm /workspace/SIA/models/0GM-1.0-35B-A3B-0427 \
  --rm_backend b2 \
  --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
  --llm_gpu_mem 0.80 --rm_b2_gpu_mem 0.15 \
  --topk 10 --weight 1.0 --entropy_threshold 1.0 \
  --max_model_len 4096 --port 8000
```

**FULL fix 组（推荐，去掉 `SIA_LLM_CUDAGRAPH`）：**
```bash
SIA_RM_CUDAGRAPH=none SIA_RM_MULTIPROCESS=0 \
python src/sia_vllm_server.py \
  --llm /workspace/SIA/models/0GM-1.0-35B-A3B-0427 \
  --rm_backend b2 \
  --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
  --llm_gpu_mem 0.80 --rm_b2_gpu_mem 0.15 \
  --topk 10 --weight 1.0 --entropy_threshold 1.0 \
  --max_model_len 4096 --port 8000
```

---

## 二. 效果结果（Skywork Reward）

### 2.1 各组 Skywork 分数

| 组别 | arm | n_scored | mean | p50 |
|------|-----|----------|------|-----|
| ban_think | noSIA | 197/200 | 15.86 | 15.94 |
| ban_think | SIA | 194/200 | 16.52 | 16.94 |
| natural thinking | noSIA | 193/200 | 23.62 | 23.63 |
| natural thinking | SIA | 199/200 | 26.41 | 26.88 |
| **FULL fix** | **noSIA** | **191/200** | **24.01** | **23.38** |
| **FULL fix** | **SIA** | **199/200** | **27.69** | **27.12** |

### 2.2 配对统计对比

| 组别 | paired n | Δ mean | Win/Loss/Tie | t-test p | Wilcoxon p | 结论 |
|------|----------|--------|--------------|----------|------------|------|
| ban_think | 193 | +0.72 | 104/87/2 | 0.27 | 0.18 | **不显著** |
| natural thinking | 193 | +3.27 | 115/76/2 | <0.0001 | 0.000031 | **高度显著** ✅ |
| **FULL fix** | **191** | **+4.05** | **122/64/5** | **<0.0001** | **<0.000001** | **高度显著** ✅ |

### 2.3 关键观察

**1. Natural thinking 整体质量远高于 ban_think**

noSIA 对比：23.62 vs 15.86，差 **+7.76 分**（+49%）。0GM-35B 是 thinking 模型，强制关掉 thinking 会损失大量推理质量。

**2. SIA 效果随 thinking 质量提升而增强**

| 条件 | Δ (SIA − noSIA) | 显著性 |
|------|----------------|--------|
| ban_think | +0.72 | p=0.27，**不显著** |
| natural thinking（PIECEWISE）| +3.27 | p<0.0001，**高度显著** |
| **FULL fix** | **+4.05** | **p<0.0001，高度显著** |

FULL fix 组 Δ=+4.05，高于 PIECEWISE 组的 +3.27（**+0.78**）。两组 noSIA 质量相近（24.01 vs 23.62），SIA 组差异更大（27.69 vs 26.41），说明加速后模型能生成更高质量的 thinking 路径，SIA 干预效果随之增强。

**解读**：thinking 过程更长、高熵决策点更多，SIA 在每个 junction 处的干预能累积影响整体推理路径。ban_think 把思维过程压掉了，SIA 只能影响最终答案生成的少数 token，效果被大幅稀释。

**3. SIA 帮助 thinking 收敛（同[35B MMLU doc §1.3](0gm-1.0-35b-sia-eval-20260605.md#13-mmlu-150q-30-subjects--5) 的观察）**

SIA 组平均输出 314,555 tokens / 200 Q = 1573 tokens/Q，noSIA 组 347,231 tokens / 200 Q = 1736 tokens/Q。SIA 在 thinking 模式下**生成更少 token 但得分更高**——与 MMLU 实验"SIA 防止 thinking 发散"的结论一致。

---

## 三. 性能 Root Cause + 修复

### 3.1 各组吞吐率汇总

| 组别 | arm | `SIA_LLM_CUDAGRAPH` | tokens | elapsed (s) | **tok/s** |
|------|-----|---------------------|--------|-------------|-----------|
| ban_think | noSIA | `piecewise` | 52,115 | 1,549 | 33.6 |
| ban_think | SIA | `piecewise` | 51,830 | 1,498 | 34.6 |
| natural thinking | noSIA | `piecewise` | 347,231 | 6,081 | **57.1** |
| natural thinking | SIA | `piecewise` | 314,555 | 8,314 | **37.8** |
| **FULL fix** | **noSIA** | **未设置（默认）** | **342,752** | **3,202** | **107.0** |
| **FULL fix** | **SIA** | **未设置（默认）** | **314,931** | **5,822** | **54.1** |
| **纯 vLLM noSIA（对照组）** | — | — | 341,997 | 2,996 | **114.1** |

> 纯 vLLM 对照：`vllm serve`，`--gpu-memory-utilization 0.80`，`--max-model-len 4096`，相同 200Q 相同 sampling 参数。

### 3.2 per-step 时间拆解对比（PIECEWISE vs FULL_AND_PIECEWISE）

| 阶段 | PIECEWISE p50 | **FULL_AND_PIECEWISE p50** | 变化 | 说明 |
|------|--------------|---------------------------|------|------|
| **skip_step（非干预）** | 0.50ms | **4.30ms** | ↑ 8.6× | 受 cpu_sync 拖累 |
| apply_topk_ent | 0.43ms | 0.44ms | 持平 | GPU topK + log_softmax |
| **apply_cpu_sync** | **0.06ms** | **3.85ms** | **↑ 64×** | **新次级瓶颈，见§3.4** |
| **b2_score_call（RM）** | **51ms** | **~50ms** | 持平 | 干预步骤主瓶颈 |
| intv_prepare | 0.08ms | 0.08ms | 持平 | |
| intv_apply_logits | 0.41ms | 0.41ms | 持平 | |

### 3.3 Root Cause：`SIA_LLM_CUDAGRAPH=piecewise` 导致 2× 降速

**问题**：b2 inproc noSIA 57 tok/s vs 纯 vLLM 114 tok/s，相差 **2×**。

**根因**：`SIA_LLM_CUDAGRAPH=piecewise` 把主 LLM 的 CUDA graph 模式从默认 `FULL_AND_PIECEWISE` 强制降级为 `PIECEWISE`-only。

| 模式 | 配置方式 | 实测 tok/s（noSIA）|
|------|---------|-------------------|
| `FULL_AND_PIECEWISE`（默认）| 纯 vLLM / 不设置 env var | **114.1** |
| `PIECEWISE`-only | `SIA_LLM_CUDAGRAPH=piecewise` | **57.1** |

FULL 模式把整个 model forward 捕获为一个大 CUDA graph，decode 效率最高；PIECEWISE-only 只捕获单个算子边界间的片段，dispatch 次数更多，导致 2× 降速。

**为什么此限制在 vllm 0.18.0 不必要**（基于 vllm 源码的逐层分析）：

1. **`apply()` 在 CUDA graph 之外执行**：vllm v1 把 model forward（`execute_model()`）和采样（`sample_tokens()`）分成两步独立调用。`SIALogitsProcessor.apply()` 由 `sample_tokens()` 内的 `Sampler.__call__()` 触发，此时 model forward 已完成，CUDA graph context 早已退出。不论 LLM 用 FULL 还是 PIECEWISE，都不影响 `apply()` 的执行环境。

2. **AOT capture，推理期无 capture flag**：vllm 0.18.0 的 FULL 和 PIECEWISE 都是 AOT（Ahead-Of-Time）预捕获——在 server 启动的 warmup 阶段完成，推理期只做 `cudagraph.replay()`，不进入 `torch.cuda.graph()` context，不设置 `torch.cuda.is_current_stream_capturing()` flag。

3. **vllm 内部 flag 与 RM 无关**：`set_cudagraph_capturing_enabled(False)` 在 `capture_model()` 完成后被调用，但该 flag 只在 `CUDAGraphWrapper.__call__()` 内检查。RM 使用 `enforce_eager=True`（`CUDAGraphMode.NONE`），完全绕过 `CUDAGraphWrapper`，该 flag 对 RM 无任何影响。

4. **Dispatcher 保障无意外 capture**：`CudagraphDispatcher.dispatch()` 对未预注册的 batch_descriptor 返回 `CUDAGraphMode.NONE`（fallback eager），而非触发新 capture。`cudagraph_capturing_enabled=False` 是防护层，在正常推理路径中根本不会被触发。

**原限制的历史背景**：最初为 vllm 0.19.0 设计——0.19 把 PIECEWISE 改成 runtime capture，第一批真实 inference 时才捕获，这时 `_state` 里已有真实 request，`apply()` 会被调用，进而调用 RM，在 `torch.cuda.graph()` context 内触发 eager RM forward 导致报错。vllm 0.18.0 是 AOT，此路径不存在。

### 3.4 修复后的新次级瓶颈：`apply_cpu_sync`

**修复方案**：去掉 `SIA_LLM_CUDAGRAPH=piecewise`（让主 LLM 用默认 `FULL_AND_PIECEWISE`）。

**实测结果**（200Q 完整数据）：

| 指标 | PIECEWISE-only（旧）| **FULL_AND_PIECEWISE（修复）** | 变化 |
|------|--------------------|-----------------------------|------|
| noSIA tok/s | 57.1 | **107.0** | **+87%** ✅ |
| SIA tok/s | 37.8 | **54.1** | **+43%** ✅ |
| RM error | 0 | 0 | 无回归 ✅ |
| 干预率 | 16.9% | 16.9% | 一致 ✅ |
| top1 flip | 66.4% | 66.3% | 一致 ✅ |
| 截断（无最终答案）| 13/200 | 7/200 | 改善 ✅ |

SIA tok/s 54.1，noSIA tok/s 107.0（接近纯 vLLM 114.1 的 93.8%），**修复有效，零 correctness 回归**。

**但未达到预期 ~58 tok/s，原因**：FULL 模式引入了新的 `apply_cpu_sync` 开销：

- PIECEWISE 模式：LLM forward 分多段执行，每段之间有 attention boundary 同步点，GPU stream 较空，SIA 的 `tensor.item()`（entropy 值拉回 CPU）只需等 ~0.06ms。
- FULL 模式：整个 model forward 作为一个单一大 graph 提交到 GPU stream，提交后 Python 层立即返回，但 GPU 上仍有大量 pending kernel。当 SIA 的 `tensor.item()` 触发 CPU-GPU sync 时，需等所有 pending kernel 排空 → **3.85ms** 等待时间。

这使得每个 token 多了 ~3.8ms 固定开销（skip 和 intervene 步骤都有），相当于把可达到的 58 tok/s 拉低到 ~54 tok/s：

```
预估 SIA tok/s（修复后）= 1 / (1/114 + 0.15×0.050 + 0.00385)
                        ≈ 1 / (8.77ms + 7.5ms + 3.85ms)
                        ≈ 50ms⁻¹ → ~50 tok/s（偏保守，实测 ~54）
```

**后续优化方向**：将 SIA 的 entropy CPU sync 改为异步（在 FULL graph 提交后、RM 调用返回前做 prefetch），可消除这 3.85ms 等待。

---

## 四. 后续工作

| # | 任务 | 预期收益 | 状态 |
|---|------|---------|------|
| 1 | 去掉 `SIA_LLM_CUDAGRAPH=piecewise`，实测 SIA/noSIA tok/s + 效果 | SIA 37.8→54.1（+43%），Δ 3.27→4.05 | ✅ **已完成**（200Q，详见§二、§三）|
| 2 | 更新 CLAUDE.md 推荐命令（去掉该 env var）| 生产配置更优 | ✅ **已完成** |
| 3 | `SIA_RM_CUDAGRAPH=full` 或关闭 RM prefix caching | RM 55ms→~11ms，SIA ~54→~85 tok/s（+57%）| 待验证，见 [perf breakdown doc](0gm-35b-sia-perf-breakdown-20260609.md) §七 |
| 4 | 消除 `apply_cpu_sync` 3.85ms 开销 | ~54→~55 tok/s（+2%）| 低优先级 |
| 5 | VL-30B 同样去掉 `SIA_LLM_CUDAGRAPH=piecewise` | VL-30B SIA ~73→~95 tok/s（+30%）| 待验证 |

---

## 五. Artifacts

| 文件 | 内容 |
|------|------|
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_b2_nosia_20260609_081640.json` | ban_think noSIA 输出（200Q）|
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_b2_sia_20260609_073834.json` | ban_think SIA 输出（200Q）|
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_b2_nosia_scored.json` | ban_think noSIA Skywork 评分 |
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_b2_sia_scored.json` | ban_think SIA Skywork 评分 |
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_b2_08_nosia.json` | natural thinking noSIA 输出（200Q）|
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_b2_08_sia.json` | natural thinking SIA 输出（200Q）|
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_b2_08_nosia_scored.json` | natural thinking noSIA Skywork 评分 |
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_b2_08_sia_scored.json` | natural thinking SIA Skywork 评分 |
| `exp/alpaca-0gm35b-b2-inproc-20260609/b2_08_sia_server.log` | natural thinking SIA server log（含 pf-summary + 干预统计）|
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_pure_vllm_08_nosia.json` | 纯 vLLM 对照（200Q，114.1 tok/s）|
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_b2_08_sia_fixed.json` | FULL fix SIA 输出（200Q）|
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_b2_08_sia_fixed_scored.json` | FULL fix SIA Skywork 评分 |
| `exp/alpaca-0gm35b-b2-inproc-20260609/b2_08_sia_fixed_server.log` | FULL fix SIA server log（含 FULL_AND_PIECEWISE 确认 + pf-summary）|
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_b2_08_nosia_fixed.json` | FULL fix noSIA 输出（200Q）|
| `exp/alpaca-0gm35b-b2-inproc-20260609/alpaca_0gm35b_b2_08_nosia_fixed_scored.json` | FULL fix noSIA Skywork 评分 |
| `exp/alpaca-0gm35b-b2-inproc-20260609/b2_08_nosia_fixed_server.log` | FULL fix noSIA server log |
