# 单卡 SIA 剩余优化空间分析 — 基于校准后的真实瓶颈

**日期**: 2026-06-02
**前置工作**:
- 已完成 InprocClient + A + D + B 三轮纯工程优化, 当前 throughput **62.4 tok/s** (90q 测得)
- 已完成 F (FP8 RM) ablation — 失败 (throughput 不变)
- 已完成详细 phase profiling ([`sia_90q_detailed_profiling_20260601.md`](../../../tmp/sia_90q_detailed_profiling_20260601.md))
- 已完成 dummy LogitsProcessor 校准 ([`sia_calibration_summary_20260601.md`](../../../tmp/sia_calibration_summary_20260601.md))

**目的**: 把"剩余还能优化什么 + 预期收益"系统化排序, 作为后续投入决策依据。

**约束**:
- 单卡 (不考虑多 GPU)
- 不动 entropy_threshold / topk (这两个改了影响 alignment 质量)
- 不动 SIA 干预语义本身 (除非 Tier 1 #3 speculative SIA 那条明确标了)

---

## 1. 当前真实瓶颈分解 (基于校准, 不是直觉)

```
SIA 当前      62.4 tok/s  (16.03 ms/token)
noSIA 上限    88.4 tok/s  (11.36 ms/token)
                          ─────────────────────────────────
                          SIA overhead = +4.67 ms/token
                          ├─ hook 框架开销  1.36 ms (跟 SIA 干预无关; LogitsProcessor 注册 + AsyncEngineArgs)
                          ├─ sync 自身      0.32 ms (entropy + cpu().tolist() 本身)
                          └─ SIA 真实工作   3.49 ms
                             ├─ b2_score_call × 33%  = 2.85 ms ← 主头 (RM 调用)
                             ├─ SKIP path Python     = 0.50 ms (output_ids/dict 访问 × 67%)
                             └─ INTERVENE wrap × 33%  = 0.14 ms (index_add_ / flip detect)
```

校准发现的关键纠正:
- 之前以为 `apply_cpu_sync 3.36 ms` 是 GPU forward 必然 sync wait — **错了**。dummy-sync 测同样代码只 0.05 ms。3.36 ms 实际是 SIA processor 工作时间长拖慢了 vLLM async pipeline 导致下次 sync 累积, 不是不可避免的等待。
- 真正的瓶颈仍是 **RM 调用** (b2_score_call 加权 2.85 ms/token), 但**已经压到 GPU 物理下限**。

---

## 2. Tier 1 — 单卡值得做（保持 SIA 干预语义，允许换模型）

| # | 优化项 | 理论预期 | 打折后预期 | 工时 | 风险 |
|---|---|---:|---:|---:|---|
| 🥇 1 | **主 LLM Qwen3-14B FP8 量化** | +20-40 tok/s | **+10-20** | 1-2 周 | 中-高 (accuracy 掉点) |
| 🥈 2 | **换更小 Value Model (4B → 1.7B 或 0.6B)** | +5-8 tok/s | **+3-5** | 1-2 天 | 中 (alignment 退化) |
| 🥉 3 | **Speculative SIA (用前一步 reward 影响下一步)** | +5-7 tok/s | **+2-4** | 1-2 周 | 高 (干预语义微变) |

### 🥇 #1 主 LLM Qwen3-14B FP8 量化 — 单卡最大杠杆

**Why**: SIA overhead 是**绝对值** (4.67 ms/token), 跟主 LLM forward 时间无关。如果主 LLM forward 从 11.36 → 7 ms (FP8 提速 1.5-2×), throughput 从 62.4 → ~85 tok/s (overhead 占比从 29% → 39%, 但 base 涨太多, 净增 20+ tok/s)。

**怎么做**: 用 `llm-compressor` 跑 oneshot FP8_DYNAMIC 量化, 流程跟 `VM-Qwen3-4B-merged-fp8-dynamic` 一样:
- `scripts/quantize_rm_fp8_dynamic.py` 是 RM 用的 (Qwen3ForSequenceClassification), 主 LLM 是 Qwen3ForCausalLM, 量化 recipe 类似但要重写一个脚本
- 14B 比 4B 模型大 3.5×, 量化耗时 ~30 min, 显存峰值要 ~50 GB

**风险**: 14B 模型 FP8 量化对 MMLU accuracy 可能掉 1-2 pp (业界数据)。**必须** 600q ablation 严格对比 bf16 vs FP8 主 LLM 的 accuracy 是否在统计误差内。

### 🥈 #2 换更小 Value Model (4B → 1.7B 或 0.6B)

**Why**: b2_score_call p50 = 8.32 ms 是 RM 端单调用瓶颈。这 ~8 ms 里 ~5 ms 是 Qwen3-4B forward GPU 时间, ~3 ms 是 vLLM dispatch + IPC。换更小 VM 直接压 GPU 部分:
- Qwen3-1.7B forward 估算 ~2-3 ms → b2_score_call 5-6 ms → 节省 2.5 ms/call × 33% = 0.8 ms/token → **+3-5 tok/s**
- Qwen3-0.6B 更小, 但小模型 reward 质量可能退化

**怎么做**: CLAUDE.md 列了 HuggingFace 上 `Runyi-Hu/SIA/VM-Qwen3-0.6B-Base` / `VM-Qwen3-1.7B-Base` checkpoint, **不需要重训**。下载 + 用 `scripts/convert_rm_for_vllm.py` 转换即可。

**风险**: SIA 论文 §5.2 表显示 Qwen3-1.7B vs Qwen3-4B 作为 VM, 在 HEx-PHI/AlpacaEval 上 alignment 差距 ~2-3 pp。但 §5.3 weak-to-strong 实验证明小 VM 仍能 guide 大 LLM。我们的 MMLU 评测不太能 catch alignment 质量差异 (eval-report.md 已经证明 MMLU 对 SIA 干预不敏感), 建议跑 alignment benchmark (HEx-PHI 至少 100 题) 验证。

### 🥉 #3 Speculative SIA

**核心思路**: 让 RM call 跟主 LLM 下一步 forward **并行** (同 GPU 多 stream) — 用上一步算的 reward 影响下一步的 logits。语义上是 1 个 token 的 stale, 但相邻 token 的 reward 几乎一样, 对 alignment 影响小。

**预期**: 把 INTERVENE step 的 RM call 从 critical path 上 hide。理论上 +5-7 tok/s, 实际取决于多 stream 在单 GPU 上是否真能 overlap (SM 共享, memory bandwidth 共享, 不一定有效)。

**风险**:
- 干预语义改了, SIA 论文 §3 公式不再严格成立
- 必须跑 alignment benchmark 验证 quality 不退化
- 多 stream 在单 GPU 上的真实 overlap 比例存疑

---

## 3. Tier 2 — 单卡纯 SIA 优化 (不动主 LLM, 不换 VM)

| # | 优化项 | 理论预期 | 打折后预期 | 工时 | 风险 |
|---|---|---:|---:|---:|---|
| 4 | **SKIP path Python wrap 优化** | +1-2 tok/s | **+0.5-1** | 1 天 | 低 |
| 5 | **vLLM 升级 0.10 → 0.15+** | 未知 | ? | 3-5 天 | 中 (兼容性) |
| 6 | **批量 update_state 处理** | <+0.5 tok/s | <+0.3 | 半天 | 低 |

### #4 SKIP path Python wrap 优化

基于校准 SKIP path Python 开销 0.6 ms × 67% = 0.4 ms/token 可省, 具体动手点:
- `self._output_ids.get(i, [])` + `list(...)` copy 每 token 都做。SKIP 时根本不需要 output_ids, 可延迟到 INTERVENE 才取。
- `self._prompt_user.get(i, "")` 在 SKIP path 也访问, 不需要
- `self._total_steps[i] = self._total_steps.get(i, 0) + 1` dict 操作 → 改 list (前提: req_idx 稠密)
- entropy 阈值判定的 Python list comprehension 改成 numpy 向量化

**预期**: +0.5-1 tok/s, 但落到 90q 噪声内 (±0.3 tok/s)。要看到信号需要跑 3 次取均值。

### #5 vLLM 升级 0.10 → 0.15+

vLLM 0.15+ 可能在 LogitsProcessor dispatch 上有优化 (针对 1.36 ms/token 的 hook overhead)。但升级 vLLM 要重新验证 ModelRegistry / InprocClient / cudagraph capture 全部兼容性。**ROI 不确定, 工作量 3-5 天**。

### #6 批量 update_state 处理

vLLM 每次 batch update (添加/移除/重排 request) 调一次 `update_state(batch_update)`。当前实现遍历 update 字段做 dict 修改。如果改成批量处理, 可省微小开销。**预期 <+0.3 tok/s, 不优先做**。

---

## 4. Tier 3 — 不推荐 (已验证或预期为零)

| 项 | 为什么不做 |
|---|---|
| F (FP8 RM) | 已实测, throughput 没变 (62.4 vs 62.0 tok/s 在噪声内, 见 [`sia_90q_metrics_B_vs_F.md`](../../../tmp/sia_90q_metrics_B_vs_F.md)) |
| E (in-process executor) | 校准证实 vLLM dispatch overhead 极小 (1.36 ms/token 已含其中), 实测预期 ≤ +0.2 tok/s |
| C (skip detokenize/sampler) | 同 E, 微小 |
| X-7 (CUDA graph batch=5) | 已实测无用 (见 [`rm-batch5-ab-test.md`](rm-batch5-ab-test.md)) |
| 2C (复用 score 时 KV) | vLLM cache manager 难 hack, 收益 < 0.5 tok/s |
| 降低 max_model_len | 微小, MMLU thinking 可能超 2048 |
| chunked_prefill 调参 | 已是默认最优值 |

---

## 5. 关键判断 — 单卡顶到哪了？

### 5.1 保持当前模型 (Qwen3-14B + VM-Qwen3-4B), 不动 entropy/topk

**单卡顶点 ~65 tok/s** (当前 62.4 + Tier 2 累计 +2-3 tok/s)。再多就需要架构性改动。

### 5.2 允许换模型 (Tier 1)

单卡上限**完全取决于换什么**:

| 配置 | 估算 SIA throughput |
|---|---:|
| 当前 (14B bf16 + VM-4B bf16) | **62.4 tok/s** |
| + VM 换 1.7B | ~67 tok/s |
| 14B 主模型 FP8 | ~85 tok/s |
| 主模型换 Qwen3-VL-30B-A3B-Instruct (MoE 3B激活, noSIA 135) | ~100 tok/s |
| 30B-A3B + VM 1.7B + SKIP path 优化 | ~110-120 tok/s |

---

## 6. 推荐路径

按"风险递增 / 收益递增 / 投入递增"次序:

### 6.1 短期 (1-2 天) — 低风险摘果

1. **#2: VM 换 Qwen3-1.7B** — 1-2 天, +3-5 tok/s, 用 HuggingFace 现成 checkpoint
2. **#4: SKIP path Python 清理** — 1 天, +0.5-1 tok/s

短期目标: 90q **65-69 tok/s** (vs 当前 62.4), 不影响 alignment 质量 (1.7B VM 论文已验证可用)。

### 6.2 中期 (1-2 周) — 真正的杠杆

3. **#1: 主 LLM 14B FP8 量化** — 1-2 周, +10-20 tok/s, 需要 600q accuracy ablation

中期目标: **80-90 tok/s** (单卡上限)。

### 6.3 长期 (架构变更, 单卡之外)

- 多 GPU NVLink: RM 跟主 LLM 真正 overlap, 直接 hide 2.85 ms/token
- 换主模型架构 (MoE / 更小 dense)
- 这些不在本文范围

---

## 7. 决策建议

**如果目标是把当前架构推到尽头**: 做 #2 + #4 (1-2 天闭环), 拿 ~3-5 tok/s, 在 90q 上不一定看得到 (噪声 ±0.3)。

**如果目标是产品级 throughput**: 直接做 #1 (主 LLM FP8), 1-2 周, 但需要严格 accuracy ablation 通过。

**如果单卡顶点 65-70 tok/s 已经够用**: 停优化, 转产品集成 / 用户场景验证。

---

## 8. 引用

- profiling 数据来源: [`sia_90q_detailed_profiling_20260601.md`](../../../tmp/sia_90q_detailed_profiling_20260601.md)
- 校准数据来源: [`sia_calibration_summary_20260601.md`](../../../tmp/sia_calibration_summary_20260601.md)
- A+D+B baseline: [`single-gpu-engineering-roadmap-20260601.md`](single-gpu-engineering-roadmap-20260601.md)
- noSIA baselines: [`noSIA-three-model-comparison.md`](noSIA-three-model-comparison.md)
- 历史 FP8 RM 失败记录: [`vllm-rm-followup-optimizations.md`](vllm-rm-followup-optimizations.md) §G4 + 本次 F 测验证
- 关键源码:
  - SIA processor apply(): `src/sia_vllm_RM.py:apply()`
  - RM client + reward buffer: `src/sia_rm/client.py`, `src/sia_rm/qwen3_with_score.py`
  - dummy LP (校准用): `src/dummy_lp.py`
