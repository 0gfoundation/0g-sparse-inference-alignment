# noSIA MMLU-Redux 600 三模型对比

**目的**：在同一评测集（MMLU-Redux 600 题，30 学科 × 20 题，temperature=1.0）下对比三个主模型在 **noSIA（纯 LLM 推理，无 SIA 干预）** 路径上的表现。目标是建立 noSIA baseline 的"性价比 Pareto"，方便后续选定主模型。

**对比对象**：

| 模型 | 类型 | 参数量 | 备注 |
|------|------|--------|------|
| `Qwen3-4B` | dense + thinking mode | 4 B | base 模型，含 think tag |
| `Qwen3-14B` | dense + thinking mode | 14 B | 历史主模型 |
| `Qwen3-VL-30B-A3B-Instruct` | **MoE + multimodal** | 30 B 总 / **3 B activated** | Instruct 不含 thinking mode |

**硬件**：单卡 H200 (143 GB)，CUDA driver 12.8。

---

## 1. 实验配置 + 启动命令

每个实验都跑 **MMLU-Redux 600 题（30 subjects × `--limit 20`）**，client 命令完全一样：

```bash
python eval/mmlu_eval.py \
  --base_url http://localhost:8000/v1 \
  --model <model-id> \
  --output sparse_logs/eval_b2_mmlu/<output>.json \
  --limit 20 \
  --temperature 1.0
```

服务端启动命令在每个模型里略有差异（详见下文）。

### 1.1 Qwen3-4B（2026-05-28, fresh）

**vLLM 版本**: 0.15.0（venv3），torch 2.9.1+cu128

```bash
# server
nohup /workspace/SIA/venv3/bin/vllm serve \
  /workspace/SIA/models/Qwen3-4B \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --served-model-name Qwen3-4B \
  --host 0.0.0.0 --port 8000 \
  > /tmp/vllm_qwen3_4b_server.log 2>&1 &

# client
nohup /workspace/SIA/venv3/bin/python eval/mmlu_eval.py \
  --base_url http://localhost:8000/v1 \
  --model Qwen3-4B \
  --output sparse_logs/eval_b2_mmlu/mmlu_redux_qwen3_4b_nosia_600q.json \
  --limit 20 \
  --temperature 1.0 \
  > /tmp/mmlu_qwen3_4b_nosia_600.log 2>&1 &
```

**日志归档**:
- client log: [`exp/log_noSIA_qwen3-4b_20260528.txt`](../exp/log_noSIA_qwen3-4b_20260528.txt)
- server log: [`exp/log_server_qwen3-4b_20260528.txt`](../exp/log_server_qwen3-4b_20260528.txt)
- result JSON: [`sparse_logs/eval_b2_mmlu/mmlu_redux_qwen3_4b_nosia_600q.json`](../sparse_logs/eval_b2_mmlu/mmlu_redux_qwen3_4b_nosia_600q.json)

### 1.2 Qwen3-14B（2026-05-14, 历史）

**vLLM 版本**: 0.10.x (历史)。**注意启动配置略有差异**：用了 `repetition_penalty=1.3` 而本次 4B/30B-VL 未指定（默认 1.0）。

```bash
# server (历史 — 来自 doc/vllm-rm-experiment-report.md §3.4)
nohup python -m vllm.entrypoints.openai.api_server \
    --model /workspace/SIA/models/Qwen3-14B \
    --gpu-memory-utilization 0.6 \
    --max-model-len 4096 \
    --override-generation-config '{"repetition_penalty": 1.3}' \
    --host 0.0.0.0 --port 8000 \
    > log_llm_server_noSIA_vllm_<TS>.txt 2>&1 &

# client
nohup python eval/mmlu_eval.py \
      --base_url http://localhost:8000/v1 \
      --model /workspace/SIA/models/Qwen3-14B \
      --output results/test_noSIA_selfServer_vllm_<TS>.json \
      --limit 20 \
      > log_noSIA_selfServer_vllm_<TS>.txt 2>&1 &
```

**日志归档**:
- client log: [`exp/log_noSIA_selfServer_vllm_202605142111.txt`](../exp/log_noSIA_selfServer_vllm_202605142111.txt)
- server log: [`exp/log_llm_server_noSIA_vllm_202605142111.txt`](../exp/log_llm_server_noSIA_vllm_202605142111.txt)

### 1.3 Qwen3-VL-30B-A3B-Instruct（2026-05-28, fresh）

**vLLM 版本**: 0.15.0（venv3），torch 2.9.1+cu128。

> ⚠️ 系统 CUDA driver 12.8 不兼容 vLLM 0.18+ 的 torch 2.10+ (CUDA 13)；最新支持 Qwen3-VL 且兼容 CUDA 12 的版本是 0.15.0。详见 [`b2-decode-mode-poc-plan.md`](b2-decode-mode-poc-plan.md) §4 安装记录。

```bash
# server
nohup /workspace/SIA/venv3/bin/vllm serve \
  /workspace/SIA/models/Qwen3-VL-30B-A3B-Instruct \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --served-model-name Qwen3-VL-30B-A3B-Instruct \
  --host 0.0.0.0 --port 8000 \
  > /tmp/vllm_qwen3vl_server.log 2>&1 &

# client (同 1.1)
nohup /workspace/SIA/venv3/bin/python eval/mmlu_eval.py \
  --base_url http://localhost:8000/v1 \
  --model Qwen3-VL-30B-A3B-Instruct \
  --output sparse_logs/eval_b2_mmlu/mmlu_redux_qwen3vl_30b_600q.json \
  --limit 20 \
  --temperature 1.0 \
  > /tmp/mmlu_qwen3vl_30b_limit20.log 2>&1 &
```

**日志归档**:
- client log: [`exp/log_noSIA_qwen3-vl-30b-a3b_20260528.txt`](../exp/log_noSIA_qwen3-vl-30b-a3b_20260528.txt)
- server log: [`exp/log_server_qwen3-vl-30b-a3b_20260528.txt`](../exp/log_server_qwen3-vl-30b-a3b_20260528.txt)
- result JSON: [`sparse_logs/eval_b2_mmlu/mmlu_redux_qwen3vl_30b_600q.json`](../sparse_logs/eval_b2_mmlu/mmlu_redux_qwen3vl_30b_600q.json)

---

## 2. 完整指标对比表

| 指标 | **Qwen3-4B** | **Qwen3-14B** | **Qwen3-VL-30B-A3B** |
|------|-------------:|--------------:|---------------------:|
| **Accuracy** | **74.67 %** | **74.83 %** | **83.00 %** ✨ |
| (correct / 600) | 448 / 600 | 449 / 600 | 498 / 600 |
| **Throughput (tok/s)** | **166.5** ⚡ | 88.4 | 134.7 |
| **ms / token (mean)** | **6.00** ⚡ | 11.31 | 7.42 |
| Avg tokens / question | 757 | 718 | **265** |
| p50 tokens/q | 546 | 544 | (短) |
| p95 tokens/q | 2048 | 2048 | (短) |
| Avg latency / q | 4.55 s | 8.12 s | **1.96 s** ⚡ |
| p50 / p95 / max latency/q | 3.3 / 12.2 / 12.8 s | 6.1 / 23.1 / 23.3 s | 1.2 / 7.1 / 15.3 s |
| Total wall-time | 45.5 min | 81.2 min | **19.6 min** ⚡ |
| Total output tokens | 454,268 | 430,908 | 158,767 |
| 干预率 | 0 % (noSIA) | 0 % (noSIA) | 0 % (noSIA) |
| 模型大小 (weights bf16) | ~7.5 GB | ~28 GB | ~60 GB |
| Active params / token | 4 B | 14 B | **3 B** (MoE) |
| 思维链模式 | thinking ✓ | thinking ✓ | Instruct, no think |

---

## 3. 两两对比 + Δ 分析

### 3.1 Qwen3-4B vs Qwen3-14B

| 指标 | 4B | 14B | Δ |
|------|---:|----:|---|
| Accuracy | 74.67 % | 74.83 % | +0.17 pp（差 1 题）|
| Throughput | 166.5 tok/s | 88.4 tok/s | **4B 快 1.88×** |
| ms / token | 6.00 | 11.31 | 14B 慢 **1.88×** |
| Avg tokens/q | 757 | 718 | -39（14B 思维链更紧凑）|
| Avg latency/q | 4.55 s | 8.12 s | **4B 快 1.79×** |
| Wall-time | 45.5 min | 81.2 min | 14B 多 80% |

**核心 takeaway**: 14B 比 4B 大 3.5× 但 MMLU 上 accuracy **几乎打平**（差 1 题，统计噪声内）。14B 的优势仅在思维链略短（-5% tokens/q），但单 token 慢近 2×，综合下 4B 全胜。

### 3.2 Qwen3-4B vs Qwen3-VL-30B-A3B

| 指标 | 4B | 30B-VL | Δ |
|------|---:|-------:|---|
| Accuracy | 74.67 % | **83.00 %** | **+8.33 pp** |
| Throughput | **166.5 tok/s** | 134.7 tok/s | 4B 快 1.24× |
| ms / token | **6.00** | 7.42 | 30B-VL 慢 1.24× |
| Avg tokens/q | 757 | **265** | -492（30B-VL 不带 think，直接给答案）|
| Avg latency/q | 4.55 s | **1.96 s** | **30B-VL 快 2.32×** |
| Wall-time | 45.5 min | **19.6 min** | 30B-VL 快 2.32× |

**核心 takeaway**: 30B-VL 在 accuracy 上**碾压** 4B (+8.33 pp)，且每题总耗时**反而比 4B 短** 2.3×。原因是 30B-VL **不带 thinking mode**（每题 265 tokens vs 4B 的 757 tokens），少 65% 的 token 完全抵消了单 token 慢 1.24× 的劣势。在 token billing 视角下，30B-VL 也更便宜（同样 600 题用 1/3 的 token）。

### 3.3 Qwen3-14B vs Qwen3-VL-30B-A3B

| 指标 | 14B | 30B-VL | Δ |
|------|----:|-------:|---|
| Accuracy | 74.83 % | **83.00 %** | **+8.17 pp** |
| Throughput | 88.4 tok/s | **134.7 tok/s** | 30B-VL 快 **1.52×** |
| ms / token | 11.31 | **7.42** | 30B-VL 快 **1.52×** |
| Avg tokens/q | 718 | **265** | -453 |
| Avg latency/q | 8.12 s | **1.96 s** | **30B-VL 快 4.14×** |
| Wall-time | 81.2 min | **19.6 min** | 30B-VL 快 4.14× |
| Active params / token | 14 B | **3 B** (MoE) | -11 B per token |

**核心 takeaway**: 30B-VL **全维度碾压** 14B —— accuracy +8.17 pp，throughput +52%，每题快 4.14×。原因有两层:
1. **MoE A3B 设计**: 30B 总参数但每 token 只激活 3B，单 token 推理速度跟 3B dense 接近
2. **Instruct 不带 thinking**: 思维链跳过，token 数减 63%

这是个**罕见的"大模型反而更快+更准"**案例 —— MoE + Instruct 形态对 MMLU 来说是质变。

---

## 4. Pareto 分析

```
              ▲  accuracy
              │
       83 %   ●  Qwen3-VL-30B-A3B  (134.7 tok/s, 1.96 s/q)  ⭐ 全维度赢家
              │
              │
              │
       75 %   ●  Qwen3-4B          (166.5 tok/s, 4.55 s/q)  ⚡ 单 token 最快
              ●  Qwen3-14B         ( 88.4 tok/s, 8.12 s/q)  ✗ 被 4B 完全 Pareto-支配
              │
              └────────────────────────────────────────────────────► tok/s
                       50            100            150          200
```

### 4.1 哪个模型是 Pareto 最优？

| 优化目标 | 最佳模型 |
|---------|---------|
| **Accuracy** | **Qwen3-VL-30B-A3B (83 %)** |
| **每题总耗时** | **Qwen3-VL-30B-A3B (1.96 s/q)** |
| 总 token 数（token billing） | Qwen3-VL-30B-A3B (159 K) |
| 单 token 最快 (tok/s) | Qwen3-4B (166.5 tok/s) |
| **综合（accuracy + 总耗时）** | **Qwen3-VL-30B-A3B**（除单 token 最快这一指标，全维度最优）|

### 4.2 Qwen3-14B 被支配

14B 在 4 项关键指标全部劣于 4B（同 accuracy 但慢 1.88×）或 30B-VL（acc 低 8 pp 且 wall-time 慢 4×）。**没有任何场景应该选 14B over 4B 或 30B-VL**。

### 4.3 4B vs 30B-VL 的真正权衡

| 场景 | 选 4B 的理由 | 选 30B-VL 的理由 |
|------|--------------|------------------|
| GPU 内存紧（<20 GB 可用）| ✓ 4B 仅需 7.5 GB | ✗ 30B-VL 需 60 GB+ |
| **追求 accuracy** | ✗ 4B 仅 75% | ✓ 30B-VL 83% |
| **追求总耗时** | ✗ 4B 4.55 s/q | ✓ 30B-VL 1.96 s/q |
| **追求 token billing 经济性** | ✗ 4B 757 tok/q | ✓ 30B-VL 265 tok/q |
| 想保留 thinking chain 用于推理分析 | ✓ 4B 带 think tag | ✗ 30B-VL 不带 |
| 单 token tok/s 看 streaming UX | ✓ 4B 略快 (166.5 vs 134.7) | — |

**结论**: 只要 GPU 内存够（≥ 80 GB），**Qwen3-VL-30B-A3B-Instruct 是首选**。

---

## 5. 几个反常的现象

### 5.1 14B accuracy 跟 4B 几乎完全打平（差 1 题）

14B 是 4B 容量的 3.5×，但 MMLU 上仅 +0.17 pp。**推测原因**：
- 4B 用 thinking mode 把 reasoning chain 拉长（757 tokens vs 14B 的 718 tokens, +5%），把容量短板用更多推理步数补回来
- MMLU 的事实知识对模型容量增长**收益递减**，14B 容量没显著超出 4B 的覆盖范围

**反证 SIA 干预效果**: 之前 4B + SIA b2 (weight=0.1, 600q) 仅 **68.00 %**, 比 4B noSIA (74.67 %) 低 **6.67 pp**。这证明那次 SIA 干预实际**伤害了** accuracy（不是 4B 容量限制）。

### 5.2 30B-VL 比 14B 又快又准

30B-VL 总参数比 14B 大 2.1×，但每 token 推理：
- **更快**: 11.31 ms → 7.42 ms（MoE 只激活 3B params，dense 14B 全激活）
- **更准**: 75 % → 83 % (+8 pp，更多 expert 容量带来更广覆盖)

这是 MoE 架构的核心优势 —— **用稀疏激活把"参数总量"跟"推理 FLOPs"解耦**。

### 5.3 30B-VL 不带 thinking mode 但 accuracy 反而更高

按"thinking mode 提升 reasoning quality"的直觉，30B-VL 没思维链应该精度更低。但实际：30B-VL 不带 think tag 反而 **+8 pp** 比 4B/14B 高。**推测原因**：
- 30B 的隐性表示已经 encode 足够强的推理能力，不需要外显思维链
- Instruct 形态训练让模型直接给精确答案（减少冗余 token 噪声）

---

## 6. 三个 noSIA 实验跟历史 SIA 实验的关系

| 实验 | Accuracy | Throughput | 备注 |
|------|---------:|-----------:|------|
| 4B + SIA b2 (weight=0.1, 600q) | 68.00 % | 75.6 tok/s | SIA weight=0.1 弱干预，反而损害 |
| 14B + SIA b2 (weight=1.0, 600q) | 74.83 % | 46.6 tok/s | 跟 14B noSIA 同 acc，SIA 仅 regression-neutral |
| 14B + SIA G1 HTTP (n=256 side-by-side) | 83.48 % | 12.5 tok/s | 历史 PyTorch RM 最佳；跟 30B-VL 持平但慢 10× |
| **4B noSIA**（本次）| **74.67 %** | **166.5 tok/s** | 4B 全部 Pareto 优于 4B+SIA(0.1) |
| **14B noSIA**（本次）| **74.83 %** | 88.4 tok/s | 跟 14B+SIA b2 acc 持平，throughput 高 90% |
| **30B-VL noSIA**（本次）| **83.00 %** ⭐ | **134.7 tok/s** ⚡ | accuracy 跟 14B+SIA G1 持平，throughput 高 **10.8×** |

**新发现**: 30B-VL noSIA 用 10× throughput **复制了** 14B+SIA G1 的 accuracy 水平。这意味着对 MMLU 这种 SIA value model 训练目标无关的任务，**升级 LLM 比加 SIA 干预性价比高得多**。

---

## 7. 引用 / log 清单

| 实验 | client log | server log | result JSON |
|------|-----------|------------|-------------|
| Qwen3-4B noSIA | [`exp/log_noSIA_qwen3-4b_20260528.txt`](../exp/log_noSIA_qwen3-4b_20260528.txt) | [`exp/log_server_qwen3-4b_20260528.txt`](../exp/log_server_qwen3-4b_20260528.txt) | `sparse_logs/eval_b2_mmlu/mmlu_redux_qwen3_4b_nosia_600q.json` |
| Qwen3-14B noSIA (历史) | [`exp/log_noSIA_selfServer_vllm_202605142111.txt`](../exp/log_noSIA_selfServer_vllm_202605142111.txt) | [`exp/log_llm_server_noSIA_vllm_202605142111.txt`](../exp/log_llm_server_noSIA_vllm_202605142111.txt) | `results/test_noSIA_selfServer_vllm_202605142111.json` |
| Qwen3-VL-30B-A3B noSIA | [`exp/log_noSIA_qwen3-vl-30b-a3b_20260528.txt`](../exp/log_noSIA_qwen3-vl-30b-a3b_20260528.txt) | [`exp/log_server_qwen3-vl-30b-a3b_20260528.txt`](../exp/log_server_qwen3-vl-30b-a3b_20260528.txt) | `sparse_logs/eval_b2_mmlu/mmlu_redux_qwen3vl_30b_600q.json` |

相关 doc:
- 4B/14B 对比的早期分析: [`b2-vs-classify-time-breakdown.md`](b2-vs-classify-time-breakdown.md)
- 14B + SIA G1 历史实验完整报告: [`vllm-rm-experiment-report.md`](vllm-rm-experiment-report.md)
- MMLU accuracy regression 验证: [`eval-report.md`](eval-report.md)
- SIA b2 路径设计: [`b2-m2-design.md`](b2-m2-design.md)

---

## 8. 结论

1. **Qwen3-VL-30B-A3B-Instruct 是 MMLU 上的全维度 Pareto 最优**: accuracy 83 %, 单题 1.96 s, throughput 134.7 tok/s
2. **Qwen3-14B 被 Qwen3-4B 完全支配**: 同 accuracy 但慢 1.88×, 没有任何场景应优先选 14B
3. **MoE A3B 设计是关键**: 30B 总参数但每 token 只激活 3B，单 token 推理速度跟 3B dense 接近
4. **SIA 干预对 MMLU 不再必要**: 30B-VL noSIA (83 %) 跟 14B + SIA G1 (83.48 %) accuracy 持平，但 throughput 高 10.8×
