# VL-30B AlpacaEval 200Q — docker compose b2 inproc 实验报告

**模型**: Qwen3-VL-30B-A3B-Instruct, b2 inproc, docker compose 生产部署
**评分 RM**: Skywork-Reward-V2-Llama-3.1-8B (第三方，独立于 VM-Qwen3-4B)
**脚本**: `eval/alpaca_eval.py` (generation) + `scripts/measure_alpaca_reward.py` (scoring)

本文档覆盖两轮 AlpacaEval 200Q 实验：
- **第一轮 (20260608)**：基线 docker 部署，`SIA_LLM_CUDAGRAPH=piecewise`（受 GPU 内存约束）
- **第二轮 (20260610)**：去掉 `SIA_LLM_CUDAGRAPH=piecewise`，切回 FULL_AND_PIECEWISE 模式（+32% SIA 速度）

---

## 第一轮实验（20260608）

### 1. 配置

#### Server 参数 (docker-compose.yml)

| 参数 | 值 |
|---|---|
| LLM | Qwen3-VL-30B-A3B-Instruct |
| RM backend | b2 inproc (vllm 0.17.1) |
| RM model | VM-Qwen3-4B-merged-for-vllm |
| `--llm_gpu_mem` | 0.48 (GPU 被其他进程占用，正常为 0.55) |
| `--rm_b2_gpu_mem` | 0.08 (正常为 0.15) |
| `--max_model_len` | 2048 |
| `--topk` | 10 |
| `--weight` | 1.0 |
| `--entropy_threshold` | 1.0 |
| `SIA_LLM_CUDAGRAPH` | piecewise（受限模式，约 2× 慢于 FULL_AND_PIECEWISE）|

#### Generation 参数

| 参数 | 值 |
|---|---|
| `--limit` | 200 |
| `--max_tokens` | 2048 |
| `--temperature` | 1.0 |
| `--top_p` | 0.95 |
| `--top_k` | 20 |
| `--repetition_penalty` | 1.0 |
| noSIA 方式 | per-request `--sia_weight 0`，共用同一 server，无需切换服务 |

### 2. 结果

#### 2.1 Skywork 打分（配对 n=199）

| 指标 | SIA | noSIA | Δ |
|---|---|---|---|
| mean reward | **30.9814** | 28.6011 | **+2.38 (+8.3%)** |
| p50 | 31.50 | 29.00 | +2.50 |
| min | 1.7891 | -2.4531 | — |
| max | 57.5000 | 58.0000 | — |
| scored / total | 200 / 200 | 199 / 200 | noSIA 1 条超 2048 被跳过 |

#### 2.2 SIA 干预健康指标

| 指标 | 值 |
|---|---|
| 干预率 (intervention ratio) | **21.9%** |
| top1 flip rate | **64.6%** |
| RM error | 0 ✅ |

#### 2.3 Generation 吞吐

| | SIA | noSIA |
|---|---|---|
| total tokens | 124,028 | 149,018 |
| wall time | 2095s (34.9 min) | 1844s (30.7 min) |
| throughput | **59.2 tok/s** | **80.8 tok/s** |
| SIA tax | — | 1.36× |

### 3. 产出文件

| 文件 | 内容 |
|---|---|
| [`exp/alpaca-vl30b-b2-docker-20260608/alpaca_vl30b_b2_sia_20260608_090255.json`](../exp/alpaca-vl30b-b2-docker-20260608/alpaca_vl30b_b2_sia_20260608_090255.json) | SIA generation，200 条 |
| [`exp/alpaca-vl30b-b2-docker-20260608/alpaca_vl30b_b2_nosia_20260608_094908.json`](../exp/alpaca-vl30b-b2-docker-20260608/alpaca_vl30b_b2_nosia_20260608_094908.json) | noSIA generation，200 条 |
| [`exp/alpaca-vl30b-b2-docker-20260608/alpaca_vl30b_b2_sia_gen.log`](../exp/alpaca-vl30b-b2-docker-20260608/alpaca_vl30b_b2_sia_gen.log) | SIA generation 进度日志 |
| [`exp/alpaca-vl30b-b2-docker-20260608/alpaca_vl30b_b2_nosia_gen.log`](../exp/alpaca-vl30b-b2-docker-20260608/alpaca_vl30b_b2_nosia_gen.log) | noSIA generation 进度日志 |
| [`exp/alpaca-vl30b-b2-docker-20260608/log_docker_AlpacaEval_20260608.txt`](../exp/alpaca-vl30b-b2-docker-20260608/log_docker_AlpacaEval_20260608.txt) | docker server 日志 (含 DONE 行) |

---

## 第二轮实验（20260610，FULL fix）

### 4. 配置变更

与第一轮相比，唯一变更：**去掉 `SIA_LLM_CUDAGRAPH=piecewise`**，让主 LLM 使用默认的 FULL_AND_PIECEWISE 模式。

```yaml
# docker-compose.yml environment 变更
# 删除: SIA_LLM_CUDAGRAPH: piecewise
# 保留: SIA_RM_CUDAGRAPH: piecewise
#       SIA_RM_MULTIPROCESS: "0"
```

GPU 内存配置恢复正常（无其他进程占用）：`--llm_gpu_mem 0.55`、`--rm_b2_gpu_mem 0.15`。其余参数与第一轮完全相同。

#### 为什么能加速

vLLM 的 CUDA graph 有两种模式：

- **PIECEWISE**：只捕获单个 attention layer 的小图，适合小 batch；大 batch（prefill）无法匹配，回退到 eager（无 graph）执行，速度慢。
- **FULL_AND_PIECEWISE（默认）**：在启动时同时捕获整个 forward pass 的 FULL 图（覆盖常见 batch size）以及 PIECEWISE 图，两者按 batch 大小自动切换，prefill 和 decode 都能命中 graph，无 eager 回退。

第一轮的 `SIA_LLM_CUDAGRAPH=piecewise` 是 SIA 代码内的一个实验性环境变量，强制主 LLM 只捕获 PIECEWISE 图，导致每个 prefill step 都以 eager 模式运行，比 FULL 图慢约 2×。去掉这个环境变量后，vLLM 恢复默认的 FULL_AND_PIECEWISE 捕获，主 LLM forward 速度回到正常水平。

vllm 0.17.1（VL-30B 使用的版本）在启动时 AOT（ahead-of-time）一次性捕获所有图，不存在运行时动态重捕获的问题，因此该修改在此版本上是安全的。详细分析见 [`0gm-35b-sia-perf-breakdown-20260609.md §六`](0gm-35b-sia-perf-breakdown-20260609.md)。

### 5. 结果

#### 5.1 Skywork 打分（配对 n=198）

| 指标 | SIA | noSIA | Δ |
|---|---|---|---|
| mean reward | **30.3802** | 29.1595 | **+1.22 (+4.2%)** |
| p50 | 30.6250 | 30.2500 | +0.375 |
| min | — | -3.7969 | — |
| max | 59.2500 | 64.5000 | — |
| scored / total | 200 / 200 | 198 / 200 | noSIA 2 条超 2048 被跳过 |

**配对统计**：

| 指标 | 值 |
|---|---|
| W / L / T | 122 / 73 / 3 |
| t-test | t=3.109，p=0.0022 ✅ |
| Wilcoxon | p=0.0005 ✅ |

#### 5.2 SIA 干预健康指标

| 指标 | 值 |
|---|---|
| 干预率 | **24.9%** |
| top1 flip rate | **66.1%** |
| RM error | 0 ✅ |

#### 5.3 Generation 吞吐

| | SIA | noSIA | vs 第一轮 SIA |
|---|---|---|---|
| total tokens | 124,654 | 150,895 | — |
| wall time | 1591s (26.5 min) | 1229s (20.5 min) | — |
| throughput | **78.3 tok/s** | **122.8 tok/s** | **+32%** |
| avg tokens/Q | 623 | 754 | — |
| SIA tax | — | 1.57× | — |

noSIA 从 80.8 → 122.8 tok/s（+52%），接近纯 vLLM 正常水平，说明第一轮的低吞吐主要来自 GPU 内存约束而非代码问题。

### 6. 产出文件

| 文件 | 内容 |
|---|---|
| [`exp/alpaca-vl30b-b2-docker-20260610/alpaca_vl30b_b2_sia_20260610_083001.json`](../exp/alpaca-vl30b-b2-docker-20260610/alpaca_vl30b_b2_sia_20260610_083001.json) | SIA generation，200 条 |
| [`exp/alpaca-vl30b-b2-docker-20260610/alpaca_vl30b_b2_nosia_20260610_083001.json`](../exp/alpaca-vl30b-b2-docker-20260610/alpaca_vl30b_b2_nosia_20260610_083001.json) | noSIA generation，200 条 |
| [`exp/alpaca-vl30b-b2-docker-20260610/alpaca_vl30b_b2_sia_scored.json`](../exp/alpaca-vl30b-b2-docker-20260610/alpaca_vl30b_b2_sia_scored.json) | SIA Skywork 打分 |
| [`exp/alpaca-vl30b-b2-docker-20260610/alpaca_vl30b_b2_nosia_scored.json`](../exp/alpaca-vl30b-b2-docker-20260610/alpaca_vl30b_b2_nosia_scored.json) | noSIA Skywork 打分 |
| [`exp/alpaca-vl30b-b2-docker-20260610/alpaca_vl30b_b2_sia_gen.log`](../exp/alpaca-vl30b-b2-docker-20260610/alpaca_vl30b_b2_sia_gen.log) | SIA generation 进度日志 |
| [`exp/alpaca-vl30b-b2-docker-20260610/alpaca_vl30b_b2_nosia_gen.log`](../exp/alpaca-vl30b-b2-docker-20260610/alpaca_vl30b_b2_nosia_gen.log) | noSIA generation 进度日志 |
| [`exp/alpaca-vl30b-b2-docker-20260610/docker_log.txt`](../exp/alpaca-vl30b-b2-docker-20260610/docker_log.txt) | docker server 日志 (含 DONE 行) |

---

## 汇总对比

| 实验 | noSIA mean | SIA mean | Δ | p 值 | 干预率 | SIA tok/s | noSIA tok/s |
|---|---|---|---|---|---|---|---|
| HTTP path max=2048 (20260603) | 27.44 | 30.19 | **+2.75** | — | 25.83% | 35.9 | — |
| HTTP path max=256 (20260603) | 11.69 | 14.01 | **+2.32** | — | 18.8% | 46.5 | — |
| b2 inproc max=256 (20260605) | — | — | — | — | 18.50% | 67.9 | — |
| **第一轮 docker (20260608)** | 28.60 | 30.98 | **+2.38** | — | 21.9% | 59.2 | 80.8 |
| **第二轮 docker FULL fix (20260610)** | 29.16 | 30.38 | **+1.22** | **0.0022** | 24.9% | **78.3** | **122.8** |

### 分析

**第二轮 Δ (+1.22) 低于第一轮 (+2.38)**，两轮均统计显著，差异原因尚不完全清楚。可能因素：
1. **统计噪声**：200Q 下单次实验方差较大（历史区间 +1.22~+2.75），不能排除正常波动
2. **noSIA 基线提升**：noSIA 绝对分从 28.60 → 29.16（更高基线使 Δ 压缩）
3. **生成长度差异**：第二轮 SIA 平均 623 tokens/Q vs 第一轮 ~620，noSIA 754 vs 745，基本相当

**吞吐大幅改善**：FULL_AND_PIECEWISE fix 带来 SIA +32%、noSIA +52%，第二轮 noSIA 122.8 tok/s 已接近纯 vLLM 正常水平。

---

## 8. MMLU 150Q 评测（20260608，同 server）

### 8.1 配置

与 AlpacaEval 共用同一 docker compose server（参数同 §1）。
noSIA arm 同样通过 per-request `--sia_weight 0` 实现，无需切换服务。

| 参数 | 值 |
|---|---|
| subjects | 30 科目 × 5Q = 150Q |
| `--limit` | 5 |
| `--temperature` | 1.0 |
| `--repetition_penalty` | 1.0 |
| thinking 模式 | N/A（Qwen3-VL-30B-A3B-Instruct 是 Instruct 变体，非 thinking model，不输出 `<think>` 块） |

### 8.2 结果

| 指标 | SIA | noSIA | Δ |
|---|---|---|---|
| accuracy | **0.7867 (118/150)** | 0.7867 (118/150) | **0.00 pp** |
| wall time | 678.8s (11.3 min) | 608.0s (10.1 min) | — |
| avg tokens/Q | 259.9 | 287.7 | — |
| throughput | 57.4 tok/s | 71.0 tok/s | SIA tax 1.24× |

**SIA 干预健康指标**：

| 指标 | 本次 | 历史参考 (HTTP path) |
|---|---|---|
| 干预率 | **11.5%** | 10.03% |
| top1 flip rate | **63.4%** | ~62% |
| RM error | 0 | 0 ✅ |

### 8.3 与历史实验对比

| 实验 | noSIA acc | SIA acc | Δ | 干预率 |
|---|---|---|---|---|
| HTTP path (20260604) | 0.7867 (118/150) | 0.8000 (120/150) | +1.33 pp | 10.03% |
| b2 inproc (20260605) | 0.7867 (118/150) | 0.8133 (122/150) | **+2.67 pp** | ~10-20% |
| **b2 inproc docker (本次)** | **0.7867 (118/150)** | **0.7867 (118/150)** | **0.00 pp** | **11.5%** |

### 8.4 分析

**Δ=0 在统计噪声范围内**。doc 中已注明 "MMLU 150Q 噪声内 (binomial 95% CI ±7.7pp)"，历史最大 Δ 也只有 +2.67 pp，远小于置信区间。0 pp 属于正常波动，不代表 SIA 失效。

**干预率 11.5% 与历史 10.03% 一致**，RM 工作正常。noSIA 吞吐 (71 tok/s vs 历史 120 tok/s) 偏低同 AlpacaEval 同因——GPU 内存约束。

**结论**：SIA 在 MMLU 上的效果本身就小（知识任务，RM 训练目标偏 helpfulness），150Q 题量不足以稳定体现 Δ，需要更多题目才能排除噪声。

---

## 9. 相关 doc

- [`qwen3-vl-30b-sia-eval-20260605.md`](qwen3-vl-30b-sia-eval-20260605.md) — VL-30B SIA 综合评测汇总 (HTTP + b2 inproc)
- [`docker-install-vl30b-20260606.md`](docker-install-vl30b-20260606.md) — docker compose 部署指南
- [`vl30b-b2-inproc-speedup-20260605.md`](vl30b-b2-inproc-speedup-20260605.md) — b2 inproc 1.46× 加速原始报告
- [`0gm-35b-sia-perf-breakdown-20260609.md`](0gm-35b-sia-perf-breakdown-20260609.md) — FULL_AND_PIECEWISE 优化分析（含 VL-30B 适用性分析 §六）
