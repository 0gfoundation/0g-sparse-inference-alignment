# VL-30B AlpacaEval 200Q — docker compose b2 inproc 实验报告

**日期**: 2026-06-08
**配置**: Qwen3-VL-30B-A3B-Instruct, b2 inproc, docker compose 生产部署
**评分 RM**: Skywork-Reward-V2-Llama-3.1-8B (第三方，独立于 VM-Qwen3-4B)
**脚本**: `eval/alpaca_eval.py` (generation) + `scripts/measure_alpaca_reward.py` (scoring)

---

## 1. 实验配置

### Server 参数 (docker-compose.yml)

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

### Generation 参数

| 参数 | 值 |
|---|---|
| `--limit` | 200 |
| `--max_tokens` | 2048 (受 max_model_len=2048 限制，实际有效约 1800-1900) |
| `--temperature` | 1.0 |
| `--top_p` | 0.95 |
| `--top_k` | 20 |
| `--repetition_penalty` | 1.0 |
| noSIA 方式 | per-request `--sia_weight 0`，共用同一 server，无需切换服务 |

---

## 2. 结果

### 2.1 Skywork 打分

| 指标 | SIA | noSIA | Δ |
|---|---|---|---|
| mean reward | **30.9814** | 28.6011 | **+2.38 (+8.3%)** |
| p50 | 31.50 | 29.00 | +2.50 |
| min | 1.7891 | -2.4531 | — |
| max | 57.5000 | 58.0000 | — |
| scored / total | 200 / 200 | 199 / 200 | noSIA 1 条超 2048 被跳过 |

### 2.2 SIA 干预健康指标

| 指标 | 本次 | 历史参考 (b2 inproc max=256) |
|---|---|---|
| 干预率 (intervention ratio) | **21.9%** | 18.50% |
| top1 flip rate | **64.6%** | ~60% |
| RM error | 0 | 0 ✅ |

干预率和 flip rate 均在健康范围（干预率 10–40%，flip rate 50–80%）。

### 2.3 Generation 吞吐

| | SIA | noSIA |
|---|---|---|
| total tokens | 124,028 | 149,018 |
| wall time | 2095s (34.9 min) | 1844s (30.7 min) |
| throughput | **59.2 tok/s** | **80.8 tok/s** |
| SIA tax | — | noSIA / SIA = 1.36× |

---

## 3. 与历史实验对比

| 实验 | noSIA mean | SIA mean | Δ | 干预率 | throughput SIA |
|---|---|---|---|---|---|
| HTTP path max=2048 (20260603) | +27.44 | +30.19 | **+2.75 (+10.0%)** | 25.83% | 35.9 tok/s |
| HTTP path max=256 (20260603) | +11.69 | +14.01 | **+2.32 (+19.8%)** | 18.8% | 46.5 tok/s |
| b2 inproc max=256 (20260605) | — | — | — | 18.50% | 67.9 tok/s |
| **b2 inproc docker max=2048 (本次)** | **28.60** | **30.98** | **+2.38 (+8.3%)** | **21.9%** | **59.2 tok/s** |

### 分析

**Δ (+2.38) 与历史一致**，绝对值落在 +2.32～+2.75 的历史区间内，SIA 在 b2 inproc docker 部署下效果稳定。

**相对提升 (8.3%) 低于 max=256 的 19.8%**，原因是 noSIA 基线更高（28.60 vs 11.69）——这是 noSIA 模型能力本身更强导致的，与 SIA 算法无关。

**吞吐低于历史 b2 inproc (59.2 vs 67.9 tok/s)**，以及 noSIA 低于历史纯 vLLM 水平 (80.8 vs 119-125 tok/s)，原因是本次 GPU 被其他进程占用，不得不将 `--llm_gpu_mem 0.48` (正常 0.55)、`--rm_b2_gpu_mem 0.08` (正常 0.15)，LLM KV cache 更小，batch 效率受限。

**noSIA 跳过 1 条 (too long > 2048)**，因 `--max_model_len 2048` + `--max_tokens 2048` 存在边界情况（prompt 较长时总长超限）。后续运行建议将 `--max_tokens` 调整为 1800。

---

## 4. 产出文件

| 文件 | 内容 |
|---|---|
| [`exp/alpaca-vl30b-b2-docker-20260608/alpaca_vl30b_b2_sia_20260608_090255.json`](../exp/alpaca-vl30b-b2-docker-20260608/alpaca_vl30b_b2_sia_20260608_090255.json) | SIA generation，200 条 |
| [`exp/alpaca-vl30b-b2-docker-20260608/alpaca_vl30b_b2_nosia_20260608_094908.json`](../exp/alpaca-vl30b-b2-docker-20260608/alpaca_vl30b_b2_nosia_20260608_094908.json) | noSIA generation，200 条 |
| [`exp/alpaca-vl30b-b2-docker-20260608/alpaca_vl30b_b2_sia_gen.log`](../exp/alpaca-vl30b-b2-docker-20260608/alpaca_vl30b_b2_sia_gen.log) | SIA generation 进度日志 (per-Q latency + throughput) |
| [`exp/alpaca-vl30b-b2-docker-20260608/alpaca_vl30b_b2_nosia_gen.log`](../exp/alpaca-vl30b-b2-docker-20260608/alpaca_vl30b_b2_nosia_gen.log) | noSIA generation 进度日志 |
| [`exp/alpaca-vl30b-b2-docker-20260608/log_docker_AlpacaEval_20260608.txt`](../exp/alpaca-vl30b-b2-docker-20260608/log_docker_AlpacaEval_20260608.txt) | docker server 日志 (含干预率/flip rate DONE 行) |

---

---

## 5. MMLU 150Q 评测 (同日，同 server)

### 5.1 配置

与 AlpacaEval 共用同一 docker compose server（参数同 §1）。
noSIA arm 同样通过 per-request `--sia_weight 0` 实现，无需切换服务。

| 参数 | 值 |
|---|---|
| subjects | 30 科目 × 5Q = 150Q |
| `--limit` | 5 |
| `--temperature` | 1.0 |
| `--repetition_penalty` | 1.0 |
| thinking 模式 | N/A（Qwen3-VL-30B-A3B-Instruct 是 Instruct 变体，非 thinking model，不输出 `<think>` 块） |

### 5.2 结果

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

### 5.3 与历史实验对比

| 实验 | noSIA acc | SIA acc | Δ | 干预率 |
|---|---|---|---|---|
| HTTP path (20260604) | 0.7867 (118/150) | 0.8000 (120/150) | +1.33 pp | 10.03% |
| b2 inproc (20260605) | 0.7867 (118/150) | 0.8133 (122/150) | **+2.67 pp** | ~10-20% |
| **b2 inproc docker (本次)** | **0.7867 (118/150)** | **0.7867 (118/150)** | **0.00 pp** | **11.5%** |

### 5.4 分析

**Δ=0 在统计噪声范围内**。doc 中已注明 "MMLU 150Q 噪声内 (binomial 95% CI ±7.7pp)"，历史最大 Δ 也只有 +2.67 pp，远小于置信区间。0 pp 属于正常波动，不代表 SIA 失效。

**干预率 11.5% 与历史 10.03% 一致**，RM 工作正常。noSIA 吞吐 (71 tok/s vs 历史 120 tok/s) 偏低同 AlpacaEval 同因——GPU 内存约束。

**结论**：SIA 在 MMLU 上的效果本身就小（知识任务，RM 训练目标偏 helpfulness），150Q 题量不足以稳定体现 Δ，需要更多题目才能排除噪声。

---

## 6. 相关 doc

- [`qwen3-vl-30b-sia-eval-20260605.md`](qwen3-vl-30b-sia-eval-20260605.md) — VL-30B SIA 综合评测汇总 (HTTP + b2 inproc)
- [`docker-install-vl30b-20260606.md`](docker-install-vl30b-20260606.md) — 本次使用的 docker compose 部署指南
- [`vl30b-b2-inproc-speedup-20260605.md`](vl30b-b2-inproc-speedup-20260605.md) — b2 inproc 1.46× 加速原始报告
