# 0GM-35B SIA vs noSIA 并发压测 — 2026-06-17

## 环境

| 项目 | 值 |
|------|-----|
| 模型 | 0GM-1.0-35B-A3B-0427 |
| 部署 | Docker, b2 inproc backend |
| GPU | 单卡 143771 MiB |
| vLLM | 0.18.0 |
| llm_gpu_mem | 0.75 |
| rm_b2_gpu_mem | 0.13 |
| max_model_len | 32768 |
| topk | 10 |
| weight | 1.0 |
| entropy_threshold | 1.0 |
| 测试脚本 | `tests/bench_35b.py` |
| 测试 prompt | "Please list all 8 planets of the solar system..." |
| max_tokens | 300 |
| 并发度 | 1 / 2 / 4 |
| 每并发度轮数 | 3 |

---

## 原始结果

### noSIA（`--no-sia`，sia_weight=0，纯 vLLM）

```
── [noSIA] 并发度 1（3 请求 = 1 并发 × 3 轮）──
  请求总数=3  成功=3  失败=0
  TTFT (ms)       min=56  p50=67  p95=103  max=103
  端到端延迟(ms)  min=2521  p50=2644  p95=2654  max=2654
  单请求吞吐(tok/s)  avg=115.2  median=113.5
  聚合吞吐(tok/s)   115.1  (wall=7.8s, total_comp=900)

── [noSIA] 并发度 2（6 请求 = 2 并发 × 3 轮）──
  请求总数=6  成功=6  失败=0
  TTFT (ms)       min=87  p50=126  p95=140  max=140
  端到端延迟(ms)  min=2942  p50=3153  p95=3188  max=3188
  单请求吞吐(tok/s)  avg=97.1  median=95.2
  聚合吞吐(tok/s)   193.8  (wall=9.3s, total_comp=1800)

── [noSIA] 并发度 4（12 请求 = 4 并发 × 3 轮）──
  请求总数=12  成功=12  失败=0
  TTFT (ms)       min=90  p50=99  p95=125  max=125
  端到端延迟(ms)  min=3078  p50=3274  p95=3457  max=3457
  单请求吞吐(tok/s)  avg=92.0  median=91.6
  聚合吞吐(tok/s)   366.8  (wall=9.8s, total_comp=3600)

── 限流探测（16 并发 × 1 轮，短 prompt）──
  wall-clock: 0.3s
  状态码分布: {200: 16}
  延迟(ms)   min=240  p50=250  max=252
  ✅ 无 429
```

### SIA（默认，entropy_threshold=1.0，干预率约 5-8%）

```
── [SIA] 并发度 1（3 请求 = 1 并发 × 3 轮）──
  请求总数=3  成功=3  失败=0
  TTFT (ms)       min=56  p50=59  p95=100  max=100
  端到端延迟(ms)  min=2928  p50=2932  p95=2985  max=2985
  单请求吞吐(tok/s)  avg=101.8  median=102.3
  聚合吞吐(tok/s)   101.7  (wall=8.8s, total_comp=900)

── [SIA] 并发度 2（6 请求 = 2 并发 × 3 轮）──
  请求总数=6  成功=6  失败=0
  TTFT (ms)       min=85  p50=127  p95=144  max=144
  端到端延迟(ms)  min=3480  p50=3734  p95=3756  max=3756
  单请求吞吐(tok/s)  avg=82.1  median=80.3
  聚合吞吐(tok/s)   164.0  (wall=11.0s, total_comp=1800)

── [SIA] 并发度 4（12 请求 = 4 并发 × 3 轮）──
  请求总数=12  成功=12  失败=0
  TTFT (ms)       min=85  p50=98  p95=136  max=136
  端到端延迟(ms)  min=4371  p50=4995  p95=5552  max=5552
  单请求吞吐(tok/s)  avg=60.9  median=60.1
  聚合吞吐(tok/s)   241.2  (wall=14.9s, total_comp=3600)

── 限流探测（16 并发 × 1 轮，短 prompt）──
  wall-clock: 0.2s
  状态码分布: {200: 16}
  延迟(ms)   min=200  p50=207  max=211
  ✅ 无 429
```

---

## 对比汇总

| 并发度 | noSIA 聚合吞吐 | SIA 聚合吞吐 | SIA 开销 | noSIA 延迟 p50 | SIA 延迟 p50 | 延迟增幅 |
|--------|--------------|------------|---------|--------------|------------|---------|
| 1 | 115.1 tok/s | 101.7 tok/s | -12% | 2644 ms | 2932 ms | +11% |
| 2 | 193.8 tok/s | 164.0 tok/s | -15% | 3153 ms | 3734 ms | +18% |
| 4 | 366.8 tok/s | 241.2 tok/s | -34% | 3274 ms | 4995 ms | +53% |

---

## 分析

1. **TTFT 几乎不受影响**：SIA 在 decode 阶段介入，不影响 prefill，TTFT 与 noSIA 基本持平（p50 均在 60-130ms）。

2. **单并发 SIA 开销 12%**：entropy_threshold=1.0 下约 5-8% 的 token 触发干预，RM 计算量较少，开销低。相比之前 AlpacaEval 记录的 42% 开销（66 vs 114 tok/s），本次更轻——原因是问题类型不同导致模型熵分布不同，干预率更低。

3. **高并发 SIA 开销放大**：并发 4 时开销升至 34%。原因：每个 decode step，RM 评分的候选 token 总数 = `并发数 × topk`，RM 计算压力随并发线性增长；而 LLM decode 有 batch 加速，两者增速不对称。

4. **聚合吞吐随并发线性增长（noSIA）**：115→194→367，vLLM batch 效率高，符合预期。SIA 下增长放缓（102→164→241），RM 成为瓶颈。

5. **无限流（429）**：16 并发全部 200，推理服务依赖 vLLM 内部队列，无速率限制机制。
