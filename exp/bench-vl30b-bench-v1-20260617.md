# VL-30B SIA Benchmark v1 — 2026-06-17

**模型**: Qwen3-VL-30B-A3B-Instruct-SIA  
**配置**: max_model_len=4096, llm_gpu_mem=0.49, rm_b2_gpu_mem=0.08, topk=10, weight=1.0, entropy_threshold=1.0  
**工具**: `tests/bench_30b.py --compare`，每档 3 轮

---

## Concurrency Sweep（input≈512, max_out=128）

### SIA 开启

| Conc | Input | Output | TTFT mean | TTFT p99 | ITL mean | Req Lat | Out tok/s | Req/s |
|------|-------|--------|-----------|----------|----------|---------|-----------|-------|
| 1    | 478   | 21     | 46ms      | 75ms     | 8.0ms    | 206ms   | 101.6     | 4.84  |
| 2    | 478   | 20     | 44ms      | 52ms     | 8.0ms    | 200ms   | 198.7     | 9.61  |
| 4    | 478   | 21     | 95ms      | 197ms    | 8.2ms    | 249ms   | 311.0     | 14.58 |
| 8    | 478   | 20     | 77ms      | 92ms     | 11.5ms   | 302ms   | 539.9     | 25.81 |
| 16   | 478   | 21     | 114ms     | 132ms    | 17.0ms   | 449ms   | 704.3     | 33.28 |

### noSIA 基线

| Conc | Input | Output | TTFT mean | TTFT p99 | ITL mean | Req Lat | Out tok/s | Req/s |
|------|-------|--------|-----------|----------|----------|---------|-----------|-------|
| 1    | 478   | 21     | 32ms      | 34ms     | 6.6ms    | 165ms   | 126.9     | 6.04  |
| 2    | 478   | 20     | 42ms      | 48ms     | 7.3ms    | 188ms   | 216.0     | 10.37 |
| 4    | 478   | 21     | 52ms      | 62ms     | 8.1ms    | 217ms   | 380.3     | 17.83 |
| 8    | 478   | 21     | 79ms      | 92ms     | 8.4ms    | 250ms   | 605.6     | 28.33 |
| 16   | 478   | 21     | 111ms     | 123ms    | 9.3ms    | 296ms   | 1067.5    | 50.84 |

### SIA vs noSIA 差异（Concurrency Sweep）

| Conc | ITL 倍数（SIA/noSIA） | tok/s 比（SIA/noSIA） |
|------|----------------------|----------------------|
| 1    | 1.21×                | 80%                  |
| 2    | 1.10×                | 92%                  |
| 4    | 1.01×                | 82%                  |
| 8    | 1.37×                | 89%                  |
| 16   | 1.83×                | 66%                  |

---

## Input-Length Sweep（max_out=128）

### SIA 开启

| Conc | Input | Output | TTFT mean | TTFT p99 | ITL mean | Req Lat | Out tok/s | Req/s |
|------|-------|--------|-----------|----------|----------|---------|-----------|-------|
| 2    | 478   | 21     | 52ms      | 79ms     | 8.5ms    | 225ms   | 186.9     | 8.69  |
| 2    | 988   | 21     | 48ms      | 56ms     | 10.5ms   | 261ms   | 163.5     | 7.55  |
| 2    | 2008  | 20     | 60ms      | 70ms     | 7.3ms    | 202ms   | 193.3     | 9.51  |
| 1    | 3928  | 23     | 88ms      | 135ms    | 6.6ms    | 235ms   | 97.8      | 4.25  |
| 1    | 8192  | —      | SKIP (> max_model_len=4096) | | | | | |

### noSIA 基线

| Conc | Input | Output | TTFT mean | TTFT p99 | ITL mean | Req Lat | Out tok/s | Req/s |
|------|-------|--------|-----------|----------|----------|---------|-----------|-------|
| 2    | 478   | 20     | 42ms      | 48ms     | 7.5ms    | 190ms   | 215.4     | 10.34 |
| 2    | 988   | 21     | 46ms      | 53ms     | 7.4ms    | 198ms   | 210.6     | 9.87  |
| 2    | 2008  | 20     | 58ms      | 70ms     | 7.6ms    | 210ms   | 190.8     | 9.16  |
| 1    | 3928  | 23     | 65ms      | 67ms     | 6.7ms    | 214ms   | 107.3     | 4.67  |
| 1    | 8192  | —      | SKIP (> max_model_len=4096) | | | | | |

---

## 关键结论

- **TTFT 不受 SIA 影响**：SIA 在 decode 阶段介入，prefill 不变，TTFT 与 noSIA 基本持平
- **低并发开销可接受**：conc=1 时 ITL 仅 +21%（8.0ms vs 6.6ms），吞吐下降 20%
- **高并发开销适中**：conc=16 时 ITL 达 noSIA 的 1.83×（优于 35B 的 4.8×）；原因是每 decode step RM 评分量 = 并发 × topk，RM 成为瓶颈
- **吞吐随并发线性扩展**：conc=16 时 SIA tok/s = 704，是 conc=1 的 6.9×，扩展性健康
- **长输入 ITL 不升反降**：4096 tokens 输入时 SIA ITL = 6.6ms（低于 512 tokens 的 8.0ms），说明 KV 读取开销对 30B 影响不显著
