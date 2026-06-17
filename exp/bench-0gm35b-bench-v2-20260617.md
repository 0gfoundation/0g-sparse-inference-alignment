# 0GM-35B 压测 v2：SIA vs noSIA — 2026-06-17

## 环境

| 项目 | 值 |
|------|-----|
| 模型 | 0GM-1.0-35B-A3B-0427-SIA |
| 部署 | Docker, b2 inproc backend |
| vLLM | 0.18.0 |
| llm_gpu_mem | 0.75 |
| rm_b2_gpu_mem | 0.13 |
| max_model_len | 32768 |
| topk | 10 |
| weight | 1.0 |
| entropy_threshold | 1.0 |
| 测试脚本 | `tests/bench_35b.py` |
| rounds | 3（每档重复 3 轮） |

---

## SIA 开启（`python tests/bench_35b.py`）

### Concurrency Sweep（input≈480, max_out=304）

| Conc | Input | Output | TTFT mean | TTFT p99 | ITL mean | Req Lat | Out tok/s | Req/s |
|------|-------|--------|-----------|----------|----------|---------|-----------|-------|
| 1    | 480   | 304    | 146ms     | 181ms    | 10.9ms   | 3434ms  | 88.5      | 0.29  |
| 2    | 480   | 304    | 232ms     | 244ms    | 19.2ms   | 6036ms  | 100.7     | 0.33  |
| 4    | 480   | 304    | 241ms     | 255ms    | 24.8ms   | 7761ms  | 156.6     | 0.52  |
| 8    | 480   | 304    | 351ms     | 391ms    | 35.8ms   | 11185ms | 217.3     | 0.71  |
| 16   | 480   | 304    | 477ms     | 607ms    | 65.4ms   | 20290ms | 239.4     | 0.79  |

### Input-Length Sweep（max_out=128）

| Conc | Input | Output | TTFT mean | TTFT p99 | ITL mean | Req Lat | Out tok/s | Req/s |
|------|-------|--------|-----------|----------|----------|---------|-----------|-------|
| 2    | 480   | 128    | 234ms     | 243ms    | 15.8ms   | 2241ms  | 114.2     | 0.89  |
| 2    | 2010  | 128    | 258ms     | 280ms    | 15.8ms   | 2266ms  | 112.9     | 0.88  |
| 2    | 4060  | 128    | 286ms     | 318ms    | 18.9ms   | 2686ms  | 95.3      | 0.74  |
| 1    | 16350 | 128    | 268ms     | 292ms    | 19.8ms   | 2759ms  | 46.4      | 0.36  |
| 1    | 32610 | 128    | 335ms     | 368ms    | 29.2ms   | 3967ms  | 32.3      | 0.25  |
| 1    | 65536 | —      | SKIP (> max_model_len=32768) | | | | | |
| 1    | 94000 | —      | SKIP (> max_model_len=32768) | | | | | |

---

## SIA 关闭（`python tests/bench_35b.py --no-sia`）

### Concurrency Sweep（input≈480, max_out=304）

| Conc | Input | Output | TTFT mean | TTFT p99 | ITL mean | Req Lat | Out tok/s | Req/s |
|------|-------|--------|-----------|----------|----------|---------|-----------|-------|
| 1    | 480   | 304    | 154ms     | 166ms    | 9.0ms    | 2882ms  | 105.4     | 0.35  |
| 2    | 480   | 304    | 261ms     | 287ms    | 9.6ms    | 3173ms  | 191.5     | 0.63  |
| 4    | 480   | 304    | 261ms     | 286ms    | 10.5ms   | 3445ms  | 352.7     | 1.16  |
| 8    | 480   | 304    | 354ms     | 384ms    | 11.8ms   | 3932ms  | 617.4     | 2.03  |
| 16   | 480   | 304    | 510ms     | 637ms    | 13.7ms   | 4658ms  | 1040.3    | 3.42  |

### Input-Length Sweep（max_out=128）

| Conc | Input | Output | TTFT mean | TTFT p99 | ITL mean | Req Lat | Out tok/s | Req/s |
|------|-------|--------|-----------|----------|----------|---------|-----------|-------|
| 2    | 480   | 128    | 251ms     | 273ms    | 9.0ms    | 1395ms  | 183.3     | 1.43  |
| 2    | 2010  | 128    | 263ms     | 295ms    | 9.3ms    | 1442ms  | 177.4     | 1.39  |
| 2    | 4060  | 128    | 275ms     | 289ms    | 8.9ms    | 1404ms  | 182.1     | 1.42  |
| 1    | 16350 | 128    | 256ms     | 258ms    | 8.6ms    | 1346ms  | 95.1      | 0.74  |
| 1    | 32610 | 128    | 358ms     | 363ms    | 8.6ms    | 1414ms  | 90.5      | 0.71  |
| 1    | 65536 | —      | SKIP (> max_model_len=32768) | | | | | |
| 1    | 94000 | —      | SKIP (> max_model_len=32768) | | | | | |

---

## SIA vs noSIA 对比分析

### Concurrency Sweep — ITL 与吞吐对比

| Conc | SIA ITL | noSIA ITL | ITL 倍数 | SIA tok/s | noSIA tok/s | SIA/noSIA |
|------|---------|-----------|----------|-----------|-------------|-----------|
| 1    | 10.9ms  | 9.0ms     | 1.2×     | 88.5      | 105.4       | 84%       |
| 2    | 19.2ms  | 9.6ms     | 2.0×     | 100.7     | 191.5       | 53%       |
| 4    | 24.8ms  | 10.5ms    | 2.4×     | 156.6     | 352.7       | 44%       |
| 8    | 35.8ms  | 11.8ms    | 3.0×     | 217.3     | 617.4       | 35%       |
| 16   | 65.4ms  | 13.7ms    | 4.8×     | 239.4     | 1040.3      | 23%       |

### Input-Length Sweep — ITL 对比（concurrency=2 for 512-4K, concurrency=1 for 16K-32K）

| Input  | SIA ITL | noSIA ITL | ITL 倍数 |
|--------|---------|-----------|----------|
| 480    | 15.8ms  | 9.0ms     | 1.8×     |
| 2010   | 15.8ms  | 9.3ms     | 1.7×     |
| 4060   | 18.9ms  | 8.9ms     | 2.1×     |
| 16350  | 19.8ms  | 8.6ms     | 2.3×     |
| 32610  | 29.2ms  | 8.6ms     | 3.4×     |

### 关键结论

1. **TTFT 不受 SIA 影响**：SIA 在 decode 阶段介入，prefill 不受影响，两侧 TTFT 基本持平

2. **SIA overhead 随并发急剧放大**：
   - conc=1：ITL 仅 +21%，吞吐下降 16%，代价合理
   - conc=16：ITL 达 noSIA 的 4.8×，吞吐降至 noSIA 的 23%
   - 原因：每个 decode step 需对 batch 内所有请求调用 RM 打分（`并发 × topk` 次），RM 是瓶颈；而 noSIA 的 LLM batch 效益随并发线性扩展

3. **SIA overhead 随输入长度增大**：32K 输入时 ITL 倍数达 3.4×，原因是 RM 需处理更长的 KV cache 上下文

4. **noSIA 吞吐扩展接近线性**：conc=16 vs conc=1 吞吐比 ≈ 10×，vLLM batching 效率高

5. **建议使用场景**：对对齐效果有要求时，conc=1~2 的场景 SIA 开销可接受（16~47%）；高吞吐优先场景建议关闭 SIA 或降低 topk
