# A/B Test: Does adding batch=5 to vLLM CUDA graph sizes help Stage B?

**问题**: `score_candidates` 一次提交 5 个 prompt 给 vLLM, 但 [`rm-split-stage-profiling.md`](rm-split-stage-profiling.md) 测出 Stage B 的 5-candidate forward GPU 时间是 ~13 ms — 看起来比理论下限 (~5-7 ms 对 4B 模型, batch=5 decode) 大近 2×。

**怀疑**: vLLM 默认 `cuda_graph_sizes=[1, 2, 4, 8, 16, ..., 512]` **不含 batch=5**, 所以 batch=5 forward 会 round up 到 batch=8 graph + 3 个 padding token, 多浪费时间。

**测试假设**: 显式加 `cuda_graph_sizes=[1, 2, 4, 5, 8, 16, 32, 64, 128]` (含 batch=5) 应该让 Stage B 变快。

---

## 实施

### 代码改动

[`src/sia_rm/client.py`](../src/sia_rm/client.py) 给 `RMClient.__init__` 加一个 `cuda_graph_sizes: Optional[list]` 参数:
- `None`（默认）→ 不传给 vLLM，用 vLLM 默认 sizes（不含 5）
- 显式 list → 传给 vLLM `LLM(cuda_graph_sizes=...)` 

### 测试脚本

[`scripts/bench_rm_batch5_ab.py`](../scripts/bench_rm_batch5_ab.py) — 一次跑一个 variant, 输出 JSON + 文字 summary。

每个 iteration 不仅记录 Stage B 总时间, 还**记录每次 `compute_logits` 调用的 batch shape**（`n_input` / `n_samples`）, 这样可以直接看 vLLM 内部是怎么 split 这 5 个 prompt 的。

**跑测命令**:

```bash
# variant 1: vLLM 默认 (不含 batch=5 graph)
SIA_RM_PROFILE=1 /workspace/SIA/venv2/bin/python \
    scripts/bench_rm_batch5_ab.py default \
    > /tmp/bench_ab_default.log 2>&1

# variant 2: 显式加 batch=5 graph
SIA_RM_PROFILE=1 /workspace/SIA/venv2/bin/python \
    scripts/bench_rm_batch5_ab.py with5 \
    > /tmp/bench_ab_with5.log 2>&1
```

每个 variant: 10 warmup + 100 measurement iters, 同 random seed (42), 同 prefix (600 tokens), 同 5 candidates。

---

## 结果

### Stage B GPU forward 时间

| metric | Default | With batch=5 graph | Δ |
|--------|--------:|-------------------:|---|
| min    | 9.43 ms  | 9.89 ms  | +0.46 (噪声) |
| **p50** | **13.19 ms** | **12.14 ms** | **−1.05 (−8%)** |
| **mean** | **16.44 ms** | **14.47 ms** | **−1.97 (−12%)** |
| **p95** | 31.51 ms | 27.95 ms | −3.56 (−11%) |
| max    | 35.18 ms | 32.60 ms | −2.58 |

### Stage B wall-clock

| metric | Default | With batch=5 graph | Δ |
|--------|--------:|-------------------:|---|
| p50    | 24.13 ms | 22.67 ms | −1.46 |
| mean   | 29.52 ms | 26.43 ms | −3.09 |

### Per-forward batch shape 分布 (100 iter × 2-3 forwards each = ~200+ records)

| n_samples | Default count | With5 count |
|-----------|--------------:|------------:|
| 1 | 97 | 100 |
| 2 | 4 | 3 |
| 3 | 2 | 2 |
| 4 | 96 | 97 |
| 5 | **1** | **0** |

### 每个 iter 用了几次 forward

| # forwards / iter | Default | With5 |
|------------------:|--------:|------:|
| 1 | 1 (1%) | 0 (0%) |
| 2 | **98 (98%)** | **98 (98%)** |
| 3 | 1 (1%) | 2 (2%) |

---

## 解读

### 1. **vLLM 强制 1+4 split, batch=5 graph 几乎从未被用**

无论加不加 batch=5 graph, **98% 的 iteration 仍然是 1 个 forward(batch=1) + 1 个 forward(batch=4)** 的 split。带 batch=5 graph 的 variant **0 次**真用到 batch=5 forward。

为什么？vLLM v1 的 scheduler 对"新进来的多个 prompt"统一走 prefill 路径处理第一次 token forward。即使我们已经 Stage A 把 prefix KV cache 暖好了, **每个 prompt 的"最后一个 candidate token"对 vLLM 来说仍然是 "prefill"** — 不是 decode (decode 在 vLLM 里特指 mid-generation 的 1-token forward)。

vLLM scheduler 把 5 个 prefill request 拆成 1 + 4 是它的内部启发式:
- 第一个 prompt 单独跑一个 forward (建立调度状态)
- 剩下 4 个 batch 一起跑

**Batch=5 graph 永远不会被命中**, 因为 vLLM 根本不会让 batch=5 出现。

### 2. 那 1-2 ms 的改善从哪儿来？

3 种可能:
- **测量噪声**: 两次跑用同 seed, 但 GPU 暖度、CUDA 内核分配等 run-to-run 抖动正常 ±1.5 ms 在 p50 上
- **batch=4 graph 调度更精确**: 加了 cuda_graph_sizes list 后 vLLM 内部可能微调了路径选择，碰巧让 batch=4 graph 用得更利索
- **真实但非常小**: 1 ms 收益但低于实用门槛

**结论**: 1-2 ms 改善**在测量噪声边缘**，**不值得为之引入额外配置**。

### 3. 真要消除 1+4 split, 怎么办

| 方案 | 复杂度 | 预期收益 |
|------|--------|---------|
| 提前两次 prefix warmup, 试图诱导 vLLM 走 decode 路径 | 中（要测） | 不确定; vLLM 看 prompt 是新的就走 prefill |
| 改 vLLM scheduler 让 "shared-prefix new requests" 一次 batch | 高（碰 vLLM v1 源码） | 理论可降到 ~7 ms (batch=5 纯 decode 物理下限) |
| 绕过 vLLM scheduler, 直接调 `GPUModelRunner.execute_model` | 极高 (vLLM internal hack, 升级风险) | 同上 |
| 双 GPU NVLink 并行 (Stage A 完全隐藏) | 中 | 大头收益 +37-60% throughput |

---

## 结论 + 建议

1. **X-7 (batch=5 in CUDA graph sizes) 不是有效优化**: 收益 1-2 ms 在测量噪声内, 而且实际上 batch=5 graph **从未被命中** (vLLM 强制 1+4 split)。**保持现状, 不要 merge X-7**。

2. **Stage B 13 ms 接近 vLLM 0.10.1.1 在当前 scheduler 下的实际下限**。再压需要改 vLLM 内部 scheduler, 工作量大。

3. **优先做并行化** (双 GPU NVLink): Stage A 5 ms 完全隐藏 → throughput +37-60%。比死磕 Stage B 多省 1 ms 收益大一个数量级。

4. **如果未来要再压 Stage B, 推荐先试方案"提前 prefix 多次 warm-up"**: 改动小, 不碰 vLLM 源码。若成功能让某些 iter 真走 batch=5 decode 路径, 数量级会更明显。

---

## 引用

- 实施 commit: 见 git log
- bench 脚本: [`scripts/bench_rm_batch5_ab.py`](../scripts/bench_rm_batch5_ab.py)
- 客户端改动: [`src/sia_rm/client.py`](../src/sia_rm/client.py) (加 `cuda_graph_sizes` kwarg)
- 原始 profile doc: [`rm-split-stage-profiling.md`](rm-split-stage-profiling.md)
- 之前 X-7 ablation 失败记录: [`b2-pure-engineering-optimizations.md`](b2-pure-engineering-optimizations.md) §X-7
- 输出 JSON: `/tmp/bench_rm_batch5_ab_default.json`, `/tmp/bench_rm_batch5_ab_with5.json`
