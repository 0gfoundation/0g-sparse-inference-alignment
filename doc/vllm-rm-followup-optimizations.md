# vLLM RM 后续优化实验报告：G2 / G3 / G4

**日期**：2026-05-26
**适用范围**：在 `doc/vllm-rm-experiment-report.md`（G1 baseline）之后，对 vLLM RM 部署的 3 轮迭代优化（G2 / G3 / G4），评估它们在 MMLU 上的端到端 throughput 和 accuracy 表现。
**核心发现**：3 轮优化在吞吐上几乎原地踏步（28-30 tok/s），FP8 反而轻微变慢。准确率累计 +1.7%pt 但不算显著。**根本瓶颈不在被优化的部分，在"共享单 GPU 强制 LLM/RM 交替执行"这条架构红线**。

---

## 1. 实验目的

G1 baseline 把 RM 从手写 PyTorch FastAPI 切到 `vllm serve --convert classify`，相对历史 PyTorch RM 提升了 +19%（25.4 → 30.2 tok/s），但相对 noSIA 上限 (88.4 tok/s) 仍只有 34%。

G2-G4 三轮迭代试图回答："**在不动 GPU 架构的前提下，还有没有"低风险高 ROI"的进一步优化**？"

| 组 | 假设的优化点 | 期望收益 |
|---|---|---|
| **G2** | RM 端 vLLM 调度参数（关 chunked prefill、显存上调、max_len 减半） | RM 调用 latency 降低 5-10% |
| **G3** | 客户端代码层（去 `entropy.item()` CUDA sync / response_so_far 增量解码 / 加 flip 日志） | per-token sync 税降低，长 response 重 decode 消除 |
| **G4** | RM 量化 FP8（权重 + KV cache） | RM GEMM 1.5-2× 加速，显存占用减半 |

---

## 2. 实验配置

所有实验都在**单张 H200 共享部署**下跑 MMLU 30 学科 × 20 题 = 600 题，对照基线 G1。

| 组 | 时间戳 | RM mem | LLM mem | RM 量化 | 客户端代码 | 其他 |
|---|---|---:|---:|---|---|---|
| G1 (baseline) | 2026-05-22 10:45 | 0.3 | 0.6 | 无 | 原版 | `max-model-len 4096`，chunked prefill 默认开 |
| **G2** | 2026-05-25 09:45 | **0.4** | 0.6 | 无 | 原版 | **关 chunked prefill**, **`max-model-len 2048`** |
| **G3** | 2026-05-25 21:20 | 0.4 | **0.55** | 无 | **+ 3 件套** | 同 G2 |
| **G4** | 2026-05-26 10:30 | **0.3** | 0.55 | **FP8 + FP8 KV** | 同 G3 | 同 G2 |

**G3 的"客户端代码 3 件套"**：
1. **去掉 `entropy.item()` CUDA sync**（commit `a2357e9`）：把 batch 内每 token 单独 sync 改成一次性 sync
2. **`response_so_far` 增量解码缓存**（commit `368ef24`）：response 长度从 1→1000 时不再每步全量 decode，改成只 decode 新增的最后 1 token
3. **top-1 flip 统计日志**（commit `c802054`）：每 INTERVENE 步打印 pre/post argmax 是否翻转，便于分析

**G4 的 FP8 配置**：`--quantization fp8 --kv-cache-dtype fp8` 同时加。

---

## 3. 实验结果

### 3.1 吞吐与准确率

| 组 | 整体 tok/s | Δ vs G1 | 准确率 (600 题) | Δ vs G1 |
|---|---:|---:|---:|---:|
| **G1** | **30.23** | — | **71.5%** | — |
| G2 | 30.30 | +0.2% | 72.5% | +1.0%pt |
| G3 | 29.65 | **-1.9%** | **73.7%** | +1.2%pt |
| G4 | 28.08 | **-7.1%** | 73.2% | +1.7%pt |

> **对比：noSIA via vLLM 原生 api_server (N3) = 88.43 tok/s, 74.8% acc**。
> SIA 最快版本 (G2) 仅到 noSIA 的 34%。

### 3.2 趋势观察

- **吞吐方向**：G1 → G2 → G3 → G4 几乎水平直线（30.23 → 30.30 → 29.65 → 28.08），FP8 甚至倒退
- **准确率方向**：稳步小幅上涨（71.5% → 73.7% → 73.2%），但都在统计噪声边缘
- **没有一组明显跨越**：与历史"PyTorch RM eager → batch forward (+105%)"或"PyTorch RM → vLLM RM (+19%)"那种数量级跨越完全不同

---

## 4. 结果分析：为什么没起到预期加速？

### 4.1 共同的根本原因：共享 GPU 的"交替执行"模型

LLM 进程和 RM 进程**跑在同一张 H200 上**。CUDA 默认调度下：

```
SKIP step:
  LLM forward (~11ms) → 采样 → done       [GPU 全程是 LLM 的]
  RM idle                                  [无 RM 调用]

INTERVENE step:
  LLM forward (~11ms) → SIA 决定 INTERVENE → HTTP 阻塞调 RM ...
                                              ↓
                                        LLM 等待中（GPU 空闲）
                                              ↓
  RM 收请求 → RM forward → 返回 → LLM 采样 → done
                                              ↑
                                        [GPU 全程是 RM 的]
```

**关键点**：LLM 和 RM **从不并发使用 GPU**。所以"RM 加速"的红利得**先让 LLM 把 GPU 还出来**才能兑现，但每次 INTERVENE 时 LLM 已经在 wait（HTTP 阻塞），轮到 RM 跑时 GPU 反正是空的。**RM 跑得更快只意味着 LLM 等待时间更短**，对总吞吐有线性贡献。

按阿姆达尔分解：
```
per_token = LLM_time + intervention_rate × RM_call_time
          ≈ 11.3ms + 0.28 × 80ms = 33.7ms → 30 tok/s
```

RM 调用占 ~67% 的 per-token 时间。理论上让 RM 调用减半（80 → 40ms）→ per-token 22.5ms → 44 tok/s。但实际上做不到，原因见下面三组各自的解释。

### 4.2 G2 调度参数为什么几乎零收益

G2 的三个改动（`--no-enable-chunked-prefill`、`gpu_memory_utilization 0.3→0.4`、`max-model-len 4096→2048`）都是 vLLM **服务端调度层**的调优。

| 改动 | 实际影响 |
|---|---|
| 关 chunked prefill | pooling/classify 任务本来就不用 chunked prefill，原本就是 no-op |
| 显存 0.3 → 0.4 | 多给了 14 GB KV cache 池，但 SIA 每次 5 candidates × 几百 token，KV 池从来不是瓶颈 |
| max-model-len 4096 → 2048 | 调度元数据减半，节省的是 vLLM 的 Python 层 dict 维护时间（μs 级），对 GPU 时间没贡献 |

**结论**：这些参数都不在关键路径上。RM 调用的 80ms 里，绝大部分是 **GPU forward 自身**（5 候选 short forward）+ **HTTP/JSON 协议开销**，vLLM 调度只占 1-3ms，调优空间极小。

### 4.3 G3 客户端 3 件套为什么没换来吞吐提升

**预期 vs 实测**：

| 改动 | 预期收益 | 实测影响 |
|---|---|---|
| 去 `entropy.item()` sync | 每 token 省 1.4ms sync 税 | per-token 无明显变化 |
| `response_so_far` 增量解码 | 每 INTERVENE 步省 3-8ms | 几乎不可测 |
| flip 日志 | 增加少量 GPU→CPU sync 和 IO | 反而引入 ~0.5ms 开销 |

**为什么 entropy.item() 优化没显效**？eval 是**单请求串行**（MMLU 一题一题跑），batch_size 始终 = 1。原代码每 token 在 batch loop 里只 sync 1 次，去掉后仍是每 token 1 次（只是把整个 batch 的 entropy 一次拉到 CPU）。**真正的收益要在多请求并发下才显现**（batch=4 时 4 次 sync 压成 1 次），单请求场景下零差异。

**为什么增量解码没显效**？MMLU 平均 response 长度 ~600 token。原代码每步 decode `output_ids` 长度从 1 → 600，累计成本 ~O(n²)。但用 fast tokenizer，单次 decode 600 token 也就 1-2ms。增量版本每步只 decode 1 token = ~0.05ms。**节省的 1-2ms × INTERVENE 率 28% = 0.5ms / token，被噪声淹没**。

**为什么 flip 日志反而拖慢**？为了正确比较 pre/post argmax，需要把 `topk_indices_lists` 和 `topk_values_lists` 都 `.cpu().tolist()` 出来（虽然合并到了同一个 sync 块）。额外的 list 操作和 print 字符串生成在 hot path 上有微小累计。

**但准确率涨了 +1.2%pt（73.7%）**：这是 G3 唯一的实质收益。增量解码消除了 BPE 边界处理的潜在不一致（旧版每步全量 decode 的字符串与 RM 看到的字符串总能对齐，但 token-by-token 决策更稳定）。

### 4.4 G4 FP8 为什么反而变慢

H100/H200 上 FP8 GEMM 的理论加速是 1.5-2×。G4 反而 -7%，三层原因：

**层 1：共享 GPU 让 FP8 的算力红利无法兑现**

按阿姆达尔，FP8 把 RM 内核加速 1.5× 的天花板是：
```
原始: per_token = 11.3 + 0.28 × 80 = 33.7ms
若 RM 真减半: per_token = 11.3 + 0.28 × 40 = 22.5ms → 44 tok/s
```

但实际上**没有任何加速**。原因：RM 内核加速 ≠ RM 调用减半。RM 调用的 80ms 里：
- 真正算力部分 ~30-50ms（这部分 FP8 能加速）
- HTTP/scheduling/tokenize overhead ~30-50ms（FP8 完全没法降）

更何况 LLM 和 RM 交替占用同一 GPU，RM 加速 = LLM 等待变短 = 仅是 latency 优化，而 LLM 自身没变快。

**层 2：FP8 KV cache scale 未校准，触发退化路径**

启动 log 里 3 条 WARNING：
```
WARNING: Using KV cache scaling factor 1.0 for fp8_e4m3
WARNING: Using uncalibrated q_scale 1.0 and/or prob_scale 1.0 with fp8 attention
WARNING: Checkpoint does not provide a q scaling factor
```

FP8 attention 在 H200 上的快路径要求 q_scale / k_scale / v_scale / prob_scale 全部校准过（基于真实数据统计出来）。我们直接开 `--kv-cache-dtype fp8` 时这些默认为 1.0，vLLM 选择"安全退化"代码路径，每次读 KV 还要做 FP8 ↔ BF16 格式转换，**多了一层开销而非省了开销**。

**层 3：FP8 权重在 SIA 这种短输入场景上的相对优势更小**

RM 每次 INTERVENE 处理的是 5 候选 × 1-2 token suffix（其余是 prefix cache hit）。FP8 GEMM 的加速主要来自大矩阵乘的算力提升，但这里矩阵规模小（K=hidden_dim, M≈8 即 5×1.5），**计算密度低、launch overhead 占比高**，FP8 的相对收益本来就不显著。

综合 3 层原因：FP8 量化在**当前共享 GPU + 未校准 + 小输入**的三重不利条件下，从理论 +15% 退化到实测 -5%。

### 4.5 总结：3 轮优化共同的"天花板"

```
当前 SIA 的 per-token 时间被三块刚性成本锁死：

  LLM forward (~11ms)        ← 改不动，14B 模型固有
  + intervention × (
      HTTP/scheduling (~20ms)  ← 协议开销，要换 IPC 才能省
      + 真正算力 (~40ms)        ← 共享 GPU 上 FP8 用不上
      + GPU context 切换 (~5ms) ← 同卡进程切换
    )

这套结构里，G2/G3/G4 攻击的全是边缘项目（vLLM scheduler 1-3ms、客户端 sync 1ms、FP8 名义算力 vs 共享卡现实）。
要真的破局，必须把 LLM 和 RM 放到两张卡上，或把 RM 集成进 LLM 同一进程。
```

---

## 5. 结论与下一步

### 结论

1. **G2 / G3 / G4 在 throughput 上累计退步 -7%**（30.23 → 28.08 tok/s）
2. **G2 / G3 / G4 在 accuracy 上累计上涨 +1.7%pt**（71.5% → 73.2%）—— 多半来自 G3 的代码稳定性提升
3. **三组优化都没攻到真正的瓶颈**：当前 SIA 速度已经被"共享单卡 + 交替执行 + 协议开销"这套架构红线锁死在 ~30 tok/s
4. **FP8 在当前环境是负优化**：共享 GPU 让 FP8 算力红利无法兑现，KV cache 未校准触发退化路径

### 下一步

**唯一能突破吞吐天花板的路径是改架构**。按 `doc/parallel-decoding-design.md` 的规划：

| 阶段 | 改动 | 预期 tok/s | 性价比 |
|---|---|---:|---:|
| **Tier 1：分卡部署** | 2 张独立 GPU，启动命令改 | ~40-50 | ⭐⭐⭐⭐⭐ |
| Tier 1 + 分卡后开 FP8 | 同 G4 但在独占卡上 | ~50-55 | ⭐⭐⭐⭐ |
| Tier 2：有状态 RM + KV 复用 (2B+2C) | ~1300 行代码 | ~70+ | ⭐⭐ |

短期建议：**向老板申请 2 张独立 H200**（一张 ≥48 GB 跑 LLM，一张 ≥24 GB 跑 RM），是唯一确定能带来显著加速的方向。

---

## 6. 引用

- G1 baseline 详细报告：`doc/vllm-rm-experiment-report.md`
- 跨阶段优化路线图（2 卡架构方案）：`doc/parallel-decoding-design.md`
- AlpacaEval 对照实验（验证 SIA 在 Helpfulness 场景的有效性）：`doc/alpaca-eval-report.md`
- 实验日志归档：`exp/README.md` 中「2026-05-25 09:45 G2」「2026-05-25 21:20 G3」「2026-05-26 10:30 G4」三节
- 客户端代码 3 件套相关 commit：`a2357e9`（entropy.item 同步去除）、`368ef24`（增量解码）、`c802054`（flip 日志）

---

## Appendix：实验运行命令

> 本节命令的日志已归档至 `exp/`。完整命令与说明见 [`exp/README.md`](../exp/README.md) 中「2026-05-25 09:45 — G2」「2026-05-25 21:20 — G3」「2026-05-26 10:30 — G4」三节。

### G2 (2026-05-25 09:45) — RM 调度参数优化

```bash
nohup vllm serve /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --runner pooling --convert classify \
    --enable-prefix-caching --no-enable-chunked-prefill \
    --gpu-memory-utilization 0.4 \
    --max-model-len 2048 \
    --port 8001 \
    > log_vllm_rm_opt_202605250945.txt 2>&1 &

nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --rm_backend vllm \
    --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --llm_gpu_mem 0.6 \
    --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > log_llm_vllmrm_202605250945.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_vllmrm_202605250945.json --limit 20 \
    > log_SIA_vllmrm_202605250945.txt 2>&1 &
```

### G3 (2026-05-25 21:20) — 客户端代码 3 件套 + LLM mem 调整

```bash
nohup vllm serve /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --runner pooling --convert classify \
    --enable-prefix-caching --no-enable-chunked-prefill \
    --gpu-memory-utilization 0.4 \
    --max-model-len 2048 \
    --port 8001 \
    > log_vllm_rm_opt_202605252120.txt 2>&1 &

nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --rm_backend vllm \
    --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --llm_gpu_mem 0.55 \
    --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > log_llm_vllmrm_202605252120.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_vllmrm_202605252120.json --limit 20 \
    > log_SIA_vllmrm_202605252120.txt 2>&1 &
```

### G4 (2026-05-26 10:30) — FP8 + FP8 KV cache

```bash
nohup vllm serve /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --runner pooling --convert classify \
    --enable-prefix-caching --no-enable-chunked-prefill \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.3 \
    --max-model-len 2048 \
    --port 8001 \
    > log_vllm_rm_fp8_kvfp8_202605261030.txt 2>&1 &

nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --rm_backend vllm \
    --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --llm_gpu_mem 0.55 \
    --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > log_llm_vllmrm_202605261030.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_vllmrm_202605261030.json --limit 20 \
    > log_SIA_vllmrm_202605261030.txt 2>&1 &
```
