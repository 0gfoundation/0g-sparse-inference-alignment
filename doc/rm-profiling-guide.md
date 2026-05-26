# RM Server Profiling 指南

针对 `src/sia_rm_server.py` 的详细 profiling 代码，用于定位 KV cache 路径下的真实性能瓶颈（forward / tokenize / kv expand / per-layer overhead）。

---

## 1. 启动命令

### 1.1 默认 profiling（推荐第一次跑）

逐次打印每个阶段耗时 + 每 50 次输出 p50/p95/max 汇总。

```bash
# 先确保旧进程已停
pkill -f "sia_rm_server.py"; pkill -f "sia_vllm_server.py"; pkill -f "mmlu_eval.py"

# 启动 RM server（profiling 默认开启）
nohup python src/sia_rm_server.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --rm_device cuda:0 \
    --port 8001 > log_rm_pf_$(date +%Y%m%d%H%M).txt 2>&1 &

# 启动 LLM server
nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --llm_gpu_mem 0.6 \
    --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 > log_llm_pf_$(date +%Y%m%d%H%M).txt 2>&1 &

# 跑评测（5-10 道就够，主要为了拿到 timing 数据）
nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 \
    --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_pf_$(date +%Y%m%d%H%M).json \
    --limit 5 1>log_SIA_pf_$(date +%Y%m%d%H%M).txt 2>&1 &
```

### 1.2 + Per-layer profiling（看是 launch overhead 还是个别层慢）

在每个 transformer decoder layer 安装 forward hook，统计单层平均/最大/最小耗时。**有额外 ~0.5ms/层 overhead，仅 debug 时开。**

```bash
RM_PROFILE_LAYERS=1 nohup python src/sia_rm_server.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --rm_device cuda:0 \
    --port 8001 > log_rm_pf_layers_$(date +%Y%m%d%H%M).txt 2>&1 &
```

### 1.3 关闭 profiling（性能基准测量场景）

profiling 代码本身有 `torch.cuda.synchronize()` 开销，做正式性能对比时关闭。

```bash
RM_PROFILE=0 nohup python src/sia_rm_server.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --rm_device cuda:0 \
    --port 8001 > log_rm_$(date +%Y%m%d%H%M).txt 2>&1 &
```

### 1.4 启用 CUDA graph（静态 bucketing 加速）

针对 profiling 发现的 per-layer kernel launch overhead（占 forward 时间的 ~95%），用 CUDA graph 把 ~360 个 kernel launch 压成 1 次，预期 HIT path forward 从 56ms → ~5-10ms。

```bash
# RM server with CUDA graph
nohup python src/sia_rm_server.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --rm_device cuda:0 \
    --port 8001 \
    --cuda_graph \
    --cg_batch 5 --cg_diff_max 64 --cg_kv_max 2048 \
    > log_rm_cg_$(date +%Y%m%d%H%M).txt 2>&1 &
```

参数说明：

| 参数 | 默认 | 含义 |
|------|------|------|
| `--cuda_graph` | off | 启用静态 bucketing + CUDA graph 捕获 |
| `--cg_batch` | 5 | bucket batch size（应等于 `--topk`） |
| `--cg_diff_max` | 64 | bucket 最大 diff token 数；实际 diff > 64 会回退到 eager |
| `--cg_kv_max` | 2048 | bucket 最大 prefix KV 长度；实际 kv > 2048 会回退到 eager |

**`--cuda_graph` 与 `--compile` 互斥。**

**日志中会出现的额外行：**
- 启动期：`[RM] CudaGraph warmup+capture: ...` 然后 `[RM] CudaGraph captured.`
- 每次走 bucket 的 HIT call：`[RM-pf-fwd-cg]` 而不是 `[RM-pf-fwd]`，`cg_run` 字段对应 forward 时间
- 回退到 eager 的 call：仍打印 `[RM-pf-fwd]`（diff 或 kv 超过 bucket 上限时触发）

**预期收益：**
- HIT path forward：56ms → **5-10ms**（5-10× 加速）
- 整体 RM 单次调用：68ms → **15-20ms**
- 整体 tok/s：约 **3-4×** 提升
- **完全 lossless**（输出 logits 与 eager 路径数值一致，仅 GPU kernel 调度方式不同）

**注意：CUDA graph + reload 不支持**，热切换 RM 会自动禁用 CUDA graph 并降级到 eager（须重启 server 重新 capture）。

---

## 2. 日志输出格式

### 2.1 单次调用 — HIT path

```
[RM-pf-hit] req=abc12345 cached_ids_len=312 stable_len=313 extension_len=1 bpe_tail=5 per_seq_diff=[7,7,7,7,7] | tokenize=3.2 prefix_calc=0.1 state_lookup=0.0ms
[RM-pf-fwd] k=5 diff=7 kv=313 full=320 update_len=1 | prep=0.5 expand=2.1 fwd=48.3 score=0.4 kvsave=2.1 total=53.4ms
[RM] score: path=kv_hit  fwd=57ms  total=58ms  seq=-1  hit_rate=99.0%(99/100)  lock_wait=0.1ms chat_tpl=0.8ms
```

字段说明：

| 字段 | 含义 |
|------|------|
| `cached_ids_len` | 上一步缓存的 prefix token 数（已经过 BPE_TRIM） |
| `stable_len` | 本次稳定前缀 token 数 |
| `extension_len` | 本步新增的 token 数（通常 = 1，即上一步刚被 vLLM 采样的 token） |
| `per_seq_diff` | 每个 candidate 实际 forward 的 token 数（= extension + bpe_tail + 1 candidate） |
| `tokenize` | 5 个 candidate text 分别 tokenize 的总耗时 |
| `prep` | 构造 diff_ids 和 attn_mask 的耗时 |
| `expand` | `_dc_expand_batch`（batch=1 → batch=k 复制 KV）耗时 |
| `fwd` | 主 forward 耗时（**含 36 层 transformer**） |
| `kvsave` | `_dc_extract_batch0_trim`（取 batch[0] 存回 cache）耗时 |
| `lock_wait` | 等待 `_rm_lock` 的时间（若 > 几 ms，说明有排队） |
| `chat_tpl` | apply_chat_template + bos 处理耗时 |

### 2.2 单次调用 — MISS path

```
[RM-pf-miss] req=abc12345 stable_len=312 bpe_tail=5 per_seq_diff=[6,6,6,6,6] | tokenize=3.1 prefix_calc=0.1 miss_prefix_fwd=42.8ms
[RM-pf-fwd] k=5 diff=6 kv=312 ...
```

| 字段 | 含义 |
|------|------|
| `miss_prefix_fwd` | batch=1, seq=stable_len 的预计算 forward 耗时（用于建 KV cache） |

### 2.3 周期性汇总（每 50 次 RM 调用）

```
[RM-pf-summary @50] tokenize: p50=3.0 p95=4.2 max=8.1 | prefix_calc: p50=0.1 p95=0.2 max=0.5 | state_lookup: p50=0.0 p95=0.1 max=0.3 | prep_tensor: p50=0.5 p95=0.8 max=1.2 | kv_expand: p50=2.0 p95=3.5 max=5.8 | forward: p50=48.0 p95=65.2 max=120.1 | score_extract: p50=0.4 p95=0.6 max=0.9 | kv_save: p50=2.0 p95=3.2 max=4.5 | miss_prefix_fwd: p50=42.0 p95=58.0 max=85.0
```

### 2.4 Per-layer 汇总（RM_PROFILE_LAYERS=1 时输出）

```
[RM-pf-layers] n_layers=36 avg_per_layer=1.35ms min=0.95 max=2.31 total_layers_time=48.6ms
```

---

## 3. 期望的分析路径

跑 5-10 道题后，从日志中找以下信号：

### 信号 A：`fwd` 占总时间 > 80%

**结论**：瓶颈在 forward 本身（per-layer overhead 主导），KV cache 节省的 compute 完全被 36 层 × 10 kernels × ~50μs launch overhead 吞掉。  
**应对**：走静态形状 bucketing + CUDA graph 路线（torch.compile mode="reduce-overhead" 或手动 capture）。

### 信号 B：`tokenize` > 10ms

**结论**：tokenizer 是瓶颈。  
**应对**：让 LLM server 直接传 token IDs，不要传 raw text；或在 RM server 用 batch tokenize 替换当前 5 次单独 tokenize。

### 信号 C：`expand` 或 `kvsave` 占 5ms+

**结论**：DynamicCache 管理本身有可观 Python overhead（每层 .contiguous() + DC.update）。  
**应对**：把 36 层的 K/V 合并成单一大张量（一次性 expand 和 trim），避免 72 次小操作。

### 信号 D：`extension_len` 总是 = 1，`per_seq_diff` 总是 7

**结论**：KV cache 已经成功复用（你的直觉正确），但 forward 仍慢 → 进一步确认是信号 A。

### 信号 E：`lock_wait` 大

**结论**：HTTP 请求在排队等 `_rm_lock`，说明某次 forward 卡住了（要么 JIT，要么真的慢）。

### 信号 F：per-layer 均匀

若 `RM_PROFILE_LAYERS=1` 显示 36 层每层都是 ~1.3ms，且 max/min 接近，**确认 launch overhead 主导**（不是某层 bug）→ 必须走 CUDA graph 路线。

---

## 4. profiling 跑完后清理

```bash
pkill -f "sia_rm_server.py"; pkill -f "sia_vllm_server.py"; pkill -f "mmlu_eval.py"
```

收集日志：
```bash
ls -lt log_rm_pf_*.txt log_llm_pf_*.txt log_SIA_pf_*.txt | head -10
```

---

## 5. profiling 代码所在位置

`src/sia_rm_server.py` 中：

- `_PROFILE_DETAIL` / `_PROFILE_LAYERS`：环境变量开关
- `_pf_now()` / `_pf_record()` / `_pf_summary_if_due()`：时间戳与统计
- `_install_layer_hooks()` / `_layer_pre_hook` / `_layer_post_hook`：per-layer hook
- 插桩位置：`_score_with_prefix_kv()`、`_try_score_with_kv_cache()`、`score()` endpoint

---

## 6. 实验日志归档

按本文方式跑过的代表性实验日志已归档至 `exp/`，完整命令与说明见 [`exp/README.md`](../exp/README.md):
- 「2026-05-20 15:00」— 开 `RM_PROFILE=True` 跑 profiling 定位瓶颈
- 「2025-05-21 00:50」— CUDA graph 加速实验（§1.4）
