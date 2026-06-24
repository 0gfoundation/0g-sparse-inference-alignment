# RM Server Profiling Guide

Detailed profiling code for `src/sia_rm_server.py`, used to identify real performance bottlenecks on the KV cache path (forward / tokenize / kv expand / per-layer overhead).

---

## 1. Launch Commands

### 1.1 Default profiling (recommended for first run)

Prints per-stage latency for each call + outputs p50/p95/max summary every 50 calls.

```bash
# Ensure old processes are stopped first
pkill -f "sia_rm_server.py"; pkill -f "sia_vllm_server.py"; pkill -f "mmlu_eval.py"

# Start RM server (profiling enabled by default)
nohup python src/sia_rm_server.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --rm_device cuda:0 \
    --port 8001 > log_rm_pf_$(date +%Y%m%d%H%M).txt 2>&1 &

# Start LLM server
nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --llm_gpu_mem 0.6 \
    --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 > log_llm_pf_$(date +%Y%m%d%H%M).txt 2>&1 &

# Run evaluation (5-10 questions is enough; mainly for timing data)
nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 \
    --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_pf_$(date +%Y%m%d%H%M).json \
    --limit 5 1>log_SIA_pf_$(date +%Y%m%d%H%M).txt 2>&1 &
```

### 1.2 + Per-layer profiling (to check whether launch overhead or specific layers are slow)

Installs forward hooks on each transformer decoder layer to record per-layer average/max/min latency. **Has ~0.5ms/layer extra overhead; enable only for debugging.**

```bash
RM_PROFILE_LAYERS=1 nohup python src/sia_rm_server.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --rm_device cuda:0 \
    --port 8001 > log_rm_pf_layers_$(date +%Y%m%d%H%M).txt 2>&1 &
```

### 1.3 Disable profiling (for formal performance baseline measurement)

The profiling code itself has `torch.cuda.synchronize()` overhead; disable it for formal performance comparisons.

```bash
RM_PROFILE=0 nohup python src/sia_rm_server.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --rm_device cuda:0 \
    --port 8001 > log_rm_$(date +%Y%m%d%H%M).txt 2>&1 &
```

### 1.4 Enable CUDA graph (static bucketing acceleration)

In response to per-layer kernel launch overhead found by profiling (accounting for ~95% of forward time), use CUDA graph to compress ~360 kernel launches into 1, expected to reduce HIT path forward from 56ms → ~5-10ms.

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

Parameter descriptions:

| Parameter | Default | Meaning |
|------|------|------|
| `--cuda_graph` | off | Enable static bucketing + CUDA graph capture |
| `--cg_batch` | 5 | Bucket batch size (should equal `--topk`) |
| `--cg_diff_max` | 64 | Maximum diff token count for bucket; actual diff > 64 falls back to eager |
| `--cg_kv_max` | 2048 | Maximum prefix KV length for bucket; actual kv > 2048 falls back to eager |

**`--cuda_graph` and `--compile` are mutually exclusive.**

**Additional log lines that will appear:**
- At startup: `[RM] CudaGraph warmup+capture: ...` then `[RM] CudaGraph captured.`
- HIT calls going through bucket: `[RM-pf-fwd-cg]` instead of `[RM-pf-fwd]`; `cg_run` field corresponds to forward time
- Calls falling back to eager: still print `[RM-pf-fwd]` (triggered when diff or kv exceeds bucket limit)

**Expected benefits:**
- HIT path forward: 56ms → **5-10ms** (5-10× speedup)
- Overall RM single call: 68ms → **15-20ms**
- Overall tok/s: approximately **3-4×** improvement
- **Completely lossless** (output logits numerically identical to eager path; only GPU kernel scheduling differs)

**Note: CUDA graph + reload is not supported**; hot-swapping the RM will automatically disable CUDA graph and downgrade to eager (server must be restarted to re-capture).

---

## 2. Log Output Format

### 2.1 Single call — HIT path

```
[RM-pf-hit] req=abc12345 cached_ids_len=312 stable_len=313 extension_len=1 bpe_tail=5 per_seq_diff=[7,7,7,7,7] | tokenize=3.2 prefix_calc=0.1 state_lookup=0.0ms
[RM-pf-fwd] k=5 diff=7 kv=313 full=320 update_len=1 | prep=0.5 expand=2.1 fwd=48.3 score=0.4 kvsave=2.1 total=53.4ms
[RM] score: path=kv_hit  fwd=57ms  total=58ms  seq=-1  hit_rate=99.0%(99/100)  lock_wait=0.1ms chat_tpl=0.8ms
```

Field descriptions:

| Field | Meaning |
|------|------|
| `cached_ids_len` | Number of prefix tokens cached in the previous step (after BPE_TRIM) |
| `stable_len` | Number of stable prefix tokens in this call |
| `extension_len` | Number of new tokens added in this step (usually = 1, the token just sampled by vLLM) |
| `per_seq_diff` | Actual number of tokens forwarded for each candidate (= extension + bpe_tail + 1 candidate) |
| `tokenize` | Total time to tokenize 5 candidate texts individually |
| `prep` | Time to construct diff_ids and attn_mask |
| `expand` | `_dc_expand_batch` (batch=1 → batch=k KV copy) time |
| `fwd` | Main forward time (**includes all 36 transformer layers**) |
| `kvsave` | `_dc_extract_batch0_trim` (extract batch[0] and save back to cache) time |
| `lock_wait` | Time waiting for `_rm_lock` (if > a few ms, there is queuing) |
| `chat_tpl` | apply_chat_template + bos processing time |

### 2.2 Single call — MISS path

```
[RM-pf-miss] req=abc12345 stable_len=312 bpe_tail=5 per_seq_diff=[6,6,6,6,6] | tokenize=3.1 prefix_calc=0.1 miss_prefix_fwd=42.8ms
[RM-pf-fwd] k=5 diff=6 kv=312 ...
```

| Field | Meaning |
|------|------|
| `miss_prefix_fwd` | Forward time for pre-computation of batch=1, seq=stable_len (to build KV cache) |

### 2.3 Periodic summary (every 50 RM calls)

```
[RM-pf-summary @50] tokenize: p50=3.0 p95=4.2 max=8.1 | prefix_calc: p50=0.1 p95=0.2 max=0.5 | state_lookup: p50=0.0 p95=0.1 max=0.3 | prep_tensor: p50=0.5 p95=0.8 max=1.2 | kv_expand: p50=2.0 p95=3.5 max=5.8 | forward: p50=48.0 p95=65.2 max=120.1 | score_extract: p50=0.4 p95=0.6 max=0.9 | kv_save: p50=2.0 p95=3.2 max=4.5 | miss_prefix_fwd: p50=42.0 p95=58.0 max=85.0
```

### 2.4 Per-layer summary (when RM_PROFILE_LAYERS=1)

```
[RM-pf-layers] n_layers=36 avg_per_layer=1.35ms min=0.95 max=2.31 total_layers_time=48.6ms
```

---

## 3. Expected Analysis Path

After running 5-10 questions, look for the following signals in the logs:

### Signal A: `fwd` accounts for > 80% of total time

**Conclusion**: Bottleneck is in the forward itself (per-layer overhead dominates); KV cache compute savings are fully absorbed by 36 layers × 10 kernels × ~50μs launch overhead.
**Response**: Use static shape bucketing + CUDA graph (torch.compile mode="reduce-overhead" or manual capture).

### Signal B: `tokenize` > 10ms

**Conclusion**: Tokenizer is the bottleneck.
**Response**: Have the LLM server pass token IDs directly instead of raw text; or replace the current 5 separate tokenize calls with batch tokenize in the RM server.

### Signal C: `expand` or `kvsave` accounts for 5ms+

**Conclusion**: DynamicCache management itself has significant Python overhead (`.contiguous()` + DC.update per layer).
**Response**: Merge all 36 layers' K/V into a single large tensor (one-shot expand and trim) to avoid 72 small operations.

### Signal D: `extension_len` is always = 1, `per_seq_diff` is always 7

**Conclusion**: KV cache is successfully being reused (your intuition is correct), but forward is still slow → further confirms Signal A.

### Signal E: `lock_wait` is large

**Conclusion**: HTTP requests are queuing for `_rm_lock`, indicating a forward got stuck (either JIT, or genuinely slow).

### Signal F: Per-layer uniform

If `RM_PROFILE_LAYERS=1` shows all 36 layers at ~1.3ms each and max/min are close, **confirms launch overhead dominates** (not a specific layer bug) → must use CUDA graph.

---

## 4. Cleanup after profiling run

```bash
pkill -f "sia_rm_server.py"; pkill -f "sia_vllm_server.py"; pkill -f "mmlu_eval.py"
```

Collect logs:
```bash
ls -lt log_rm_pf_*.txt log_llm_pf_*.txt log_SIA_pf_*.txt | head -10
```

---

## 5. Profiling Code Locations

In `src/sia_rm_server.py`:

- `_PROFILE_DETAIL` / `_PROFILE_LAYERS`: environment variable switches
- `_pf_now()` / `_pf_record()` / `_pf_summary_if_due()`: timestamps and statistics
- `_install_layer_hooks()` / `_layer_pre_hook` / `_layer_post_hook`: per-layer hooks
- Instrumentation points: `_score_with_prefix_kv()`, `_try_score_with_kv_cache()`, `score()` endpoint

---

## 6. Experiment Log Archive

Representative experiment logs run in the manner described in this document have been archived to `exp/`; complete commands and descriptions are in [`exp/README.md`](../exp/README.md):
- "2026-05-20 15:00" — ran with `RM_PROFILE=True` for profiling to identify bottleneck
- "2025-05-21 00:50" — CUDA graph acceleration experiment (§1.4)

Bottleneck analysis and next optimization steps based on the above profiling data are in [`doc/profiling-bottleneck-analysis.md`](profiling-bottleneck-analysis.md).
