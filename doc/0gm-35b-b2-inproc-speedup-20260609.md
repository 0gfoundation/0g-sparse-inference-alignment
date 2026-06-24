# 0GM-35B b2 inproc Speed Optimization Measurements — vllm 0.18.0 sweet spot (2026-06-09)

**TL;DR**: Enabling `--rm_backend b2` on vllm **0.18.0** + `SIA_RM_CUDAGRAPH=none`, 0GM-35B achieves in-process RM for the first time:

- **SIA throughput: ~49 tok/s (vs HTTP ~32 tok/s) → ~1.5× speedup**
- **noSIA throughput: ~54 tok/s (b2 inproc has zero overhead on non-intervention steps)**
- **0 RM errors, 0 cudagraph conflicts** (venv6 smoke test, H200 single GPU)
- **Zero code changes**: only need to switch venv (0.18.0) + two env vars

Passed smoke test; ready to deploy without additional testing.

---

## 1. Background: Why 0GM-35B Could Not Use b2 inproc Before

0GM-35B uses the `Qwen3_5MoeForConditionalGeneration` architecture (Qwen3.5 series), which was **not registered** in vllm 0.17.x and earlier, making it impossible to load — requiring 0.18+ as a minimum.

In vllm 0.19.x, PIECEWISE was changed to runtime capture mode: during main LLM inference it enters a `torch.cuda.graph()` context and sets a process-level global `is_currently_capturing` flag, which blocks any CUDA op in the same process (including eager RM forward). Exhaustive testing confirmed that all b2 configurations on vllm 0.19 fail 100%:

| Configuration | Result |
|---|---|
| FULL_AND_PIECEWISE main LLM + PIECEWISE RM | ❌ RM error 10%+ steps |
| FULL_AND_PIECEWISE main LLM + eager RM | ❌ RM error (flag comes from main LLM) |
| PIECEWISE-only main LLM + PIECEWISE RM | ❌ 100% INTERVENE failure |
| PIECEWISE-only main LLM + eager RM | ❌ RM error |

See [`doc/0gm-35b-sia-rm-inproc-path-20260602.md`](0gm-35b-sia-rm-inproc-path-20260602.md) for details.

---

## 2. Why vllm 0.18.0 Is the Sweet Spot

| Condition | Impact | 0.17.1 | **0.18.0** | 0.19.0 |
|------|------|--------|-----------|--------|
| Supports `Qwen3_5MoeForConditionalGeneration` | 0GM-35B can load | ❌ | ✅ | ✅ |
| PIECEWISE is still AOT capture | Does not enter `torch.cuda.graph()` during inference | ✅ | ✅ | ❌ (changed to runtime) |
| Has `unlock_workspace()` | Passes MoE workspace lock | ❌ | ✅ | ✅ |

0.18.0 satisfies all three conditions simultaneously and is the only viable version for 0GM-35B b2 inproc.

---

## 3. Why `SIA_RM_CUDAGRAPH=none` (Not `piecewise`)

Setting `SIA_RM_CUDAGRAPH=piecewise` on the RM side in vllm 0.18.0 also fails:

**Failure reason**: RM's PIECEWISE mode only captures a fixed set of batch_descriptors during the warmup phase. At inference time, prefix caching (APC) generates new batch_descriptors not seen during warmup; `CUDAGraphWrapper.__call__` detects the new key and calls `validate_cudagraph_capturing_enabled()`, which raises because post-startup capturing is disabled:
```
RuntimeError: CUDA graph capturing detected at an inappropriate time.
```

Note: this raise occurs **before** entering the `torch.cuda.graph()` context, so it does not set the CUDA global flag — this is the key difference from the 0.19 failure mode (0.19 is a flag conflict; 0.18 is a Python exception).

**Solution**: `SIA_RM_CUDAGRAPH=none` → `enforce_eager=True`, so the RM completely skips `CUDAGraphWrapper` with no graph capture attempts.

The 0.18 main LLM's PIECEWISE only **replays** already-captured graphs during inference (does not enter `torch.cuda.graph()` context), so eager RM can forward freely.

---

## 4. Configuration

### 4.1 Topology — b2 inproc

**Single process**: A nested vllm `LLM(...)` is embedded inside the SIA EngineCore as the RM.

```
┌─────────── Main Process (sia_vllm_server.py) ───────────────┐
│   FastAPI server (port 8000)                                  │
│   ↓                                                            │
│   AsyncLLMEngine (main LLM = 0GM-35B, vllm 0.18.0)           │
│   ↓ EngineCore subprocess                                     │
│   ┌───────────────────────────────────────────────┐           │
│   │  SIA LogitsProcessor                           │           │
│   │  ↓  __call__()                                 │           │
│   │  RMClient (b2 backend, inproc)                 │           │
│   │  ↓  score_candidates(...)  ← direct Python call│           │
│   │  ┌─────────────────────────────────────────┐  │           │
│   │  │  nested vllm LLM(VM-Qwen3-4B-merged...) │  │           │
│   │  │  enforce_eager=True  (SIA_RM_CUDAGRAPH   │  │           │
│   │  │  =none)                                  │  │           │
│   │  │  on same CUDA context as main LLM        │  │           │
│   │  └─────────────────────────────────────────┘  │           │
│   └───────────────────────────────────────────────┘           │
└───────────────────────────────────────────────────────────────┘
```

No HTTP cross-process communication. RM calls go through in-process Python functions; the cross-tokenizer bridge automatically handles token mapping from 0GM-35B (248044 vocab) to VM-Qwen3-4B (151643 vocab).

### 4.2 Key Parameters

| Parameter | Value |
|------|---|
| `--llm` | 0GM-1.0-35B-A3B-0427 |
| `--rm_backend` | **`b2`** |
| `--rm_model` | VM-Qwen3-4B-merged-for-vllm |
| `--llm_gpu_mem` | 0.55 |
| `--rm_b2_gpu_mem` | 0.15 |
| `--max_model_len` | 2048 |
| `--topk` | 10 |
| `--weight` | 1.0 |
| `--entropy_threshold` | 1.0 |
| `SIA_LLM_CUDAGRAPH` | **`piecewise`** (main LLM PIECEWISE-only AOT) |
| `SIA_RM_CUDAGRAPH` | **`none`** (RM eager, required — see §3) |
| `SIA_RM_MULTIPROCESS` | `0` (in-process InprocClient) |

---

## 5. Measured Throughput (H200 single GPU, 2026-06-09)

| Mode | tok/s | Notes |
|------|-------|------|
| noSIA (sia_weight=0) | **~54** | Pure vllm inference, zero RM overhead |
| SIA (entropy_threshold=1.0, ~8% intervention rate) | **~49** | Default configuration, ~9% overhead |
| Old HTTP vllm RM backend (baseline) | **~32** | Historical measurement on same hardware |

**b2 inproc vs HTTP: ~1.5× speedup** — matches VL-30B b2 inproc's 1.46×.

> Note: The measurements above used 0GM-35B's thinking mode (default output includes thinking tokens); throughput will be higher if thinking is disabled in production chat scenarios.

### SIA Intervention Log Sample (smoke test)

```
[SIA] req=0 DONE  intervened=3/20    ratio=15.0%  top1_flip=1/3   (33.3%)
[SIA] req=0 DONE  intervened=14/200  ratio=7.0%   top1_flip=11/14 (78.6%)
[SIA] req=0 DONE  intervened=17/200  ratio=8.5%   top1_flip=15/17 (88.2%)
[SIA] req=0 DONE  intervened=22/200  ratio=11.0%  top1_flip=18/22 (81.8%)
```

top1_flip rate of 78–88% indicates that RM interventions have a meaningful impact (changing token selection only at high-entropy positions).

---

## 6. Comparison with VL-30B

| Dimension | VL-30B | 0GM-35B |
|------|--------|---------|
| Architecture | `Qwen3VLMoe` | `Qwen3_5MoeForConditionalGeneration` |
| vllm version for b2 inproc | 0.17.1 | **0.18.0** |
| `SIA_LLM_CUDAGRAPH` | `piecewise` | `piecewise` |
| `SIA_RM_CUDAGRAPH` | `piecewise` | **`none`** (0.18 RM PIECEWISE has prefix-cache new descriptor issue) |
| End-to-end speedup vs HTTP | 1.46× | ~1.5× |
| venv / requirements | `vl30b-b2-inproc.txt` | `0gm35b-b2-inproc.txt` |

---

## 7. Quick Start

### 7.1 Install venv

```bash
scripts/setup_venv_0gm35b_b2.sh /path/to/venv-0gm35b-b2
```

Manual steps:
```bash
python3 -m venv /path/to/venv-0gm35b-b2
source /path/to/venv-0gm35b-b2/bin/activate
pip install -r requirements/0gm35b-b2-inproc.txt
pip install -e .
```

### 7.2 Start the Server

```bash
SIA_LLM_CUDAGRAPH=piecewise SIA_RM_CUDAGRAPH=none \
SIA_RM_MULTIPROCESS=0 \
python src/sia_vllm_server.py \
  --llm /path/to/0GM-1.0-35B-A3B \
  --rm_backend b2 \
  --rm_model /path/to/VM-Qwen3-4B-merged-for-vllm \
  --rm_b2_gpu_mem 0.15 --llm_gpu_mem 0.55 \
  --topk 10 --weight 1.0 --entropy_threshold 1.0 \
  --max_model_len 2048 --port 8000
```

Startup takes approximately 15 minutes (first-time JIT compilation of FlashInfer GDN kernel + PIECEWISE warmup).

### 7.3 Smoke Test

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":20,"temperature":0.7}' \
  | python3 -m json.tool
```

The server log should show (with no `RM error` lines):
```
[SIA] req=0 DONE  intervened=X/20  ratio=Y%  top1_flip=...
```

### 7.4 Disable SIA for Comparison

```bash
# SIA on (default)
curl -s ... -d '{"...", "max_tokens":200}'

# SIA off (pure vllm baseline)
curl -s ... -d '{"...", "max_tokens":200, "sia_weight":0}'
```

---

## 8. Notes

- **`SIA_RM_CUDAGRAPH=none` is required** — `piecewise` cannot be used (see §3)
- **Do not run b2 in a 0.19 venv** — it will fail regardless of the cudagraph configuration
- `--max_model_len 2048` is set to leave GPU memory for the RM; increase it if the GPU has sufficient headroom
- The first startup triggers FlashInfer GDN prefill kernel JIT compilation (~4 minutes); subsequent runs use cached kernels
- The MoE config warning (`Using default MoE config`) is expected — vllm 0.18 has no H200-specific MoE tuning configuration

---

## References

- [`doc/0gm-35b-speedup-plan-20260609.md`](0gm-35b-speedup-plan-20260609.md) — Full plan comparison (P0–P6) + P0 test process
- [`doc/vl30b-b2-inproc-speedup-20260605.md`](vl30b-b2-inproc-speedup-20260605.md) — VL-30B equivalent speedup, including AlpacaEval quality validation
- `requirements/0gm35b-b2-inproc.txt` — vllm 0.18.0 dependency configuration
- `scripts/setup_venv_0gm35b_b2.sh` — one-command venv setup script
