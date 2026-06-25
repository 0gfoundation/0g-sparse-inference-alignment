# OverflowError Crash Fix at High Concurrency (2026-06-25)

## Symptom

Running `python tests/bench_35b.py --stress` caused the SIA server to crash at
conc=128–192. After the crash, all subsequent requests returned `EngineDeadError`
and the server had to be restarted.

## Root Cause

**`OverflowError: out of range integral type conversion attempted`**

At high concurrency, vLLM's `output_tok_ids` list or the top-k candidate indices
occasionally contained a token ID outside the valid u32 range (negative or
> 4,294,967,295). HuggingFace's Rust-backed tokenizer raises `OverflowError` when
it tries to convert such a value to `u32` during `decode()`.

The crash propagated in two stages:

**Stage 1 — caught, but triggers fallback into the broken path**

Inside `score_candidates_batch`, the cross-tokenizer path calls
`self._llm_tok.decode([c])` for each candidate token ID `c`. When `c` is invalid
this raises `OverflowError`. The surrounding `except Exception` handler catches it
and prints:

```
[SIA] b2 batch scoring failed (falling back to sequential): out of range integral type conversion attempted
```

The code then falls back to per-request sequential scoring.

**Stage 2 — uncaught, kills the EngineCore**

In the sequential fallback loop, `_get_response_so_far` is called for the same
request that has the invalid token ID in its `output_ids`. It calls:

```python
tail_text = self._llm_tok.decode(output_ids[last_n:], skip_special_tokens=True)
```

This raises `OverflowError` again. There was **no try-except** at this call site.
The exception propagated all the way up to vLLM's `apply_logits_processors` →
`sample_tokens` → `EngineCore.run_busy_loop`, causing the entire EngineCore
process to exit. All subsequent requests then received `EngineDeadError`.

## Fix

Added `try-except (OverflowError, ValueError)` at all `decode()` call sites that
accept externally-sourced token IDs. On error, invalid token IDs are filtered out,
a `[SIA] WARNING` log line is emitted with the offending values for future
diagnosis, and execution continues without crashing.

**Files changed:**

| File | Location | Scenario |
|------|----------|----------|
| `src/sia_vllm_RM.py` | `_get_response_so_far` | Decoding accumulated output token sequence |
| `src/sia_rm/client.py` | `score_candidates` candidate decode | Single-request scoring path |
| `src/sia_rm/client.py` | `score_candidates_batch` candidate decode | Batch scoring path |
| `src/sia_rm/client.py` | `_get_stable_rm_prefix` incremental decode | Per-step prefix cache update |
| `src/sia_rm/client.py` | `_get_stable_rm_prefix` full decode | First call / negative-delta fallback |

Example warning log emitted on a bad token:

```
[SIA] WARNING req=5: token decode error (out of range integral type conversion attempted);
bad ids=[<value>] (first 5 of 1), tail_len=45. Filtering and retrying.
```

## Verification

After the fix, the stress test ran cleanly from conc=1 to conc=432 with zero
failures (stopped only because it reached the `--stress-max-conc 512` limit):

```
conc=  84  ok=168/168  ITL=115.3ms  tok/s=684.0  TPM=41040  ✓  (fail=0%)
conc= 128  ok=256/256  ITL=117.3ms  tok/s=635.9  TPM=38153  ✓  (fail=0%)
conc= 192  ok=384/384  ITL=135.5ms  tok/s=682.3  TPM=40939  ✓  (fail=0%)
conc= 288  ok=576/576  ITL=141.4ms  tok/s=664.7  TPM=39881  ✓  (fail=0%)
conc= 432  ok=864/864  ITL=145.7ms  tok/s=631.9  TPM=37912  ✓  (fail=0%)
```

Peak throughput: **conc=84, tok/s=684, TPM=41,040**.

## Open Question

The exact origin of the invalid token ID (why vLLM occasionally produces an
out-of-u32-range value at high concurrency) is not yet identified. The `[SIA]
WARNING` log will capture the specific bad token ID value the next time it occurs,
which should make the root cause easier to trace.

Likely candidates:
- A CUDA graph padding artifact in vLLM's sampled token list
- A numerical edge case in top-k sampling under NaN/inf logits at large batch sizes
- A vLLM v1 internal sentinel value leaking into `output_tok_ids`

## Commit

`39d9564` — `fix: guard against OverflowError from invalid token IDs at high concurrency`
