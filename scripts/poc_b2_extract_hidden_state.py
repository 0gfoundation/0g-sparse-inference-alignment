"""
M1a PoC — Verify that hidden_state can be extracted from vLLM 0.10.1.1's decode
loop without breaking CUDA graph optimization (per-token time still ~7ms).

Core insight (confirmed from source):
  - vLLM's Qwen3ForCausalLM.forward() returns hidden_states (does NOT apply lm_head)
  - Qwen3ForCausalLM.compute_logits() is where lm_head is applied
  - compute_logits is called outside the CUDA graph
  - → We can hook into compute_logits to capture hidden_state without affecting the graph

PoC steps:
  1. Monkey-patch Qwen3ForCausalLM.compute_logits to add a side channel writing hidden_state
  2. Run continuous decode with vllm.LLM
  3. Verify that hidden_state is captured
  4. Measure per-token time to confirm CUDA graph is still active (~7ms)

See doc/b2-decode-mode-poc-plan.md §5.1
"""
import statistics
import time

import torch

# Monkey-patch before importing vllm
import vllm.model_executor.models.qwen3 as _qwen3_mod

# Class-level side channel — shared reference across instances
_qwen3_mod.Qwen3ForCausalLM.last_hidden_state = None

_original_compute_logits = _qwen3_mod.Qwen3ForCausalLM.compute_logits


def _patched_compute_logits(self, hidden_states, sampling_metadata):
    """Hook: record hidden_state to class attribute, then apply lm_head as normal."""
    # hidden_states at the compute_logits entry point are already at "sample positions",
    # i.e. shape (n_samples, hidden_dim) rather than (batch, seq, hidden_dim).
    # We keep the reference directly (no clone, saves GPU memory).
    type(self).last_hidden_state = hidden_states
    return _original_compute_logits(self, hidden_states, sampling_metadata)


_qwen3_mod.Qwen3ForCausalLM.compute_logits = _patched_compute_logits
print("[patch] Qwen3ForCausalLM.compute_logits hooked", flush=True)


# ---- Now import the vllm main module + run tests ----
from vllm import LLM, SamplingParams  # noqa: E402

MODEL = "/workspace/SIA/models/Qwen3-4B"
PROMPT = "The quick brown fox jumps over the lazy dog. " * 30
WARMUP = 2
N = 5


def time_generate(llm, prompts, sp):
    t = time.perf_counter()
    _ = llm.generate(prompts, sp, use_tqdm=False)
    return time.perf_counter() - t


def main():
    print(f"Loading vLLM with patched Qwen3ForCausalLM ...", flush=True)
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.3,
        max_model_len=1024,
        enforce_eager=False,  # Critical: preserve CUDA graph optimization
        disable_log_stats=True,
    )
    print(f"vLLM loaded in {time.perf_counter()-t0:.1f}s\n", flush=True)

    sp_short = SamplingParams(temperature=0.0, max_tokens=10,
                              min_tokens=10, ignore_eos=True)
    sp_long = SamplingParams(temperature=0.0, max_tokens=110,
                             min_tokens=110, ignore_eos=True)

    # warmup
    print("Warming up...", flush=True)
    for _ in range(WARMUP):
        _ = time_generate(llm, [PROMPT], sp_short)
        _ = time_generate(llm, [PROMPT], sp_long)

    # Measure latency
    print("Timing (batch=1)...", flush=True)
    ts_short = [time_generate(llm, [PROMPT], sp_short) for _ in range(N)]
    ts_long  = [time_generate(llm, [PROMPT], sp_long)  for _ in range(N)]
    T_short = statistics.median(ts_short) * 1000
    T_long  = statistics.median(ts_long)  * 1000
    per_tok = (T_long - T_short) / 100
    print(f"  T(max_tokens=10):  median={T_short:.1f}ms")
    print(f"  T(max_tokens=110): median={T_long:.1f}ms")
    print(f"  per-token decode:  {per_tok:.2f}ms")
    print(f"  (baseline without patch: 7.02ms. Allows +1-2ms patch overhead)")

    # Verify hidden_state is captured
    print(f"\n=== Verify last_hidden_state is captured ===", flush=True)
    hs = _qwen3_mod.Qwen3ForCausalLM.last_hidden_state
    if hs is None:
        print(f"  last_hidden_state is still None — monkey-patch did not take effect (possible subprocess isolation)")
        print(f"     Try an alternative: register custom class via ModelRegistry, or patch vLLM source")
        return
    print(f"  hidden_state captured!")
    print(f"     shape:  {tuple(hs.shape)}")
    print(f"     dtype:  {hs.dtype}")
    print(f"     device: {hs.device}")
    print(f"     mean:   {hs.mean().item():.4f}")
    print(f"     std:    {hs.std().item():.4f}")

    # ==== M1a overall verdict ====
    print(f"\n=== M1a Acceptance ===")
    if hs is not None and per_tok < 10:
        print(f"  PASS: hidden_state captured + per-token = {per_tok:.2f}ms < 10ms")
        print(f"  -> Proceed to M1b: add score_head, measure 5-candidate scoring")
    elif hs is not None and per_tok < 15:
        print(f"  PARTIAL PASS: hidden_state captured, but per-token = {per_tok:.2f}ms is slow")
        print(f"     CUDA graph may be partially degraded. Can still proceed to M1b with lower expectations")
    elif hs is not None:
        print(f"  WARNING: hidden_state captured, but per-token = {per_tok:.2f}ms shows clear regression")
        print(f"     CUDA graph is likely not active. Check whether monkey-patch introduced ops outside the graph")
    else:
        print(f"  FAIL: hidden_state not captured.")
        print(f"     Possible cause: vLLM EngineCore starts in subprocess, monkey-patch does not cross process boundary")
        print(f"     Next step: try registering a subclass via vllm.ModelRegistry.register_model")


if __name__ == "__main__":
    main()
