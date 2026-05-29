"""
RM split-stage profiling for the LLM/RM parallelization study.

Goal: measure the GPU time of each stage independently so we can decide
whether the Reward Model can be hidden behind the main LLM forward (by
running the two models concurrently on a multi-GPU node).

Three stages of an RM intervention call:

  A. "Non-intervention KV"
     Catch the RM KV cache up to the latest LLM-generated tokens
     (the K tokens accumulated since the last RM call via fix_a_token).
     This work happens *before* the main LLM has produced its next
     intervention point — it is potentially **overlap-able with the
     main LLM forward**.

  B. "5-candidate KV"
     Forward the 5 candidate tokens (one per topk choice) appended to
     the now-cached prefix. This work happens **after** the main LLM
     produced its topk decision, so it sits on the **critical path**
     and cannot be overlapped with the main LLM forward.

  C. "5-candidate score head"
     Apply the score head Linear(hidden, 1) to each of the 5 sample
     positions. Sits on the same critical path as B.

If A's GPU time can fit inside the main LLM forward time (~12 ms for
Qwen3-14B at batch=1), we can fully hide A and only B+C count toward
SIA overhead.

Usage:
    SIA_RM_PROFILE=1 python scripts/bench_rm_split_profile.py

The SIA_RM_PROFILE=1 env var is required — it enables cuda.Event-based
GPU-side timings inside Qwen3WithScoreForCausalLM. Wall-clock timings
are always reported.
"""
import os
import random
import statistics
import sys
import time
import uuid

# Make sia_rm importable in EngineCore subprocess (spawned via python -m,
# does NOT inherit parent's sys.path).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
sys.path.insert(0, _SRC)
_pp = os.environ.get("PYTHONPATH", "")
if _SRC not in _pp.split(":"):
    os.environ["PYTHONPATH"] = (_SRC + ":" + _pp) if _pp else _SRC

# Force profile mode on (the model's hook reads this env var at module
# import time in the EngineCore subprocess).
os.environ["SIA_RM_PROFILE"] = "1"

# Unique reward + timing file id so concurrent runs don't trample each other.
os.environ["SIA_REWARD_FILE_ID"] = f"bench_split_{uuid.uuid4().hex[:8]}"

import sia_rm  # noqa: F401 — triggers ModelRegistry registration
from sia_rm import RMClient

MODEL_VM = "/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm"

# Realistic SIA decode pattern parameters
N_WARMUP = 5         # warmup intervene calls (excluded from stats)
N_MEASURE = 100      # measured intervene calls
PREFIX_INIT = 600    # initial chat prompt size (~MMLU question length)
SKIP_LO = 1          # min SKIP tokens between two INTERVENE calls
SKIP_HI = 5          # max SKIP tokens between two INTERVENE calls
N_CANDIDATES = 5     # SIA topk=5


def percentile(values, p):
    vs = sorted(values)
    return vs[int(len(vs) * p)]


def report(label, values, unit="ms"):
    if not values:
        print(f"  {label:<40s}  (no data)")
        return
    vs = sorted(values)
    n = len(vs)
    print(f"  {label:<40s}  "
          f"min={vs[0]:6.2f}  "
          f"p50={vs[n // 2]:6.2f}  "
          f"mean={sum(vs) / n:6.2f}  "
          f"p95={vs[int(n * 0.95)]:6.2f}  "
          f"max={vs[-1]:6.2f}  {unit}")


def main():
    print("=" * 78)
    print("RM split-stage profiling — Qwen3WithScoreForCausalLM (b2 backend)")
    print("=" * 78)
    print(f"  SIA_RM_PROFILE   : {os.environ.get('SIA_RM_PROFILE')}")
    print(f"  reward/timing id : {os.environ['SIA_REWARD_FILE_ID']}")
    print(f"  model            : {MODEL_VM}")
    print(f"  warmup iters     : {N_WARMUP}")
    print(f"  measure iters    : {N_MEASURE}")
    print(f"  init prefix len  : {PREFIX_INIT} tokens")
    print(f"  skip tokens/iter : {SKIP_LO}..{SKIP_HI} (random)")
    print(f"  topk candidates  : {N_CANDIDATES}")
    print()

    print("Loading RMClient (vLLM 0.10.1.1 venv2)...")
    t0 = time.perf_counter()
    rm = RMClient(model_path=MODEL_VM, gpu_mem=0.3, max_model_len=4096)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s\n")

    # Simulate a single SIA session
    random.seed(42)
    initial_prefix = [random.randint(1000, 100000)
                      for _ in range(PREFIX_INIT)]
    sid = rm.new_session(initial_prefix)
    # The 5 candidate tokens are picked once and reused per iteration. In
    # real SIA they'd be the LLM's topk output (variable per step); the
    # exact ids don't change GPU time here.
    candidates = [random.randint(1000, 100000) for _ in range(N_CANDIDATES)]

    # ---- Warmup ----
    print(f"Warmup ({N_WARMUP} iterations) ...")
    for i in range(N_WARMUP):
        n_skip = random.randint(SKIP_LO, SKIP_HI)
        for _ in range(n_skip):
            rm.fix_a_token(sid, random.randint(1000, 100000))
        r = rm.score_candidates_profiled(sid, candidates)
        print(f"  warmup #{i}: K={r['n_new_prefix']:<2d} "
              f"A_wall={r['a_wall_ms']:6.2f} "
              f"B_wall={r['b_wall_ms']:6.2f} "
              f"A_fwd={r['a_forward_gpu_ms']:6.2f} "
              f"B_fwd={r['b_forward_gpu_ms']:6.2f} "
              f"score={r['b_score_gpu_ms']:6.2f} ms")
    print()

    # ---- Measurement ----
    print(f"Measuring ({N_MEASURE} iterations) ...")
    records = []
    for i in range(N_MEASURE):
        n_skip = random.randint(SKIP_LO, SKIP_HI)
        for _ in range(n_skip):
            rm.fix_a_token(sid, random.randint(1000, 100000))
        r = rm.score_candidates_profiled(sid, candidates)
        records.append(r)
        if i < 5 or (i + 1) % 25 == 0:
            print(f"  iter #{i:3d}: K={r['n_new_prefix']:<2d} "
                  f"A_wall={r['a_wall_ms']:6.2f} "
                  f"B_wall={r['b_wall_ms']:6.2f} "
                  f"A_fwd={r['a_forward_gpu_ms']:6.2f} "
                  f"B_fwd={r['b_forward_gpu_ms']:6.2f} "
                  f"score={r['b_score_gpu_ms']:6.2f} ms "
                  f"prefix_len={initial_prefix.__len__() + sum(rr['n_new_prefix'] for rr in records[:i+1])}")
    print()

    # ---- Aggregate ----
    n_new = [r['n_new_prefix'] for r in records]
    a_wall = [r['a_wall_ms'] for r in records]
    b_wall = [r['b_wall_ms'] for r in records]
    a_fwd = [r['a_forward_gpu_ms'] for r in records]
    b_fwd = [r['b_forward_gpu_ms'] for r in records]
    b_score = [r['b_score_gpu_ms'] for r in records]

    print("=" * 78)
    print("Summary")
    print("=" * 78)
    print(f"  iterations         : {len(records)}")
    print(f"  prefix grew        : {PREFIX_INIT} -> "
          f"{PREFIX_INIT + sum(n_new)} tokens "
          f"(avg {sum(n_new) / len(n_new):.1f} new per call)")
    print()
    print("  Stage A  — non-intervention KV (1 prompt of prefix only)")
    report("    A wall-clock", a_wall)
    report("    A GPU forward (KV computation)", a_fwd)
    print()
    print("  Stage B  — 5-candidate forward (5 prompts each = prefix + 1)")
    report("    B wall-clock", b_wall)
    report("    B GPU forward (5 candidate KV + score)", b_fwd)
    report("    B GPU score head only", b_score)
    # B forward minus score gives KV-only cost for the 5 candidates
    b_kv_only = [f - s for f, s in zip(b_fwd, b_score)]
    report("    B GPU candidate KV only (fwd - score)", b_kv_only)
    print()

    # ---- Parallelization viability analysis ----
    print("=" * 78)
    print("Parallelization viability (with 2-GPU NVLink node target)")
    print("=" * 78)
    a_p50 = percentile(a_fwd, 0.5)
    a_p95 = percentile(a_fwd, 0.95)
    b_kv_p50 = percentile(b_kv_only, 0.5)
    b_score_p50 = percentile(b_score, 0.5)
    critical_p50 = b_kv_p50 + b_score_p50
    # Typical main-LLM (Qwen3-14B) per-token forward time, from the
    # historical no-intervention measurement.
    LLM_FORWARD_MS = 11.31
    print(f"  Main LLM (Qwen3-14B) per-token forward (historical): "
          f"~{LLM_FORWARD_MS:.2f} ms\n")
    print(f"  Overlap-able work (Stage A — non-intervention KV):")
    print(f"    p50 GPU = {a_p50:.2f} ms,  p95 = {a_p95:.2f} ms")
    if a_p50 <= LLM_FORWARD_MS:
        budget_used = a_p50 / LLM_FORWARD_MS * 100
        print(f"    → p50 fits inside one LLM forward ({budget_used:.1f}% of "
              f"{LLM_FORWARD_MS:.2f} ms budget). Can be fully hidden.")
    else:
        excess = a_p50 - LLM_FORWARD_MS
        print(f"    → p50 EXCEEDS LLM forward budget by {excess:.2f} ms; "
              f"need (a_p50 / LLM_FORWARD_MS) ≈ {a_p50 / LLM_FORWARD_MS:.1f}x "
              f"LLM steps' worth of slack to hide.")
    print()
    print(f"  Critical-path work (Stage B — 5 candidate KV + score):")
    print(f"    p50 GPU = {critical_p50:.2f} ms  "
          f"(KV {b_kv_p50:.2f} + score {b_score_p50:.2f})")
    print(f"    → cannot overlap with current LLM step; sits on per-token "
          f"critical path.")
    print()
    print(f"  Expected per-token cost with full A-overlap, partial intervene:")
    for r_int in [0.17, 0.27, 0.33]:
        per_tok = LLM_FORWARD_MS + r_int * critical_p50
        tps = 1000 / per_tok
        print(f"    intervene rate {r_int * 100:4.1f}%:  "
              f"{per_tok:.2f} ms/tok  →  {tps:.1f} tok/s")
    print()


if __name__ == "__main__":
    main()
