"""
A/B comparison of Stage B (5-candidate forward) with and without an
explicit batch=5 entry in vLLM's CUDA graph capture sizes.

Default vLLM capture sizes: [1, 2, 4, 8, 16, ..., 512]
  → SIA's batch=5 is padded up to the batch=8 graph (3 dummy tokens),
    AND/OR vLLM scheduler splits the 5 prompts into 1 + 4 separate
    forwards. Both adds overhead.

Variant tested: cuda_graph_sizes=[1, 2, 4, 5, 8, 16, 32, 64, 128]
  → batch=5 has a dedicated CUDA graph; we want to see whether
    Stage B GPU time drops, and how often vLLM still splits into
    multiple forwards.

Per-iteration we record:
  - n_forwards_b   : how many compute_logits calls happened in Stage B
  - per-forward shapes (n_input, n_samples)
  - Stage B total GPU forward ms (sum across forwards)
  - Stage B score head ms (sum across forwards)

Usage:
    SIA_RM_PROFILE=1 python scripts/bench_rm_batch5_ab.py default
    SIA_RM_PROFILE=1 python scripts/bench_rm_batch5_ab.py with5

The script runs ONE variant per invocation (so the two vLLM instances
don't share GPU state); compare the two output logs afterwards.

Output JSON: /tmp/bench_rm_batch5_ab_<variant>.json
"""
import json
import os
import random
import statistics
import sys
import time
import uuid

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
sys.path.insert(0, _SRC)
_pp = os.environ.get("PYTHONPATH", "")
if _SRC not in _pp.split(":"):
    os.environ["PYTHONPATH"] = (_SRC + ":" + _pp) if _pp else _SRC

os.environ["SIA_RM_PROFILE"] = "1"
os.environ["SIA_REWARD_FILE_ID"] = f"bench_ab_{uuid.uuid4().hex[:8]}"

import sia_rm  # noqa: F401
from sia_rm import RMClient, read_all_timings, truncate_timings, truncate_rewards, read_rewards


MODEL_VM = "/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm"
N_WARMUP = 10
N_MEASURE = 100
PREFIX_INIT = 600
SKIP_LO = 1
SKIP_HI = 5
N_CANDIDATES = 5


def pct(values, p):
    vs = sorted(values)
    return vs[int(len(vs) * p)] if vs else 0.0


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("default", "with5"):
        print(f"Usage: {sys.argv[0]} {{default|with5}}")
        sys.exit(1)
    variant = sys.argv[1]

    cuda_graph_sizes = None
    if variant == "with5":
        cuda_graph_sizes = [1, 2, 4, 5, 8, 16, 32, 64, 128]

    out_path = f"/tmp/bench_rm_batch5_ab_{variant}.json"

    print("=" * 78)
    print(f"A/B bench — variant = {variant}")
    print(f"  cuda_graph_sizes  : {cuda_graph_sizes}")
    print(f"  N_WARMUP / N_MEASURE: {N_WARMUP} / {N_MEASURE}")
    print(f"  PREFIX_INIT       : {PREFIX_INIT}")
    print(f"  SKIP per INTERVENE: {SKIP_LO}..{SKIP_HI}")
    print(f"  output JSON       : {out_path}")
    print("=" * 78)

    print("Loading RMClient ...")
    t0 = time.perf_counter()
    rm = RMClient(
        model_path=MODEL_VM,
        gpu_mem=0.3,
        max_model_len=4096,
        cuda_graph_sizes=cuda_graph_sizes,
    )
    print(f"  loaded in {time.perf_counter() - t0:.1f}s\n")

    random.seed(42)
    initial_prefix = [random.randint(1000, 100000)
                      for _ in range(PREFIX_INIT)]
    sid = rm.new_session(initial_prefix)
    candidates = [random.randint(1000, 100000) for _ in range(N_CANDIDATES)]

    # Use the lower-level mechanics directly (not score_candidates_profiled)
    # so we can capture per-record info from the timing channel for Stage B.

    def one_iter():
        prefix = rm._sessions[sid]
        n_already = rm._last_prefilled.get(sid, 0)
        n_new = len(prefix) - n_already
        TP = rm._TokensPrompt

        # Stage A: prefill new prefix
        truncate_rewards()
        truncate_timings()
        a_t0 = time.perf_counter()
        if n_new > 0:
            rm.llm.generate([TP(prompt_token_ids=prefix)],
                            rm._sp, use_tqdm=False)
        a_wall = (time.perf_counter() - a_t0) * 1000
        a_timings = read_all_timings()

        # Stage B: 5 candidates
        truncate_rewards()
        n_a = len(a_timings)
        b_t0 = time.perf_counter()
        prompts = [TP(prompt_token_ids=prefix + [c]) for c in candidates]
        rm.llm.generate(prompts, rm._sp, use_tqdm=False)
        b_wall = (time.perf_counter() - b_t0) * 1000
        all_t = read_all_timings()
        b_timings = all_t[n_a:]

        rewards = read_rewards()
        assert rewards is not None and rewards.numel() == N_CANDIDATES

        rm._last_prefilled[sid] = len(prefix)
        return {
            "n_new": n_new,
            "a_wall": a_wall,
            "a_fwd": sum(t["forward_ms"] for t in a_timings),
            "b_wall": b_wall,
            "b_n_forwards": len(b_timings),
            "b_forwards": [
                {"forward_ms": t["forward_ms"], "score_ms": t["score_ms"],
                 "n_input": t["n_input"], "n_samples": t["n_samples"]}
                for t in b_timings
            ],
            "b_fwd_sum": sum(t["forward_ms"] for t in b_timings),
            "b_score_sum": sum(t["score_ms"] for t in b_timings),
        }

    # Warmup
    print(f"Warmup ({N_WARMUP}) ...")
    for i in range(N_WARMUP):
        for _ in range(random.randint(SKIP_LO, SKIP_HI)):
            rm.fix_a_token(sid, random.randint(1000, 100000))
        r = one_iter()
        print(f"  warmup #{i}: n_new={r['n_new']} b_n_fwd={r['b_n_forwards']} "
              f"b_fwd_sum={r['b_fwd_sum']:.2f}ms shapes={[f['n_samples'] for f in r['b_forwards']]}")
    print()

    # Measurement
    print(f"Measuring ({N_MEASURE}) ...")
    records = []
    for i in range(N_MEASURE):
        for _ in range(random.randint(SKIP_LO, SKIP_HI)):
            rm.fix_a_token(sid, random.randint(1000, 100000))
        r = one_iter()
        records.append(r)
        if i < 5 or (i + 1) % 25 == 0:
            shapes = [f["n_samples"] for f in r["b_forwards"]]
            print(f"  iter #{i:3d}: n_new={r['n_new']} "
                  f"b_n_fwd={r['b_n_forwards']:1d} shapes={shapes!s:<10s} "
                  f"b_fwd={r['b_fwd_sum']:6.2f}ms score={r['b_score_sum']:.2f}ms "
                  f"b_wall={r['b_wall']:6.2f}ms")

    # Aggregate
    b_n_fwd = [r["b_n_forwards"] for r in records]
    b_fwd_sum = [r["b_fwd_sum"] for r in records]
    b_score_sum = [r["b_score_sum"] for r in records]
    b_wall = [r["b_wall"] for r in records]

    from collections import Counter
    nfwd_counter = Counter(b_n_fwd)

    # Per-forward shape distribution (across all records)
    shape_counter = Counter()
    for r in records:
        for f in r["b_forwards"]:
            shape_counter[f["n_samples"]] += 1

    # Latency by # forwards (1-forward iters vs 2-forward iters)
    fwd_by_count = {}
    for r in records:
        nf = r["b_n_forwards"]
        fwd_by_count.setdefault(nf, []).append(r["b_fwd_sum"])

    print()
    print("=" * 78)
    print(f"Summary — variant = {variant}")
    print("=" * 78)
    print(f"  Stage B # forwards distribution (over {N_MEASURE} iter):")
    for k in sorted(nfwd_counter.keys()):
        print(f"    {k} forward(s) per iter: {nfwd_counter[k]:>3d}x  "
              f"({nfwd_counter[k]/N_MEASURE*100:.0f}%)")
    print(f"  Per-forward batch-size distribution (n_samples):")
    for k in sorted(shape_counter.keys()):
        print(f"    n_samples = {k}: {shape_counter[k]:>3d}x")
    print()
    print(f"  Stage B GPU forward ms:")
    print(f"    min={min(b_fwd_sum):6.2f}  p50={pct(b_fwd_sum, 0.5):6.2f}  "
          f"mean={sum(b_fwd_sum)/len(b_fwd_sum):6.2f}  "
          f"p95={pct(b_fwd_sum, 0.95):6.2f}  max={max(b_fwd_sum):6.2f}")
    print(f"  Stage B score head ms:")
    print(f"    min={min(b_score_sum):6.2f}  p50={pct(b_score_sum, 0.5):6.2f}  "
          f"mean={sum(b_score_sum)/len(b_score_sum):6.2f}  "
          f"p95={pct(b_score_sum, 0.95):6.2f}  max={max(b_score_sum):6.2f}")
    print(f"  Stage B wall-clock ms:")
    print(f"    min={min(b_wall):6.2f}  p50={pct(b_wall, 0.5):6.2f}  "
          f"mean={sum(b_wall)/len(b_wall):6.2f}  "
          f"p95={pct(b_wall, 0.95):6.2f}  max={max(b_wall):6.2f}")
    print()
    print("  Stage B GPU forward ms broken down by # forwards/iter:")
    for k in sorted(fwd_by_count.keys()):
        vs = fwd_by_count[k]
        print(f"    {k} forward iters (n={len(vs)}):  "
              f"min={min(vs):6.2f}  p50={pct(vs, 0.5):6.2f}  "
              f"mean={sum(vs)/len(vs):6.2f}  max={max(vs):6.2f}")

    # Write JSON for cross-variant comparison
    summary = {
        "variant": variant,
        "cuda_graph_sizes": cuda_graph_sizes,
        "n_warmup": N_WARMUP, "n_measure": N_MEASURE,
        "prefix_init": PREFIX_INIT,
        "n_candidates": N_CANDIDATES,
        "b_fwd_sum": {
            "min": min(b_fwd_sum), "p50": pct(b_fwd_sum, 0.5),
            "mean": sum(b_fwd_sum)/len(b_fwd_sum),
            "p95": pct(b_fwd_sum, 0.95), "max": max(b_fwd_sum),
            "values": b_fwd_sum,
        },
        "b_score_sum": {
            "min": min(b_score_sum), "p50": pct(b_score_sum, 0.5),
            "mean": sum(b_score_sum)/len(b_score_sum),
            "p95": pct(b_score_sum, 0.95), "max": max(b_score_sum),
        },
        "b_wall": {
            "min": min(b_wall), "p50": pct(b_wall, 0.5),
            "mean": sum(b_wall)/len(b_wall),
            "p95": pct(b_wall, 0.95), "max": max(b_wall),
        },
        "n_forwards_dist": dict(nfwd_counter),
        "shape_dist": dict(shape_counter),
        "fwd_by_count_stats": {
            str(k): {
                "n": len(v), "min": min(v), "p50": pct(v, 0.5),
                "mean": sum(v)/len(v), "max": max(v),
            } for k, v in fwd_by_count.items()
        },
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved {out_path}")


if __name__ == "__main__":
    main()
