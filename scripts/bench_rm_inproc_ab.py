"""
A/B test: VLLM_ENABLE_V1_MULTIPROCESSING=0 (InprocClient) vs default (SyncMPClient)

The 1+4 split observed in `bench_rm_batch5_ab.py` is caused by a ZMQ race
in vLLM's default subprocess-based EngineCoreProc: the 5 add_request calls
serialize over ZMQ, and req#1 arrives at the EngineCore subprocess before
req#2-5 do, triggering a single-prompt scheduler step on req#1 alone.

Setting VLLM_ENABLE_V1_MULTIPROCESSING=0 makes vLLM use InprocClient
(EngineCore in the same process, plain method calls, no ZMQ). All 5
add_request calls complete synchronously before _run_engine starts
stepping the scheduler, so the scheduler picks up all 5 in one schedule()
call and runs a single batch=5 forward.

Run:
    SIA_RM_PROFILE=1 python scripts/bench_rm_inproc_ab.py default
    SIA_RM_PROFILE=1 python scripts/bench_rm_inproc_ab.py inproc

Output: /tmp/bench_rm_inproc_ab_<variant>.json
"""
import json
import os
import random
import sys
import time
import uuid

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
sys.path.insert(0, _SRC)
_pp = os.environ.get("PYTHONPATH", "")
if _SRC not in _pp.split(":"):
    os.environ["PYTHONPATH"] = (_SRC + ":" + _pp) if _pp else _SRC

if len(sys.argv) != 2 or sys.argv[1] not in ("default", "inproc"):
    print(f"Usage: {sys.argv[0]} {{default|inproc}}")
    sys.exit(1)
variant = sys.argv[1]

# IMPORTANT: must be set BEFORE importing vllm
if variant == "inproc":
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

os.environ["SIA_RM_PROFILE"] = "1"
os.environ["SIA_REWARD_FILE_ID"] = f"bench_inproc_{uuid.uuid4().hex[:8]}"

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
    out_path = f"/tmp/bench_rm_inproc_ab_{variant}.json"

    print("=" * 78)
    print(f"InprocClient A/B bench — variant = {variant}")
    print(f"  VLLM_ENABLE_V1_MULTIPROCESSING = "
          f"{os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING', '<unset, defaults to 1>')}")
    print(f"  N_WARMUP / N_MEASURE: {N_WARMUP} / {N_MEASURE}")
    print(f"  PREFIX_INIT       : {PREFIX_INIT}")
    print(f"  SKIP per INTERVENE: {SKIP_LO}..{SKIP_HI}")
    print(f"  output JSON       : {out_path}")
    print("=" * 78)

    print("Loading RMClient ...")
    t0 = time.perf_counter()
    rm = RMClient(model_path=MODEL_VM, gpu_mem=0.3, max_model_len=4096)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s\n")

    random.seed(42)
    initial_prefix = [random.randint(1000, 100000)
                      for _ in range(PREFIX_INIT)]
    sid = rm.new_session(initial_prefix)
    candidates = [random.randint(1000, 100000) for _ in range(N_CANDIDATES)]

    def one_iter():
        prefix = rm._sessions[sid]
        n_already = rm._last_prefilled.get(sid, 0)
        n_new = len(prefix) - n_already
        TP = rm._TokensPrompt

        truncate_rewards()
        truncate_timings()
        a_t0 = time.perf_counter()
        if n_new > 0:
            rm.llm.generate([TP(prompt_token_ids=prefix)],
                            rm._sp, use_tqdm=False)
        a_wall = (time.perf_counter() - a_t0) * 1000
        a_timings = read_all_timings()

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

    print(f"Warmup ({N_WARMUP}) ...")
    for i in range(N_WARMUP):
        for _ in range(random.randint(SKIP_LO, SKIP_HI)):
            rm.fix_a_token(sid, random.randint(1000, 100000))
        r = one_iter()
        print(f"  warmup #{i}: n_new={r['n_new']} b_n_fwd={r['b_n_forwards']} "
              f"b_fwd_sum={r['b_fwd_sum']:.2f}ms shapes={[f['n_samples'] for f in r['b_forwards']]}")
    print()

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

    b_n_fwd = [r["b_n_forwards"] for r in records]
    b_fwd_sum = [r["b_fwd_sum"] for r in records]
    b_score_sum = [r["b_score_sum"] for r in records]
    b_wall = [r["b_wall"] for r in records]

    from collections import Counter
    nfwd_counter = Counter(b_n_fwd)
    shape_counter = Counter()
    for r in records:
        for f in r["b_forwards"]:
            shape_counter[f["n_samples"]] += 1

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

    summary = {
        "variant": variant,
        "multiprocessing": variant != "inproc",
        "n_warmup": N_WARMUP, "n_measure": N_MEASURE,
        "prefix_init": PREFIX_INIT,
        "n_candidates": N_CANDIDATES,
        "b_fwd_sum": {
            "min": min(b_fwd_sum), "p50": pct(b_fwd_sum, 0.5),
            "mean": sum(b_fwd_sum)/len(b_fwd_sum),
            "p95": pct(b_fwd_sum, 0.95), "max": max(b_fwd_sum),
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
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved {out_path}")


if __name__ == "__main__":
    main()
