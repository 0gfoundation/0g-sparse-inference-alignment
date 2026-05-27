"""
M2 §7.A 验证脚本.

测 prefix caching 命中率: 维护一个不断增长的 prefix, 每次给它 append 1 token,
然后跑 5 prompt batch generate (max_tokens=1).

期望: 因为 5 个 prompt 共享前 N-1 token, 第 1 次后 KV cache 命中, 每次 forward
只算 1 个新 token, p50 latency ~10-15ms (vs 第 1 次 cold prefill ~50-100ms).

PASS 条件:
  - 稳态 p50 < 20ms : ✅ M2 设计可行
  - 稳态 p50 30-50ms: ⚠️ 仍可投但要手动 warm 或换 stateful API
  - 稳态 p50 > 50ms : ❌ prefix caching 没省到, 改方案

跑法:
  python scripts/bench_b2_prefix_cache.py
"""
import statistics
import time
import torch

MODEL = "/workspace/SIA/models/Qwen3-4B"  # 用 base 模型即可, 不需要 VM
START_PREFIX_LEN = 50  # 起始 prompt 长度 (tokens)
N_ITER = 100  # 测多少次
N_WARMUP = 5
N_CANDIDATES = 5  # 每次 5 个 candidate (跟 SIA 实际场景一致)


def main():
    from vllm import LLM, SamplingParams, TokensPrompt

    print("=" * 60)
    print("§7.A: prefix caching 命中率 + score_candidates latency")
    print("=" * 60)

    print(f"\nLoading vLLM (prefix_caching=ON) ...", flush=True)
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.3,
        max_model_len=2048,
        enable_prefix_caching=True,
        enforce_eager=False,
        disable_log_stats=True,
    )
    print(f"Loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1,
                        ignore_eos=True)

    # 初始 prefix
    import random
    random.seed(42)
    prefix = [random.randint(1000, 5000) for _ in range(START_PREFIX_LEN)]
    # 5 个 candidate token (随便)
    candidates = [100, 200, 300, 400, 500]

    def one_iter():
        """Score N_CANDIDATES candidates: build N prompts = prefix + [c]"""
        prompts = [TokensPrompt(prompt_token_ids=prefix + [c])
                   for c in candidates]
        t = time.perf_counter()
        _ = llm.generate(prompts, sp, use_tqdm=False)
        return (time.perf_counter() - t) * 1000

    # warmup: 头几次会 cold prefill
    print(f"\nWarmup ({N_WARMUP} iter)...", flush=True)
    warmup_ts = []
    for i in range(N_WARMUP):
        t = one_iter()
        warmup_ts.append(t)
        prefix.append(random.randint(1000, 5000))
        print(f"  warmup #{i}: {t:.1f}ms (prefix len now {len(prefix)})",
              flush=True)

    # 正式测量
    print(f"\nMeasuring ({N_ITER} iter, prefix grows from "
          f"{len(prefix)} → {len(prefix)+N_ITER})...", flush=True)
    timings = []
    for i in range(N_ITER):
        t = one_iter()
        timings.append(t)
        prefix.append(random.randint(1000, 5000))
        if i < 5 or i % 20 == 0:
            print(f"  iter #{i:3d}: {t:.1f}ms (prefix len {len(prefix)})",
                  flush=True)

    # 统计
    p50 = statistics.median(timings)
    p95 = sorted(timings)[int(0.95 * len(timings))]
    p99 = sorted(timings)[int(0.99 * len(timings))]
    avg = sum(timings) / len(timings)
    mn = min(timings)
    mx = max(timings)

    print(f"\n=== Latency stats (5 candidates batched, {N_ITER} iters) ===")
    print(f"  min  : {mn:6.2f} ms")
    print(f"  p50  : {p50:6.2f} ms")
    print(f"  avg  : {avg:6.2f} ms")
    print(f"  p95  : {p95:6.2f} ms")
    print(f"  p99  : {p99:6.2f} ms")
    print(f"  max  : {mx:6.2f} ms")

    print(f"\n=== §7.A 判定 ===")
    if p50 < 20:
        print(f"  ✅ PASS: p50 = {p50:.1f}ms < 20ms")
        print(f"  → prefix caching 命中良好, M2 设计可行")
    elif p50 < 50:
        print(f"  ⚠️ MARGINAL: p50 = {p50:.1f}ms (20-50ms)")
        print(f"  → 仍可投 M2, 但实际收益打折; 考虑手动 warm KV")
    else:
        print(f"  ❌ FAIL: p50 = {p50:.1f}ms > 50ms")
        print(f"  → prefix caching 不够, 需换方案 (stateful API / 手动 KV 管理)")

    # 关键: trajectory 看延迟是否随 prefix 增长稳定
    print(f"\n=== Latency trajectory (每 10 个 iter 的 p50) ===")
    for chunk_start in range(0, N_ITER, 10):
        chunk = timings[chunk_start:chunk_start+10]
        p = statistics.median(chunk)
        print(f"  iter {chunk_start:3d}-{chunk_start+9:3d}: "
              f"p50={p:.1f}ms (prefix len ~{START_PREFIX_LEN+N_WARMUP+chunk_start})")


if __name__ == "__main__":
    main()
