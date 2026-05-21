"""
模拟真实 SIA 调用：从 response_so_far=空开始，每步 5 个 candidate 打分，
response_so_far 每步加 1 个 token。共 ~1000 步，覆盖 prefix 1~1000 token 长度。

对比当前 PyTorch RM server 的实测数据（来自最近一次 baseline 跑）：
  - Forward p50 = 57ms (无 cuda graph) / 36ms (cuda graph 工作时)
  - 完整 RM 调用 p50 = ~80ms (含 tokenize + chat_template + kv_save)
"""

import requests
import time
import statistics
from collections import defaultdict

URL = "http://localhost:8001/classify"
MODEL = "/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm"
N_STEPS = 1000          # 总步数
N_CANDIDATES = 5        # SIA topk

# 用一段简单可控的内容，每步加 1 个 token-ish 单词
USER = (
    "Please write a long detailed essay explaining the history of artificial "
    "intelligence research, including key milestones and influential people."
)

# 构造每个 step 的 candidate token（差异在最后 1 个 token）
def make_candidates(idx: int):
    # idx 决定真实加进 response 的"token"内容
    # 5 个 candidate 只在最后 1 个不同
    base_cands = [" the", " of", " and", " a", " research"]
    return base_cands  # 不依赖 idx，简化测试

def fmt(user: str, response_so_far: str, candidate: str) -> str:
    return (
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{response_so_far}{candidate}<|im_end|>"
    )


def main():
    session = requests.Session()

    # Warmup
    print("Warmup (5 calls)...")
    sample_texts = [fmt(USER, "", c) for c in make_candidates(0)]
    for _ in range(5):
        session.post(URL, json={"model": MODEL, "input": sample_texts,
                                 "activation": False}).json()

    print(f"\nRunning {N_STEPS} steps, {N_CANDIDATES} candidates per step...\n")

    # 真实推理：response 每步追加一个 token 的内容
    # 用单词模拟，每步追加一个英文词；最后 prefix 长度大致跟单词数对齐
    growing_response = ""
    word_pool = [
        "artificial", "intelligence", "research", "began", "with", "early",
        "experiments", "in", "logic", "and", "problem-solving", "during",
        "the", "mid", "twentieth", "century", "researchers", "like", "Alan",
        "Turing", "proposed", "fundamental", "ideas", "about", "machine",
        "thinking", "later", "John", "McCarthy", "coined", "the", "term",
        "at", "Dartmouth", "in", "1956", "establishing", "the", "field",
        "as", "a", "formal", "discipline", "subsequent", "decades", "saw",
        "developments", "in", "expert", "systems", "neural", "networks",
        "and", "knowledge", "representation",
    ]

    # 每步 latency
    latencies_per_step: list = []
    # 每步对应的 input token 长度（取第一个 candidate 的 token 数作代表）
    seq_lens: list = []

    t_total_start = time.perf_counter()
    for step in range(N_STEPS):
        # 每步加一个新单词到 response_so_far
        new_word = word_pool[step % len(word_pool)]
        growing_response += " " + new_word

        cands = make_candidates(step)
        texts = [fmt(USER, growing_response, c) for c in cands]

        t0 = time.perf_counter()
        r = session.post(
            URL,
            json={"model": MODEL, "input": texts, "activation": False},
        )
        ms = (time.perf_counter() - t0) * 1000
        latencies_per_step.append(ms)
        resp = r.json()
        # 取 vLLM 报的 prompt token 数（5 个 candidate token 数总和）
        total_tokens = resp.get("usage", {}).get("prompt_tokens", 0)
        seq_lens.append(total_tokens // N_CANDIDATES)

        if (step + 1) % 100 == 0:
            recent = latencies_per_step[-100:]
            print(
                f"  step {step + 1:4d}  approx_kv_len={seq_lens[-1]:4d}  "
                f"recent-100 p50={statistics.median(recent):.1f}ms  "
                f"p95={sorted(recent)[int(len(recent)*0.95)]:.1f}ms"
            )

    t_total = time.perf_counter() - t_total_start

    # === 分桶分析 ===
    buckets = defaultdict(list)
    for ms, sl in zip(latencies_per_step, seq_lens):
        if sl < 50:
            b = "<50"
        elif sl < 100:
            b = "50-100"
        elif sl < 200:
            b = "100-200"
        elif sl < 300:
            b = "200-300"
        elif sl < 500:
            b = "300-500"
        elif sl < 800:
            b = "500-800"
        else:
            b = "800+"
        buckets[b].append(ms)

    print(f"\n{'=' * 70}")
    print(f"Total: {N_STEPS} RM calls, each batch={N_CANDIDATES}")
    print(f"Total wall time: {t_total:.1f}s = {t_total / N_STEPS * 1000:.1f}ms per call")
    print(f"{'=' * 70}")
    print(f"\n{'kv_len':>10}  {'n':>6}  {'p50':>8}  {'p90':>8}  {'p99':>8}  {'max':>8}")
    print("-" * 60)
    bucket_order = ["<50", "50-100", "100-200", "200-300", "300-500", "500-800", "800+"]
    for b in bucket_order:
        if b not in buckets:
            continue
        v = sorted(buckets[b])
        n = len(v)
        if n == 0:
            continue
        print(
            f"{b:>10}  {n:>6}  "
            f"{v[n // 2]:>7.1f}ms  "
            f"{v[int(n * 0.9)]:>7.1f}ms  "
            f"{v[min(int(n * 0.99), n - 1)]:>7.1f}ms  "
            f"{v[-1]:>7.1f}ms"
        )

    # === Overall ===
    v = sorted(latencies_per_step)
    n = len(v)
    print(f"\n{'overall':>10}  {n:>6}  "
          f"{v[n // 2]:>7.1f}ms  "
          f"{v[int(n * 0.9)]:>7.1f}ms  "
          f"{v[min(int(n * 0.99), n - 1)]:>7.1f}ms  "
          f"{v[-1]:>7.1f}ms")

    # === Comparison ===
    print(f"\n{'=' * 70}")
    print("Comparison vs current PyTorch RM (from previous profiling runs):")
    print(f"{'=' * 70}")
    print(f"  Current RM (Flash eager + KV cache):")
    print(f"    forward p50 = ~57ms, total per call p50 = ~80ms (含 tokenize + chat_tpl + kv_save)")
    print(f"    1000 calls ≈ 80s wall time")
    print()
    print(f"  Current RM (CG bucket 工作时):")
    print(f"    forward p50 = ~36ms, total per call p50 = ~40ms")
    print(f"    但只在前 ~270 次调用工作，之后 NaN 触发 disable")
    print()
    print(f"  vLLM (this benchmark):")
    print(f"    total per call p50 = {statistics.median(latencies_per_step):.1f}ms")
    print(f"    1000 calls = {t_total:.1f}s")


if __name__ == "__main__":
    main()
