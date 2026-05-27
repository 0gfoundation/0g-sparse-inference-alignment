"""测量 F_vllm: Qwen3-4B 在 vLLM continuous decode 模式下的 per-token decode 时间。

跟 transformers eager 测试（F=47ms）不同，这里走的是 vLLM 的 CUDA graph 优化路径，
这是 B2 (有状态 RM) 路线的真实算力下限。

方法：用"差分法"消除 prefill 影响 — 同一 prompt 跑 max_tokens=K1 和 max_tokens=K2 两次，
per-token = (T(K2) - T(K1)) / (K2 - K1)。

测两个 batch size：
  - batch=1: 模拟 B2 的 fix_a_token（每 LLM SKIP 步异步推 1 token）
  - batch=5: 模拟 B2 的 get_candidates_scores（INTERVENE 时同时评 5 个候选）

注意：用 Qwen3-4B base（生成模型）而非 VM SequenceClassification 版本，因为：
  - vLLM 只能对 generative 模型做 continuous decode
  - 两者层数 / hidden_dim / KV head 数完全相同，per-token decode 时间等价
  - 唯一区别是 VM 末尾换 score head，但 forward 时间几乎不变
"""
import time
import statistics
from vllm import LLM, SamplingParams

MODEL = "/workspace/SIA/models/Qwen3-4B"
PROMPT = "The quick brown fox jumps over the lazy dog. " * 30  # ~270 tokens
L_LLM = 11.3  # LLM 14B 单 token 时间，对照基线
WARMUP_RUNS = 2
TEST_RUNS = 5


def time_generate(llm, prompts, sampling_params):
    """跑一次 generate，返回 wall-clock 秒数。"""
    t = time.perf_counter()
    _ = llm.generate(prompts, sampling_params, use_tqdm=False)
    return time.perf_counter() - t


def main():
    print(f"Loading vLLM with {MODEL} ...", flush=True)
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.3,
        max_model_len=1024,
        enforce_eager=False,  # 启用 CUDA graph (默认已是)
        disable_log_stats=True,
    )
    print(f"vLLM loaded in {time.perf_counter()-t0:.1f}s\n", flush=True)

    # ==== batch=1 测试 ====
    print("=== Phase A: batch=1 (模拟 B2 fix_a_token, +1 token/step) ===")
    sp_short = SamplingParams(temperature=0.0, max_tokens=10, min_tokens=10, ignore_eos=True)
    sp_long  = SamplingParams(temperature=0.0, max_tokens=110, min_tokens=110, ignore_eos=True)

    # Warmup
    for _ in range(WARMUP_RUNS):
        _ = time_generate(llm, [PROMPT], sp_short)
        _ = time_generate(llm, [PROMPT], sp_long)

    # 测短和长
    short_times = [time_generate(llm, [PROMPT], sp_short) for _ in range(TEST_RUNS)]
    long_times  = [time_generate(llm, [PROMPT], sp_long)  for _ in range(TEST_RUNS)]
    T_short = statistics.median(short_times) * 1000
    T_long  = statistics.median(long_times) * 1000
    per_tok_batch1 = (T_long - T_short) / (110 - 10)
    print(f"  T(max_tokens=10):    median={T_short:.1f}ms (n={TEST_RUNS})")
    print(f"  T(max_tokens=110):   median={T_long:.1f}ms (n={TEST_RUNS})")
    print(f"  per-token decode:    (T110 - T10) / 100 = ({T_long:.1f} - {T_short:.1f}) / 100 = "
          f"**{per_tok_batch1:.2f}ms**")

    # ==== batch=5 测试 ====
    print("\n=== Phase B: batch=5 (模拟 B2 get_candidates_scores) ===")
    prompts_b5 = [PROMPT] * 5

    # Warmup
    for _ in range(WARMUP_RUNS):
        _ = time_generate(llm, prompts_b5, sp_short)
        _ = time_generate(llm, prompts_b5, sp_long)

    short_times_b5 = [time_generate(llm, prompts_b5, sp_short) for _ in range(TEST_RUNS)]
    long_times_b5  = [time_generate(llm, prompts_b5, sp_long)  for _ in range(TEST_RUNS)]
    T_short_b5 = statistics.median(short_times_b5) * 1000
    T_long_b5  = statistics.median(long_times_b5) * 1000
    per_tok_batch5 = (T_long_b5 - T_short_b5) / (110 - 10)
    print(f"  T(max_tokens=10, b=5):  median={T_short_b5:.1f}ms (n={TEST_RUNS})")
    print(f"  T(max_tokens=110, b=5): median={T_long_b5:.1f}ms (n={TEST_RUNS})")
    print(f"  per-token decode (batch=5):  (T110 - T10) / 100 = "
          f"**{per_tok_batch5:.2f}ms**")

    # ==== 判定 ====
    print(f"\n=== 判定（对照 L = {L_LLM}ms LLM forward 时间）===")
    F = per_tok_batch1
    S = per_tok_batch5
    print(f"  F (fix_a_token, batch=1): {F:.2f}ms")
    print(f"  S (get_candidates_scores, batch=5): {S:.2f}ms")
    print()
    if F < L_LLM * 0.5:
        print(f"  ✅ F={F:.2f}ms << L={L_LLM}ms → B2 路线必赢，RM 完全藏在 LLM 影子里")
    elif F < L_LLM:
        print(f"  ✅ F={F:.2f}ms < L={L_LLM}ms → B2 路线可行，RM 能藏在 LLM 影子里")
    elif F < L_LLM * 1.5:
        print(f"  ⚠️ F={F:.2f}ms 略 > L={L_LLM}ms → 临界区，部分时段 RM 会拖累 LLM")
    else:
        print(f"  ❌ F={F:.2f}ms >> L={L_LLM}ms → 即便分卡 RM 也是瓶颈，B2 不工作")

    if S < L_LLM * 2:
        print(f"  ✅ S={S:.2f}ms 较低，INTERVENE 步骤的同步阻塞 cost 可接受")
    elif S < L_LLM * 4:
        print(f"  ⚠️ S={S:.2f}ms 偏高，每个 INTERVENE 步骤会 stall LLM 一段时间")
    else:
        print(f"  ❌ S={S:.2f}ms >> L，INTERVENE 同步代价大")


if __name__ == "__main__":
    main()
