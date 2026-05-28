"""
M1a PoC — 验证能否从 vLLM 0.10.1.1 的 decode loop 里提取 hidden_state，
且不破坏 CUDA graph 优化（per-token 时间仍 ≈ 7ms）。

核心 insight (源码确认)：
  - vLLM 的 Qwen3ForCausalLM.forward()  返回 hidden_states（不 apply lm_head）
  - Qwen3ForCausalLM.compute_logits() 才 apply lm_head
  - compute_logits 在 CUDA graph 外面调用
  - → 我们可以在 compute_logits 里挂钩，拿到 hidden_state，且不影响图

PoC 步骤：
  1. Monkey-patch Qwen3ForCausalLM.compute_logits，加入 side channel 写 hidden_state
  2. 用 vllm.LLM 跑 continuous decode
  3. 验证 hidden_state 是否被捕获
  4. 测 per-token 时间，确认 CUDA graph 仍生效（~7ms）

详见 doc/b2-decode-mode-poc-plan.md §5.1
"""
import statistics
import time

import torch

# 在 import vllm 之前做 monkey-patch
import vllm.model_executor.models.qwen3 as _qwen3_mod

# 类级别 side channel —— 不同实例共享同一个引用
_qwen3_mod.Qwen3ForCausalLM.last_hidden_state = None

_original_compute_logits = _qwen3_mod.Qwen3ForCausalLM.compute_logits


def _patched_compute_logits(self, hidden_states, sampling_metadata):
    """Hook：记录 hidden_state 到类属性，然后照常 apply lm_head。"""
    # hidden_states 在 compute_logits 入口已经是 "sample position" 的，
    # 即 shape (n_samples, hidden_dim) 而不是 (batch, seq, hidden_dim)。
    # 我们直接保留引用（不 clone，省 GPU 内存）
    type(self).last_hidden_state = hidden_states
    return _original_compute_logits(self, hidden_states, sampling_metadata)


_qwen3_mod.Qwen3ForCausalLM.compute_logits = _patched_compute_logits
print("[patch] Qwen3ForCausalLM.compute_logits hooked", flush=True)


# ---- 现在再 import vllm 主模块 + 跑测试 ----
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
        enforce_eager=False,  # 关键：保留 CUDA graph 优化
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

    # 测延迟
    print("Timing (batch=1)...", flush=True)
    ts_short = [time_generate(llm, [PROMPT], sp_short) for _ in range(N)]
    ts_long  = [time_generate(llm, [PROMPT], sp_long)  for _ in range(N)]
    T_short = statistics.median(ts_short) * 1000
    T_long  = statistics.median(ts_long)  * 1000
    per_tok = (T_long - T_short) / 100
    print(f"  T(max_tokens=10):  median={T_short:.1f}ms")
    print(f"  T(max_tokens=110): median={T_long:.1f}ms")
    print(f"  per-token decode:  {per_tok:.2f}ms")
    print(f"  (对照 baseline 无 patch: 7.02ms。允许 +1-2ms 的 patch overhead)")

    # 验证 hidden_state 被捕获
    print(f"\n=== 验证 last_hidden_state 被捕获 ===", flush=True)
    hs = _qwen3_mod.Qwen3ForCausalLM.last_hidden_state
    if hs is None:
        print(f"  ❌ last_hidden_state 仍是 None — monkey-patch 未生效（可能是 subprocess 隔离）")
        print(f"     需要换方案：register custom class via ModelRegistry，或 patch vLLM 源码")
        return
    print(f"  ✅ hidden_state 抓到了!")
    print(f"     shape:  {tuple(hs.shape)}")
    print(f"     dtype:  {hs.dtype}")
    print(f"     device: {hs.device}")
    print(f"     mean:   {hs.mean().item():.4f}")
    print(f"     std:    {hs.std().item():.4f}")

    # ==== M1a 总判定 ====
    print(f"\n=== M1a 验收 ===")
    if hs is not None and per_tok < 10:
        print(f"  ✅ PASS: hidden_state 捕获 + per-token = {per_tok:.2f}ms < 10ms")
        print(f"  → 进 M1b: 加 score_head, 实测 5-candidate scoring")
    elif hs is not None and per_tok < 15:
        print(f"  ⚠️ 部分通过: hidden_state 捕获，但 per-token = {per_tok:.2f}ms 偏慢")
        print(f"     可能 CUDA graph 部分降级。仍可进 M1b，但调低预期")
    elif hs is not None:
        print(f"  ⚠️ hidden_state 捕获，但 per-token = {per_tok:.2f}ms 明显退化")
        print(f"     CUDA graph 大概率没启用。需检查 monkey-patch 是否引入了未计入图的 op")
    else:
        print(f"  ❌ FAIL: hidden_state 没拿到。")
        print(f"     原因可能：vLLM EngineCore 在 subprocess 启动，monkey-patch 不跨进程")
        print(f"     下一步：尝试 vllm.ModelRegistry.register_model 注册子类")


if __name__ == "__main__":
    main()
