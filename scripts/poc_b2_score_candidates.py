"""
M1b PoC — 验证 B2 路径下 score head 输出的 reward
跟 BF16 baseline (transformers + VM SequenceClassification) 数值一致。

流程：
  1. 准备 5 个 candidate（chat-formatted "answer A/B/C/D/E"）
  2. 跑 vLLM patched 路径（compute_logits hook 算 score）→ rewards_vllm
  3. 跑 BF16 transformers baseline (VM full forward) → rewards_bf16
  4. 计算 Pearson + per-prompt argmax 一致性
"""
import json
import os
import statistics
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REWARDS_FILE = "/tmp/b2_rewards.txt"
# 关键: vLLM 也必须用 VM 模型 (LoRA merged backbone), 否则 hidden_state 不在同一空间
# 前提: config.json 的 architectures 改成 Qwen3ForCausalLM, qwen3.py patch 里 skip score.weight
MODEL_BASE = "/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm"
MODEL_VM = "/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm"


# ============== 5 个 candidate prompts ==============
def build_candidates(tok):
    """5 个 chat-formatted candidate strings, 模拟 SIA INTERVENE 步给 RM 评分的输入。"""
    base_user = "What is 2 + 2?"
    candidates_text = [" 4", " 5", " 3", " 6", " 100"]  # 不同回答
    prompts = []
    for cand in candidates_text:
        convs = [
            {"role": "user", "content": base_user},
            {"role": "assistant", "content": cand},
        ]
        text = tok.apply_chat_template(convs, tokenize=False)
        bos = tok.bos_token
        if bos and text.startswith(bos):
            text = text[len(bos):]
        prompts.append(text)
    return prompts, candidates_text


# ============== vLLM patched 路径跑 reward ==============
def run_vllm_patched(prompts):
    """删除旧 /tmp/b2_rewards.txt，跑 5 prompts 一次 forward (max_tokens=1)，收 rewards"""
    # 清旧
    if os.path.exists(REWARDS_FILE):
        os.remove(REWARDS_FILE)

    from vllm import LLM, SamplingParams
    print(f"[vLLM] Loading {MODEL_BASE} ...", flush=True)
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL_BASE,
        dtype="bfloat16",
        gpu_memory_utilization=0.3,
        max_model_len=1024,
        enforce_eager=False,
        disable_log_stats=True,
    )
    print(f"[vLLM] Loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1,
                        ignore_eos=True)

    # 跑 5 prompts batch
    print(f"[vLLM] Generating reward for {len(prompts)} candidates...",
          flush=True)
    t0 = time.perf_counter()
    _ = llm.generate(prompts, sp, use_tqdm=False)
    dt = (time.perf_counter() - t0) * 1000
    print(f"[vLLM] Generate done in {dt:.1f}ms (含 prefill)", flush=True)

    # 测稳态 batch=5 latency（warmup 3 次后 测 5 次）
    for _ in range(3):
        _ = llm.generate(prompts, sp, use_tqdm=False)
    timings = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = llm.generate(prompts, sp, use_tqdm=False)
        timings.append((time.perf_counter() - t0) * 1000)
    p50 = statistics.median(timings)
    print(f"[vLLM] batch=5 generate p50 = {p50:.1f}ms (含 prefill, 5 prompts)",
          flush=True)

    # 释放 GPU 给 transformers baseline
    del llm
    torch.cuda.empty_cache()

    return p50


def parse_rewards_file():
    """读 /tmp/b2_rewards.txt 提取每次 compute_logits 调用的 rewards 列表"""
    if not os.path.exists(REWARDS_FILE):
        return []
    out = []
    for line in open(REWARDS_FILE):
        # 格式: call#N shape=(...) rewards=[r1, r2, ...]
        if "rewards=" not in line:
            continue
        try:
            rewards_str = line.split("rewards=")[1].strip()
            rewards = json.loads(rewards_str)
            out.append(rewards)
        except Exception:
            pass
    return out


# ============== BF16 baseline ==============
def run_bf16_baseline(prompts):
    print(f"[BF16] Loading {MODEL_VM} ...", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_VM, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(MODEL_VM, trust_remote_code=True)
    print(f"[BF16] Loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    rewards = []
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt").input_ids.to("cuda")
            out = model(ids)
            # out.logits shape (1, 1)
            r = out.logits.squeeze().float().item()
            rewards.append(r)
    return rewards


def main():
    print("=" * 60)
    print("M1b PoC: vLLM patched score vs BF16 transformers baseline")
    print("=" * 60)

    # 准备 candidates
    print("\n[1/3] 准备 5 个 candidate prompts ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_VM, trust_remote_code=True)
    prompts, candidates_text = build_candidates(tok)
    for i, (p, c) in enumerate(zip(prompts, candidates_text)):
        print(f"  candidate {i}: response='{c.strip()}' "
              f"(prompt {len(p)} chars)")

    # vLLM 路径
    print("\n[2/3] 跑 vLLM patched 路径 ...", flush=True)
    vllm_latency = run_vllm_patched(prompts)
    vllm_calls = parse_rewards_file()
    print(f"\n[vLLM] 收到 {len(vllm_calls)} 次 compute_logits 调用", flush=True)
    if not vllm_calls:
        print("  ❌ 没收到 rewards — patch 可能没生效")
        return

    # 取最后一次包含 5 个 reward 的调用（应是 batch=5 的 sampling call）
    # 注意 prefill 的 hidden_states shape 不是 (5, ...)，所以过滤
    vllm_rewards = None
    for r_list in reversed(vllm_calls):
        if len(r_list) == 5:
            vllm_rewards = r_list
            break
    if vllm_rewards is None:
        print(f"  ❌ 找不到 batch=5 的调用。所有调用:")
        for r in vllm_calls[-10:]:
            print(f"     len={len(r)}: {r[:3]}...")
        return
    print(f"  vLLM rewards (batch=5): {[f'{r:.3f}' for r in vllm_rewards]}")

    # BF16 baseline
    print(f"\n[3/3] 跑 BF16 transformers baseline ...", flush=True)
    bf16_rewards = run_bf16_baseline(prompts)
    print(f"  BF16 rewards: {[f'{r:.3f}' for r in bf16_rewards]}")

    # 对比
    print(f"\n=== 数值对比 ===")
    print(f"  candidate  vLLM        BF16        |Δ|")
    diffs = []
    for i, (c, v, b) in enumerate(zip(candidates_text, vllm_rewards, bf16_rewards)):
        d = abs(v - b)
        diffs.append(d)
        print(f"  '{c.strip():>5s}'    {v:+8.4f}    {b:+8.4f}    {d:.4f}")
    print(f"  mean |Δ|: {statistics.mean(diffs):.4f}")
    print(f"  max  |Δ|: {max(diffs):.4f}")

    # Pearson
    try:
        from scipy.stats import pearsonr
        r, p = pearsonr(vllm_rewards, bf16_rewards)
        print(f"  Pearson correlation: {r:.4f} (p={p:.3g})")
    except Exception:
        # 手算
        import math
        mv = sum(vllm_rewards) / len(vllm_rewards)
        mb = sum(bf16_rewards) / len(bf16_rewards)
        cov = sum((v - mv) * (b - mb) for v, b in zip(vllm_rewards, bf16_rewards))
        var_v = sum((v - mv) ** 2 for v in vllm_rewards)
        var_b = sum((b - mb) ** 2 for b in bf16_rewards)
        r = cov / math.sqrt(var_v * var_b + 1e-9)
        print(f"  Pearson correlation (manual): {r:.4f}")

    # Top-1 一致
    vllm_top1 = max(range(5), key=lambda i: vllm_rewards[i])
    bf16_top1 = max(range(5), key=lambda i: bf16_rewards[i])
    print(f"  Top-1 candidate: vLLM='{candidates_text[vllm_top1].strip()}' "
          f"BF16='{candidates_text[bf16_top1].strip()}' "
          f"{'✅ MATCH' if vllm_top1 == bf16_top1 else '❌ MISMATCH'}")

    print(f"\n=== M1b 验收 ===")
    pass_corr = r > 0.99
    pass_top1 = vllm_top1 == bf16_top1
    pass_diff = max(diffs) < 0.5
    if pass_corr and pass_top1 and pass_diff:
        print(f"  ✅ PASS: Pearson={r:.4f} > 0.99, Top-1 一致, max|Δ|={max(diffs):.3f} < 0.5")
        print(f"  → 投 M2/M3, 预期单卡 ~49 tok/s")
    elif r > 0.95 and pass_top1:
        print(f"  ⚠️ 部分通过: Pearson={r:.4f}, Top-1 一致 — 可投 M2，但需复测更多 prompt 验证")
    else:
        print(f"  ❌ FAIL: Pearson={r:.4f}, Top-1 {'同' if pass_top1 else '不同'} "
              f"— 需排查 hidden_state 提取位置或 score head 加载")
    print(f"\n  vLLM batch=5 generate p50 latency: {vllm_latency:.1f}ms "
          f"(含 prefill；纯 decode part ~7ms 来自 measure_rm_continuous_decode.py)")


if __name__ == "__main__":
    main()
