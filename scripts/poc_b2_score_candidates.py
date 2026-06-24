"""
M1b PoC — Verify that the reward output by the score head on the B2 path
is numerically consistent with the BF16 baseline (transformers + VM SequenceClassification).

Flow:
  1. Prepare 5 candidates (chat-formatted "answer A/B/C/D/E")
  2. Run vLLM patched path (compute_logits hook computes score) -> rewards_vllm
  3. Run BF16 transformers baseline (VM full forward) -> rewards_bf16
  4. Compute Pearson correlation + per-prompt argmax consistency
"""
import json
import os
import statistics
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REWARDS_FILE = "/tmp/b2_rewards.txt"
# Key: vLLM must also use the VM model (LoRA merged backbone), otherwise hidden_state is not in the same space
# Prerequisite: config.json architectures changed to Qwen3ForCausalLM, qwen3.py patch skips score.weight
MODEL_BASE = "/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm"
MODEL_VM = "/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm"


# ============== 5 candidate prompts ==============
def build_candidates(tok):
    """5 chat-formatted candidate strings, simulating the input scored by RM at SIA INTERVENE steps."""
    base_user = "What is 2 + 2?"
    candidates_text = [" 4", " 5", " 3", " 6", " 100"]  # different answers
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


# ============== Run reward via vLLM patched path ==============
def run_vllm_patched(prompts):
    """Delete old /tmp/b2_rewards.txt, run 5 prompts in one forward (max_tokens=1), collect rewards"""
    # Clean up old file
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

    # Run 5-prompt batch
    print(f"[vLLM] Generating reward for {len(prompts)} candidates...",
          flush=True)
    t0 = time.perf_counter()
    _ = llm.generate(prompts, sp, use_tqdm=False)
    dt = (time.perf_counter() - t0) * 1000
    print(f"[vLLM] Generate done in {dt:.1f}ms (including prefill)", flush=True)

    # Measure steady-state batch=5 latency (3 warmup runs then 5 measurement runs)
    for _ in range(3):
        _ = llm.generate(prompts, sp, use_tqdm=False)
    timings = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = llm.generate(prompts, sp, use_tqdm=False)
        timings.append((time.perf_counter() - t0) * 1000)
    p50 = statistics.median(timings)
    print(f"[vLLM] batch=5 generate p50 = {p50:.1f}ms (including prefill, 5 prompts)",
          flush=True)

    # Release GPU for the transformers baseline
    del llm
    torch.cuda.empty_cache()

    return p50


def parse_rewards_file():
    """Read /tmp/b2_rewards.txt and extract the rewards list from each compute_logits call"""
    if not os.path.exists(REWARDS_FILE):
        return []
    out = []
    for line in open(REWARDS_FILE):
        # Format: call#N shape=(...) rewards=[r1, r2, ...]
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

    # Prepare candidates
    print("\n[1/3] Preparing 5 candidate prompts ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_VM, trust_remote_code=True)
    prompts, candidates_text = build_candidates(tok)
    for i, (p, c) in enumerate(zip(prompts, candidates_text)):
        print(f"  candidate {i}: response='{c.strip()}' "
              f"(prompt {len(p)} chars)")

    # vLLM path
    print("\n[2/3] Running vLLM patched path ...", flush=True)
    vllm_latency = run_vllm_patched(prompts)
    vllm_calls = parse_rewards_file()
    print(f"\n[vLLM] Received {len(vllm_calls)} compute_logits calls", flush=True)
    if not vllm_calls:
        print("  No rewards received — patch may not have taken effect")
        return

    # Take the last call containing 5 rewards (should be the batch=5 sampling call)
    # Note: prefill hidden_states shape is not (5, ...), so filter those out
    vllm_rewards = None
    for r_list in reversed(vllm_calls):
        if len(r_list) == 5:
            vllm_rewards = r_list
            break
    if vllm_rewards is None:
        print(f"  No batch=5 call found. All calls:")
        for r in vllm_calls[-10:]:
            print(f"     len={len(r)}: {r[:3]}...")
        return
    print(f"  vLLM rewards (batch=5): {[f'{r:.3f}' for r in vllm_rewards]}")

    # BF16 baseline
    print(f"\n[3/3] Running BF16 transformers baseline ...", flush=True)
    bf16_rewards = run_bf16_baseline(prompts)
    print(f"  BF16 rewards: {[f'{r:.3f}' for r in bf16_rewards]}")

    # Compare
    print(f"\n=== Numerical Comparison ===")
    print(f"  candidate  vLLM        BF16        |delta|")
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
        # compute manually
        import math
        mv = sum(vllm_rewards) / len(vllm_rewards)
        mb = sum(bf16_rewards) / len(bf16_rewards)
        cov = sum((v - mv) * (b - mb) for v, b in zip(vllm_rewards, bf16_rewards))
        var_v = sum((v - mv) ** 2 for v in vllm_rewards)
        var_b = sum((b - mb) ** 2 for b in bf16_rewards)
        r = cov / math.sqrt(var_v * var_b + 1e-9)
        print(f"  Pearson correlation (manual): {r:.4f}")

    # Top-1 consistency
    vllm_top1 = max(range(5), key=lambda i: vllm_rewards[i])
    bf16_top1 = max(range(5), key=lambda i: bf16_rewards[i])
    print(f"  Top-1 candidate: vLLM='{candidates_text[vllm_top1].strip()}' "
          f"BF16='{candidates_text[bf16_top1].strip()}' "
          f"{'MATCH' if vllm_top1 == bf16_top1 else 'MISMATCH'}")

    print(f"\n=== M1b Acceptance ===")
    pass_corr = r > 0.99
    pass_top1 = vllm_top1 == bf16_top1
    pass_diff = max(diffs) < 0.5
    if pass_corr and pass_top1 and pass_diff:
        print(f"  PASS: Pearson={r:.4f} > 0.99, Top-1 consistent, max|delta|={max(diffs):.3f} < 0.5")
        print(f"  -> Proceed to M2/M3, expected ~49 tok/s on single GPU")
    elif r > 0.95 and pass_top1:
        print(f"  PARTIAL PASS: Pearson={r:.4f}, Top-1 consistent — can proceed to M2, but verify on more prompts")
    else:
        print(f"  FAIL: Pearson={r:.4f}, Top-1 {'consistent' if pass_top1 else 'inconsistent'} "
              f"— investigate hidden_state extraction position or score head loading")
    print(f"\n  vLLM batch=5 generate p50 latency: {vllm_latency:.1f}ms "
          f"(including prefill; pure decode part ~7ms from measure_rm_continuous_decode.py)")


if __name__ == "__main__":
    main()
