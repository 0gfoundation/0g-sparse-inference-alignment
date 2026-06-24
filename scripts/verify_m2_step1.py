"""
M2 step 1 verification: sia_rm.qwen3_with_score runs 5 candidates, compared against BF16 baseline.

Uses a /dev/shm file channel to cross the vLLM EngineCore subprocess boundary.

PASS criteria (same as M1b):
  Pearson > 0.99 + Top-1 match + max|Δ| < 0.5

Run (from SIA repo root):
  python scripts/verify_m2_step1.py
"""
import json
import os
import statistics
import sys
import time
import uuid

# Add src/ to sys.path + PYTHONPATH (the latter is for the vLLM subprocess)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
sys.path.insert(0, _SRC)
_pp = os.environ.get("PYTHONPATH", "")
if _SRC not in _pp.split(":"):
    os.environ["PYTHONPATH"] = (_SRC + ":" + _pp) if _pp else _SRC

# Set a unique id for the reward file to avoid conflicts with other RMClient instances
_FID = f"verify_{uuid.uuid4().hex[:8]}"
os.environ["SIA_REWARD_FILE_ID"] = _FID
print(f"[verify] PYTHONPATH = {os.environ['PYTHONPATH']}", flush=True)
print(f"[verify] SIA_REWARD_FILE_ID = {_FID}", flush=True)
print(f"[verify] reward channel path = /dev/shm/sia_reward_{_FID}.bin",
      flush=True)

import torch  # noqa: E402
import sia_rm  # noqa: E402, F401 — triggers ModelRegistry.register_model
from sia_rm import read_rewards, read_all_rewards, truncate_rewards  # noqa: E402
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # noqa

MODEL_VM = "/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm"


def build_candidates(tok):
    base_user = "What is 2 + 2?"
    candidates_text = [" 4", " 5", " 3", " 6", " 100"]
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


def run_vllm(prompts):
    from vllm import LLM, SamplingParams

    # Clear old reward file
    truncate_rewards()

    print(f"[vLLM] Loading VM with Qwen3WithScoreForCausalLM ...", flush=True)
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL_VM,
        hf_overrides={"architectures": ["Qwen3WithScoreForCausalLM"]},
        dtype="bfloat16",
        gpu_memory_utilization=0.3,
        max_model_len=1024,
        enable_prefix_caching=True,
        enforce_eager=False,
        disable_log_stats=True,
    )
    print(f"[vLLM] Loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1,
                        ignore_eos=True)

    # batch generate 5 prompts
    print(f"[vLLM] Generating reward for {len(prompts)} candidates...",
          flush=True)
    truncate_rewards()  # clear warmup records from vLLM startup
    _ = llm.generate(prompts, sp, use_tqdm=False)

    # View all compute_logits calls (debug multiple forward calls issue)
    records = read_all_rewards()
    print(f"  compute_logits called {len(records)} times:")
    for i, r in enumerate(records):
        print(f"    record #{i}: shape={tuple(r.shape)} values={r.tolist()}")

    rewards = read_rewards()  # concat all records
    if rewards is None or rewards.numel() != len(prompts):
        print(f"  ⚠️ Got {0 if rewards is None else rewards.numel()} rewards, "
              f"expected {len(prompts)}")
        if rewards is None:
            return None, None
    print(f"  read_rewards (concat): shape={tuple(rewards.shape)}: "
          f"{rewards.tolist()}", flush=True)

    # Steady-state latency
    for _ in range(3):
        _ = llm.generate(prompts, sp, use_tqdm=False)
    timings = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = llm.generate(prompts, sp, use_tqdm=False)
        timings.append((time.perf_counter() - t0) * 1000)
    p50 = statistics.median(timings)
    print(f"[vLLM] batch=5 generate p50 = {p50:.1f}ms", flush=True)

    del llm
    torch.cuda.empty_cache()
    # Return the full reward list (5 items); subsequent comparison with BF16
    return rewards.tolist() if rewards is not None else None, p50


def run_bf16(prompts):
    print(f"[BF16] Loading VM SequenceClassification ...", flush=True)
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
            rewards.append(out.logits.squeeze().float().item())
    return rewards


def main():
    print("=" * 60)
    print("M2 step 1 verify: sia_rm vs BF16 baseline (via /dev/shm channel)")
    print("=" * 60)

    tok = AutoTokenizer.from_pretrained(MODEL_VM, trust_remote_code=True)
    prompts, candidates_text = build_candidates(tok)

    print(f"\n[1/2] vLLM (Qwen3WithScoreForCausalLM)")
    vllm_rewards, vllm_latency = run_vllm(prompts)
    if vllm_rewards is None:
        print(f"\n❌ FAIL: reward channel not working")
        sys.exit(1)
    if len(vllm_rewards) != 5:
        print(f"\n⚠️ vLLM got {len(vllm_rewards)} rewards (expected 5)")
        print(f"  This means vLLM batched 5 prompts did not yield all 5 rows in one compute_logits call")
        print(f"  Continuing with partial comparison (first {len(vllm_rewards)})")

    print(f"\n[2/2] BF16 transformers baseline")
    bf16_rewards = run_bf16(prompts)

    # Compare (align to min(len_vllm, len_bf16))
    n_compare = min(len(vllm_rewards), len(bf16_rewards))
    print(f"\n=== Numerical comparison (first {n_compare}) ===")
    print(f"  candidate  vLLM        BF16        |Δ|")
    diffs = []
    for i in range(n_compare):
        v, b = vllm_rewards[i], bf16_rewards[i]
        c = candidates_text[i]
        d = abs(v - b)
        diffs.append(d)
        print(f"  '{c.strip():>5s}'    {v:+8.4f}    {b:+8.4f}    {d:.4f}")
    max_d = max(diffs)
    mean_d = sum(diffs) / len(diffs)
    print(f"  mean |Δ|: {mean_d:.4f}, max |Δ|: {max_d:.4f}")

    # Pearson + Top-1 (over first n_compare)
    v_sub = vllm_rewards[:n_compare]
    b_sub = bf16_rewards[:n_compare]
    try:
        from scipy.stats import pearsonr
        r, _ = pearsonr(v_sub, b_sub)
    except Exception:
        import math
        mv = sum(v_sub) / len(v_sub)
        mb = sum(b_sub) / len(b_sub)
        cov = sum((v - mv) * (b - mb) for v, b in zip(v_sub, b_sub))
        var_v = sum((v - mv) ** 2 for v in v_sub)
        var_b = sum((b - mb) ** 2 for b in b_sub)
        r = cov / math.sqrt(var_v * var_b + 1e-9)
    print(f"  Pearson: {r:.4f}")
    vllm_top1 = max(range(n_compare), key=lambda i: v_sub[i])
    bf16_top1 = max(range(n_compare), key=lambda i: b_sub[i])
    top1_ok = vllm_top1 == bf16_top1
    print(f"  Top-1: vLLM='{candidates_text[vllm_top1].strip()}' "
          f"BF16='{candidates_text[bf16_top1].strip()}' "
          f"{'✅' if top1_ok else '❌'}")

    print(f"\n=== M2 step 1 acceptance ===")
    pass_corr = r > 0.99
    pass_diff = max_d < 0.5
    if pass_corr and top1_ok and pass_diff:
        print(f"  ✅ PASS")
        print(f"  ModelRegistry + custom class + /dev/shm channel all work,"
              f" values equivalent to BF16")
        print(f"  → Proceed to step 2: write RMClient (session management + fix_a_token API)")
    else:
        print(f"  ❌ FAIL: Pearson={r:.4f} (need>0.99), "
              f"Top-1 {'OK' if top1_ok else 'BAD'}, "
              f"max|Δ|={max_d:.3f} (need<0.5)")

    # Clean up reward file
    p = f"/dev/shm/sia_reward_{_FID}.bin"
    if os.path.exists(p):
        os.remove(p)


if __name__ == "__main__":
    main()
