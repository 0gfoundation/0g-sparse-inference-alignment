"""
M2 step 2 verification: RMClient end-to-end functionality + chained latency.

Test cases:
  T1. Single score_candidates numerics vs BF16 baseline
  T2. fix_a_token chained: simulates SIA multi-step decode, fixing one token per step
      then scoring 5 candidates; verify values against BF16 (with accumulated prefix)
  T3. Steady-state latency: 100 score_candidates calls with a growing prefix,
      measure p50/p95; compare with §7.A bench's 15.2ms

Run (from repo root):
  python scripts/verify_m2_step2.py
"""
import os
import statistics
import sys
import time

# Add src/ to sys.path + PYTHONPATH
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
sys.path.insert(0, _SRC)
_pp = os.environ.get("PYTHONPATH", "")
if _SRC not in _pp.split(":"):
    os.environ["PYTHONPATH"] = (_SRC + ":" + _pp) if _pp else _SRC

import torch  # noqa: E402
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # noqa
from sia_rm import RMClient  # noqa: E402

MODEL_VM = "/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm"


def build_one_prompt_ids(tok, user: str, assistant: str) -> list[int]:
    convs = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    text = tok.apply_chat_template(convs, tokenize=False)
    bos = tok.bos_token
    if bos and text.startswith(bos):
        text = text[len(bos):]
    return tok(text).input_ids


def bf16_score(bf16_model, tok, prompt_ids: list[int]) -> float:
    ids = torch.tensor([prompt_ids], device="cuda")
    with torch.no_grad():
        out = bf16_model(ids)
    return out.logits.squeeze().float().item()


def t1_basic_numerics(rm: RMClient, bf16_model, tok) -> bool:
    print("\n" + "=" * 60)
    print("T1. Single score_candidates vs BF16 baseline")
    print("=" * 60)

    user = "What is 2 + 2?"
    candidates_text = [" 4", " 5", " 3", " 6", " 100"]
    # candidates are the differences at the end of a chat-formatted assistant response
    # each candidate is a full different prompt → comparing 5 independent sessions is cumbersome
    # simplification: single session, prefix is user message part, candidates are different next tokens
    # but chat template doesn't allow this; we follow step 1 and directly score 5 full prompts

    # Using RMClient's session API:
    # session prefix = (full user message tokens), each candidate in score_candidates
    # is the tokens of a different complete assistant response, but this doesn't match
    # the prefix+[c] assumption (each candidate is multi-token, not single token)
    #
    # True SIA simulation: prefix = (user + N already-generated tokens), candidates = 5 choices of next 1 token
    # Simplified here by constructing 5 "nearly identical prefix, only last 1 token differs" sessions
    # Directly using 5 complete prompts from step 1 for one score_candidates is the cleanest comparison

    prompts_ids = [
        build_one_prompt_ids(tok, user, c) for c in candidates_text
    ]
    # 5 prompts differ in the last few tokens, common prefix up to user message
    # Find longest common prefix
    common = []
    for k in range(min(len(p) for p in prompts_ids)):
        if all(p[k] == prompts_ids[0][k] for p in prompts_ids):
            common.append(prompts_ids[0][k])
        else:
            break
    print(f"  prompt lengths: {[len(p) for p in prompts_ids]}, "
          f"common prefix length: {len(common)}")
    # Remaining tokens after common prefix for each prompt
    suffixes = [p[len(common):] for p in prompts_ids]
    print(f"  suffix lengths: {[len(s) for s in suffixes]}")

    # Since candidate in RMClient is a single token, we cannot directly reuse RMClient's
    # score_candidates interface (it assumes candidate = 1 token).
    # → This is a step 2 limitation: RMClient.score_candidates assumes candidates
    # are single tokens. But the actual SIA working mode is indeed single-token candidates
    # (next token IDs from the main LLM model's topk), so this limitation is OK.
    #
    # Verification strategy: give RMClient.session prefix = common prefix, then score_candidates
    # 5 candidates = [s[0] for s in suffixes] (but suffix may not be single token).
    # Fall back to: use prompts with a single different token.

    # Redesign: use base prefix + 5 different single tokens
    base_prefix_ids = build_one_prompt_ids(tok, user, "")
    # Strip the <|im_end|>\n auto-added by chat template
    # Use base_prefix_ids as session prefix, candidates are 5 token ids
    cand_token_ids = [
        tok(" 4").input_ids[0] if len(tok(" 4").input_ids) >= 1 else 220,
        tok(" 5").input_ids[0] if len(tok(" 5").input_ids) >= 1 else 220,
        tok(" 3").input_ids[0],
        tok(" 6").input_ids[0],
        tok(" 100").input_ids[0],
    ]
    print(f"  base prefix len: {len(base_prefix_ids)}, "
          f"candidate token ids: {cand_token_ids}")

    sid = rm.new_session(base_prefix_ids)
    vllm_rewards = rm.score_candidates(sid, cand_token_ids)
    print(f"  vLLM rewards: {[f'{r:+.3f}' for r in vllm_rewards]}")

    # BF16 baseline: each prompt = base_prefix + [cand], independent forward
    bf16_rewards = []
    for c in cand_token_ids:
        bf16_rewards.append(
            bf16_score(bf16_model, tok, base_prefix_ids + [c])
        )
    print(f"  BF16 rewards: {[f'{r:+.3f}' for r in bf16_rewards]}")

    diffs = [abs(v - b) for v, b in zip(vllm_rewards, bf16_rewards)]
    max_d = max(diffs)
    print(f"  max |Δ|: {max_d:.4f}")
    ok = max_d < 0.5
    print(f"  {'✅ PASS' if ok else '❌ FAIL'} (max |Δ| < 0.5)")
    rm.end_session(sid)
    return ok


def t2_chained_fix_a_token(rm: RMClient, bf16_model, tok) -> bool:
    print("\n" + "=" * 60)
    print("T2. fix_a_token chained: simulates SIA 5-step decode")
    print("=" * 60)

    user = "Compute 7 * 8 step by step."
    base_prefix_ids = build_one_prompt_ids(tok, user, "")
    # SIA 5-step decode: fix [token_a, token_b, ...] simulating "we chose these tokens"
    chain = [220, 56, 220, 23, 220]  # arbitrary 5 token ids (used for prefix growth)
    cand_token_ids = [100, 200, 300, 400, 500]

    sid = rm.new_session(base_prefix_ids)
    all_ok = True
    for step, t_to_fix in enumerate(chain):
        # First score 5 candidates under the current prefix
        vllm_r = rm.score_candidates(sid, cand_token_ids)
        # BF16 baseline (accumulated prefix + each cand)
        cur_prefix = list(rm._sessions[sid])  # equivalent to base + tokens fixed in prior steps
        bf16_r = [
            bf16_score(bf16_model, tok, cur_prefix + [c])
            for c in cand_token_ids
        ]
        diffs = [abs(v - b) for v, b in zip(vllm_r, bf16_r)]
        max_d = max(diffs)
        ok = max_d < 0.5
        all_ok = all_ok and ok
        print(f"  step {step}: prefix_len={len(cur_prefix):3d} "
              f"max|Δ|={max_d:.4f} {'✅' if ok else '❌'}")

        # fix_a_token advances the session
        rm.fix_a_token(sid, t_to_fix)

    rm.end_session(sid)
    print(f"  {'✅ PASS' if all_ok else '❌ FAIL'} (all 5 steps within 0.5)")
    return all_ok


def t3_latency(rm: RMClient, tok) -> bool:
    print("\n" + "=" * 60)
    print("T3. Chained 100 score_candidates steady-state latency")
    print("=" * 60)

    user = "Write a brief paragraph about quantum computing."
    base_prefix_ids = build_one_prompt_ids(tok, user, "")
    sid = rm.new_session(base_prefix_ids)
    cand_token_ids = [100, 200, 300, 400, 500]

    # warmup 5
    for _ in range(5):
        rm.score_candidates(sid, cand_token_ids)
        rm.fix_a_token(sid, 220)

    # 100 iterations
    timings = []
    for i in range(100):
        _, ms = rm.time_score_candidates(sid, cand_token_ids)
        timings.append(ms)
        rm.fix_a_token(sid, 220)  # advance prefix length

    p50 = statistics.median(timings)
    p95 = sorted(timings)[int(0.95 * len(timings))]
    mn = min(timings)
    mx = max(timings)
    avg = sum(timings) / len(timings)
    print(f"  prefix grew {len(base_prefix_ids)+5} → "
          f"{len(rm._sessions[sid])} tokens")
    print(f"  min: {mn:.2f}, p50: {p50:.2f}, avg: {avg:.2f}, "
          f"p95: {p95:.2f}, max: {mx:.2f} ms")

    target_max = 25  # allow RMClient 5ms more than §7.A bench for session overhead
    ok = p50 < target_max
    print(f"  {'✅ PASS' if ok else '❌ FAIL'} (p50 < {target_max}ms; "
          f"§7.A pure-bench p50 was 15.2ms)")
    rm.end_session(sid)
    return ok


def main():
    print("=" * 60)
    print("M2 step 2 verify: RMClient")
    print("=" * 60)

    tok = AutoTokenizer.from_pretrained(MODEL_VM, trust_remote_code=True)

    print(f"\nLoading RMClient ...", flush=True)
    t0 = time.perf_counter()
    rm = RMClient(model_path=MODEL_VM, gpu_mem=0.3, max_model_len=2048)
    print(f"RMClient loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    print(f"\nLoading BF16 baseline ...", flush=True)
    bf16_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_VM, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()

    results = []
    results.append(("T1", t1_basic_numerics(rm, bf16_model, tok)))
    results.append(("T2", t2_chained_fix_a_token(rm, bf16_model, tok)))

    # Release GPU before T3 latency test by clearing BF16 baseline
    del bf16_model
    torch.cuda.empty_cache()
    results.append(("T3", t3_latency(rm, tok)))

    print("\n" + "=" * 60)
    print("M2 step 2 summary")
    print("=" * 60)
    for name, ok in results:
        print(f"  {name}: {'✅ PASS' if ok else '❌ FAIL'}")
    all_ok = all(r[1] for r in results)
    print(f"\n  {'✅ ALL PASS, proceed to step 3' if all_ok else '❌ FAIL, needs investigation'}")


if __name__ == "__main__":
    main()
