"""
M2 step 2 验证: RMClient 端到端功能 + 链式 latency.

测试用例:
  T1. 单次 score_candidates 数值 vs BF16 baseline
  T2. fix_a_token 链式: 模拟 SIA 多步 decode, 每步 fix 一个 token 然后 score 5 candidates
      每次 score 拿出来的数值跟 BF16 (用累积的 prefix) 一致
  T3. 稳态 latency: 100 次 score_candidates, prefix 不断增长, 测 p50/p95
      跟 §7.A bench 的 15.2ms 对比

跑法 (从仓库根目录):
  python scripts/verify_m2_step2.py
"""
import os
import statistics
import sys
import time

# src/ 到 sys.path + PYTHONPATH
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
    print("T1. 单次 score_candidates vs BF16 baseline")
    print("=" * 60)

    user = "What is 2 + 2?"
    candidates_text = [" 4", " 5", " 3", " 6", " 100"]
    # candidate 是 chat-formatted assistant response 末尾的差异
    # 每个 candidate 整 prompt 不同 → 5 个独立 session 比较繁琐
    # 简化: 单 session, prefix 是 user message 部分, candidates 是不同的下一个 token
    # 但 chat template 不允许这种, 我们仿照 step 1 用 5 个完整 prompts 直接 score

    # 用 RMClient 的 session API:
    # session prefix = (user message 完整 token), score_candidates 中每个
    # candidate 就是不同 assistant 完整 response 的 token, 但这跟 prefix+[c]
    # 假设不符 (每个 candidate 是多 token 不是 single token)
    #
    # 真正模拟 SIA: prefix = (user + 已生成 N token), candidates = next 1 token 的 5 choice
    # 这里简化为构造 5 个"几乎相同 prefix 只末尾 1 token 不同"的 session
    # 直接用 step 1 的 5 个完整 prompts 跑一次 score_candidates 是最干净的对比

    prompts_ids = [
        build_one_prompt_ids(tok, user, c) for c in candidates_text
    ]
    # 5 个 prompts 末尾几个 token 不同, prefix 公共部分到 user message
    # 找最长 common prefix
    common = []
    for k in range(min(len(p) for p in prompts_ids)):
        if all(p[k] == prompts_ids[0][k] for p in prompts_ids):
            common.append(prompts_ids[0][k])
        else:
            break
    print(f"  prompts 长度: {[len(p) for p in prompts_ids]}, "
          f"common prefix 长度: {len(common)}")
    # 每个 prompt 在 common prefix 之后的剩余 token
    suffixes = [p[len(common):] for p in prompts_ids]
    print(f"  suffix 长度: {[len(s) for s in suffixes]}")

    # 由于 candidate 在 RMClient 里是 single token, 我们无法直接复用 RMClient
    # 的 score_candidates 接口 (它假设 candidate = 1 token).
    # → 这是 step 2 的局限性: RMClient.score_candidates 假定 candidate
    # 就是单个 token. 但实际 SIA 工作模式确实是 single-token candidates
    # (从 LLM 主模型的 topk 出来的下一 token id), 所以这个限制 OK.
    #
    # 验证策略: 给 RMClient.session prefix = common prefix, 然后 score_candidates
    # 5 candidates = [s[0] for s in suffixes] (但 suffix 不一定单 token).
    # 退而求其次: 用单 token 不同的 prompts.

    # 重新设计: 用 base prefix + 5 个不同的 single token
    base_prefix_ids = build_one_prompt_ids(tok, user, "")
    # 去掉 chat template 自动加的 <|im_end|>\n
    # 我们就用 base_prefix_ids 作 session prefix, 然后 candidates 是 5 个 token id
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

    # BF16 baseline: 各个 prompt = base_prefix + [cand], 独立 forward
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
    print("T2. fix_a_token 链式: 模拟 SIA 5 step decode")
    print("=" * 60)

    user = "Compute 7 * 8 step by step."
    base_prefix_ids = build_one_prompt_ids(tok, user, "")
    # SIA 5 步 decode: fix [token_a, token_b, ...] 模拟"我们选了这些 token"
    chain = [220, 56, 220, 23, 220]  # arbitrary 5 token ids (任意, 用作 prefix 增长)
    cand_token_ids = [100, 200, 300, 400, 500]

    sid = rm.new_session(base_prefix_ids)
    all_ok = True
    for step, t_to_fix in enumerate(chain):
        # 先 score 当前 prefix 下 5 candidates
        vllm_r = rm.score_candidates(sid, cand_token_ids)
        # BF16 baseline (累积 prefix + 各 cand)
        cur_prefix = list(rm._sessions[sid])  # 等价 base + 之前 step fix 过的
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

        # fix_a_token 推进 session
        rm.fix_a_token(sid, t_to_fix)

    rm.end_session(sid)
    print(f"  {'✅ PASS' if all_ok else '❌ FAIL'} (all 5 steps within 0.5)")
    return all_ok


def t3_latency(rm: RMClient, tok) -> bool:
    print("\n" + "=" * 60)
    print("T3. 链式 100 次 score_candidates 稳态 latency")
    print("=" * 60)

    user = "Write a brief paragraph about quantum computing."
    base_prefix_ids = build_one_prompt_ids(tok, user, "")
    sid = rm.new_session(base_prefix_ids)
    cand_token_ids = [100, 200, 300, 400, 500]

    # warmup 5
    for _ in range(5):
        rm.score_candidates(sid, cand_token_ids)
        rm.fix_a_token(sid, 220)

    # 100 iter
    timings = []
    for i in range(100):
        _, ms = rm.time_score_candidates(sid, cand_token_ids)
        timings.append(ms)
        rm.fix_a_token(sid, 220)  # 推进 prefix 长度

    p50 = statistics.median(timings)
    p95 = sorted(timings)[int(0.95 * len(timings))]
    mn = min(timings)
    mx = max(timings)
    avg = sum(timings) / len(timings)
    print(f"  prefix grew {len(base_prefix_ids)+5} → "
          f"{len(rm._sessions[sid])} tokens")
    print(f"  min: {mn:.2f}, p50: {p50:.2f}, avg: {avg:.2f}, "
          f"p95: {p95:.2f}, max: {mx:.2f} ms")

    target_max = 25  # 给 RMClient 比 §7.A bench 多 5ms session overhead 预算
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

    # T3 latency 测试前清理 BF16 baseline 释放 GPU
    del bf16_model
    torch.cuda.empty_cache()
    results.append(("T3", t3_latency(rm, tok)))

    print("\n" + "=" * 60)
    print("M2 step 2 总结")
    print("=" * 60)
    for name, ok in results:
        print(f"  {name}: {'✅ PASS' if ok else '❌ FAIL'}")
    all_ok = all(r[1] for r in results)
    print(f"\n  {'✅ ALL PASS, 进 step 3' if all_ok else '❌ FAIL, 需调查'}")


if __name__ == "__main__":
    main()
