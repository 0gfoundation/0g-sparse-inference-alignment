"""
Skywork-Reward-V2 third-party scoring — assigns reward scores to AlpacaEval generation results.
Simplified from /workspace/SIA/git/SIA/src/measure_reward.py, targeting only the AlpacaEval output format.

Usage:
    python scripts/measure_alpaca_reward.py \\
        --input_file exp/alpaca_results.json \\
        --output_file exp/alpaca_results_scored.json \\
        --rm /workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B \\
        --device cuda:0

Optional:
    --strip_think    Strip <think>...</think> content before feeding to Skywork
                     (for 0GM-thinking models; also applicable to 14B which also outputs <think>)
    --max_length     RM tokenizer truncation length, default 2048
"""
import argparse
import json
import re

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Handle thinking model output stripping, covering two cases:
# 1) Paired blocks `<think>...</think>` or `<thinking>...</thinking>` (including nested/custom variants)
# 2) Leading thinking — when chat_template injects the opening `<think>` at the end of the prompt,
#    the output is actually "{thinking content}</think>{answer}" with no opening tag → the old
#    paired regex won't match. Use `_LEAD_THINK_END_RE` to find the first `</think>` (or variant)
#    and strip everything before it.
_PAIR_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>\s*", flags=re.DOTALL)
_LEAD_THINK_END_RE = re.compile(r"^.*?</think(?:ing)?>\s*", flags=re.DOTALL)


def maybe_strip_think(text: str, strip: bool) -> str:
    if not strip:
        return text
    # Step 1: strip leading thinking (artifact of chat_template-injected <think> at prompt end)
    text = _LEAD_THINK_END_RE.sub("", text, count=1)
    # Step 2: strip any remaining nested paired think/thinking blocks
    return _PAIR_THINK_RE.sub("", text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", required=True,
                   help="JSON output from alpaca_eval.py")
    p.add_argument("--output_file", required=True,
                   help="Save here after adding reward field")
    p.add_argument("--rm",
                   default="/workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B",
                   help="Skywork RM path")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_length", type=int, default=2048,
                   help="Skywork RM tokenizer truncation (max_model_len), default 2048")
    p.add_argument("--strip_think", action="store_true",
                   help="For thinking models, strip <think>...</think> before scoring")
    args = p.parse_args()

    print(f"loading Skywork RM: {args.rm}")
    tok = AutoTokenizer.from_pretrained(args.rm, use_fast=True)
    # transformers 4.48 (venv) uses torch_dtype=, 4.55+ (venv4) uses dtype=
    # Using old API (torch_dtype) here — newer transformers will emit deprecation warning but still works
    rm = AutoModelForSequenceClassification.from_pretrained(
        args.rm, torch_dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="sdpa", num_labels=1,
    )
    rm.eval()
    print(f"RM ready on {args.device}")

    data = json.load(open(args.input_file))
    print(f"scoring {len(data)} samples from {args.input_file}")
    print(f"strip_think = {args.strip_think}")

    scored = []
    rewards = []
    n_skip_err = 0
    n_skip_long = 0

    for i, row in enumerate(data):
        instr = row["instruction"]
        output = row["output"]

        if str(output).startswith("ERROR:") or not str(output).strip():
            row["reward"] = None
            row["skip_reason"] = "empty_or_error"
            n_skip_err += 1
            scored.append(row)
            continue

        # Optional: strip <think>...</think>
        output_to_score = maybe_strip_think(output, args.strip_think)
        if not output_to_score.strip():
            # Empty after stripping (entire response was inside think, no final answer)
            row["reward"] = None
            row["skip_reason"] = "empty_after_strip_think"
            n_skip_err += 1
            scored.append(row)
            continue

        # Build Skywork chat: user + assistant
        convs = [
            {"role": "user", "content": instr},
            {"role": "assistant", "content": output_to_score},
        ]
        text = tok.apply_chat_template(convs, tokenize=False)
        if tok.bos_token and text.startswith(tok.bos_token):
            text = text[len(tok.bos_token):]

        enc = tok(text, return_tensors="pt").to(args.device)
        if enc.input_ids.shape[1] >= args.max_length:
            row["reward"] = None
            row["skip_reason"] = f"length>={args.max_length}"
            n_skip_long += 1
            scored.append(row)
            continue

        with torch.no_grad():
            out = rm(**enc)
            r = out.logits[0][0].item()

        row["reward"] = r
        rewards.append(r)
        scored.append(row)

        if (i + 1) % 50 == 0:
            cur_mean = sum(rewards) / max(len(rewards), 1)
            print(f"  [{i+1}/{len(data)}] scored {len(rewards)} "
                  f"(skip_err={n_skip_err} skip_long={n_skip_long}), "
                  f"cur_mean_reward={cur_mean:.4f}")

    json.dump(scored, open(args.output_file, "w"), ensure_ascii=False, indent=2)
    mean_reward = sum(rewards) / max(len(rewards), 1)
    print()
    print(f"=== summary ===")
    print(f"  total samples: {len(data)}")
    print(f"  scored: {len(rewards)}")
    print(f"  skipped (error/empty): {n_skip_err}")
    print(f"  skipped (too long > {args.max_length}): {n_skip_long}")
    print(f"  mean reward: {mean_reward:.4f}")
    if rewards:
        rewards_sorted = sorted(rewards)
        print(f"  p50: {rewards_sorted[len(rewards_sorted)//2]:.4f}")
        print(f"  min: {min(rewards):.4f}  max: {max(rewards):.4f}")
    print(f"  saved scored JSON to: {args.output_file}")


if __name__ == "__main__":
    main()
