"""
Skywork-Reward-V2 第三方打分 — 给 AlpacaEval 生成结果打 reward 分。
基于 /workspace/SIA/git/SIA/src/measure_reward.py 简化, 只针对 AlpacaEval 的输出格式。

用法:
    python scripts/measure_alpaca_reward.py \\
        --input_file exp/alpaca_results.json \\
        --output_file exp/alpaca_results_scored.json \\
        --rm /workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B \\
        --device cuda:0

可选:
    --strip_think    在喂给 Skywork 前剥离 <think>...</think> 内容
                     (0GM-thinking 模型用; 14B 也可用因为它也会输出 <think>)
    --max_length     RM tokenizer 截断长度, 默认 2048
"""
import argparse
import json
import re

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# 处理 thinking 模型的 output 剥离, 涵盖两类情况:
# 1) 配对块 `<think>...</think>` 或 `<thinking>...</thinking>` (含嵌套/自创变体)
# 2) Leading thinking — 当 chat_template 把 opening `<think>` 注入 prompt 末尾时, output
#    实际形如 "{thinking content}</think>{answer}", 没有开 tag → 旧的配对正则不匹配。
#    用 `_LEAD_THINK_END_RE` 找第一个 `</think>` (或变体), 把它之前的所有内容剥掉。
_PAIR_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>\s*", flags=re.DOTALL)
_LEAD_THINK_END_RE = re.compile(r"^.*?</think(?:ing)?>\s*", flags=re.DOTALL)


def maybe_strip_think(text: str, strip: bool) -> str:
    if not strip:
        return text
    # Step 1: 剥 leading thinking (chat_template-injected <think> 在 prompt 末尾的产物)
    text = _LEAD_THINK_END_RE.sub("", text, count=1)
    # Step 2: 剥任何剩余的嵌套配对 think/thinking 块
    return _PAIR_THINK_RE.sub("", text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", required=True,
                   help="alpaca_eval.py 输出的 JSON")
    p.add_argument("--output_file", required=True,
                   help="加 reward 字段后保存到这里")
    p.add_argument("--rm",
                   default="/workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B",
                   help="Skywork RM 路径")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_length", type=int, default=2048,
                   help="Skywork RM tokenizer 截断 (max_model_len), 默认 2048")
    p.add_argument("--strip_think", action="store_true",
                   help="对 thinking 模型, 先剥离 <think>...</think> 再打分")
    args = p.parse_args()

    print(f"loading Skywork RM: {args.rm}")
    tok = AutoTokenizer.from_pretrained(args.rm, use_fast=True)
    # transformers 4.48 (venv) 用 torch_dtype=, 4.55+ (venv4) 用 dtype=
    # 这里走旧 API (torch_dtype) — 新版 transformers 会发 deprecation warning 但仍 work
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

        # 可选: 剥离 <think>...</think>
        output_to_score = maybe_strip_think(output, args.strip_think)
        if not output_to_score.strip():
            # 剥离后空了 (整个回答全在 think 内, 没 final answer)
            row["reward"] = None
            row["skip_reason"] = "empty_after_strip_think"
            n_skip_err += 1
            scored.append(row)
            continue

        # 构造 Skywork chat: user + assistant
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
