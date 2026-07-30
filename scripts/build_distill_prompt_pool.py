"""
Build a deduplicated, shuffled prompt pool for SIA distillation (Fix 1) data
collection, from the original VM training data sources (WildGuardMix +
UltraFeedback-ShareGPT + UltraFeedback-UltraChat) instead of repeating the
805-question AlpacaEval set — those sources give ~157K unique, diverse
prompts (safety/red-team + general instruction-following/creative/dialogue),
vs. AlpacaEval's fixed helpfulness-only 805.

Usage:
    python scripts/build_distill_prompt_pool.py \\
        --input /path/to/vm-training-data/wildguardmix-Qwen3.json \\
                /path/to/vm-training-data/ultrafeedback-sharegpt-Qwen3.json \\
                /path/to/vm-training-data/ultrafeedback-ultrachat-Qwen3.json \\
        --output data/distill_prompts_pool.json \\
        --limit 2000 --seed 42

Output: JSON list of {"instruction": ...} records — directly usable as
eval/alpaca_eval.py's --dataset argument.
"""
import argparse
import json
import random
import re

# Every source record's "prompt" field is "Human:\n{instruction}\nAssistant:\n"
# (verified: 157,432 / 157,442 records match this exactly; the ~10 that don't
# contain a second "Human:" turn are just used as-is, raw).
_PROMPT_RE = re.compile(r"^Human:\n(.*)\nAssistant:\n$", re.DOTALL)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True,
                    help="One or more vm-training-data/*.json files")
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=None,
                    help="Sample at most this many unique instructions (default: all)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seen = set()
    instructions = []
    for path in args.input:
        data = json.load(open(path, encoding="utf-8"))
        n_before = len(instructions)
        for rec in data:
            m = _PROMPT_RE.match(rec["prompt"])
            instr = m.group(1) if m else rec["prompt"]
            if instr in seen:
                continue
            seen.add(instr)
            instructions.append(instr)
        print(f"{path}: {len(data)} records -> "
              f"{len(instructions) - n_before} new unique instructions")

    print(f"total unique instructions across all inputs: {len(instructions)}")

    rng = random.Random(args.seed)
    rng.shuffle(instructions)
    if args.limit is not None:
        instructions = instructions[:args.limit]

    out = [{"instruction": instr} for instr in instructions]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved {len(out)} instructions to {args.output}")


if __name__ == "__main__":
    main()
