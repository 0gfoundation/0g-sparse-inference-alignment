"""
Compare new InprocClient MMLU 600q result against historical baseline.

Historical baseline (b2 backend, multiprocess RM EngineCore subprocess):
  /tmp/mmlu_14b_limit20.log → 46.6 tok/s, 7964.4s total latency,
  371,251 tokens, 600 questions, accuracy 74.83%
  Saved at sparse_logs/eval_b2_mmlu/mmlu_redux_14b_600q_weight1.json

New (b2 backend, InprocClient — VLLM_ENABLE_V1_MULTIPROCESSING=0):
  Eval log: /tmp/mmlu_14b_inproc_20260529_131755.log
  Saved at sparse_logs/eval_b2_mmlu/mmlu_redux_14b_600q_inproc.json
"""
import argparse
import json
import re
import sys


def parse_eval_log(path):
    """Pull the final summary lines from an mmlu_eval.py log."""
    out = {}
    with open(path) as f:
        text = f.read()
    m = re.search(r"^Overall accuracy:\s*([\d.]+)\s*\((\d+)/(\d+)\)", text, re.M)
    if m:
        out["accuracy"] = float(m.group(1))
        out["correct"] = int(m.group(2))
        out["total"] = int(m.group(3))
    m = re.search(r"^Total latency\s*:\s*([\d.]+)s\s+avg=([\d.]+)s/q", text, re.M)
    if m:
        out["total_latency_s"] = float(m.group(1))
        out["avg_latency_per_q"] = float(m.group(2))
    m = re.search(r"^Avg token length:\s*([\d.]+)\s*tokens/q\s+\(total=(\d+)\)", text, re.M)
    if m:
        out["avg_tokens_per_q"] = float(m.group(1))
        out["total_tokens"] = int(m.group(2))
    m = re.search(r"^Throughput\s*:\s*([\d.]+)\s*tokens/s", text, re.M)
    if m:
        out["throughput_tok_s"] = float(m.group(1))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="/tmp/mmlu_14b_limit20.log")
    p.add_argument("--new", default="/tmp/mmlu_14b_inproc_20260529_131755.log")
    args = p.parse_args()

    b = parse_eval_log(args.baseline)
    n = parse_eval_log(args.new)

    print("="*78)
    print("MMLU-Redux 600q — multiprocess RM (baseline) vs InprocClient RM (new)")
    print("="*78)
    print()
    print(f"  {'metric':<28} {'baseline':>14} {'inproc':>14} {'Δ':>14}")
    print(f"  {'-'*28} {'-'*14} {'-'*14} {'-'*14}")

    def row(name, key, unit="", fmt="{:.2f}"):
        bv = b.get(key)
        nv = n.get(key)
        if bv is None or nv is None:
            print(f"  {name:<28} {'(n/a)':>14} {'(n/a)':>14} {'(n/a)':>14}")
            return
        delta = nv - bv
        pct = (delta / bv * 100) if bv else 0.0
        sign = "+" if delta >= 0 else ""
        bv_s = fmt.format(bv) + unit
        nv_s = fmt.format(nv) + unit
        d_s = f"{sign}{fmt.format(delta)} ({sign}{pct:.1f}%)"
        print(f"  {name:<28} {bv_s:>14} {nv_s:>14} {d_s:>14}")

    row("accuracy",           "accuracy",          fmt="{:.4f}")
    row("correct",            "correct",           fmt="{:.0f}")
    row("total_questions",    "total",             fmt="{:.0f}")
    row("total_latency_s",    "total_latency_s",   fmt="{:.1f}")
    row("avg_latency_per_q",  "avg_latency_per_q", fmt="{:.2f}")
    row("avg_tokens_per_q",   "avg_tokens_per_q",  fmt="{:.1f}")
    row("total_tokens",       "total_tokens",      fmt="{:.0f}")
    row("throughput_tok_s",   "throughput_tok_s",  fmt="{:.2f}")

    print()
    if "throughput_tok_s" in b and "throughput_tok_s" in n:
        delta = n["throughput_tok_s"] - b["throughput_tok_s"]
        pct = delta / b["throughput_tok_s"] * 100
        sign = "+" if delta >= 0 else ""
        print(f"  >>> Throughput Δ: {sign}{delta:.2f} tok/s ({sign}{pct:.1f}%)")
    if "accuracy" in b and "accuracy" in n:
        d_acc = (n["accuracy"] - b["accuracy"]) * 100
        sign = "+" if d_acc >= 0 else ""
        print(f"  >>> Accuracy Δ:   {sign}{d_acc:.2f} pp  (baseline {b['accuracy']*100:.2f}% → {n['accuracy']*100:.2f}%)")


if __name__ == "__main__":
    main()
