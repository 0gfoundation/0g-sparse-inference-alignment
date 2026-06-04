"""Score Run 1 outputs with Skywork, pair against noSIA, compute paired stats."""
import json
import sys
import time
import math
import statistics

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEVICE = "cuda:0"
SKYWORK = "/workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B"
SIA_IN = "/tmp/vl30b_runs/vllm_sia_outputs.json"
SIA_OUT = "/tmp/vl30b_runs/vllm_sia_scored.json"
NOSIA = "/tmp/alpaca_qwen3vl30b_nosia_200q_scored_20260603_034614.json"

print(f"[scorer] loading Skywork on {DEVICE} …", flush=True)
TOK = AutoTokenizer.from_pretrained(SKYWORK)
MODEL = AutoModelForSequenceClassification.from_pretrained(
    SKYWORK, torch_dtype=torch.bfloat16,
    device_map=DEVICE, attn_implementation="sdpa",
).eval()
print(f"[scorer] ready", flush=True)


def score(instr, resp):
    if not resp:
        return None
    convs = [{"role": "user", "content": instr}, {"role": "assistant", "content": resp}]
    text = TOK.apply_chat_template(convs, tokenize=False)
    if TOK.bos_token and text.startswith(TOK.bos_token):
        text = text[len(TOK.bos_token):]
    enc = TOK(text, return_tensors="pt", truncation=True, max_length=4096).to(DEVICE)
    with torch.no_grad():
        return float(MODEL(**enc).logits[0][0].item())


# Score Run 1
sia = json.load(open(SIA_IN))
print(f"[scorer] scoring {len(sia)} Run 1 records …", flush=True)
t0 = time.time()
for i, rec in enumerate(sia):
    out = rec.get("output")
    try:
        rec["reward"] = score(rec["instruction"], out)
    except Exception as e:
        rec["reward"] = None
        rec["reward_error"] = str(e)
    if (i + 1) % 25 == 0:
        with open(SIA_OUT, "w") as f:
            json.dump(sia, f, ensure_ascii=False, indent=2)
        print(f"[scorer] {i+1}/{len(sia)} ({time.time()-t0:.0f}s)", flush=True)

with open(SIA_OUT, "w") as f:
    json.dump(sia, f, ensure_ascii=False, indent=2)
print(f"[scorer] DONE Run 1 scoring in {time.time()-t0:.0f}s", flush=True)

# Pair vs noSIA on instruction text
nosia = json.load(open(NOSIA))
nosia_by_instr = {r["instruction"]: r for r in nosia}

paired = []
for sia_rec in sia:
    nosia_rec = nosia_by_instr.get(sia_rec["instruction"])
    if nosia_rec and sia_rec.get("reward") is not None and nosia_rec.get("reward") is not None:
        paired.append({
            "id": sia_rec["id"],
            "instr": sia_rec["instruction"],
            "sia_reward": sia_rec["reward"],
            "nosia_reward": nosia_rec["reward"],
            "delta": sia_rec["reward"] - nosia_rec["reward"],
            "sia_tokens": sia_rec.get("usage", {}).get("completion_tokens", 0),
            "nosia_tokens": nosia_rec.get("tokens", 0),
        })

print()
print(f"=== Paired comparison: Run 1 (vllm RM SIA) vs noSIA ===")
print(f"Paired records: {len(paired)}")
print()

if paired:
    sia_r = [p["sia_reward"] for p in paired]
    nos_r = [p["nosia_reward"] for p in paired]
    deltas = [p["delta"] for p in paired]
    n = len(deltas)

    print(f"--- Distribution ---")
    print(f"  SIA reward:    mean={statistics.mean(sia_r):+.3f}  median={statistics.median(sia_r):+.3f}  std={statistics.stdev(sia_r):.3f}")
    print(f"  noSIA reward:  mean={statistics.mean(nos_r):+.3f}  median={statistics.median(nos_r):+.3f}  std={statistics.stdev(nos_r):.3f}")
    print()
    print(f"--- Paired Δ (SIA - noSIA) ---")
    print(f"  mean Δ:        {statistics.mean(deltas):+.3f}")
    print(f"  median Δ:      {statistics.median(deltas):+.3f}")
    print(f"  std(Δ):        {statistics.stdev(deltas):.3f}")
    pct_change = statistics.mean(deltas) / abs(statistics.mean(nos_r)) * 100
    print(f"  rel Δ:         {pct_change:+.1f}%")
    print()

    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    print(f"  Win/Loss/Tie:  {pos}/{neg}/{n-pos-neg}  ({pos/n*100:.1f}% win rate)")

    # Paired t-test
    mean_d = statistics.mean(deltas)
    std_d = statistics.stdev(deltas)
    t = mean_d / (std_d / math.sqrt(n))
    # Normal approx p-value
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    print(f"  paired t:      {t:+.3f}  (p ≈ {p:.6f})")
    print()

    # Save report
    report = f"""=== VL-30B SIA Run 1 (vllm RM, path A direct-token-ids) vs noSIA ===

Source: {SIA_IN}
        {NOSIA}
Paired records: {n}

Reward distribution:
  SIA   mean={statistics.mean(sia_r):+.3f} median={statistics.median(sia_r):+.3f} std={statistics.stdev(sia_r):.3f}
  noSIA mean={statistics.mean(nos_r):+.3f} median={statistics.median(nos_r):+.3f} std={statistics.stdev(nos_r):.3f}

Paired Δ (SIA - noSIA):
  mean   {statistics.mean(deltas):+.3f}
  median {statistics.median(deltas):+.3f}
  rel    {pct_change:+.1f}%
  win    {pos}/{n} = {pos/n*100:.1f}%
  loss   {neg}/{n} = {neg/n*100:.1f}%
  t      {t:+.3f}  (p ≈ {p:.6f})
"""
    with open("/tmp/vl30b_runs/run1_vs_nosia_partial.txt", "w") as f:
        f.write(report)
    print(f"Report: /tmp/vl30b_runs/run1_vs_nosia_partial.txt")

    json.dump(paired, open("/tmp/vl30b_runs/run1_vs_nosia_paired.json", "w"), ensure_ascii=False, indent=2)
    print(f"Paired records: /tmp/vl30b_runs/run1_vs_nosia_paired.json")
