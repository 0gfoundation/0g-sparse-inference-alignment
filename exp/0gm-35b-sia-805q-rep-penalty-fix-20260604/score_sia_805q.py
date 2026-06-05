"""Skywork score 0GM-35B SIA-no-think 805Q + paired vs prior noSIA baselines."""
import json, statistics, math, time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda:0"
SKY = "/workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B"

SIA_INPUT = "/tmp/0gm_runs/sia_0gm_805q_max2048_nothink.json"
NOSIA_THINKBRIEF_805 = "/tmp/alpaca_0gm_nosia_thinkbrief_805q_scored_v2_20260603_023431.json"
NOSIA_BANTHINK_100 = "/tmp/alpaca_0gm_nosia_banthink_scored_20260602_161059.json"

print("loading Skywork …", flush=True)
TOK = AutoTokenizer.from_pretrained(SKY)
M = AutoModelForSequenceClassification.from_pretrained(
    SKY, torch_dtype=torch.bfloat16, device_map=DEVICE, attn_implementation="sdpa",
).eval()
print("ready", flush=True)


def score(instr, resp):
    if not resp:
        return None
    convs = [{"role": "user", "content": instr},
             {"role": "assistant", "content": resp}]
    txt = TOK.apply_chat_template(convs, tokenize=False)
    if TOK.bos_token and txt.startswith(TOK.bos_token):
        txt = txt[len(TOK.bos_token):]
    enc = TOK(txt, return_tensors="pt", truncation=True, max_length=4096).to(DEVICE)
    with torch.no_grad():
        return float(M(**enc).logits[0][0].item())


# Score SIA
print(f"\nScoring SIA-no-think 805Q ...", flush=True)
sia = json.load(open(SIA_INPUT))
t0 = time.time()
for r in sia:
    o = r.get("output")
    r["reward"] = score(r["instruction"], o) if o and not str(o).startswith("ERROR:") else None
print(f"done in {time.time()-t0:.0f}s", flush=True)
with open(SIA_INPUT.replace(".json", "_scored.json"), "w") as f:
    json.dump(sia, f, ensure_ascii=False, indent=2)


def aggregate(records, label):
    rewards = [r["reward"] for r in records if r.get("reward") is not None]
    print(f"\n=== {label} (n={len(rewards)}) ===")
    print(f"  mean={statistics.mean(rewards):+.3f}  median={statistics.median(rewards):+.3f}  std={statistics.stdev(rewards):.3f}")
    return rewards


# Aggregates
print("\n" + "=" * 60)
sia_r = aggregate(sia, "SIA-no-think (this run, rep=1.0)")

nosia_tb = json.load(open(NOSIA_THINKBRIEF_805))
nosia_tb_r = aggregate(nosia_tb, "noSIA-thinkbrief 805Q (prior, rep=1.0 driver, rep=1.3 server default?)")

nosia_bt = json.load(open(NOSIA_BANTHINK_100))
nosia_bt_r = aggregate(nosia_bt, "noSIA-banthink 100Q (prior, apples-to-apples to this SIA)")


def paired_t(a_list, b_list, a_records, b_records, label):
    """Pair by instruction."""
    a_by = {r["instruction"]: r["reward"] for r in a_records if r.get("reward") is not None}
    b_by = {r["instruction"]: r["reward"] for r in b_records if r.get("reward") is not None}
    common = set(a_by) & set(b_by)
    if not common:
        print(f"  {label}: no overlap")
        return
    diffs = [a_by[k] - b_by[k] for k in common]
    mean_d = statistics.mean(diffs)
    std_d = statistics.stdev(diffs) if len(diffs) > 1 else 0
    t = mean_d / (std_d / math.sqrt(len(diffs))) if std_d > 0 else 0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))) if abs(t) > 0 else 1
    pos = sum(1 for x in diffs if x > 0)
    rel = mean_d / abs(statistics.mean([b_by[k] for k in common])) * 100 if any(b_by[k] for k in common) else 0
    print(f"  {label:<50}: n={len(diffs)}  Δmean={mean_d:+.3f}  rel={rel:+6.1f}%  win={pos:3d}/{len(diffs)} ({pos/len(diffs)*100:3.0f}%)  t={t:+6.2f} p≈{p:.4f}")


print("\n=== Paired comparisons ===")
paired_t(sia_r, nosia_tb_r, sia, nosia_tb, "SIA-no-think vs noSIA-thinkbrief (805Q both)")
paired_t(sia_r, nosia_bt_r, sia, nosia_bt, "SIA-no-think vs noSIA-banthink (100Q intersect, same setup)")
