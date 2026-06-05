"""Skywork score noSIA 805Q + paired vs SIA 805Q (real apples-to-apples)."""
import json, statistics, math, time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda:0"
SKY = "/workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B"

NOSIA_INPUT = "/tmp/0gm_runs/nosia_0gm_805q_max2048_nothink.json"
SIA_SCORED  = "/tmp/0gm_runs/sia_0gm_805q_max2048_nothink_scored.json"

print("loading Skywork …", flush=True)
TOK = AutoTokenizer.from_pretrained(SKY)
M = AutoModelForSequenceClassification.from_pretrained(
    SKY, torch_dtype=torch.bfloat16, device_map=DEVICE, attn_implementation="sdpa",
).eval()
print("ready", flush=True)


def score(instr, resp):
    if not resp: return None
    convs = [{"role": "user", "content": instr},
             {"role": "assistant", "content": resp}]
    txt = TOK.apply_chat_template(convs, tokenize=False)
    if TOK.bos_token and txt.startswith(TOK.bos_token):
        txt = txt[len(TOK.bos_token):]
    enc = TOK(txt, return_tensors="pt", truncation=True, max_length=4096).to(DEVICE)
    with torch.no_grad():
        return float(M(**enc).logits[0][0].item())


# Score noSIA
nosia = json.load(open(NOSIA_INPUT))
print(f"\nScoring noSIA 805Q ...", flush=True)
t0 = time.time()
for r in nosia:
    o = r.get("output")
    r["reward"] = score(r["instruction"], o) if o and not str(o).startswith("ERROR:") else None
print(f"done in {time.time()-t0:.0f}s", flush=True)
with open(NOSIA_INPUT.replace(".json", "_scored.json"), "w") as f:
    json.dump(nosia, f, ensure_ascii=False, indent=2)

# Load SIA (already scored)
sia = json.load(open(SIA_SCORED))

def aggregate(records, label):
    rewards = [r["reward"] for r in records if r.get("reward") is not None]
    print(f"\n=== {label} (n={len(rewards)}) ===")
    print(f"  mean={statistics.mean(rewards):+.3f}  median={statistics.median(rewards):+.3f}  std={statistics.stdev(rewards):.3f}")


print("\n" + "=" * 60)
aggregate(sia, "SIA 805Q (rep=1.0 fix, no-think)")
aggregate(nosia, "noSIA 805Q (rep=1.0, no-think, apples-to-apples baseline)")

# Paired comparison
sia_by = {r["instruction"]: r["reward"] for r in sia if r.get("reward") is not None}
nos_by = {r["instruction"]: r["reward"] for r in nosia if r.get("reward") is not None}
common = list(set(sia_by) & set(nos_by))
diffs = [sia_by[k] - nos_by[k] for k in common]
mean_d = statistics.mean(diffs)
std_d = statistics.stdev(diffs)
t = mean_d / (std_d / math.sqrt(len(diffs))) if std_d > 0 else 0
p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))) if abs(t) > 0 else 1
pos = sum(1 for x in diffs if x > 0)
nos_mean = statistics.mean([nos_by[k] for k in common])
rel = mean_d / abs(nos_mean) * 100

print(f"\n=== PAIRED: SIA vs noSIA (n={len(common)}, real apples-to-apples) ===")
print(f"  Δmean = {mean_d:+.3f}  rel = {rel:+.1f}%")
print(f"  win = {pos}/{len(diffs)} ({pos/len(diffs)*100:.0f}%)")
print(f"  t = {t:+.2f}  p ≈ {p:.6f}")

# Driver perf stats
def driver_stats(records, label):
    toks = [r['tokens'] for r in records if r.get('tokens', -1) > 0]
    elapsed = [r['elapsed'] for r in records if r.get('elapsed') is not None]
    print(f"\n{label}:")
    print(f"  n={len(records)}  avg_tok={sum(toks)/len(toks):.0f}  sum_elapsed={sum(elapsed)/60:.1f} min  tok/s={sum(toks)/sum(elapsed):.1f}")

driver_stats(sia, "SIA driver")
driver_stats(nosia, "noSIA driver")
