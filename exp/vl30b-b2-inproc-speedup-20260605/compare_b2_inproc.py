"""Skywork score b2 inproc 200Q + paired vs prior HTTP path 200Q."""
import json, statistics, math, time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda:0"
SKY = "/workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B"

B2_INPUT = "/tmp/vl30b_runs/sia_VL30B_b2inproc_200q_max256.json"
HTTP_SCORED = "/tmp/vl30b_runs/sia_VL30B_vllmRM_200q_max256_scored.json"  # prior HTTP path 200Q max=256

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


# Score b2 inproc
b2 = json.load(open(B2_INPUT))
print(f"\nScoring b2 inproc 200Q ...", flush=True)
t0 = time.time()
for r in b2:
    r["reward"] = score(r["instruction"], r.get("output"))
print(f"done in {time.time()-t0:.0f}s", flush=True)
with open(B2_INPUT.replace(".json", "_scored.json"), "w") as f:
    json.dump(b2, f, ensure_ascii=False, indent=2)


def aggregate(records, label):
    rewards = [r.get("reward") for r in records if r.get("reward") is not None]
    print(f"\n=== {label} (n={len(rewards)}) ===")
    print(f"  mean={statistics.mean(rewards):+.3f}  median={statistics.median(rewards):+.3f}  std={statistics.stdev(rewards):.3f}")


http = json.load(open(HTTP_SCORED))
print("\n" + "=" * 60)
aggregate(b2, "b2 inproc (this run, vllm 0.17.1)")
aggregate(http, "HTTP path (prior, vllm 0.19)")

# Paired by instruction
b2_by = {r["instruction"]: r.get("reward") for r in b2 if r.get("reward") is not None}
http_by = {r["instruction"]: r.get("reward") for r in http if r.get("reward") is not None}
common = list(set(b2_by) & set(http_by))
diffs = [b2_by[k] - http_by[k] for k in common]
mean_d = statistics.mean(diffs)
std_d = statistics.stdev(diffs)
t = mean_d / (std_d / math.sqrt(len(diffs))) if std_d > 0 else 0
p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))) if abs(t) > 0 else 1
pos = sum(1 for x in diffs if x > 0)
http_mean = statistics.mean([http_by[k] for k in common])
rel = mean_d / abs(http_mean) * 100

print(f"\n=== PAIRED: b2 inproc vs HTTP path (n={len(common)}, expected ≈ 0 if equivalent) ===")
print(f"  Δmean = {mean_d:+.3f}  rel = {rel:+.1f}%")
print(f"  win = {pos}/{len(diffs)} ({pos/len(diffs)*100:.0f}%)")
print(f"  t = {t:+.2f}  p ≈ {p:.4f}")
print(f"  (large p > 0.05 means statistically equivalent → no quality regression)")
