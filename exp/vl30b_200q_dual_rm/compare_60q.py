"""Compare first 60 questions across Run 2 (PyTorch RM), Run 1 (vllm RM), noSIA — both at full length and trunc=256."""
import json, time, math, statistics
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda:0"
SKY = "/workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B"
LLM = "/workspace/SIA/models/Qwen3-VL-30B-A3B-Instruct"

RUN2 = "/tmp/vl30b_runs/pytorch_sia_outputs.json"
RUN1 = "/tmp/vl30b_runs/vllm_sia_scored.json"
NOS = "/tmp/vl30b_runs/nosia_scored_with_trunc256.json"

print("Loading LLM tok …", flush=True)
LTOK = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)
print("Loading Skywork …", flush=True)
STOK = AutoTokenizer.from_pretrained(SKY)
M = AutoModelForSequenceClassification.from_pretrained(
    SKY, torch_dtype=torch.bfloat16, device_map=DEVICE, attn_implementation="sdpa",
).eval()
print("Ready", flush=True)


def trunc(text, n):
    if not text:
        return text
    ids = LTOK.encode(text, add_special_tokens=False)
    return text if len(ids) <= n else LTOK.decode(ids[:n], skip_special_tokens=True)


def score(instr, resp):
    if not resp:
        return None
    convs = [{"role": "user", "content": instr}, {"role": "assistant", "content": resp}]
    txt = STOK.apply_chat_template(convs, tokenize=False)
    if STOK.bos_token and txt.startswith(STOK.bos_token):
        txt = txt[len(STOK.bos_token):]
    enc = STOK(txt, return_tensors="pt", truncation=True, max_length=4096).to(DEVICE)
    with torch.no_grad():
        return float(M(**enc).logits[0][0].item())


# Load Run 2, take first 60 records
run2 = json.load(open(RUN2))[:60]
print(f"Run 2 (PyTorch RM): {len(run2)} records to score", flush=True)
t0 = time.time()
for i, r in enumerate(run2):
    out = r.get("output")
    r["reward"] = score(r["instruction"], out) if out else None
    r["reward_trunc256"] = score(r["instruction"], trunc(out, 256)) if out else None
print(f"  Run 2 done in {time.time()-t0:.0f}s", flush=True)

# Save
with open("/tmp/vl30b_runs/pytorch_sia_scored_partial60.json", "w") as f:
    json.dump(run2, f, ensure_ascii=False, indent=2)

# Load other arms
run1 = json.load(open(RUN1))
nos = json.load(open(NOS))
run1_by = {r["instruction"]: r for r in run1}
nos_by = {r["instruction"]: r for r in nos}

# Pair on instruction
triples = []
for r2 in run2:
    instr = r2["instruction"]
    r1 = run1_by.get(instr)
    n_rec = nos_by.get(instr)
    if r1 and n_rec:
        triples.append((r2, r1, n_rec))

n = len(triples)
print(f"\nPaired: {n}/{len(run2)}")
print()

def stats_block(label, r2_vals, r1_vals, ns_vals):
    if not r2_vals: return
    print(f"=== {label} ===")
    print(f"  Run 2 (PyTorch RM):  mean={statistics.mean(r2_vals):+.3f}  median={statistics.median(r2_vals):+.3f}  std={statistics.stdev(r2_vals):.3f}")
    print(f"  Run 1 (vllm RM):     mean={statistics.mean(r1_vals):+.3f}  median={statistics.median(r1_vals):+.3f}  std={statistics.stdev(r1_vals):.3f}")
    print(f"  noSIA:               mean={statistics.mean(ns_vals):+.3f}  median={statistics.median(ns_vals):+.3f}  std={statistics.stdev(ns_vals):.3f}")
    print()
    # Compare Run 2 vs noSIA
    d_r2 = [r2 - ns for r2, ns in zip(r2_vals, ns_vals)]
    pos = sum(1 for d in d_r2 if d > 0)
    neg = sum(1 for d in d_r2 if d < 0)
    mean_d = statistics.mean(d_r2); std_d = statistics.stdev(d_r2) if len(d_r2)>1 else 0
    rel = mean_d / abs(statistics.mean(ns_vals)) * 100 if statistics.mean(ns_vals) else 0
    t = mean_d / (std_d / math.sqrt(len(d_r2))) if std_d > 0 else 0
    p = 2*(1-0.5*(1+math.erf(abs(t)/math.sqrt(2)))) if abs(t)>0 else 1
    print(f"  Run 2 vs noSIA: Δ mean={mean_d:+.3f} rel={rel:+.1f}% win={pos}/{n} ({pos/n*100:.0f}%) t={t:+.2f} p≈{p:.4f}")
    # Run 1 vs noSIA
    d_r1 = [r1 - ns for r1, ns in zip(r1_vals, ns_vals)]
    pos1 = sum(1 for d in d_r1 if d > 0)
    mean_d1 = statistics.mean(d_r1); std_d1 = statistics.stdev(d_r1) if len(d_r1)>1 else 0
    rel1 = mean_d1 / abs(statistics.mean(ns_vals)) * 100 if statistics.mean(ns_vals) else 0
    t1 = mean_d1 / (std_d1 / math.sqrt(len(d_r1))) if std_d1 > 0 else 0
    p1 = 2*(1-0.5*(1+math.erf(abs(t1)/math.sqrt(2)))) if abs(t1)>0 else 1
    print(f"  Run 1 vs noSIA: Δ mean={mean_d1:+.3f} rel={rel1:+.1f}% win={pos1}/{n} ({pos1/n*100:.0f}%) t={t1:+.2f} p≈{p1:.4f}")
    # Run 2 vs Run 1
    d_r21 = [r2 - r1 for r2, r1 in zip(r2_vals, r1_vals)]
    pos21 = sum(1 for d in d_r21 if d > 0)
    mean_d21 = statistics.mean(d_r21); std_d21 = statistics.stdev(d_r21) if len(d_r21)>1 else 0
    t21 = mean_d21 / (std_d21 / math.sqrt(len(d_r21))) if std_d21 > 0 else 0
    p21 = 2*(1-0.5*(1+math.erf(abs(t21)/math.sqrt(2)))) if abs(t21)>0 else 1
    print(f"  Run 2 vs Run 1: Δ mean={mean_d21:+.3f}  Run 2 wins {pos21}/{n} ({pos21/n*100:.0f}%) t={t21:+.2f} p≈{p21:.4f}")
    print()


# Full length
r2_full = [t[0]["reward"] for t in triples if t[0]["reward"] is not None]
r1_full = [t[1]["reward"] for t in triples if t[0]["reward"] is not None]
ns_full = [t[2]["reward"] for t in triples if t[0]["reward"] is not None]
stats_block("FULL length", r2_full, r1_full, ns_full)

# Trunc 256
r2_t256 = [t[0]["reward_trunc256"] for t in triples if t[0]["reward_trunc256"] is not None]
r1_t256 = [t[1].get("reward_trunc256") for t in triples if t[0]["reward_trunc256"] is not None]
ns_t256 = [t[2].get("reward_trunc256") for t in triples if t[0]["reward_trunc256"] is not None]
# Filter ones with all three values
keep = [(a,b,c) for a,b,c in zip(r2_t256, r1_t256, ns_t256) if a is not None and b is not None and c is not None]
if keep:
    r2_t256, r1_t256, ns_t256 = zip(*keep)
    stats_block("TRUNC 256 tokens", r2_t256, r1_t256, ns_t256)

# Also: length comparison
print("=== Output length (generated tokens) ===")
r2_lens = [t[0].get("usage", {}).get("completion_tokens", 0) for t in triples]
r1_lens = [t[1].get("usage", {}).get("completion_tokens", 0) for t in triples]
ns_lens = [t[2].get("tokens", 0) for t in triples]
print(f"  Run 2 (PyTorch RM):  mean={statistics.mean(r2_lens):.0f} median={statistics.median(r2_lens):.0f}")
print(f"  Run 1 (vllm RM):     mean={statistics.mean(r1_lens):.0f} median={statistics.median(r1_lens):.0f}")
print(f"  noSIA:               mean={statistics.mean(ns_lens):.0f} median={statistics.median(ns_lens):.0f}")
