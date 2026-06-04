"""Score outputs truncated to first 180 LLM tokens."""
import json, time, math, statistics
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda:0"
SKY = "/workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B"
LLM = "/workspace/SIA/models/Qwen3-VL-30B-A3B-Instruct"
T = 180

SIA_F = "/tmp/vl30b_runs/vllm_sia_scored.json"
NOS_F = "/tmp/vl30b_runs/nosia_scored_with_trunc256.json"

print(f"loading LLM tok …", flush=True)
LTOK = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)
print(f"loading Skywork …", flush=True)
STOK = AutoTokenizer.from_pretrained(SKY)
M = AutoModelForSequenceClassification.from_pretrained(
    SKY, torch_dtype=torch.bfloat16, device_map=DEVICE, attn_implementation="sdpa",
).eval()
print(f"ready", flush=True)

def trunc(text):
    if not text: return text
    ids = LTOK.encode(text, add_special_tokens=False)
    return text if len(ids) <= T else LTOK.decode(ids[:T], skip_special_tokens=True)

def score(instr, resp):
    if not resp: return None
    convs = [{"role":"user","content":instr},{"role":"assistant","content":resp}]
    txt = STOK.apply_chat_template(convs, tokenize=False)
    if STOK.bos_token and txt.startswith(STOK.bos_token):
        txt = txt[len(STOK.bos_token):]
    enc = STOK(txt, return_tensors="pt", truncation=True, max_length=4096).to(DEVICE)
    with torch.no_grad():
        return float(M(**enc).logits[0][0].item())

sia = json.load(open(SIA_F))
print(f"scoring SIA n={len(sia)} at trunc={T}", flush=True)
t0=time.time()
for i,r in enumerate(sia):
    o = r.get("output"); tr = trunc(o) if o else None
    try: r[f"reward_trunc{T}"] = score(r["instruction"], tr)
    except: r[f"reward_trunc{T}"] = None
print(f"  SIA done in {time.time()-t0:.0f}s", flush=True)

nos = json.load(open(NOS_F))
print(f"scoring noSIA n={len(nos)} at trunc={T}", flush=True)
t0=time.time()
for r in nos:
    o = r.get("output"); tr = trunc(o) if o else None
    try: r[f"reward_trunc{T}"] = score(r["instruction"], tr)
    except: r[f"reward_trunc{T}"] = None
print(f"  noSIA done in {time.time()-t0:.0f}s", flush=True)

with open(SIA_F, "w") as f: json.dump(sia, f, ensure_ascii=False, indent=2)
with open(NOS_F, "w") as f: json.dump(nos, f, ensure_ascii=False, indent=2)

nos_by = {r["instruction"]: r for r in nos}
paired=[]
for s in sia:
    n_rec = nos_by.get(s["instruction"])
    if n_rec and s.get(f"reward_trunc{T}") is not None and n_rec.get(f"reward_trunc{T}") is not None:
        paired.append({"id": s["id"], "sia": s[f"reward_trunc{T}"], "nos": n_rec[f"reward_trunc{T}"]})
deltas = [p["sia"]-p["nos"] for p in paired]
n=len(deltas)
mean_d = statistics.mean(deltas); std_d = statistics.stdev(deltas)
sia_m = statistics.mean(p["sia"] for p in paired); nos_m = statistics.mean(p["nos"] for p in paired)
t = mean_d / (std_d/math.sqrt(n))
p = 2*(1-0.5*(1+math.erf(abs(t)/math.sqrt(2))))
pos = sum(1 for d in deltas if d>0); neg=sum(1 for d in deltas if d<0)
rel = mean_d/abs(nos_m)*100

print(f"\n=== Run 1 (vllm RM SIA) vs noSIA at first {T} tokens (n={n}) ===")
print(f"  SIA   mean={sia_m:+.3f}")
print(f"  noSIA mean={nos_m:+.3f}")
print(f"  Δ     mean={mean_d:+.3f}  median={statistics.median(deltas):+.3f}  rel={rel:+.1f}%")
print(f"  win   {pos}/{n} = {pos/n*100:.1f}%   loss={neg}")
print(f"  t={t:+.3f}  (p≈{p:.6f})")
