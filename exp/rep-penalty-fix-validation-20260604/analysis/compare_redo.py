"""Skywork score REDO SIA-2048-v2 (full + trunc256) + paired vs noSIA-2048."""
import json, statistics, math, time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda:0"
SKY = "/workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B"
LLM_TOK = "/workspace/SIA/models/Qwen3-VL-30B-A3B-Instruct"
T = 256

SIA_V2     = "/tmp/vl30b_runs/sia_VL30B_vllmRM_200q_max2048_v2.json"
NOSIA_2048 = "/tmp/vl30b_runs/nosia_VL30B_200q_max2048_scored_dual.json"
SIA_256    = "/tmp/vl30b_runs/sia_VL30B_vllmRM_200q_max256_scored.json"
NOSIA_256  = "/tmp/vl30b_runs/nosia_VL30B_200q_max256_scored.json"

print("loading tokenizers …", flush=True)
LTOK = AutoTokenizer.from_pretrained(LLM_TOK, trust_remote_code=True)
TOK = AutoTokenizer.from_pretrained(SKY)
print("loading Skywork …", flush=True)
M = AutoModelForSequenceClassification.from_pretrained(
    SKY, torch_dtype=torch.bfloat16, device_map=DEVICE, attn_implementation="sdpa",
).eval()
print("ready", flush=True)


def trunc(text, n=T):
    if not text: return text
    ids = LTOK.encode(text, add_special_tokens=False)
    if len(ids) <= n: return text
    return LTOK.decode(ids[:n], skip_special_tokens=True)


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


sia_v2 = json.load(open(SIA_V2))
print(f"\nScoring SIA-2048-v2 full + trunc{T}, n={len(sia_v2)} ...", flush=True)
t0 = time.time()
for r in sia_v2:
    out = r.get("output")
    if out:
        r["reward"]              = score(r["instruction"], out)
        r[f"reward_trunc{T}"]    = score(r["instruction"], trunc(out, T))
    else:
        r["reward"] = None
        r[f"reward_trunc{T}"] = None
print(f"done in {time.time()-t0:.0f}s", flush=True)
with open(SIA_V2.replace(".json", "_scored.json"), "w") as f:
    json.dump(sia_v2, f, ensure_ascii=False, indent=2)

# Verify SIA-v2 is NOT identical to noSIA-2048 (the broken-run check)
nosia_2048 = json.load(open(NOSIA_2048))
nosia_2048_by = {r["instruction"]: r for r in nosia_2048}
sia_256_by    = {r["instruction"]: r for r in json.load(open(SIA_256))}
nosia_256_by  = {r["instruction"]: r for r in json.load(open(NOSIA_256))}

# Sanity check: outputs should differ
same_count = 0
for r in sia_v2[:10]:
    instr = r["instruction"]
    n = nosia_2048_by.get(instr)
    if n and r.get("output") == n.get("output"):
        same_count += 1
print(f"\nSanity check (first 10 Q): {same_count}/10 outputs byte-identical to noSIA-2048")
print(f"  (Expected: 0/10 — SIA should produce different outputs from noSIA)")

paired = []
for r in sia_v2:
    if r.get("reward") is None or r.get(f"reward_trunc{T}") is None: continue
    instr = r["instruction"]
    n2 = nosia_2048_by.get(instr); s256 = sia_256_by.get(instr); n256 = nosia_256_by.get(instr)
    if n2 and s256 and n256:
        d = {
            "sia_2048_full":    r["reward"],
            "sia_2048_trunc":   r[f"reward_trunc{T}"],
            "nosia_2048_full":  n2.get("reward"),
            "nosia_2048_trunc": n2.get(f"reward_trunc{T}"),
            "sia_256":          s256.get("reward"),
            "nosia_256":        n256.get("reward"),
        }
        if all(v is not None for v in d.values()):
            paired.append(d)

N = len(paired)
print(f"\nfully paired across all arms: {N}/{len(sia_v2)}")


def paired_t(a, b, label):
    d = [x - y for x, y in zip(a, b)]
    mean_d = statistics.mean(d)
    std_d  = statistics.stdev(d) if len(d) > 1 else 0
    t = mean_d / (std_d / math.sqrt(len(d))) if std_d > 0 else 0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))) if abs(t) > 0 else 1
    pos = sum(1 for x in d if x > 0)
    rel = mean_d / abs(statistics.mean(b)) * 100 if statistics.mean(b) else 0
    print(f"  {label:<50}: Δmean={mean_d:+.3f}  rel={rel:+6.1f}%  win={pos:3d}/{len(d)} ({pos/len(d)*100:3.0f}%)  t={t:+6.2f} p≈{p:.4f}")


print(f"\n=== VL-30B Skywork mean reward (n={N}) ===")
for key, lbl in [
    ("sia_2048_full",   "SIA-2048-v2 full      (REDO, intervention 25.8%)"),
    ("nosia_2048_full", "noSIA-2048 full       (baseline, rep=1.0)"),
    ("sia_2048_trunc",  "SIA-2048-v2 trunc256"),
    ("nosia_2048_trunc","noSIA-2048 trunc256"),
    ("sia_256",         "SIA-256 native"),
    ("nosia_256",       "noSIA-256 native"),
]:
    vals = [p[key] for p in paired]
    print(f"  {lbl:<46}: mean={statistics.mean(vals):+.3f}  median={statistics.median(vals):+.3f}")

print(f"\n=== Apples-to-apples ===")
print("\n[Full-length, max=2048 regime — THE KEY TEST]")
paired_t([p["sia_2048_full"] for p in paired],   [p["nosia_2048_full"] for p in paired],   "SIA-2048 full vs noSIA-2048 full")

print("\n[Trunc-256, max=2048 outputs cut to first 256]")
paired_t([p["sia_2048_trunc"] for p in paired],  [p["nosia_2048_trunc"] for p in paired],  "SIA-2048 trunc256 vs noSIA-2048 trunc256")

print("\n[256-cap reference]")
paired_t([p["sia_256"] for p in paired], [p["nosia_256"] for p in paired], "SIA-256 vs noSIA-256")

print("\n[Cross-length on SIA: same arm, different max_tokens]")
paired_t([p["sia_2048_trunc"] for p in paired], [p["sia_256"] for p in paired], "SIA-2048-trunc256 vs SIA-256-native")
