"""Score current PyTorch-RM partial run + 3-way paired compare."""
import json, statistics, math, time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda:0"
SKY = "/workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B"

OURS_REP10 = "/tmp/qwen3_14b_runs/sia_REP10_200q_max256.json"
OURS_VLLM    = "/tmp/qwen3_14b_runs/sia_805q_outputs_max256_scored.json"  # 125Q already scored
PAPER_SIA    = "/workspace/SIA/git/SIA/assets/generation_results/alpaca_sia_scored_Qwen3-14B_202605152354.json"
PAPER_NOSIA  = "/workspace/SIA/git/SIA/assets/generation_results/alpaca_base_scored_Qwen3-14B_202605152354.json"

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

# === Load ===
ours_pt = json.load(open(OURS_REP10))
ours_vl = json.load(open(OURS_VLLM))
paper_s = json.load(open(PAPER_SIA))
paper_n = json.load(open(PAPER_NOSIA))

print(f"Loaded: pytorch-RM={len(ours_pt)}  vllm-RM={len(ours_vl)}  paper_sia={len(paper_s)}  paper_nosia={len(paper_n)}")

ps_by = {r["prompt"]: r for r in paper_s}
pn_by = {r["prompt"]: r for r in paper_n}
ovl_by = {r["instruction"]: r for r in ours_vl}

def find_paper(instr, by):
    for p, r in by.items():
        if instr in p:
            return r
    return None

# === Score ours_pt ===
print(f"\nScoring {len(ours_pt)} PyTorch-RM outputs ...", flush=True)
t0 = time.time()
for r in ours_pt:
    o = r.get("output")
    r["reward"] = score(r["instruction"], o) if o else None
print(f"done in {time.time()-t0:.0f}s", flush=True)

# Save scored
with open(OURS_REP10.replace(".json", "_scored.json"), "w") as f:
    json.dump(ours_pt, f, ensure_ascii=False, indent=2)

# === Paired records ===
print(f"\nBuilding 4-way paired records ...")
paired = []
for our_pt in ours_pt:
    if our_pt.get("reward") is None:
        continue
    instr = our_pt["instruction"]
    pr_s = find_paper(instr, ps_by)
    pr_n = find_paper(instr, pn_by)
    our_vl = ovl_by.get(instr)
    if pr_s and pr_n:
        ps_r = pr_s.get("reward")
        pn_r = pr_n.get("reward")
        # ours_vllm rewards were already scored in earlier run; lookup
        ovl_r = our_vl.get("reward") if our_vl else None
        if ps_r is not None and pn_r is not None:
            paired.append({
                "instr": instr,
                "ours_pt": our_pt["reward"],
                "ours_vl": ovl_r,    # may be None if not in vllm partial
                "paper_sia": ps_r,
                "paper_nos": pn_r,
            })

n = len(paired)
n_with_vl = sum(1 for p in paired if p["ours_vl"] is not None)
print(f"paired pytorch-RM with paper: {n}/{len(ours_pt)}")
print(f"  of those, also have ours-vllm-RM scored: {n_with_vl}")

def paired_t(a, b, label):
    d = [x - y for x, y in zip(a, b)]
    if not d:
        print(f"  {label}: empty")
        return
    mean_d = statistics.mean(d)
    std_d = statistics.stdev(d) if len(d) > 1 else 0
    t = mean_d / (std_d / math.sqrt(len(d))) if std_d > 0 else 0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))) if abs(t) > 0 else 1
    pos = sum(1 for x in d if x > 0)
    rel = mean_d / abs(statistics.mean(b)) * 100 if statistics.mean(b) else 0
    print(f"  {label:<40}: Δmean={mean_d:+.3f}  rel={rel:+6.1f}%  win={pos:3d}/{len(d)} ({pos/len(d)*100:3.0f}%)  t={t:+6.2f} p≈{p:.4f}")

print(f"\n=== Skywork mean reward (n={n}, all max_tokens=256) ===")
opt_r = [p["ours_pt"] for p in paired]
ps_r  = [p["paper_sia"] for p in paired]
pn_r  = [p["paper_nos"] for p in paired]
print(f"  Ours REP=1.0  (rep_penalty fix applied):                   mean={statistics.mean(opt_r):+.3f}  median={statistics.median(opt_r):+.3f}")
print(f"  Paper SIA     (official code, rep_penalty=none):           mean={statistics.mean(ps_r):+.3f}  median={statistics.median(ps_r):+.3f}")
print(f"  Paper noSIA   (official code, no SIA):                     mean={statistics.mean(pn_r):+.3f}  median={statistics.median(pn_r):+.3f}")

print(f"\n=== Paired comparisons ===")
paired_t(opt_r, ps_r, "Ours-REP10 vs Paper-SIA")
paired_t(opt_r, pn_r, "Ours-REP10 vs Paper-noSIA")
paired_t(ps_r, pn_r, "Paper-SIA vs Paper-noSIA")

# Critical: Ours-REP10 vs Ours-REP13 (vllm-RM 125Q) — direct A/B on rep_penalty
if n_with_vl > 0:
    sub = [p for p in paired if p["ours_vl"] is not None]
    a = [p["ours_pt"] for p in sub]
    b = [p["ours_vl"] for p in sub]
    print(f"\n=== HYPOTHESIS TEST: REP=1.0 vs REP=1.3 (same Q's, isolates rep_penalty, n={len(sub)}) ===")
    print(f"  Ours-REP10 (1.0): mean={statistics.mean(a):+.3f}  median={statistics.median(a):+.3f}")
    print(f"  Ours-REP13 (1.3): mean={statistics.mean(b):+.3f}  median={statistics.median(b):+.3f}")
    paired_t(a, b, "Ours-REP10 vs Ours-REP13")
