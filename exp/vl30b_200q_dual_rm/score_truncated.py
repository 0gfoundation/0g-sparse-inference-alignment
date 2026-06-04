"""Score truncated outputs (first 256 LLM tokens) with Skywork, compare paired."""
import json
import time
import math
import statistics

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda:0"
SKYWORK = "/workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B"
LLM_PATH = "/workspace/SIA/models/Qwen3-VL-30B-A3B-Instruct"
TRUNC = 256

SIA_IN = "/tmp/vl30b_runs/vllm_sia_scored.json"     # has full-length Skywork rewards
NOSIA_IN = "/tmp/alpaca_qwen3vl30b_nosia_200q_scored_20260603_034614.json"

# Load LLM tokenizer to apply 256-token truncation
print("[scorer] loading LLM tokenizer to truncate outputs to 256 tokens …", flush=True)
LLM_TOK = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)


def truncate_256(text):
    if not text:
        return text
    ids = LLM_TOK.encode(text, add_special_tokens=False)
    if len(ids) <= TRUNC:
        return text
    return LLM_TOK.decode(ids[:TRUNC], skip_special_tokens=True)


# Load Skywork
print(f"[scorer] loading Skywork on {DEVICE} …", flush=True)
SKY_TOK = AutoTokenizer.from_pretrained(SKYWORK)
MODEL = AutoModelForSequenceClassification.from_pretrained(
    SKYWORK, torch_dtype=torch.bfloat16,
    device_map=DEVICE, attn_implementation="sdpa",
).eval()
print(f"[scorer] ready", flush=True)


def score(instr, resp):
    if not resp:
        return None
    convs = [{"role": "user", "content": instr}, {"role": "assistant", "content": resp}]
    text = SKY_TOK.apply_chat_template(convs, tokenize=False)
    if SKY_TOK.bos_token and text.startswith(SKY_TOK.bos_token):
        text = text[len(SKY_TOK.bos_token):]
    enc = SKY_TOK(text, return_tensors="pt", truncation=True, max_length=4096).to(DEVICE)
    with torch.no_grad():
        return float(MODEL(**enc).logits[0][0].item())


# Score SIA Run 1 truncated outputs
sia = json.load(open(SIA_IN))
print(f"[scorer] scoring {len(sia)} SIA records (truncated to first {TRUNC} LLM tokens) …", flush=True)
t0 = time.time()
for i, rec in enumerate(sia):
    out = rec.get("output")
    trunc = truncate_256(out) if out else None
    try:
        rec["reward_trunc256"] = score(rec["instruction"], trunc)
    except Exception as e:
        rec["reward_trunc256"] = None
    if (i + 1) % 50 == 0:
        print(f"  SIA {i+1}/{len(sia)} ({time.time()-t0:.0f}s)", flush=True)
print(f"[scorer] SIA truncated scoring done in {time.time()-t0:.0f}s", flush=True)

# Score noSIA truncated outputs
nosia = json.load(open(NOSIA_IN))
print(f"[scorer] scoring {len(nosia)} noSIA records (truncated) …", flush=True)
t0 = time.time()
for i, rec in enumerate(nosia):
    # noSIA file uses 'output' field
    out = rec.get("output")
    trunc = truncate_256(out) if out else None
    try:
        rec["reward_trunc256"] = score(rec["instruction"], trunc)
    except Exception as e:
        rec["reward_trunc256"] = None
    if (i + 1) % 50 == 0:
        print(f"  noSIA {i+1}/{len(nosia)} ({time.time()-t0:.0f}s)", flush=True)
print(f"[scorer] noSIA truncated scoring done in {time.time()-t0:.0f}s", flush=True)

# Save updated files
with open(SIA_IN, "w") as f:
    json.dump(sia, f, ensure_ascii=False, indent=2)
nosia_out = "/tmp/vl30b_runs/nosia_scored_with_trunc256.json"
with open(nosia_out, "w") as f:
    json.dump(nosia, f, ensure_ascii=False, indent=2)

# Pair and compare
nosia_by_instr = {r["instruction"]: r for r in nosia}
paired = []
for sia_rec in sia:
    nosia_rec = nosia_by_instr.get(sia_rec["instruction"])
    if nosia_rec and sia_rec.get("reward_trunc256") is not None and nosia_rec.get("reward_trunc256") is not None:
        paired.append({
            "id": sia_rec["id"],
            "sia_full": sia_rec.get("reward"),
            "sia_t256": sia_rec["reward_trunc256"],
            "nosia_full": nosia_rec.get("reward"),
            "nosia_t256": nosia_rec["reward_trunc256"],
            "delta_full": (sia_rec.get("reward") or 0) - (nosia_rec.get("reward") or 0),
            "delta_t256": sia_rec["reward_trunc256"] - nosia_rec["reward_trunc256"],
        })

n = len(paired)
print()
print(f"=== Comparison: full-length vs first-{TRUNC}-tokens Skywork scoring (paired n={n}) ===\n")


def summarize(label, sia_vals, nos_vals, deltas):
    n_p = sum(1 for d in deltas if d > 0)
    n_n = sum(1 for d in deltas if d < 0)
    mean_d = statistics.mean(deltas)
    std_d = statistics.stdev(deltas)
    t = mean_d / (std_d / math.sqrt(len(deltas))) if std_d > 0 else float("nan")
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))) if math.isfinite(t) else float("nan")
    rel = mean_d / abs(statistics.mean(nos_vals)) * 100 if statistics.mean(nos_vals) else float("nan")
    print(f"{label}:")
    print(f"  SIA   mean={statistics.mean(sia_vals):+.3f}  median={statistics.median(sia_vals):+.3f}  std={statistics.stdev(sia_vals):.3f}")
    print(f"  noSIA mean={statistics.mean(nos_vals):+.3f}  median={statistics.median(nos_vals):+.3f}  std={statistics.stdev(nos_vals):.3f}")
    print(f"  Δ     mean={mean_d:+.3f}  median={statistics.median(deltas):+.3f}  rel={rel:+.1f}%")
    print(f"  win={n_p}/{len(deltas)} ({n_p/len(deltas)*100:.1f}%), loss={n_n}, t={t:+.3f} (p≈{p:.6f})")
    print()


summarize("FULL length",
          [p["sia_full"] for p in paired if p["sia_full"] is not None],
          [p["nosia_full"] for p in paired if p["nosia_full"] is not None],
          [p["delta_full"] for p in paired if p["sia_full"] is not None and p["nosia_full"] is not None])

summarize(f"FIRST {TRUNC} TOKENS",
          [p["sia_t256"] for p in paired],
          [p["nosia_t256"] for p in paired],
          [p["delta_t256"] for p in paired])

json.dump(paired, open("/tmp/vl30b_runs/run1_vs_nosia_trunc256_paired.json", "w"), ensure_ascii=False, indent=2)
print("Paired data saved: /tmp/vl30b_runs/run1_vs_nosia_trunc256_paired.json")
