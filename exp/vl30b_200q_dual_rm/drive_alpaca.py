"""Driver: 200 AlpacaEval questions through SIA server.

Saves outputs incrementally (every 10 Q) so a crash mid-run still yields partial JSON.
"""
import json
import os
import sys
import time

import requests

if len(sys.argv) < 3:
    sys.exit("usage: drive_alpaca.py <out_path> <model_id> [max_tokens=2048] [n=200]")
OUT = sys.argv[1]
MODEL = sys.argv[2]
MAX_TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
N = int(sys.argv[4]) if len(sys.argv) > 4 else 200
TEMP = 1.0

ALPACA = "/workspace/SIA/data/alpaca_eval/alpaca_eval.json"
SERVER = "http://localhost:8000/v1/chat/completions"
data = json.load(open(ALPACA))[:N]

print(f"[driver] model={MODEL} n={len(data)} max_tokens={MAX_TOK} → {OUT}", flush=True)
results = []
t0 = time.time()
for i, item in enumerate(data):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": item["instruction"]}],
        "max_tokens": MAX_TOK,
        "temperature": TEMP,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t_start = time.time()
    rec = {"id": i, "instruction": item["instruction"]}
    try:
        r = requests.post(SERVER, json=payload, timeout=700)
        r.raise_for_status()
        j = r.json()
        rec["output"] = j["choices"][0]["message"]["content"]
        rec["finish_reason"] = j["choices"][0].get("finish_reason")
        rec["usage"] = j.get("usage", {})
        rec["elapsed_s"] = time.time() - t_start
        print(f"[{i+1:3d}/{len(data)}] {rec['elapsed_s']:5.1f}s "
              f"toks={rec['usage'].get('completion_tokens','?')} "
              f"finish={rec['finish_reason']}", flush=True)
    except Exception as e:
        rec["output"] = None
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["elapsed_s"] = time.time() - t_start
        print(f"[{i+1:3d}/{len(data)}] {rec['elapsed_s']:5.1f}s ERR: {e}", flush=True)
    results.append(rec)
    if (i + 1) % 10 == 0 or (i + 1) == len(data):
        with open(OUT, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  ...saved {i+1}/{len(data)} ({(time.time()-t0)/60:.1f} min)", flush=True)

print(f"[driver] DONE total={(time.time()-t0)/60:.1f}min → {OUT}", flush=True)
