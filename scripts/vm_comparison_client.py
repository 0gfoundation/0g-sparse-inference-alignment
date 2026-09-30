"""VM comparison client — scores 200 AlpacaEval responses with scalar and vocab_lowrank VMs.

Reads (instruction, response, skywork_reward) from an existing scored JSON file,
sends each pair to both VM servers concurrently, then prints a comparison table
with Spearman correlations vs Skywork.

Usage
-----
    python scripts/vm_comparison_client.py \\
        --data    exp/alpaca-vl30b-farma-bt-2048-seq-20260705/nosia_scored.json \\
        --scalar  http://localhost:8001 \\
        --vocab   http://localhost:8002 \\
        --output  exp/alpaca-vl30b-vm-comparison-20260706/vm_scores.json

    # If Skywork rewards are already in --data, they are reused.
    # Use --limit N to test with a subset first.

Endpoints called
----------------
    POST {url}/score   {"instruction": "...", "response": "..."}  → {"score": X}
    GET  {url}/health  → {"status": "ok", "vm_type": "..."}
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_health(url: str, label: str) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=5)
        r.raise_for_status()
        info = r.json()
        print(f"  [{label}] {url}/health → {info}", flush=True)
        return True
    except Exception as e:
        print(f"  [{label}] health check FAILED: {e}", flush=True)
        return False


def score_one(url: str, instruction: str, response: str, timeout: int = 120) -> float:
    payload = {"instruction": instruction, "response": response}
    r = requests.post(f"{url}/score", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["score"]


def score_all(url: str, items: list, label: str) -> list:
    """Score all items against one VM server, with a progress bar."""
    scores = [None] * len(items)
    n_err  = 0

    def _score_item(idx_item):
        idx, item = idx_item
        try:
            return idx, score_one(url, item["instruction"], item["output"])
        except Exception as e:
            return idx, None

    # Use a thread pool so we can show progress; server is sequential anyway.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_score_item, (i, it)): i for i, it in enumerate(items)}
        done = 0
        t0 = time.time()
        for fut in as_completed(futures):
            idx, score = fut.result()
            scores[idx] = score
            if score is None:
                n_err += 1
            done += 1
            if done % 20 == 0 or done == len(items):
                elapsed = time.time() - t0
                rate = done / elapsed
                print(f"  [{label}] {done}/{len(items)}  "
                      f"{elapsed:.0f}s  {rate:.1f} req/s  errors={n_err}",
                      flush=True)

    return scores


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _stats(scores, skywork):
    """Return (mean, n, rho_vs_skywork, p_value, pairwise_acc)."""
    paired = [(s, sw) for s, sw in zip(scores, skywork)
              if s is not None and sw is not None]
    if not paired:
        return None, 0, None, None, None

    sc, sw = zip(*paired)
    mean  = sum(sc) / len(sc)
    n     = len(sc)
    rho, p = spearmanr(sw, sc)

    # Pairwise accuracy vs Skywork ordering
    correct = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if sw[i] == sw[j]:
                continue
            total += 1
            prefer_i = sw[i] > sw[j]
            if (sc[i] > sc[j]) == prefer_i:
                correct += 1
    pair_acc = correct / total if total else None

    return mean, n, rho, p, pair_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",   required=True,
                   help="scored JSON (instruction + output + reward fields)")
    p.add_argument("--scalar", default="http://localhost:8001",
                   help="scalar head VM server URL")
    p.add_argument("--vocab",  default="http://localhost:8002",
                   help="vocab_lowrank VM server URL")
    p.add_argument("--output", default=None,
                   help="save raw scores here (optional)")
    p.add_argument("--limit",  type=int, default=0,
                   help="only score first N items (0 = all)")
    args = p.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────
    data = json.load(open(args.data))
    if args.limit:
        data = data[: args.limit]
    # Keep only items with a valid Skywork reward and non-empty output
    items = [
        it for it in data
        if it.get("reward") is not None
        and str(it.get("output", "")).strip()
        and not str(it.get("output", "")).startswith("ERROR:")
    ]
    print(f"Loaded {len(items)} valid items from {args.data}", flush=True)

    # ── Health checks ──────────────────────────────────────────────────────
    print("\nHealth checks:")
    ok_scalar = check_health(args.scalar, "scalar")
    ok_vocab  = check_health(args.vocab,  "vocab ")
    if not (ok_scalar and ok_vocab):
        print("ERROR: one or both servers not reachable. Start them first.")
        return

    skywork_scores = [float(it["reward"]) for it in items]

    # ── Score with both VMs ────────────────────────────────────────────────
    print(f"\nScoring {len(items)} items with scalar head VM ({args.scalar}) ...")
    t0 = time.time()
    scalar_scores = score_all(args.scalar, items, "scalar")
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"\nScoring {len(items)} items with vocab_lowrank VM ({args.vocab}) ...")
    t0 = time.time()
    vocab_scores = score_all(args.vocab, items, "vocab ")
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Save raw scores ────────────────────────────────────────────────────
    if args.output:
        out_data = []
        for it, sc, vc in zip(items, scalar_scores, vocab_scores):
            out_data.append({
                "instruction": it["instruction"],
                "output":      it["output"],
                "skywork":     it["reward"],
                "scalar":      sc,
                "vocab":       vc,
            })
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out_data, open(args.output, "w"), ensure_ascii=False, indent=2)
        print(f"\nRaw scores saved → {args.output}")

    # ── Statistics ────────────────────────────────────────────────────────
    sc_mean, sc_n, sc_rho, sc_p, sc_acc = _stats(scalar_scores, skywork_scores)
    vc_mean, vc_n, vc_rho, vc_p, vc_acc = _stats(vocab_scores,  skywork_scores)
    cr_rho, _  = spearmanr(
        [s for s, v in zip(scalar_scores, vocab_scores) if s is not None and v is not None],
        [v for s, v in zip(scalar_scores, vocab_scores) if s is not None and v is not None],
    )

    def _fmt_rho(rho, p):
        if rho is None: return "   n/a"
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        return f"{rho:+.4f}{stars}"

    print(f"\n{'='*66}")
    print(f"  {'Metric':<38} {'Scalar':>12}  {'VocabLR':>12}")
    print(f"  {'-'*62}")
    print(f"  {'n scored':<38} {sc_n:>12}  {vc_n:>12}")
    print(f"  {'mean VM score':<38} {sc_mean:>12.4f}  {vc_mean:>12.4f}")
    print(f"  {'Spearman ρ vs Skywork':<38} {_fmt_rho(sc_rho,sc_p):>12}  {_fmt_rho(vc_rho,vc_p):>12}")
    print(f"  {'Pairwise acc vs Skywork':<38} {sc_acc:>11.1%}  {vc_acc:>11.1%}")
    print(f"  {'-'*62}")
    print(f"  {'Spearman ρ  scalar vs vocab':<38} {cr_rho:>+12.4f}")
    print(f"{'='*66}")
    print(f"\n  *** p<0.001  ** p<0.01  * p<0.05\n")


if __name__ == "__main__":
    main()
