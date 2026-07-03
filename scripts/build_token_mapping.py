#!/usr/bin/env python3
"""
Build a token ID mapping from a source tokenizer (e.g. 0GM-35B) to a target
tokenizer (e.g. Qwen3-4B-Base) using the text-round-trip method:

    source_token_id  →  decode to text  →  encode with target tokenizer

Mapping strategy:
    1 target token  → exact 1-to-1 mapping  (arr[src_id] = tgt_id)
    N target tokens → approximate: arr[src_id] = tgt_ids[0]  (first token)
    0 target tokens → unmapped: arr[src_id] = -1  (special/empty token only)

For multi-token cases, using the first target token as a proxy is sound because
(a) 0GM merged the substring into one token precisely because it appears as a
    unit; the VM score for tgt_ids[0] carries most of the signal, and
(b) any approximation beats the fallback of score=0 (no VM intervention).

Output files:
  <output>.npy   — int32 numpy array of shape (source_vocab_size,)
                   arr[src_id] = tgt_id  (or -1 only for empty/error tokens)
  <output>.json  — metadata (vocab sizes, coverage stats, model paths)

Usage:
  python scripts/build_token_mapping.py \\
    --src_model /workspace/SIA/models/0GM-1.0-35B-A3B-0427 \\
    --tgt_model /workspace/sia-repo/models/Qwen3-4B-Base \\
    --output /workspace/sia-repo/0g-sparse-inference-alignment/token_mapping_ogm35b_to_qwen3_4b

NOTE: loads tokenizers only (CPU/RAM); does NOT load model weights; safe to run
alongside an active training process.
"""

import argparse
import json
import os
import time

import numpy as np
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Build source→target tokenizer token ID mapping")
    p.add_argument("--src_model", required=True,
                   help="Source model dir (0GM-35B tokenizer lives here)")
    p.add_argument("--tgt_model", required=True,
                   help="Target model dir (Qwen3-4B-Base tokenizer lives here)")
    p.add_argument("--output", required=True,
                   help="Output file prefix (no extension). Two files are written: .npy and .json")
    p.add_argument("--log_interval", type=int, default=10000,
                   help="Print progress every N tokens (default 10000)")
    return p.parse_args()


def main():
    args = parse_args()

    # ---- Load tokenizers (CPU only; no model weights loaded) ----
    print(f"[1/4] Loading source tokenizer: {args.src_model}")
    src_tok = AutoTokenizer.from_pretrained(args.src_model, trust_remote_code=True)
    src_vocab_size = len(src_tok)
    print(f"      source vocab size: {src_vocab_size}")

    print(f"[2/4] Loading target tokenizer: {args.tgt_model}")
    tgt_tok = AutoTokenizer.from_pretrained(args.tgt_model, trust_remote_code=True)
    tgt_vocab_size = len(tgt_tok)
    print(f"      target vocab size: {tgt_vocab_size}")

    # ---- Build mapping ----
    print(f"[3/4] Building mapping ({src_vocab_size} source tokens) ...")
    mapping = np.full(src_vocab_size, -1, dtype=np.int32)

    exact = 0         # 1-to-1 exact match
    approx = 0        # multi-token → first token approximation
    empty = 0         # decode produced empty string → encode gives [] → -1
    errors = 0        # exceptions (rare byte-level edge cases)

    t0 = time.perf_counter()

    for src_id in range(src_vocab_size):
        try:
            text = src_tok.decode([src_id], skip_special_tokens=False)
            tgt_ids = tgt_tok.encode(text, add_special_tokens=False)

            if len(tgt_ids) == 1:
                tgt_id = int(tgt_ids[0])
                if 0 <= tgt_id < tgt_vocab_size:
                    mapping[src_id] = tgt_id
                    exact += 1
                else:
                    errors += 1
            elif len(tgt_ids) > 1:
                # Approximate: use first target token as proxy score
                tgt_id = int(tgt_ids[0])
                if 0 <= tgt_id < tgt_vocab_size:
                    mapping[src_id] = tgt_id
                    approx += 1
                else:
                    errors += 1
            else:
                empty += 1               # truly empty (special token with no text repr)

        except Exception:
            errors += 1

        if (src_id + 1) % args.log_interval == 0:
            elapsed = time.perf_counter() - t0
            pct = (src_id + 1) / src_vocab_size * 100
            eta = elapsed / (src_id + 1) * (src_vocab_size - src_id - 1)
            mapped = exact + approx
            print(
                f"  {src_id+1:>7}/{src_vocab_size}  {pct:5.1f}%  "
                f"exact={exact}  approx={approx}  empty={empty}  err={errors}  "
                f"elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                flush=True,
            )

    elapsed_total = time.perf_counter() - t0
    mapped = exact + approx
    coverage_pct = mapped / src_vocab_size * 100
    exact_pct = exact / src_vocab_size * 100
    approx_pct = approx / src_vocab_size * 100

    print(f"\n  Done in {elapsed_total:.1f}s")
    print(f"  exact (1-to-1):  {exact:>7}  ({exact_pct:.2f}%)")
    print(f"  approx (first):  {approx:>7}  ({approx_pct:.2f}%)")
    print(f"  total mapped:    {mapped:>7}  ({coverage_pct:.2f}%)")
    print(f"  empty (unmapped):{empty:>7}  ({empty/src_vocab_size*100:.2f}%)")
    print(f"  errors:          {errors:>7}")

    # ---- Save ----
    print(f"\n[4/4] Saving ...")
    npy_path  = args.output + ".npy"
    json_path = args.output + ".json"

    np.save(npy_path, mapping)
    size_kb = os.path.getsize(npy_path) / 1024
    print(f"  mapping array → {npy_path}  ({size_kb:.0f} KB)")

    meta = {
        "src_model":        args.src_model,
        "tgt_model":        args.tgt_model,
        "src_vocab_size":   src_vocab_size,
        "tgt_vocab_size":   tgt_vocab_size,
        "exact_count":      exact,
        "approx_count":     approx,
        "mapped_count":     mapped,
        "empty_count":      empty,
        "error_count":      errors,
        "exact_pct":        round(exact_pct, 4),
        "approx_pct":       round(approx_pct, 4),
        "coverage_pct":     round(coverage_pct, 4),
        "mapping_strategy": "first-token-approximation",
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  metadata       → {json_path}")

    # ---- Quick sanity check ----
    print("\nSanity check (10 sample mapped tokens):")
    shown = 0
    for src_id in range(src_vocab_size):
        if mapping[src_id] < 0:
            continue
        src_text = src_tok.decode([src_id],            skip_special_tokens=False)
        tgt_text = tgt_tok.decode([int(mapping[src_id])], skip_special_tokens=False)
        match = "✓" if src_text == tgt_text else "≈"
        print(f"  {match} src[{src_id:6d}]={repr(src_text):30s}  →  tgt[{mapping[src_id]:6d}]={repr(tgt_text)}")
        shown += 1
        if shown >= 10:
            break

    # Show a few approximate-mapped examples
    print("\nSample approximate-mapped tokens (multi-token, first used as proxy):")
    shown = 0
    for src_id in range(src_vocab_size):
        if mapping[src_id] < 0:
            continue
        try:
            src_text = src_tok.decode([src_id], skip_special_tokens=False)
            tgt_ids  = tgt_tok.encode(src_text, add_special_tokens=False)
            if len(tgt_ids) > 1:
                proxy_text = tgt_tok.decode([int(tgt_ids[0])], skip_special_tokens=False)
                print(f"  ~ src[{src_id:6d}]={repr(src_text):30s}  →  {len(tgt_ids)} tgt tokens {tgt_ids[:4]}  proxy={repr(proxy_text)}")
                shown += 1
        except Exception:
            pass
        if shown >= 10:
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
