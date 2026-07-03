#!/usr/bin/env python3
"""
Build a token ID mapping from a source tokenizer (e.g. 0GM-35B) to a target
tokenizer (e.g. Qwen3-4B-Base) using the text-round-trip method:

    source_token_id  →  decode to text  →  encode with target tokenizer
    If result is exactly 1 target token: record mapping.
    Otherwise: mark as unmapped (-1).

Output files:
  <output>.npy   — int32 numpy array of shape (source_vocab_size,)
                   arr[src_id] = tgt_id  (or -1 if unmapped)
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

    mapped = 0
    multi_token = 0  # src token decodes to text that tgt tokenizer splits into >1 tokens
    empty = 0        # decode produced empty string → encode gives []
    errors = 0       # exceptions (rare byte-level edge cases)

    t0 = time.perf_counter()

    for src_id in range(src_vocab_size):
        try:
            text = src_tok.decode([src_id], skip_special_tokens=False)
            tgt_ids = tgt_tok.encode(text, add_special_tokens=False)

            if len(tgt_ids) == 1:
                tgt_id = int(tgt_ids[0])
                if 0 <= tgt_id < tgt_vocab_size:
                    mapping[src_id] = tgt_id
                    mapped += 1
                else:
                    errors += 1           # tgt_id out of range (shouldn't happen)
            elif len(tgt_ids) == 0:
                empty += 1               # empty text (special token with no text repr)
            else:
                multi_token += 1         # text maps to multiple tgt tokens → no 1-to-1

        except Exception:
            errors += 1

        if (src_id + 1) % args.log_interval == 0:
            elapsed = time.perf_counter() - t0
            pct = (src_id + 1) / src_vocab_size * 100
            eta = elapsed / (src_id + 1) * (src_vocab_size - src_id - 1)
            print(
                f"  {src_id+1:>7}/{src_vocab_size}  {pct:5.1f}%  "
                f"mapped={mapped}  multi={multi_token}  empty={empty}  err={errors}  "
                f"elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                flush=True,
            )

    elapsed_total = time.perf_counter() - t0
    coverage_pct = mapped / src_vocab_size * 100

    print(f"\n  Done in {elapsed_total:.1f}s")
    print(f"  mapped:      {mapped:>7}  ({coverage_pct:.2f}%)")
    print(f"  multi-token: {multi_token:>7}  ({multi_token/src_vocab_size*100:.2f}%)")
    print(f"  empty:       {empty:>7}  ({empty/src_vocab_size*100:.2f}%)")
    print(f"  errors:      {errors:>7}")

    # ---- Save ----
    print(f"\n[4/4] Saving ...")
    npy_path  = args.output + ".npy"
    json_path = args.output + ".json"

    np.save(npy_path, mapping)
    size_kb = os.path.getsize(npy_path) / 1024
    print(f"  mapping array → {npy_path}  ({size_kb:.0f} KB)")

    meta = {
        "src_model":       args.src_model,
        "tgt_model":       args.tgt_model,
        "src_vocab_size":  src_vocab_size,
        "tgt_vocab_size":  tgt_vocab_size,
        "mapped_count":    mapped,
        "multi_token_count": multi_token,
        "empty_count":     empty,
        "error_count":     errors,
        "coverage_pct":    round(coverage_pct, 4),
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

    # Also show a few unmapped examples
    print("\nSample unmapped tokens (multi-token in target):")
    shown = 0
    for src_id in range(src_vocab_size):
        if mapping[src_id] >= 0:
            continue
        try:
            src_text = src_tok.decode([src_id], skip_special_tokens=False)
            tgt_ids  = tgt_tok.encode(src_text, add_special_tokens=False)
            if len(tgt_ids) > 1:
                print(f"  src[{src_id:6d}]={repr(src_text):30s}  →  {len(tgt_ids)} tgt tokens: {tgt_ids[:5]}")
                shown += 1
        except Exception:
            pass
        if shown >= 5:
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
