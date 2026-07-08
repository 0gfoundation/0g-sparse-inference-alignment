"""
Token mapping table quality test.

Tests the ogm35b → qwen3_4b mapping table by:
  1. Loading 35B and VM tokenizers (CPU only, no GPU needed)
  2. Reading existing 0GM-35B inference result files
  3. For each generated output token:
       35B tokenize → map via table → VM decode single token
     Compare VM-decoded text with 35B-decoded text for that token.
  4. Report per-token match rate, mismatch categories, and per-sample stats.

Usage:
    python scripts/test_token_mapping.py [--files <glob>] [--limit N]
"""

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MAPPING_NPY  = REPO / "token_mapping_ogm35b_to_qwen3_4b.npy"
MAPPING_JSON = REPO / "token_mapping_ogm35b_to_qwen3_4b.json"
LLM_TOK_PATH = "/workspace/sia-repo/models/0GM-1.0-35B-A3B-tokenizer-only"
VM_TOK_PATH  = "/workspace/sia-repo/models/Qwen3-4B-Base"

DEFAULT_GLOBS = [
    str(REPO / "exp/alpaca-0gm35b-b2-inproc-20260609/*.json"),
    str(REPO / "exp/alpaca-0gm35b-piecewise-fix3-20260610/*.json"),
    str(REPO / "exp/0gm-35b-sia-mmlu-150q-thinking-20260605/*.json"),
    str(REPO / "exp/0gm-35b-nosia-mmlu-150q-thinking-20260605/*.json"),
    str(REPO / "exp/0gm-35b-sia-mmlu-150q-no-think-20260605/*.json"),
]


def load_result_files(patterns: list[str]) -> list[dict]:
    """Load all matching JSON result files, return list of samples."""
    samples = []
    for pat in patterns:
        for path in glob.glob(pat):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        item["_source_file"] = path
                    samples.extend(data)
                    print(f"  loaded {len(data):4d} samples  ← {path}")
            except Exception as e:
                print(f"  WARNING: could not load {path}: {e}")
    return samples


def extract_generated_text(sample: dict) -> str:
    """Return the model-generated text from a result sample."""
    # AlpacaEval format: 'output' is the generated response
    # MMLU format: 'result' contains the full conversation; 'output' may also exist
    if "output" in sample and sample["output"]:
        return str(sample["output"])
    if "result" in sample and sample["result"]:
        return str(sample["result"])
    return ""


def categorize_mismatch(text_35b: str, text_vm: str) -> str:
    """Return a short label for why text_35b != text_vm."""
    if not text_35b and not text_vm:
        return "both_empty"
    if not text_35b:
        return "35b_empty"
    if not text_vm:
        return "vm_empty"
    # Check if it's a whitespace difference
    if text_35b.strip() == text_vm.strip():
        return "whitespace_diff"
    # Check if one is a prefix of the other (first-token approximation artifact)
    if text_vm and text_35b.startswith(text_vm):
        return "vm_is_prefix"
    if text_35b and text_vm.startswith(text_35b):
        return "35b_is_prefix"
    # Unicode/encoding category
    try:
        text_35b.encode("utf-8"); text_vm.encode("utf-8")
    except UnicodeEncodeError:
        return "unicode_error"
    if any(ord(c) > 0x10000 for c in text_35b + text_vm):
        return "high_codepoint"
    # Punctuation / symbol vs word
    if len(text_35b) <= 2 or len(text_vm) <= 2:
        return "short_token_mismatch"
    return "content_mismatch"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", nargs="+", default=None,
                        help="Glob patterns for result JSON files (default: all 0GM exp files)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples to process")
    parser.add_argument("--show_mismatches", type=int, default=20,
                        help="Print this many mismatch examples (default 20)")
    args = parser.parse_args()

    # ── Load mapping table ────────────────────────────────────────────────────
    print(f"\nLoading mapping table: {MAPPING_NPY}")
    if not MAPPING_NPY.exists():
        print("ERROR: mapping .npy not found. Run scripts/build_token_mapping.py first.")
        sys.exit(1)
    mapping = np.load(str(MAPPING_NPY))  # shape (src_vocab_size,), dtype int32
    print(f"  shape: {mapping.shape}, dtype: {mapping.dtype}")
    if MAPPING_JSON.exists():
        with open(MAPPING_JSON) as f:
            meta = json.load(f)
        print(f"  coverage: {meta.get('coverage_pct', '?')}%  "
              f"strategy: {meta.get('mapping_strategy', '?')}  "
              f"src_vocab: {meta.get('src_vocab_size', '?')}")

    # ── Load tokenizers ───────────────────────────────────────────────────────
    print(f"\nLoading tokenizers...")
    from transformers import AutoTokenizer
    print(f"  35B LLM tokenizer: {LLM_TOK_PATH}")
    llm_tok = AutoTokenizer.from_pretrained(LLM_TOK_PATH, trust_remote_code=True)
    print(f"  VM tokenizer:      {VM_TOK_PATH}")
    vm_tok  = AutoTokenizer.from_pretrained(VM_TOK_PATH,  trust_remote_code=True)
    print(f"  LLM vocab_size={llm_tok.vocab_size}  VM vocab_size={vm_tok.vocab_size}")

    # ── Load result files ─────────────────────────────────────────────────────
    patterns = args.files if args.files else DEFAULT_GLOBS
    print(f"\nLoading result files...")
    samples = load_result_files(patterns)
    if not samples:
        print("ERROR: no result samples found. Check file paths.")
        sys.exit(1)
    if args.limit:
        samples = samples[:args.limit]
    print(f"\n  total samples to process: {len(samples)}")

    # ── Per-token analysis ────────────────────────────────────────────────────
    total_tokens      = 0
    matched_tokens    = 0
    unmapped_tokens   = 0   # mapping value == -1 (should be 0 with 100% coverage)
    mismatch_counter  = Counter()   # mismatch category → count
    mismatch_examples = []          # list of dicts for --show_mismatches

    per_sample_all_match = 0
    per_sample_partial   = 0
    per_sample_no_match  = 0

    for sample in samples:
        text = extract_generated_text(sample)
        if not text:
            continue

        token_ids = llm_tok.encode(text, add_special_tokens=False)
        if not token_ids:
            continue

        sample_match = 0
        sample_total = 0

        for tid in token_ids:
            if tid >= len(mapping):
                # Token ID out of mapping table range (shouldn't happen with 0GM-35B)
                mismatch_counter["out_of_range"] += 1
                total_tokens += 1
                continue

            vm_id = int(mapping[tid])
            total_tokens += 1
            sample_total += 1

            # Decode single token with each tokenizer
            text_35b = llm_tok.decode([tid], skip_special_tokens=False,
                                      clean_up_tokenization_spaces=False)
            if vm_id < 0:
                unmapped_tokens += 1
                cat = "unmapped_neg1"
                mismatch_counter[cat] += 1
                if len(mismatch_examples) < args.show_mismatches:
                    mismatch_examples.append({
                        "35b_id": tid, "vm_id": vm_id,
                        "35b_text": repr(text_35b), "vm_text": "(unmapped)",
                        "category": cat,
                    })
                continue

            text_vm = vm_tok.decode([vm_id], skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)

            if text_35b == text_vm:
                matched_tokens += 1
                sample_match += 1
            else:
                cat = categorize_mismatch(text_35b, text_vm)
                mismatch_counter[cat] += 1
                if len(mismatch_examples) < args.show_mismatches:
                    mismatch_examples.append({
                        "35b_id": tid, "vm_id": vm_id,
                        "35b_text": repr(text_35b), "vm_text": repr(text_vm),
                        "category": cat,
                    })

        # Per-sample classification
        if sample_total == 0:
            continue
        ratio = sample_match / sample_total
        if ratio == 1.0:
            per_sample_all_match += 1
        elif ratio >= 0.5:
            per_sample_partial += 1
        else:
            per_sample_no_match += 1

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TOKEN MAPPING QUALITY REPORT")
    print("=" * 60)

    mismatched_tokens = total_tokens - matched_tokens - unmapped_tokens
    match_rate = matched_tokens / total_tokens * 100 if total_tokens else 0
    mismatch_rate = mismatched_tokens / total_tokens * 100 if total_tokens else 0

    print(f"\n── Per-token statistics ──")
    print(f"  Total tokens analysed : {total_tokens:,}")
    print(f"  Matched (exact)       : {matched_tokens:,}  ({match_rate:.2f}%)")
    print(f"  Mismatched            : {mismatched_tokens:,}  ({mismatch_rate:.2f}%)")
    if unmapped_tokens:
        print(f"  Unmapped (vm_id < 0)  : {unmapped_tokens:,}  ({unmapped_tokens/total_tokens*100:.2f}%)")

    if mismatch_counter:
        print(f"\n── Mismatch breakdown ──")
        for cat, cnt in mismatch_counter.most_common():
            pct = cnt / total_tokens * 100
            print(f"  {cat:<28s}: {cnt:6,}  ({pct:.3f}%)")

    total_samples = per_sample_all_match + per_sample_partial + per_sample_no_match
    if total_samples:
        print(f"\n── Per-sample statistics ──")
        print(f"  Total samples           : {total_samples:,}")
        print(f"  All tokens matched      : {per_sample_all_match:,}  "
              f"({per_sample_all_match/total_samples*100:.1f}%)")
        print(f"  ≥50% tokens matched     : {per_sample_partial:,}  "
              f"({per_sample_partial/total_samples*100:.1f}%)")
        print(f"  <50% tokens matched     : {per_sample_no_match:,}  "
              f"({per_sample_no_match/total_samples*100:.1f}%)")

    if mismatch_examples:
        print(f"\n── First {len(mismatch_examples)} mismatch examples ──")
        for ex in mismatch_examples:
            print(f"  [{ex['category']}]  35b_id={ex['35b_id']}  vm_id={ex['vm_id']}")
            print(f"    35B: {ex['35b_text']}")
            print(f"    VM : {ex['vm_text']}")

    print("\n── SIA relevance summary ──")
    print(f"  In SIA inference, the mapping is applied to topk=10 *candidate* tokens")
    print(f"  (not the full output sequence). Candidates are typically high-frequency")
    print(f"  tokens, so the effective match rate for candidates is likely higher")
    print(f"  than the {match_rate:.1f}% measured here on full output sequences.")
    print(f"  Mismatched candidates receive score=0 (neutral, not wrong).")
    print()


if __name__ == "__main__":
    main()
