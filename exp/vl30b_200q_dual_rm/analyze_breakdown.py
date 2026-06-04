"""Detect where 'breakdown' patterns first appear in each Run 1 output.

Heuristics for breakdown (any of these triggers a flag at the position where it first occurs):
1. Foreign-language tail (CJK > 80% in any 200-char window) when the question is in English
2. Run-on word chain: 8+ consecutive single-line words with no punctuation
3. Heading repetition: 'Final Answer:' / 'Corrected version' / 'Note:' / 'Summary:' appearing 3+ times
4. Apologetic-loop: 'I apologize' / 'Let me try again' / 'Wait,' (3+ instances)
5. Same-prefix repetition: same 30-char prefix appearing 3+ times consecutively
"""
import json
import re
import statistics

DATA = "/tmp/vl30b_runs/vllm_sia_outputs.json"
records = json.load(open(DATA))

CJK_RE = re.compile(r'[一-鿿぀-ヿ가-힯]')
APOL_RE = re.compile(r'(?:I apologize|Let me try again|Wait,|Let me restart|Actually, let me|Hmm,?\s+actually|Let me reconsider)', re.IGNORECASE)
HEADING_RE = re.compile(r'(?:^|\n)\s*(?:Final Answer|Corrected version|Note:|Summary:|Revised version|Final response|Updated version|Re-summarized)', re.IGNORECASE | re.MULTILINE)


def is_mostly_english(s):
    """Detect question/early answer language."""
    sample = s[:200]
    cjk = len(CJK_RE.findall(sample))
    ascii_letters = sum(1 for c in sample if c.isascii() and c.isalpha())
    return ascii_letters > 50 and cjk < 10


def detect_breakdown_position(text, question_is_english):
    """Returns (position_chars, position_pct, reason) for first breakdown, or (None, None, None)."""
    if not text:
        return None, None, None
    n = len(text)

    # 1. Sudden CJK injection in a previously English answer (200-char windows)
    if question_is_english:
        for i in range(200, n - 200, 100):
            window = text[i:i+200]
            cjk = len(CJK_RE.findall(window))
            if cjk > 50:  # 25% of 200 chars are CJK
                return i, i / n, "foreign-language injection (CJK)"

    # 2. Run-on word chain: 25+ consecutive English-word tokens with no comma/period/semicolon/colon.
    # Pathological "pretentious word soup" without sentence structure.
    chain_re = re.compile(r'(?:[A-Za-z]{3,}\s){25,}[A-Za-z]{3,}')
    for m in chain_re.finditer(text):
        return m.start(), m.start() / n, "run-on word chain"

    # 3. Heading repetition (3+ of these markers)
    headings = list(HEADING_RE.finditer(text))
    if len(headings) >= 3:
        return headings[2].start(), headings[2].start() / n, f"heading repetition ({len(headings)} markers)"

    # 4. Apology / restart loop
    apols = list(APOL_RE.finditer(text))
    if len(apols) >= 3:
        return apols[2].start(), apols[2].start() / n, f"apologetic loop ({len(apols)} retries)"

    # 5. Same prefix consecutive
    lines = text.split('\n')
    for i in range(len(lines) - 2):
        a = lines[i].strip()[:40]
        if len(a) > 15 and a == lines[i+1].strip()[:40] == lines[i+2].strip()[:40]:
            pos = sum(len(l) + 1 for l in lines[:i])
            return pos, pos / n, "consecutive line repetition"

    return None, None, None


# Classify each record
results = []
for r in records:
    out = r.get('output') or ''
    instr = r.get('instruction', '')
    q_eng = is_mostly_english(instr)
    pos, pct, reason = detect_breakdown_position(out, q_eng)
    results.append({
        "id": r["id"],
        "tokens": r.get('usage', {}).get('completion_tokens', 0),
        "finish": r.get('finish_reason'),
        "len_chars": len(out),
        "breakdown_pos_chars": pos,
        "breakdown_pct": pct,
        "reason": reason,
    })

# Summary
n_total = len(results)
n_breakdown = sum(1 for r in results if r['breakdown_pos_chars'] is not None)
print(f"=== VL-30B SIA Run 1 (vllm RM) 200Q breakdown analysis ===")
print(f"Total records: {n_total}")
print(f"Records with detected breakdown: {n_breakdown} ({n_breakdown/n_total*100:.1f}%)")
print(f"Records clean (no breakdown detected): {n_total - n_breakdown} ({(n_total-n_breakdown)/n_total*100:.1f}%)")
print()

# Position distribution
pcts = [r['breakdown_pct'] for r in results if r['breakdown_pct'] is not None]
if pcts:
    print(f"--- Breakdown position (as % of output length) ---")
    pcts_sorted = sorted(pcts)
    print(f"  n = {len(pcts)}")
    print(f"  min:    {min(pcts)*100:.1f}%")
    print(f"  p25:    {pcts_sorted[len(pcts)//4]*100:.1f}%")
    print(f"  median: {statistics.median(pcts)*100:.1f}%")
    print(f"  mean:   {statistics.mean(pcts)*100:.1f}%")
    print(f"  p75:    {pcts_sorted[3*len(pcts)//4]*100:.1f}%")
    print(f"  max:    {max(pcts)*100:.1f}%")
    print()

# Buckets of % position
print(f"--- Position bucket distribution (where breakdown first appears) ---")
buckets = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
for lo, hi in buckets:
    n = sum(1 for p in pcts if lo <= p < hi)
    bar = '█' * (n * 50 // max(1, len(pcts)))
    print(f"  [{lo*100:>3.0f}%-{hi*100:>3.0f}%]: {n:3d}  {bar}")
print()

# Token positions
tok_buckets = [(0, 256), (256, 512), (512, 1024), (1024, 1536), (1536, 2048)]
breakdown_tok_positions = []
for r in results:
    if r['breakdown_pos_chars'] is not None and r['len_chars'] > 0:
        # Approximate: tokens ∝ chars / 4 (rough English avg)
        approx_tok = int(r['breakdown_pos_chars'] * r['tokens'] / r['len_chars'])
        breakdown_tok_positions.append(approx_tok)
if breakdown_tok_positions:
    print(f"--- Approximate breakdown position (tokens, rough) ---")
    for lo, hi in tok_buckets:
        n = sum(1 for t in breakdown_tok_positions if lo <= t < hi)
        bar = '█' * (n * 50 // max(1, len(breakdown_tok_positions)))
        print(f"  [{lo:>4d}-{hi:>4d}]: {n:3d}  {bar}")
    print()

# Reasons
print("--- Breakdown reasons ---")
from collections import Counter
reasons = Counter(r['reason'] for r in results if r['reason'])
for reason, count in reasons.most_common():
    print(f"  {count:3d}  {reason}")
print()

# Breakdown by finish_reason (does max_tokens=2048 hit correlate with breakdown?)
print("--- Breakdown rate by finish_reason ---")
for fr in ['stop', 'length']:
    subset = [r for r in results if r['finish'] == fr]
    if subset:
        n_bd = sum(1 for r in subset if r['breakdown_pos_chars'] is not None)
        print(f"  finish={fr}: {n_bd}/{len(subset)} ({n_bd/len(subset)*100:.1f}%) breakdown rate")
print()

# Some example records
print("--- 5 random examples WITH breakdown ---")
import itertools
examples = [r for r in results if r['breakdown_pos_chars'] is not None][:5]
for ex in examples:
    print(f"  id={ex['id']:3d} tokens={ex['tokens']:4d} pct={ex['breakdown_pct']*100:5.1f}% reason={ex['reason']}")

print()
print("--- 5 examples WITHOUT breakdown (clean outputs) ---")
clean = [r for r in results if r['breakdown_pos_chars'] is None][:5]
for ex in clean:
    print(f"  id={ex['id']:3d} tokens={ex['tokens']:4d} finish={ex['finish']}")

# Save full table
json.dump(results, open('/tmp/vl30b_runs/run1_breakdown_analysis.json', 'w'), ensure_ascii=False, indent=2)
print(f"\nFull per-record analysis saved: /tmp/vl30b_runs/run1_breakdown_analysis.json")
