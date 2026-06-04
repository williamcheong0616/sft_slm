"""
Rebalance curated_data.jsonl by hard-capping over-represented assistant-turn
opening tokens.

Why hard-cap instead of probabilistic mitigation with synonyms
--------------------------------------------------------------
The original approach used synonym substitution (Strategy C) as one of three
mitigation strategies. However, when synonyms are drawn from a small fixed set
(e.g. three tokens), the synonyms themselves accumulate the same probability mass
that was removed from the original prefixes, creating new attractors of equal
magnitude. Mathematically, synonym-based replacement cannot reduce the total
attractor coverage regardless of mitigation probability.

The corrected approach:
  1. Hard-cap: allow at most MAX_PER_PREFIX instances of each over-threshold
     token to survive in the output.  Excess instances are mitigated using only
     deletion (Strategy A) or clause-end perturbation (Strategy B) — neither of
     which introduces a new sentence-initial attractor.
  2. The corpus is shuffled before the cap is applied so the retained examples
     are a random, representative sample of contexts in which the prefix appeared.
"""

import json
import re
import random
from collections import Counter

INPUT_FILE  = "curated_data.jsonl"
OUTPUT_FILE = "curated_data_rebalanced.jsonl"

TARGET_PREFIXES = ['aah', 'aiyoh', 'eh', 'seriously lah', 'actually']

# Only prefixes with counts exceeding THRESHOLD are capped.
THRESHOLD = 5000

# After rebalancing, each over-threshold prefix may appear at most this many
# times in sentence-initial position across the entire corpus.
# Setting this to ~2 % of the total dataset (≈ 2 000 for a 100 k corpus)
# keeps each opener present but prevents any single token dominating the
# response-initial position distribution.
MAX_PER_PREFIX = 2000


def get_starting_prefix(text: str):
    """Return (normalized_prefix, full_matched_string, punct) or (None,None,None)."""
    pattern = r"^\s*(aah|aiyoh|eh|seriously\s+lah|actually)([\s,\.!]*)(?=\s|\w|$)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        normalized = re.sub(r'\s+', ' ', match.group(1).lower())
        return normalized, match.group(0), match.group(2)
    return None, None, None


def apply_mitigation(text: str, full_match: str, normalized_prefix: str) -> tuple[str, str]:
    """
    Apply Strategy A or B (chosen 50/50).  Strategy C (synonym swap) is
    intentionally excluded: fixed synonyms become new sentence-initial
    attractors and leave total attractor coverage unchanged.
    """
    rest = text[len(full_match):].strip()
    if rest:
        rest = rest[0].upper() + rest[1:]

    if random.random() < 0.5:
        # Strategy A — Delete prefix entirely.
        return rest, "Deleted"
    else:
        # Strategy B — Move prefix to the end of the first clause.
        boundary = re.search(r'[.!?]', rest)
        if boundary:
            idx = boundary.start()
            new_text = rest[:idx] + f", {normalized_prefix}" + rest[idx:]
        else:
            # No sentence boundary: append as a trailing tag.
            new_text = rest + f", {normalized_prefix}."
        return new_text, "Perturbed"


def main():
    random.seed(42)

    # ── Pass 1: count prefix occurrences ─────────────────────────────────────
    print(f"--- Pass 1: Counting prefixes in {INPUT_FILE} ---")
    prefix_counts: Counter = Counter()
    all_records = []

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            all_records.append(data)
            for msg in data.get("conversations", []):
                if msg.get("role") == "assistant":
                    prefix, _, _ = get_starting_prefix(msg.get("content", ""))
                    if prefix:
                        prefix_counts[prefix] += 1

    print(f"Total records: {len(all_records):,}")
    print("Prefix frequencies (assistant-turn-initial):")
    for p, c in prefix_counts.most_common():
        flag = f"  <-- OVER THRESHOLD (cap → {MAX_PER_PREFIX:,})" if c > THRESHOLD else ""
        print(f"  '{p}': {c:,}{flag}")

    targets = {p for p, c in prefix_counts.items() if c > THRESHOLD and p in TARGET_PREFIXES}
    print(f"\nTargets to cap (>{THRESHOLD}): {targets}")
    if not targets:
        print("Nothing to rebalance. Exiting.")
        return

    # ── Shuffle so the retained examples are random, not just the earliest ──
    random.shuffle(all_records)

    # ── Pass 2: apply hard-cap ───────────────────────────────────────────────
    print(f"\n--- Pass 2: Hard-cap at {MAX_PER_PREFIX:,} per prefix → {OUTPUT_FILE} ---")

    kept_counts:      Counter = Counter()   # how many retained per prefix
    mitigated_counts: Counter = Counter()
    strategy_counts:  Counter = Counter()

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        for data in all_records:
            for msg in data.get("conversations", []):
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content", "")
                prefix, full_match, _ = get_starting_prefix(content)
                if prefix not in targets:
                    continue

                if kept_counts[prefix] < MAX_PER_PREFIX:
                    # Still under cap — keep verbatim.
                    kept_counts[prefix] += 1
                else:
                    # Over cap — mitigate without synonyms.
                    new_content, strategy = apply_mitigation(content, full_match, prefix)
                    msg["content"] = new_content
                    mitigated_counts[prefix] += 1
                    strategy_counts[strategy] += 1

            fout.write(json.dumps(data, ensure_ascii=False) + '\n')

    # ── Summary ──────────────────────────────────────────────────────────────
    total = len(all_records)
    print("\n--- Rebalancing complete ---")
    print(f"Output: {OUTPUT_FILE}")
    print(f"\nRetained (kept at sentence-initial position):")
    for p in sorted(targets):
        original = prefix_counts[p]
        retained = kept_counts[p]
        print(f"  '{p}': {original:,} → {retained:,} ({retained/total*100:.1f}% of corpus)")

    total_retained = sum(kept_counts[p] for p in targets)
    total_original = sum(prefix_counts[p] for p in targets)
    print(f"\nTotal attractor turns:  {total_original:,} ({total_original/total*100:.1f}%) → "
          f"{total_retained:,} ({total_retained/total*100:.1f}%)")
    print(f"Reduction: {(total_original - total_retained):,} turns mitigated")

    print(f"\nMitigation by prefix:")
    for p, c in mitigated_counts.most_common():
        print(f"  '{p}': {c:,}")
    print(f"\nMitigation by strategy:")
    for s, c in strategy_counts.most_common():
        print(f"  {s}: {c:,}")


if __name__ == "__main__":
    main()
