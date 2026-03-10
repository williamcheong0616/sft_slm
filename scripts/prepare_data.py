#!/usr/bin/env python3
"""
Prepare curated_data.jsonl for SFT training.

Input format (curated_data.jsonl):
    {"id": 1, "conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Output format (train.jsonl / eval.jsonl):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --input curated_data.jsonl --output-dir data --eval-ratio 0.05
    python scripts/prepare_data.py --input curated_data.jsonl --eval-ratio 0.1 --seed 42 --min-turns 2
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter


def validate_example(example: dict, idx: int) -> tuple[bool, str]:
    """Validate a single example. Returns (is_valid, reason)."""

    # Must have conversations
    if "conversations" not in example:
        return False, "missing 'conversations' key"

    convos = example["conversations"]

    if not isinstance(convos, list):
        return False, "'conversations' is not a list"

    if len(convos) < 2:
        return False, f"too few messages ({len(convos)}, need ≥ 2)"

    # Check message structure
    for i, msg in enumerate(convos):
        if not isinstance(msg, dict):
            return False, f"message {i} is not a dict"
        if "role" not in msg or "content" not in msg:
            return False, f"message {i} missing 'role' or 'content'"
        if msg["role"] not in ("system", "user", "assistant"):
            return False, f"message {i} has invalid role: {msg['role']}"

    # Must have at least one user and one assistant message
    roles = [m["role"] for m in convos]
    if "user" not in roles:
        return False, "no user message"
    if "assistant" not in roles:
        return False, "no assistant message"

    # Check for empty content
    for i, msg in enumerate(convos):
        content = msg["content"]
        if isinstance(content, str) and not content.strip():
            return False, f"message {i} has empty content"

    return True, "ok"


def convert_example(example: dict) -> dict:
    """Convert from curated format to SFT messages format."""
    messages = []
    for msg in example["conversations"]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"].strip() if isinstance(msg["content"], str) else msg["content"],
        })
    return {"messages": messages}


def compute_stats(examples: list[dict]) -> dict:
    """Compute dataset statistics."""
    num_turns = [len(ex["conversations"]) for ex in examples]
    user_lengths = []
    assistant_lengths = []

    for ex in examples:
        for msg in ex["conversations"]:
            content = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
            word_count = len(content.split())
            if msg["role"] == "user":
                user_lengths.append(word_count)
            elif msg["role"] == "assistant":
                assistant_lengths.append(word_count)

    return {
        "total_examples": len(examples),
        "avg_turns": sum(num_turns) / len(num_turns) if num_turns else 0,
        "min_turns": min(num_turns) if num_turns else 0,
        "max_turns": max(num_turns) if num_turns else 0,
        "avg_user_words": sum(user_lengths) / len(user_lengths) if user_lengths else 0,
        "avg_assistant_words": sum(assistant_lengths) / len(assistant_lengths) if assistant_lengths else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare curated data for SFT")
    parser.add_argument("--input", "-i", default="curated_data.jsonl",
                        help="Input JSONL file (default: curated_data.jsonl)")
    parser.add_argument("--output-dir", "-o", default="data",
                        help="Output directory (default: data)")
    parser.add_argument("--eval-ratio", type=float, default=0.05,
                        help="Fraction held out for eval (default: 0.05 = 5%%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--min-turns", type=int, default=2,
                        help="Minimum number of messages per example (default: 2)")
    parser.add_argument("--max-assistant-words", type=int, default=None,
                        help="Drop examples where assistant response > N words")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print stats, don't write files")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return

    # -------------------------------------------------------------------------
    # 1. Load and validate
    # -------------------------------------------------------------------------
    print(f"📂 Loading: {input_path}")

    raw_examples = []
    parse_errors = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_examples.append(json.loads(line))
            except json.JSONDecodeError:
                parse_errors += 1
                if parse_errors <= 5:
                    print(f"   ⚠️  Line {line_num}: JSON parse error")

    print(f"   Loaded:       {len(raw_examples):,} examples")
    if parse_errors:
        print(f"   Parse errors: {parse_errors:,}")

    # Validate
    valid_examples = []
    invalid_reasons = Counter()
    for i, ex in enumerate(raw_examples):
        is_valid, reason = validate_example(ex, i)
        if is_valid:
            # Apply filters
            num_msgs = len(ex["conversations"])
            if num_msgs < args.min_turns:
                invalid_reasons[f"< {args.min_turns} turns"] += 1
                continue

            if args.max_assistant_words:
                too_long = False
                for msg in ex["conversations"]:
                    if msg["role"] == "assistant":
                        content = msg["content"] if isinstance(msg["content"], str) else ""
                        if len(content.split()) > args.max_assistant_words:
                            too_long = True
                            break
                if too_long:
                    invalid_reasons["assistant too long"] += 1
                    continue

            valid_examples.append(ex)
        else:
            invalid_reasons[reason] += 1

    print(f"   Valid:        {len(valid_examples):,}")
    if invalid_reasons:
        print(f"   Dropped:      {sum(invalid_reasons.values()):,}")
        for reason, count in invalid_reasons.most_common(10):
            print(f"     - {reason}: {count:,}")

    if not valid_examples:
        print("❌ No valid examples found!")
        return

    # -------------------------------------------------------------------------
    # 2. Compute stats
    # -------------------------------------------------------------------------
    stats = compute_stats(valid_examples)
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total examples:      {stats['total_examples']:,}")
    print(f"   Avg turns/example:   {stats['avg_turns']:.1f}")
    print(f"   Turn range:          {stats['min_turns']}-{stats['max_turns']}")
    print(f"   Avg user words:      {stats['avg_user_words']:.0f}")
    print(f"   Avg assistant words: {stats['avg_assistant_words']:.0f}")

    if args.dry_run:
        print("\n🔍 Dry run — no files written.")
        return

    # -------------------------------------------------------------------------
    # 3. Shuffle and split
    # -------------------------------------------------------------------------
    random.seed(args.seed)
    random.shuffle(valid_examples)

    eval_size = max(1, int(len(valid_examples) * args.eval_ratio))
    train_size = len(valid_examples) - eval_size

    eval_examples = valid_examples[:eval_size]
    train_examples = valid_examples[eval_size:]

    print(f"\n✂️  Split (seed={args.seed}, eval_ratio={args.eval_ratio}):")
    print(f"   Train: {len(train_examples):,}")
    print(f"   Eval:  {len(eval_examples):,}")

    # -------------------------------------------------------------------------
    # 4. Convert and write
    # -------------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_examples:
            converted = convert_example(ex)
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for ex in eval_examples:
            converted = convert_example(ex)
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")

    print(f"\n✅ Done!")
    print(f"   Train: {train_path} ({train_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"   Eval:  {eval_path} ({eval_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Show a sample
    print(f"\n📋 Sample output (first train example):")
    sample = convert_example(train_examples[0])
    for msg in sample["messages"]:
        role = msg["role"].upper()
        content = msg["content"][:100] + ("..." if len(msg["content"]) > 100 else "")
        print(f"   [{role}] {content}")


if __name__ == "__main__":
    main()
