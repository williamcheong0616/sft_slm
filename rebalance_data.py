import json
import re
import random
from collections import Counter

INPUT_FILE = "curated_data.jsonl"
OUTPUT_FILE = "curated_data_rebalanced.jsonl"

TARGET_PREFIXES = ['aah', 'aiyoh', 'eh', 'seriously lah', 'actually']
THRESHOLD = 5000
MITIGATION_PROBABILITY = 0.70

SYNONYMS = ['Alahai', 'Aduh', 'Memang']

def get_starting_prefix(text):
    """
    Checks if the text starts with one of the target prefixes.
    Returns (normalized_prefix, full_matched_string) or (None, None).
    """
    # Regex breakdown:
    # ^\s* = optional leading whitespace
    # (aah|aiyoh|eh|seriously\s+lah|actually) = target words
    # ([\s,\.!]*?) = optional trailing punctuation/spaces (non-greedy)
    # (?=\s|\w|$) = lookahead to ensure we matched a full word/phrase
    
    pattern = r"^\s*(aah|aiyoh|eh|seriously\s+lah|actually)([\s,\.!]*)(?=\s|\w|$)"
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        normalized_prefix = re.sub(r'\s+', ' ', match.group(1).lower())
        full_match = match.group(0)
        punct = match.group(2)
        return normalized_prefix, full_match, punct
    return None, None, None

def apply_mitigation(text, normalized_prefix, full_match, punct):
    rest_of_text = text[len(full_match):].strip()
    
    # Capitalize the first letter of the remaining text
    if rest_of_text:
        rest_of_text = rest_of_text[0].upper() + rest_of_text[1:]
        
    strategy = random.choice(['A', 'B', 'C'])
    
    if strategy == 'A':
        # Strategy A: Delete Prefix
        return rest_of_text, "Deleted"
        
    elif strategy == 'B':
        # Strategy B: Perturb (move to end of first sentence)
        prefix_clean = normalized_prefix.capitalize()
        # Find first sentence boundary
        match_boundary = re.search(r'[\.\!\?]', rest_of_text)
        
        if match_boundary:
            idx = match_boundary.start()
            new_text = rest_of_text[:idx] + f", {prefix_clean.lower()}" + rest_of_text[idx:]
        else:
            new_text = rest_of_text + f", {prefix_clean.lower()}."
        return new_text, "Perturbed"
        
    elif strategy == 'C':
        # Strategy C: Synonym Swap
        synonym = random.choice(SYNONYMS)
        new_punct = punct.strip()
        if not new_punct:
            new_punct = ", "
        elif not new_punct.endswith(" "):
            new_punct += " "
            
        new_text = synonym + new_punct + rest_of_text
        return new_text, "Swapped"

def main():
    random.seed(42) # For reproducibility
    
    print(f"--- Pass 1: Counting Prefixes in {INPUT_FILE} ---")
    prefix_counts = Counter()
    total_rows = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            total_rows += 1
            data = json.loads(line)
            
            for msg in data.get("conversations", []):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    prefix, _, _ = get_starting_prefix(content)
                    if prefix:
                        prefix_counts[prefix] += 1
                        
    print(f"Total rows scanned: {total_rows}")
    print("Prefix frequencies:")
    for prefix, count in prefix_counts.most_common():
        print(f"  - '{prefix}': {count}")
        
    targets_over_threshold = {p for p, c in prefix_counts.items() if c > THRESHOLD and p in TARGET_PREFIXES}
    print(f"\nTargets exceeding threshold ({THRESHOLD}): {targets_over_threshold}")
    
    print(f"\n--- Pass 2: Rebalancing and Generating {OUTPUT_FILE} ---")
    
    stats_mitigated = Counter()
    stats_strategies = Counter()
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
             
        for line in fin:
            if not line.strip(): continue
            data = json.loads(line)
            
            for msg in data.get("conversations", []):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    prefix, full_match, punct = get_starting_prefix(content)
                    
                    if prefix in targets_over_threshold:
                        # 70% probability to mitigate
                        if random.random() < MITIGATION_PROBABILITY:
                            new_content, strategy_used = apply_mitigation(content, prefix, full_match, punct)
                            msg["content"] = new_content
                            stats_mitigated[prefix] += 1
                            stats_strategies[strategy_used] += 1
                            
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')
            
    print("\n--- Rebalancing Complete ---")
    print(f"Output saved to: {OUTPUT_FILE}")
    print("\nMitigation Stats by Prefix:")
    for prefix, count in stats_mitigated.most_common():
        print(f"  - '{prefix}': mitigated {count} times")
        
    print("\nMitigation Stats by Strategy:")
    for strat, count in stats_strategies.most_common():
        print(f"  - {strat}: {count} times")

if __name__ == "__main__":
    main()
