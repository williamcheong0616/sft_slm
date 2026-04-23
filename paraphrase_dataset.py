#!/usr/bin/env python3
"""
Paraphrase curated_data.jsonl using Ollama (gemma3:4b model).
Preserves Bahasa Rojak (mixed Malay-English) style while using different words.
Batch processes to avoid overloading Ollama.
"""

import json
import time
import argparse
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: 'requests' library not found. Install it with: pip install requests")
    sys.exit(1)

# ─────────────────────────── CONFIG ───────────────────────────
OLLAMA_URL      = "http://localhost:11434/api/generate"
MODEL           = "gemma3:4b"          # adjust if your server tag differs
BATCH_SIZE      = 10                   # conversations processed per batch
DELAY_BETWEEN   = 1.0                  # seconds to sleep between API calls
TIMEOUT         = 120                  # seconds per request
INPUT_FILE      = "curated_data.jsonl"
OUTPUT_FILE     = "paraphrased_data.jsonl"
PROGRESS_FILE   = "paraphrase_progress.json"  # tracks last completed id
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
Kau adalah seorang penulis yang mahir dalam Bahasa Rojak Malaysia — campuran \
organik antara Bahasa Melayu, English, dan kadang-kadang Mandarin atau Tamil. \
Tugas kau: paraphrase teks yang diberi. Gantikan perkataan-perkataan dengan \
sinonim atau frasa lain yang serupa maknanya, TAPI kekalkan:
1. Nada & gaya Bahasa Rojak yang sama (contoh: kalau asal ada "lah", "kan", \
"meh", "wei", "bro", "gila", kekalkan slang tersebut)
2. Kode-suis (code-switching) antara Malay ↔ English pada kadar yang sama
3. Makna/mesej asal
4. Panjang ayat yang lebih kurang sama
Jangan tambah, jangan buang idea penting. Output HANYA teks yang sudah \
diparaphrase, tanpa penjelasan atau label tambahan.\
"""


def load_progress() -> set:
    """Load set of already-processed conversation IDs."""
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE, "r") as f:
            data = json.load(f)
        return set(data.get("done_ids", []))
    return set()


def save_progress(done_ids: set) -> None:
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"done_ids": sorted(done_ids)}, f)


def load_existing_output(output_path: str) -> dict:
    """Return dict of {id: record} already written to output file."""
    existing = {}
    if Path(output_path).exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        existing[rec["id"]] = rec
                    except json.JSONDecodeError:
                        pass
    return existing


def call_ollama(prompt: str, model: str, ollama_url: str) -> str | None:
    """Call Ollama generate endpoint. Returns model text or None on failure."""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 1024,
        }
    }
    try:
        resp = requests.post(ollama_url, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.Timeout:
        print("    [WARN] Request timed out.")
        return None
    except requests.exceptions.ConnectionError:
        print("    [ERROR] Cannot connect to Ollama. Is it running?")
        return None
    except Exception as e:
        print(f"    [ERROR] {e}")
        return None


def paraphrase_turn(role: str, content: str, model: str, ollama_url: str) -> str:
    """Paraphrase a single conversation turn, keeping the role label."""
    label = "User" if role == "user" else "Assistant"
    prompt = f"[{label}]: {content}\n\nParaphrase teks [{label}] di atas:"
    result = call_ollama(prompt, model, ollama_url)
    return result if result else content   # fall back to original on failure


def process_record(record: dict, model: str, ollama_url: str, delay: float) -> dict:
    """Paraphrase all turns in one JSONL record."""
    new_convos = []
    for turn in record.get("conversations", []):
        role    = turn["role"]
        content = turn["content"]
        print(f"      → paraphrasing [{role}] …", end=" ", flush=True)
        new_content = paraphrase_turn(role, content, model, ollama_url)
        print("done")
        new_convos.append({"role": role, "content": new_content})
        time.sleep(delay)                  # gentle pacing between turns

    return {"id": record["id"], "conversations": new_convos}


def main():
    parser = argparse.ArgumentParser(description="Paraphrase curated_data.jsonl with Ollama")
    parser.add_argument("--input",       default=INPUT_FILE,    help="Input JSONL file")
    parser.add_argument("--output",      default=OUTPUT_FILE,   help="Output JSONL file")
    parser.add_argument("--model",       default=MODEL,         help="Ollama model tag")
    parser.add_argument("--batch-size",  type=int, default=BATCH_SIZE,
                        help="Number of records per batch before progress save")
    parser.add_argument("--delay",       type=float, default=DELAY_BETWEEN,
                        help="Seconds between API calls")
    parser.add_argument("--start-id",   type=int, default=None,
                        help="Skip records with id < this value (override auto-resume)")
    parser.add_argument("--limit",       type=int, default=None,
                        help="Only process this many records (for testing)")
    parser.add_argument("--ollama-url",  default=OLLAMA_URL,   help="Ollama API base URL")
    args = parser.parse_args()

    # Resolve config from parsed args
    model      = args.model
    delay      = args.delay
    ollama_url = args.ollama_url

    # ── Load state ──────────────────────────────────────────
    done_ids = load_progress()
    existing = load_existing_output(args.output)
    done_ids.update(existing.keys())   # sync both sources

    print(f"=== Paraphrase Dataset ===")
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")
    print(f"  Model  : {model}")
    print(f"  Already done: {len(done_ids)} records")

    # ── Read input ──────────────────────────────────────────
    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Filter out already-processed
    if args.start_id is not None:
        records = [r for r in records if r["id"] >= args.start_id]
    else:
        records = [r for r in records if r["id"] not in done_ids]

    if args.limit:
        records = records[:args.limit]

    print(f"  To process: {len(records)} records\n")

    if not records:
        print("Nothing left to do. Exiting.")
        return

    # ── Open output in append mode ──────────────────────────
    out_f = open(args.output, "a", encoding="utf-8")

    try:
        batch_count = 0
        for i, record in enumerate(records, 1):
            rec_id = record["id"]
            print(f"[{i}/{len(records)}] Processing id={rec_id} …")

            try:
                new_record = process_record(record, model, ollama_url, delay)
                out_f.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                out_f.flush()
                done_ids.add(rec_id)
                batch_count += 1
            except KeyboardInterrupt:
                print("\n[!] Interrupted by user. Saving progress …")
                save_progress(done_ids)
                print(f"    Progress saved to {PROGRESS_FILE}. Re-run to resume.")
                sys.exit(0)
            except Exception as e:
                print(f"    [ERROR] Failed on id={rec_id}: {e} — skipping.")

            # Save progress every BATCH_SIZE records
            if batch_count % args.batch_size == 0:
                save_progress(done_ids)
                print(f"  [✓] Batch checkpoint saved ({batch_count} done this run)\n")
                time.sleep(2)  # brief pause between batches

    finally:
        out_f.close()
        save_progress(done_ids)
        print(f"\n=== Done! ===")
        print(f"  Total processed this run : {batch_count}")
        print(f"  Output file              : {args.output}")
        print(f"  Progress file            : {PROGRESS_FILE}")


if __name__ == "__main__":
    main()
