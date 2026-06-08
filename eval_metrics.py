#!/usr/bin/env python3
"""
Evaluation pipeline for Bahasa Rojak QLoRA fine-tuned models.

Phase 1 — Generation:
  Each model variant (zero-shot / fine-tuned) runs in an isolated subprocess
  so CUDA memory is fully released between loads.

Phase 2 — Metrics:
  BLEU-4, ROUGE-L, BERTScore (xlm-roberta-large), chrF, CMI

Output files  (all in ./eval_results/):
  test_samples.json        fixed test set  (shared with human_eval_app.py)
  responses_<label>.json   raw responses per model variant
  benchmark_raw.json       all responses + per-sample metrics
  benchmark_results.xlsx   human-readable Excel (3 sheets)
  benchmark_summary.csv    aggregate scores per model variant

Model variants (9 total, 3 per model family):
  <model>_base       vanilla globally pre-trained checkpoint (no SEA-LION)
  <model>_zeroshot   SEA-LION regionally pre-trained, zero-shot
  <model>_finetuned  SEA-LION + QLoRA SFT adapter

Usage:
  python eval_metrics.py                      # full run, all 9 variants
  python eval_metrics.py --model gemma        # gemma base/zero-shot/fine-tuned only
  python eval_metrics.py --skip-generate      # recompute metrics from saved files
  python eval_metrics.py --force              # regenerate even if file exists
  python eval_metrics.py --no-base            # skip vanilla base variants
  python eval_metrics.py --no-zeroshot        # skip SEA-LION zero-shot variants
  python eval_metrics.py --no-finetuned       # skip fine-tuned variants
  python eval_metrics.py --samples 75         # number of test samples (default)

Requirements:
  pip install sacrebleu rouge-score bert-score openpyxl
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime

# ─── Configuration ────────────────────────────────────────────────────────────

DATA_FILE   = "./curated_data.jsonl"
OUTPUT_DIR  = "./eval_results"
SAMPLE_SEED = 42
N_SAMPLES   = 75

ADAPTER_PATHS = {
    "gemma": "./output/gemma-4b/final",
    "llama": "./output/llama-8b/final",
    "qwen":  "./output/qwen-8b/final",
}

BASE_MODEL_IDS = {
    "gemma": "aisingapore/Gemma-SEA-LION-v4-4B-VL",
    "llama": "aisingapore/Llama-SEA-LION-v3.5-8B-R",
    "qwen":  "aisingapore/Qwen-SEA-LION-v4-8B-VL",
}

# Globally pre-trained base checkpoints (no SEA-LION regional pre-training).
# Used for the base_zeroshot condition that isolates the contribution of
# regional pre-training from that of supervised fine-tuning.
VANILLA_MODEL_IDS = {
    "gemma": "google/gemma-3-4b-it",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen":  "Qwen/Qwen2.5-7B-Instruct",
}

_CHILD_FLAG = "--_child-generate"

# ─── CMI (Code-Mixing Index) heuristic lexicon ────────────────────────────────
# Word counts per language label; CMI = (N - dominant_lang_count) / N * 100

_MALAY_TOKENS = {
    "lah", "mah", "kan", "lor", "wey", "wei", "nak", "tak", "tapi", "macam",
    "mana", "ni", "tu", "ke", "je", "la", "eh", "ah", "dah", "pun", "boleh",
    "kat", "dengan", "atau", "yang", "untuk", "dalam", "bila", "kalau", "sebab",
    "bukan", "jadi", "ada", "buat", "kena", "perlu", "fikir", "rasa", "orang",
    "kerja", "benda", "tengah", "memang", "ingat", "cakap", "tau", "pakai",
    "hari", "masa", "mesti", "langsung", "sampai", "selalu", "sekarang",
    "lepas", "sebelum", "balik", "pergi", "ambil", "letak", "bagi", "cuba",
    "diorang", "korang", "kami", "kita", "dia", "saya", "aku", "kau", "awak",
    "dorang", "dekat", "jauh", "besar", "kecil", "cepat", "lambat", "betul",
    "salah", "bagus", "teruk", "takpe", "takpa", "oklah", "gitulah", "nilah",
    "tulah", "jelah", "sekolah", "universiti", "hospital", "kedai", "rumah",
    "jalan", "kereta", "bas", "sini", "sana", "takde", "takkan", "harusnya",
    "patutnya", "alahai", "aisehman", "alamak", "walaupun", "sungguh", "sangat",
    "amat", "terlalu", "agak", "lebih", "kurang", "semua", "sikit", "banyak",
    "sedikit", "ramai", "lagi", "akan", "pernah", "belum", "masih", "terus",
    "jangan", "tolong", "harap", "best", "syok", "penat", "lapar", "kenyang",
    "abang", "kakak", "adik", "ibu", "ayah", "kawan", "sahabat", "pasal",
    "selepas", "sebelum", "mrt", "lrt", "tren", "mana-mana", "betoi",
}

_ENGLISH_TOKENS = {
    "the", "a", "an", "is", "it", "in", "of", "to", "and", "that",
    "this", "for", "with", "not", "but", "have", "be", "are", "was",
    "were", "by", "from", "at", "as", "on", "we", "you", "i", "they",
    "he", "she", "my", "your", "his", "her", "our", "their", "its",
    "will", "would", "could", "should", "can", "may", "might", "shall",
    "do", "does", "did", "has", "had", "been", "being", "get", "got",
    "make", "made", "go", "went", "come", "came", "see", "saw", "know",
    "think", "want", "use", "need", "help", "work", "time", "day",
    "way", "how", "what", "when", "where", "who", "why", "which",
    "more", "also", "just", "even", "still", "here", "there", "then",
    "than", "so", "very", "really", "well", "now", "only", "too",
    "most", "some", "all", "any", "no", "if", "about", "up", "out",
    "like", "first", "new", "good", "great", "right", "big", "high",
    "such", "because", "between", "through", "after", "before", "while",
    "both", "each", "those", "these", "same", "other", "into", "since",
    "without", "always", "never", "often", "usually", "again", "already",
    "actually", "basically", "seriously", "honestly", "apparently",
}


def compute_cmi(text: str) -> float:
    """
    Code-Mixing Index = (N - max_lang_count) / N * 100.
    Uses heuristic lexicon; treat as approximate, consistent across models.
    """
    tokens = re.findall(r"\b[a-zA-Z']+\b", text.lower())
    if not tokens:
        return 0.0
    n_malay   = sum(1 for t in tokens if t in _MALAY_TOKENS)
    n_english = sum(1 for t in tokens if t in _ENGLISH_TOKENS)
    n_total   = len(tokens)
    dominant  = max(n_malay, n_english)
    if dominant == 0:
        return 50.0
    return round((n_total - dominant) / n_total * 100, 2)


# ─── Dataset helpers ──────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sample_test_set(data_file: str, n: int, seed: int, output_dir: str) -> list:
    test_path = os.path.join(output_dir, "test_samples.json")
    if os.path.exists(test_path):
        print(f"  Using existing test set: {test_path}")
        with open(test_path, "r", encoding="utf-8") as f:
            return json.load(f)

    all_records = load_jsonl(data_file)
    valid = [
        r for r in all_records
        if len(r.get("conversations", [])) >= 2
        and r["conversations"][0]["role"] == "user"
        and r["conversations"][1]["role"] == "assistant"
        and r["conversations"][1]["content"].strip()
    ]
    random.seed(seed)
    chosen = random.sample(valid, min(n, len(valid)))
    samples = [
        {
            "id": i,
            "question":  s["conversations"][0]["content"],
            "reference": s["conversations"][1]["content"],
        }
        for i, s in enumerate(chosen)
    ]
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(samples)} test samples → {test_path}")
    return samples


# ─── Child mode: generate responses for ONE model variant ─────────────────────

def child_generate(args):
    import torch

    model_key    = args.child_model_key
    mode         = args.child_mode_str        # "base" | "zeroshot" | "finetuned"
    output_file  = args.child_output_file
    samples_file = args.child_samples_file

    with open(samples_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    adapter_path = ADAPTER_PATHS[model_key]

    from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Choose model ID based on mode ──────────────────────────────────────────
    if mode == "base":
        source_id = VANILLA_MODEL_IDS[model_key]
    else:
        source_id = BASE_MODEL_IDS[model_key]

    print(f"\n{'='*60}")
    print(f"  Model : {model_key.upper()} ({mode})")
    print(f"  Source: {source_id}")
    print(f"  PID   : {os.getpid()}")
    print(f"{'='*60}\n")

    # ── Load model ─────────────────────────────────────────────────────────────
    if mode == "base":
        # Vanilla base checkpoints are all text-only instruction models — use
        # standard AutoModelForCausalLM; no ViT removal needed.
        model     = AutoModelForCausalLM.from_pretrained(
            source_id, quantization_config=bnb, device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(source_id)

    elif model_key == "gemma":
        try:
            from transformers import Gemma3ForConditionalGeneration, AutoProcessor
            model     = Gemma3ForConditionalGeneration.from_pretrained(
                source_id, quantization_config=bnb, device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            tokenizer = AutoProcessor.from_pretrained(source_id).tokenizer
        except (ImportError, Exception):
            model     = AutoModelForCausalLM.from_pretrained(
                source_id, quantization_config=bnb, device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(source_id)

    elif model_key == "llama":
        model     = AutoModelForCausalLM.from_pretrained(
            source_id, quantization_config=bnb, device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(source_id)

    else:  # qwen + SEA-LION
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            model     = Qwen3VLForConditionalGeneration.from_pretrained(
                source_id, quantization_config=bnb, device_map="auto",
            )
            tokenizer = AutoProcessor.from_pretrained(source_id).tokenizer
        except (ImportError, Exception):
            model     = AutoModelForCausalLM.from_pretrained(
                source_id, quantization_config=bnb, device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(source_id)

        # Drop ViT to save VRAM (SEA-LION VL checkpoints only)
        if hasattr(model, "visual"):
            print("  Dropping ViT to save VRAM …")
            del model.visual
            torch.cuda.empty_cache()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load adapter (fine-tuned only) ─────────────────────────────────────────
    if mode == "finetuned":
        from peft import PeftModel
        print(f"  Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    # ── Generate ───────────────────────────────────────────────────────────────
    results = []
    for i, sample in enumerate(samples):
        question = sample["question"]
        print(f"  [{i+1:02d}/{len(samples)}] {question[:70]}…")

        messages = [{"role": "user", "content": question}]
        extra_kw = {}
        # thinking_mode is a SEA-LION-Llama-specific kwarg; skip for vanilla base
        if model_key == "llama" and mode != "base":
            extra_kw["thinking_mode"] = "off"

        try:
            prompt  = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **extra_kw
            )
            inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    do_sample=True,
                )
            response = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
        except Exception as e:
            response = f"ERROR: {e}"

        results.append({"id": sample["id"], "question": question, "response": response})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved {len(results)} responses → {output_file}")
    print("  Child done — CUDA memory released on exit.")


# ─── Metric computation (runs in parent after all children finish) ─────────────

def compute_metrics(hypotheses: list, references: list) -> dict:
    import sacrebleu
    from rouge_score import rouge_scorer as rs_module

    bleu       = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf       = sacrebleu.corpus_chrf(hypotheses, [references])

    scorer     = rs_module.RougeScorer(["rougeL"], use_stemmer=False)
    rougeL_per = [
        scorer.score(ref, hyp)["rougeL"].fmeasure
        for hyp, ref in zip(hypotheses, references)
    ]
    rougeL_avg = sum(rougeL_per) / len(rougeL_per) * 100

    bertscore_avg = None
    try:
        from bert_score import score as bscore
        _, _, F = bscore(
            hypotheses, references,
            model_type="xlm-roberta-large",
            lang="others",
            verbose=False,
            batch_size=8,
        )
        bertscore_avg = F.mean().item() * 100
    except ImportError:
        print("  [WARN] bert-score not installed — BERTScore skipped.")

    cmi_per = [compute_cmi(h) for h in hypotheses]
    cmi_avg = sum(cmi_per) / len(cmi_per)

    return {
        "bleu4":             round(bleu.score,   2),
        "chrf":              round(chrf.score,   2),
        "rougeL":            round(rougeL_avg,   2),
        "bertscore":         round(bertscore_avg, 2) if bertscore_avg is not None else None,
        "cmi":               round(cmi_avg,       2),
        "_per_rougeL":       [round(s * 100, 2) for s in rougeL_per],
        "_per_cmi":          cmi_per,
    }


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def orchestrate(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_keys = [args.model] if args.model else ["gemma", "llama", "qwen"]

    modes = []
    if not args.no_base:
        modes.append("base")
    if not args.no_zeroshot:
        modes.append("zeroshot")
    if not args.no_finetuned:
        modes.append("finetuned")

    if not modes:
        print("Nothing to do: both --no-zeroshot and --no-finetuned are set.")
        sys.exit(1)

    # ── Phase 1: fixed test set ────────────────────────────────────────────────
    print("\n[Phase 1]  Sampling test set …")
    samples      = sample_test_set(DATA_FILE, args.samples, SAMPLE_SEED, OUTPUT_DIR)
    samples_file = os.path.join(OUTPUT_DIR, "test_samples.json")
    references   = [s["reference"] for s in samples]
    ids          = [s["id"]        for s in samples]
    print(f"  {len(samples)} test samples ready.")

    # ── Phase 2: generate responses ────────────────────────────────────────────
    variants       = [(mk, mode) for mk in model_keys for mode in modes]
    response_files = {}

    if not args.skip_generate:
        print(f"\n[Phase 2]  Generating responses ({len(variants)} model variant(s)) …")
        for model_key, mode in variants:
            label    = f"{model_key}_{mode}"
            out_file = os.path.join(OUTPUT_DIR, f"responses_{label}.json")

            if mode == "finetuned" and not os.path.exists(ADAPTER_PATHS[model_key]):
                print(f"  SKIP {label}  — adapter not found at {ADAPTER_PATHS[model_key]}")
                continue

            if os.path.exists(out_file) and not args.force:
                print(f"  SKIP {label}  — already exists  (use --force to regenerate)")
                response_files[label] = out_file
                continue

            print(f"\n  Spawning child for: {label}")
            proc = subprocess.run([
                sys.executable, __file__,
                _CHILD_FLAG,
                "--child-model-key",    model_key,
                "--child-mode-str",     mode,
                "--child-output-file",  out_file,
                "--child-samples-file", samples_file,
            ])
            if proc.returncode == 0:
                print(f"  ✅  {label}")
                response_files[label] = out_file
            else:
                print(f"  ❌  {label}  (exit {proc.returncode})")
    else:
        print("\n[Phase 2]  Skipped (--skip-generate). Loading existing files …")
        for model_key, mode in variants:
            label    = f"{model_key}_{mode}"
            out_file = os.path.join(OUTPUT_DIR, f"responses_{label}.json")
            if os.path.exists(out_file):
                response_files[label] = out_file
                print(f"  Found: {out_file}")
            else:
                print(f"  Missing: {out_file}")

    if not response_files:
        print("\nNo response files available. Run without --skip-generate first.")
        sys.exit(1)

    # ── Phase 3: load all responses ────────────────────────────────────────────
    print("\n[Phase 3]  Loading responses …")
    all_responses = {}
    for label, fpath in response_files.items():
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_responses[label] = {item["id"]: item["response"] for item in data}
        print(f"  {label}: {len(data)} responses")

    # per-sample lookup
    per_sample = {s["id"]: {"id": s["id"], "question": s["question"], "reference": s["reference"]}
                  for s in samples}

    # ── Phase 4: compute metrics ───────────────────────────────────────────────
    print("\n[Phase 4]  Computing metrics …")
    summary_rows = []

    for label in sorted(all_responses.keys()):
        resp_map = all_responses[label]
        hyps     = [resp_map.get(sid, "") for sid in ids]

        print(f"  {label} …")
        m = compute_metrics(hyps, references)

        for i, sid in enumerate(ids):
            per_sample[sid][f"{label}_response"] = hyps[i]
            per_sample[sid][f"{label}_rougeL"]   = m["_per_rougeL"][i]
            per_sample[sid][f"{label}_cmi"]       = m["_per_cmi"][i]

        summary_rows.append({
            "model_variant": label,
            "bleu4":         m["bleu4"],
            "chrf":          m["chrf"],
            "rougeL":        m["rougeL"],
            "bertscore":     m["bertscore"],
            "cmi":           m["cmi"],
        })

    # ── Phase 5: save outputs ──────────────────────────────────────────────────
    print("\n[Phase 5]  Saving outputs …")

    raw_path = os.path.join(OUTPUT_DIR, "benchmark_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "generated":      datetime.now().isoformat(),
                "n_samples":      len(samples),
                "seed":           SAMPLE_SEED,
                "model_variants": list(all_responses.keys()),
            },
            "samples": list(per_sample.values()),
            "summary": summary_rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {raw_path}")

    try:
        import pandas as pd

        # Sheet 1 — responses side by side (easy for human inspection)
        resp_cols = {
            "ID":        ids,
            "Question":  [per_sample[sid]["question"]  for sid in ids],
            "Reference": [per_sample[sid]["reference"] for sid in ids],
        }
        for label in sorted(all_responses.keys()):
            resp_map = all_responses[label]
            resp_cols[label] = [resp_map.get(sid, "") for sid in ids]
        df_resp = pd.DataFrame(resp_cols)

        # Sheet 2 — aggregate metrics
        df_sum = pd.DataFrame(summary_rows)

        # Sheet 3 — per-sample CMI (code-mixing quality)
        cmi_cols = {"ID": ids, "Question": [per_sample[sid]["question"] for sid in ids]}
        for label in sorted(all_responses.keys()):
            cmi_cols[f"{label}_cmi"] = [per_sample[sid].get(f"{label}_cmi", "") for sid in ids]
        df_cmi = pd.DataFrame(cmi_cols)

        xlsx_path = os.path.join(OUTPUT_DIR, "benchmark_results.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df_resp.to_excel(writer, sheet_name="Responses", index=False)
            df_sum.to_excel(writer,  sheet_name="Summary Metrics", index=False)
            df_cmi.to_excel(writer,  sheet_name="CMI per Sample", index=False)
        print(f"  Saved: {xlsx_path}")

        csv_path = os.path.join(OUTPUT_DIR, "benchmark_summary.csv")
        df_sum.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    except ImportError as e:
        print(f"  [WARN] {e} — Excel output skipped.")

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  Evaluation Summary  ({len(samples)} samples, seed={SAMPLE_SEED})")
    print(f"{'='*72}")
    print(f"  {'Variant':<25}  {'BLEU-4':>7}  {'chrF':>7}  {'ROUGE-L':>8}  {'BERTScore':>10}  {'CMI':>6}")
    print("  " + "-" * 70)
    for row in summary_rows:
        bs = f"{row['bertscore']:.2f}" if row["bertscore"] is not None else "   N/A"
        print(
            f"  {row['model_variant']:<25}"
            f"  {row['bleu4']:>7.2f}"
            f"  {row['chrf']:>7.2f}"
            f"  {row['rougeL']:>8.2f}"
            f"  {bs:>10}"
            f"  {row['cmi']:>6.2f}"
        )
    print(f"{'='*72}")
    print(f"\nAll outputs → {os.path.abspath(OUTPUT_DIR)}/")


# ─── Argument parsing ─────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Evaluation pipeline for Bahasa Rojak QLoRA models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",          type=str, choices=["gemma", "llama", "qwen"],
                   help="Run only this model (default: all three)")
    p.add_argument("--samples",        type=int, default=N_SAMPLES,
                   help="Number of test samples to draw")
    p.add_argument("--skip-generate",  action="store_true",
                   help="Skip generation — recompute metrics from saved response files")
    p.add_argument("--force",          action="store_true",
                   help="Re-generate responses even if file already exists")
    p.add_argument("--no-base",        action="store_true",
                   help="Skip vanilla base model zero-shot variants (google/gemma-3-4b-it etc.)")
    p.add_argument("--no-zeroshot",    action="store_true",
                   help="Skip SEA-LION zero-shot variants")
    p.add_argument("--no-finetuned",   action="store_true",
                   help="Skip fine-tuned adapter variants")

    # Internal child flags — not for direct use
    p.add_argument(_CHILD_FLAG,             dest="child_mode",        action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--child-model-key",     dest="child_model_key",   type=str,            help=argparse.SUPPRESS)
    p.add_argument("--child-mode-str",      dest="child_mode_str",    type=str,            help=argparse.SUPPRESS)
    p.add_argument("--child-output-file",   dest="child_output_file", type=str,            help=argparse.SUPPRESS)
    p.add_argument("--child-samples-file",  dest="child_samples_file",type=str,            help=argparse.SUPPRESS)
    return p


def main():
    args = build_parser().parse_args()
    if args.child_mode:
        child_generate(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
