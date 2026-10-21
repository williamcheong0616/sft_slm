#!/usr/bin/env python3
"""
Evaluation pipeline for Bahasa Rojak QLoRA fine-tuned models.

Phase 1 — Test set:
  Samples a fixed test set from the HELD-OUT split (data/eval.jsonl, written by
  scripts/prepare_data.py with seed 42) and checks it for overlap with the
  training file (data/train.jsonl).

Phase 2 — Generation:
  Each model variant (zero-shot / fine-tuned) runs in an isolated subprocess
  so CUDA memory is fully released between loads.

Phase 3 — Metrics:
  BLEU-4, chrF, ROUGE-L, BERTScore (xlm-roberta-large), CMI (Gambäck–Das
  standard + the original heuristic), each with a bootstrap 95% CI over items,
  plus a paired-bootstrap p-value for finetuned vs zeroshot per model family.

Output files  (all in ./eval_results/):
  test_samples.json          fixed test set + the source file path / sha256 it came from
  overlap_report.json        exact / normalised test–train overlap
  responses_<label>.json     raw responses per model variant (+ generation metadata)
  benchmark_raw.json         all responses + per-sample metrics + run metadata
  benchmark_results.xlsx     human-readable Excel (4 sheets)
  benchmark_summary.csv      aggregate scores + 95% CIs (+ p vs zeroshot) per variant
  benchmark_paired.csv       finetuned − zeroshot deltas, delta CIs and p-values

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
  python eval_metrics.py --test-file data/eval.jsonl --train-file data/train.jsonl
  python eval_metrics.py --resample --force   # rebuild test set (new source) + regenerate
  python eval_metrics.py --seed 42            # generation seed (transformers.set_seed)
  python eval_metrics.py --vanilla-model qwen=<upstream base from the model card>
  python eval_metrics.py --n-bootstrap 1000 --bootstrap-seed 12345

Requirements:
  pip install sacrebleu rouge-score bert-score openpyxl numpy
"""

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime

# ─── Configuration ────────────────────────────────────────────────────────────

# The adapters in ADAPTER_PATHS are trained by {gemma,llama,qwen}/train.py on
# data/train.jsonl. The test set MUST come from the held-out split, never from
# curated_data.jsonl (which contains the training rows).
TEST_FILE   = "./data/eval.jsonl"    # 5% held-out split, scripts/prepare_data.py --seed 42
TRAIN_FILE  = "./data/train.jsonl"   # used only for the overlap check
OUTPUT_DIR  = "./eval_results"
SAMPLE_SEED = 42
GEN_SEED    = 42
N_SAMPLES   = 75

N_BOOTSTRAP    = 1000
BOOTSTRAP_SEED = 12345
CI_LEVEL       = 0.95

METRICS = ["bleu4", "chrf", "rougeL", "bertscore", "cmi_standard", "cmi_heuristic"]

# (treatment, control) pairs tested with the paired bootstrap, per model family
PAIRED_COMPARISONS = [("finetuned", "zeroshot")]

GENERATION_KWARGS = {
    "max_new_tokens":     512,
    "temperature":        0.7,
    "top_p":              0.9,
    "top_k":              50,
    "repetition_penalty": 1.1,
    "do_sample":          True,
}

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
# Override per family with  --vanilla-model <family>=<hf_model_id>.
#
# WARNING: Qwen/Qwen2.5-7B-Instruct is NOT the upstream base of
# aisingapore/Qwen-SEA-LION-v4-8B-VL (a Qwen3-VL-family model), so the
# qwen_base condition as configured here is a mismatched control. Look up the
# actual base checkpoint on the SEA-LION model card and pass it via
# --vanilla-model qwen=<id> before reporting qwen_base numbers.
VANILLA_MODEL_IDS = {
    "gemma": "google/gemma-3-4b-it",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen":  "Qwen/Qwen2.5-7B-Instruct",
}
_MISMATCHED_VANILLA_DEFAULTS = {"qwen": "Qwen/Qwen2.5-7B-Instruct"}

_CHILD_FLAG = "--_child-generate"

# ─── CMI (Code-Mixing Index) lexicon ──────────────────────────────────────────
# Small hand-built Malay / English word lists used to tag tokens by language.

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


_TOKEN_RE = re.compile(r"\b[a-zA-Z']+\b")


def _cmi_tokens(text: str) -> list:
    return _TOKEN_RE.findall(text.lower())


def compute_cmi_heuristic(text: str) -> float:
    """
    Original heuristic index (kept for comparability with earlier runs):
    (N - max_lang_count) / N * 100, where N counts ALL tokens (including
    tokens matched by neither lexicon) and texts with no lexicon hits score 50.
    This is NOT the Gambäck–Das CMI; report compute_cmi_standard() instead.
    """
    tokens = _cmi_tokens(text)
    if not tokens:
        return 0.0
    n_malay   = sum(1 for t in tokens if t in _MALAY_TOKENS)
    n_english = sum(1 for t in tokens if t in _ENGLISH_TOKENS)
    n_total   = len(tokens)
    dominant  = max(n_malay, n_english)
    if dominant == 0:
        return 50.0
    return round((n_total - dominant) / n_total * 100, 2)


def compute_cmi_standard(text: str) -> float:
    """
    Utterance-level Code-Mixing Index of Gambäck & Das:
        CMI = 100 * (1 - max_i(w_i) / (n - u))    if n > u
        CMI = 0                                    if n == u
    n = number of tokens, w_i = tokens tagged as language i (Malay / English),
    u = language-independent tokens, here every token matched by neither
    lexicon. The lexicons are small, so most content words land in u: the score
    measures mixing among lexicon-tagged tokens only.
    """
    tokens = _cmi_tokens(text)
    n_malay   = sum(1 for t in tokens if t in _MALAY_TOKENS)
    n_english = sum(1 for t in tokens if t in _ENGLISH_TOKENS)
    n_tagged  = n_malay + n_english            # = n - u
    if n_tagged == 0:
        return 0.0
    return round(100.0 * (1.0 - max(n_malay, n_english) / n_tagged), 2)


# ─── Reasoning-trace stripping ────────────────────────────────────────────────

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE  = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


def strip_think(text: str) -> tuple:
    """
    Remove reasoning traces before scoring. Returns (clean_text, kind) where
    kind is None (nothing removed) or one of:
      "block"       complete <think>…</think> block(s)
      "orphan"      reasoning ended by </think> with no opener (the chat
                    template put <think> in the prompt) — everything up to and
                    including the last </think> is dropped
      "unclosed"    <think> with no closer (generation hit max_new_tokens
                    while still thinking) — everything from <think> is dropped
    """
    kind  = None
    clean = text
    if _THINK_BLOCK_RE.search(clean):
        clean = _THINK_BLOCK_RE.sub("", clean)
        kind  = "block"
    if _THINK_CLOSE_RE.search(clean):
        clean = _THINK_CLOSE_RE.split(clean)[-1]
        kind  = kind or "orphan"
    if _THINK_OPEN_RE.search(clean):
        clean = _THINK_OPEN_RE.split(clean)[0]
        kind  = kind or "unclosed"
    return clean.strip(), kind


# ─── Dataset helpers ──────────────────────────────────────────────────────────

class EvalDataError(RuntimeError):
    """Test-set / response-file problem that would make the scores invalid."""


def load_jsonl(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_turns(record: dict) -> list:
    """
    Chat turns of a record. scripts/prepare_data.py writes train/eval.jsonl
    under "messages"; curated_data.jsonl uses "conversations".
    """
    turns = record.get("messages") or record.get("conversations") or []
    return turns if isinstance(turns, list) else []


def extract_qa(record: dict):
    """
    Return {"question", "reference"[, "system"]} from the first user turn and
    the assistant turn that follows it (a leading system turn is kept as
    "system"), or None if the record is unusable.
    """
    turns  = get_turns(record)
    system = None
    if turns and isinstance(turns[0], dict) and turns[0].get("role") == "system":
        system = turns[0].get("content")
        turns  = turns[1:]
    if len(turns) < 2 or not all(isinstance(t, dict) for t in turns[:2]):
        return None
    user, assistant = turns[0], turns[1]
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        return None
    question, reference = user.get("content"), assistant.get("content")
    if not isinstance(question, str) or not isinstance(reference, str):
        return None
    if not question.strip() or not reference.strip():
        return None
    qa = {"question": question, "reference": reference}
    if isinstance(system, str) and system.strip():
        qa["system"] = system
    return qa


def sample_test_set(test_file: str, n: int, seed: int, output_dir: str,
                    resample: bool = False) -> list:
    """
    Draw (or reuse) the fixed test set. The cache records the source file and
    its sha256; a cache built from a different source, size or seed is refused
    unless resample=True.
    """
    test_path = os.path.join(output_dir, "test_samples.json")
    if not os.path.exists(test_file):
        raise EvalDataError(
            f"Test file not found: {test_file}\n"
            "Create the held-out split with:\n"
            "  python scripts/prepare_data.py --input curated_data.jsonl "
            "--output-dir data --eval-ratio 0.05 --seed 42"
        )
    source = {
        "test_file": os.path.abspath(test_file),
        "sha256":    file_sha256(test_file),
        "n_samples": n,
        "seed":      seed,
    }

    if os.path.exists(test_path) and not resample:
        with open(test_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cached_source = cached.get("source") if isinstance(cached, dict) else None
        if cached_source is None:
            raise EvalDataError(
                f"{test_path} has no source record — it was written by an older "
                "version of this script that sampled from curated_data.jsonl (the "
                "training data). Re-run with --resample --force to rebuild the test "
                "set from the held-out split and regenerate all responses."
            )
        mismatched = [k for k in ("sha256", "n_samples", "seed")
                      if cached_source.get(k) != source[k]]
        if mismatched:
            raise EvalDataError(
                f"{test_path} was built from a different source "
                f"({', '.join(mismatched)} differ):\n"
                f"  cached : {cached_source.get('test_file')}  "
                f"sha256={str(cached_source.get('sha256'))[:12]}  "
                f"n={cached_source.get('n_samples')}  seed={cached_source.get('seed')}\n"
                f"  current: {source['test_file']}  sha256={source['sha256'][:12]}  "
                f"n={n}  seed={seed}\n"
                "Re-run with --resample --force to rebuild it and regenerate responses."
            )
        print(f"  Using existing test set: {test_path}  "
              f"(source {cached_source['test_file']}, sha256 {source['sha256'][:12]})")
        return cached["samples"]

    all_records = load_jsonl(test_file)
    valid = [qa for qa in (extract_qa(r) for r in all_records) if qa is not None]
    if len(valid) < n:
        raise EvalDataError(
            f"Only {len(valid)} usable user→assistant records in {test_file} "
            f"(of {len(all_records)}), but --samples={n}. Lower --samples or "
            "regenerate the split with scripts/prepare_data.py."
        )
    chosen  = random.Random(seed).sample(valid, n)
    samples = [{"id": i, **qa} for i, qa in enumerate(chosen)]
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump({"source": source, "samples": samples}, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(samples)} test samples from {test_file} → {test_path}")
    return samples


def normalise_text(text: str) -> str:
    return " ".join(text.lower().split())


def check_overlap(samples: list, train_file: str) -> dict:
    """
    Exact and normalised (lowercase, whitespace-collapsed) matches of test
    questions against training user turns and test references against
    training assistant turns. Paraphrased duplicates are NOT detected.
    """
    if not os.path.exists(train_file):
        print(f"  [WARN] Train file not found: {train_file} — overlap check skipped.")
        return {"train_file": os.path.abspath(train_file), "checked": False}

    train_text = {"user": set(), "assistant": set()}
    for record in load_jsonl(train_file):
        for turn in get_turns(record):
            if (isinstance(turn, dict) and turn.get("role") in train_text
                    and isinstance(turn.get("content"), str)):
                train_text[turn["role"]].add(turn["content"])
    train_norm = {role: {normalise_text(t) for t in texts}
                  for role, texts in train_text.items()}

    report = {
        "train_file":   os.path.abspath(train_file),
        "train_sha256": file_sha256(train_file),
        "checked":      True,
        "n_samples":    len(samples),
    }
    for field, role in (("question", "user"), ("reference", "assistant")):
        report[f"{field}_exact"] = [
            s["id"] for s in samples if s[field] in train_text[role]]
        report[f"{field}_normalised"] = [
            s["id"] for s in samples if normalise_text(s[field]) in train_norm[role]]
    return report


def parse_model_overrides(pairs: list) -> dict:
    """['qwen=Org/Model', …] → {'qwen': 'Org/Model', …}"""
    overrides = {}
    for pair in pairs or []:
        key, sep, model_id = pair.partition("=")
        key, model_id = key.strip(), model_id.strip()
        if not sep or key not in VANILLA_MODEL_IDS or not model_id:
            raise ValueError(
                f"Bad --vanilla-model {pair!r}; expected <family>=<model_id> with "
                f"family in {sorted(VANILLA_MODEL_IDS)}")
        overrides[key] = model_id
    return overrides


def load_responses(path: str) -> tuple:
    """Return (metadata, items) from a responses_<label>.json file (legacy files are plain lists)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("metadata", {}), data["responses"]
    return {}, data


def stale_reason(path: str, samples: list):
    """
    None if the response file answers exactly the current test questions,
    otherwise a short reason. Ids are 0..n-1 in every test set, so ids alone
    cannot tell an old file from a new one — compare the question text.
    """
    _, items = load_responses(path)
    by_id = {item.get("id"): item.get("question") for item in items}
    missing   = [s["id"] for s in samples if s["id"] not in by_id]
    different = [s["id"] for s in samples
                 if s["id"] in by_id and by_id[s["id"]] != s["question"]]
    if missing or different:
        return f"{len(different)} question(s) differ, {len(missing)} missing"
    return None


# ─── Child mode: generate responses for ONE model variant ─────────────────────

def child_generate(args):
    import torch

    model_key    = args.child_model_key
    mode         = args.child_mode_str        # "base" | "zeroshot" | "finetuned"
    output_file  = args.child_output_file
    samples_file = args.child_samples_file

    with open(samples_file, "r", encoding="utf-8") as f:
        test_set = json.load(f)
    samples = test_set["samples"]

    adapter_path = ADAPTER_PATHS[model_key]

    from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, set_seed

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Choose model ID based on mode ──────────────────────────────────────────
    # The parent resolves the ID (incl. --vanilla-model overrides) and passes it
    # in, since this subprocess re-parses argv from scratch.
    if args.child_source_id:
        source_id = args.child_source_id
    elif mode == "base":
        source_id = VANILLA_MODEL_IDS[model_key]
    else:
        source_id = BASE_MODEL_IDS[model_key]

    print(f"\n{'='*60}")
    print(f"  Model : {model_key.upper()} ({mode})")
    print(f"  Source: {source_id}")
    print(f"  Seed  : {args.child_seed}")
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

    template_kw = {}
    # thinking_mode is a SEA-LION-Llama-specific kwarg; skip for vanilla base
    if model_key == "llama" and mode != "base":
        template_kw["thinking_mode"] = "off"
    # Qwen3-family templates default to emitting a <think> trace; turn it off.
    # Any residual trace is stripped by the parent before scoring.
    chat_template = getattr(tokenizer, "chat_template", None)
    if model_key == "qwen" or (isinstance(chat_template, str) and "enable_thinking" in chat_template):
        template_kw["enable_thinking"] = False

    def render_prompt(messages):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **template_kw
            )
        except TypeError:
            if "enable_thinking" not in template_kw:
                raise
            template_kw.pop("enable_thinking")
            print("  [WARN] chat template rejected enable_thinking — continuing without it.")
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **template_kw
            )

    if args.child_seed is not None:
        set_seed(args.child_seed)

    # ── Generate ───────────────────────────────────────────────────────────────
    results = []
    for i, sample in enumerate(samples):
        question = sample["question"]
        print(f"  [{i+1:02d}/{len(samples)}] {question[:70]}…")

        messages = [{"role": "user", "content": question}]
        if sample.get("system"):
            messages.insert(0, {"role": "system", "content": sample["system"]})

        try:
            prompt  = render_prompt(messages)
            inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, **GENERATION_KWARGS)
            response = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
        except Exception as e:
            response = f"ERROR: {e}"

        results.append({"id": sample["id"], "question": question, "response": response})

    import transformers
    metadata = {
        "label":                f"{model_key}_{mode}",
        "source_model_id":      source_id,
        "adapter_path":         os.path.abspath(adapter_path) if mode == "finetuned" else None,
        "generation_seed":      args.child_seed,
        "generation_kwargs":    GENERATION_KWARGS,
        "chat_template_kwargs": template_kw,
        "test_set_source":      test_set.get("source"),
        "transformers_version": transformers.__version__,
        "generated":            datetime.now().isoformat(),
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "responses": results}, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved {len(results)} responses → {output_file}")
    print("  Child done — CUDA memory released on exit.")


# ─── Bootstrap statistics ─────────────────────────────────────────────────────

def bootstrap_indices(n_items: int, n_resamples: int = N_BOOTSTRAP,
                      seed: int = BOOTSTRAP_SEED):
    """
    (n_resamples × n_items) matrix of item indices drawn with replacement.
    The same matrix is reused for every model variant, so resample b is the
    same multiset of items for all variants — required for the paired test.
    """
    import numpy as np
    return np.random.default_rng(seed).integers(0, n_items, size=(n_resamples, n_items))


def bootstrap_distribution(stat_fn, indices):
    """stat_fn(index_array) -> float, evaluated on every resample."""
    import numpy as np
    return np.array([stat_fn(idx) for idx in indices], dtype=float)


def percentile_ci(distribution, level: float = CI_LEVEL) -> tuple:
    import numpy as np
    tail   = (1.0 - level) / 2 * 100
    lo, hi = np.percentile(distribution, [tail, 100 - tail])
    return float(lo), float(hi)


def bootstrap_ci(stat_fn, n_items: int, n_resamples: int = N_BOOTSTRAP,
                 seed: int = BOOTSTRAP_SEED, level: float = CI_LEVEL) -> tuple:
    """Percentile bootstrap CI of stat_fn over items resampled with replacement."""
    indices = bootstrap_indices(n_items, n_resamples, seed)
    return percentile_ci(bootstrap_distribution(stat_fn, indices), level)


def paired_bootstrap_pvalue(dist_a, dist_b, observed_delta: float) -> float:
    """
    Two-sided paired-bootstrap p-value for H0: metric(A) == metric(B).
    dist_a / dist_b hold the statistics of A and B on the SAME resamples. The
    resampled deltas are centred on 0 to simulate H0 (Koehn 2004; cf. sacrebleu
    --paired-bs), and
        p = (1 + #{b : |delta_b - mean(delta)| >= |observed_delta|}) / (B + 1)
    """
    import numpy as np
    deltas  = np.asarray(dist_a, dtype=float) - np.asarray(dist_b, dtype=float)
    centred = deltas - deltas.mean()
    exceed  = int(np.sum(np.abs(centred) >= abs(observed_delta) - 1e-12))
    return (1 + exceed) / (len(deltas) + 1)


# ─── Metric computation (runs in parent after all children finish) ─────────────

def _corpus_stat_fn(metric_cls, hypotheses: list, references: list):
    """
    stat_fn(idx) -> corpus-level score (BLEU / chrF) on the resampled items.
    Corpus BLEU is not a mean of sentence scores, so each resample is rescored
    from summed per-sentence sufficient statistics — identical to re-running
    corpus_bleu / corpus_chrf on the resample, but ~1000× faster. Falls back to
    full recomputation if sacrebleu's private API changes.
    """
    import numpy as np
    metric = metric_cls()
    try:
        stats = np.asarray(metric._extract_corpus_statistics(hypotheses, [references]))
        metric._compute_score_from_stats(stats.sum(axis=0).tolist())
        return lambda idx: metric._compute_score_from_stats(stats[idx].sum(axis=0).tolist()).score
    except (AttributeError, TypeError):
        return lambda idx: metric.corpus_score(
            [hypotheses[i] for i in idx], [[references[i] for i in idx]]).score


def per_item_scores(hypotheses: list, references: list) -> dict:
    """Sentence-level scores on a 0–100 scale (bertscore is None if not installed)."""
    from rouge_score import rouge_scorer as rs_module

    scorer = rs_module.RougeScorer(["rougeL"], use_stemmer=False)
    rougeL = [
        scorer.score(ref, hyp)["rougeL"].fmeasure * 100
        for hyp, ref in zip(hypotheses, references)
    ]

    bertscore = None
    try:
        from bert_score import score as bscore
        _, _, F = bscore(
            hypotheses, references,
            model_type="xlm-roberta-large",
            lang="others",
            verbose=False,
            batch_size=8,
        )
        bertscore = [f * 100 for f in F.tolist()]
    except ImportError:
        print("  [WARN] bert-score not installed — BERTScore skipped.")

    return {
        "rougeL":        rougeL,
        "bertscore":     bertscore,
        "cmi_standard":  [compute_cmi_standard(h)  for h in hypotheses],
        "cmi_heuristic": [compute_cmi_heuristic(h) for h in hypotheses],
    }


def compute_metrics(hypotheses: list, references: list, indices) -> dict:
    """
    Point estimate, bootstrap distribution and percentile CI for every metric
    in METRICS. `indices` comes from bootstrap_indices() and must be shared by
    all variants.
    """
    import numpy as np
    from sacrebleu.metrics import BLEU, CHRF

    per_item = per_item_scores(hypotheses, references)
    stat_fns = {
        "bleu4": _corpus_stat_fn(BLEU, hypotheses, references),
        "chrf":  _corpus_stat_fn(CHRF, hypotheses, references),
    }
    for name, values in per_item.items():
        if values is not None:
            arr = np.asarray(values, dtype=float)
            stat_fns[name] = lambda idx, arr=arr: float(arr[idx].mean())

    all_items = np.arange(len(hypotheses))
    result = {"score": {}, "ci": {}, "dist": {}, "per_item": per_item}
    for name in METRICS:
        fn = stat_fns.get(name)
        if fn is None:
            result["score"][name], result["ci"][name], result["dist"][name] = None, (None, None), None
            continue
        result["score"][name] = float(fn(all_items))
        result["dist"][name]  = bootstrap_distribution(fn, indices)
        result["ci"][name]    = percentile_ci(result["dist"][name])
    return result


def paired_comparisons(results: dict) -> list:
    """Paired bootstrap for each (treatment, control) in PAIRED_COMPARISONS, per family."""
    rows = []
    families = sorted({label.rsplit("_", 1)[0] for label in results})
    for family in families:
        for treat_mode, ctrl_mode in PAIRED_COMPARISONS:
            treat, ctrl = f"{family}_{treat_mode}", f"{family}_{ctrl_mode}"
            if treat not in results or ctrl not in results:
                continue
            a, b = results[treat], results[ctrl]
            for metric in METRICS:
                if a["dist"][metric] is None or b["dist"][metric] is None:
                    continue
                delta  = a["score"][metric] - b["score"][metric]
                lo, hi = percentile_ci(a["dist"][metric] - b["dist"][metric])
                rows.append({
                    "family":          family,
                    "treatment":       treat,
                    "control":         ctrl,
                    "metric":          metric,
                    "treatment_score": round(a["score"][metric], 2),
                    "control_score":   round(b["score"][metric], 2),
                    "delta":           round(delta, 2),
                    "delta_ci_low":    round(lo, 2),
                    "delta_ci_high":   round(hi, 2),
                    "p_value":         round(paired_bootstrap_pvalue(
                                           a["dist"][metric], b["dist"][metric], delta), 4),
                })
    return rows


def _r2(x):
    return round(x, 2) if x is not None else None


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def orchestrate(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_keys  = [args.model] if args.model else ["gemma", "llama", "qwen"]
    vanilla_ids = {**VANILLA_MODEL_IDS, **args.vanilla_overrides}

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

    if "base" in modes:
        for mk in model_keys:
            if vanilla_ids[mk] == _MISMATCHED_VANILLA_DEFAULTS.get(mk):
                print(f"\n  [WARN] {mk}_base uses {vanilla_ids[mk]}, which is NOT the upstream base of "
                      f"{BASE_MODEL_IDS[mk]}.\n         Set --vanilla-model {mk}=<base from the model card> "
                      f"before reporting {mk}_base.")

    def source_id_for(model_key, mode):
        return vanilla_ids[model_key] if mode == "base" else BASE_MODEL_IDS[model_key]

    # ── Phase 1: fixed test set ────────────────────────────────────────────────
    print("\n[Phase 1]  Sampling test set …")
    samples      = sample_test_set(args.test_file, args.samples, SAMPLE_SEED, OUTPUT_DIR,
                                   resample=args.resample)
    samples_file = os.path.join(OUTPUT_DIR, "test_samples.json")
    with open(samples_file, "r", encoding="utf-8") as f:
        test_source = json.load(f)["source"]
    references   = [s["reference"] for s in samples]
    ids          = [s["id"]        for s in samples]
    print(f"  {len(samples)} test samples ready.")

    print(f"\n  Checking overlap with training file {args.train_file} …")
    overlap = check_overlap(samples, args.train_file)
    if overlap["checked"]:
        counts = {k: len(overlap[k]) for k in
                  ("question_exact", "question_normalised", "reference_exact", "reference_normalised")}
        print(f"  questions:  exact {counts['question_exact']}, normalised {counts['question_normalised']}"
              f"   references:  exact {counts['reference_exact']}, normalised {counts['reference_normalised']}"
              f"   (of {len(samples)})")
        if any(counts.values()):
            print("  [WARN] Test items also occur in the training file — see overlap_report.json.")
    with open(os.path.join(OUTPUT_DIR, "overlap_report.json"), "w", encoding="utf-8") as f:
        json.dump(overlap, f, indent=2)

    # ── Phase 2: generate responses ────────────────────────────────────────────
    variants       = [(mk, mode) for mk in model_keys for mode in modes]
    response_files = {}

    if not args.skip_generate:
        print(f"\n[Phase 2]  Generating responses ({len(variants)} model variant(s)) …")

        stale = []
        for model_key, mode in variants:
            label    = f"{model_key}_{mode}"
            out_file = os.path.join(OUTPUT_DIR, f"responses_{label}.json")
            if os.path.exists(out_file) and not args.force:
                reason = stale_reason(out_file, samples)
                generated_with = load_responses(out_file)[0].get("source_model_id")
                if reason is None and generated_with and generated_with != source_id_for(model_key, mode):
                    reason = f"generated with {generated_with}, requested {source_id_for(model_key, mode)}"
                if reason:
                    stale.append(f"  {out_file}: {reason}")
        if stale:
            raise EvalDataError(
                "These response files do not match the current test set / model:\n"
                + "\n".join(stale) + "\nRe-run with --force to regenerate them.")

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
                "--child-source-id",    source_id_for(model_key, mode),
                "--child-seed",         str(args.seed),
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
        stale = [f"  {path}: {reason}" for path, reason in
                 ((p, stale_reason(p, samples)) for p in response_files.values()) if reason]
        if stale:
            raise EvalDataError(
                "These response files were generated for a different test set:\n"
                + "\n".join(stale) + "\nRe-run without --skip-generate and with --force.")

    if not response_files:
        print("\nNo response files available. Run without --skip-generate first.")
        sys.exit(1)

    # ── Phase 3: load all responses ────────────────────────────────────────────
    print("\n[Phase 3]  Loading responses (stripping <think> traces) …")
    all_responses   = {}
    generation_meta = {}
    think_stripped  = {}
    for label, fpath in response_files.items():
        meta, items = load_responses(fpath)
        generation_meta[label] = meta or {"legacy_file": True}
        counts   = {"block": 0, "orphan": 0, "unclosed": 0}
        resp_map = {}
        n_errors = 0
        for item in items:
            text, kind = strip_think(item["response"])
            if kind:
                counts[kind] += 1
            if item["response"].startswith("ERROR:"):
                n_errors += 1
            resp_map[item["id"]] = text
        all_responses[label]  = resp_map
        think_stripped[label] = counts
        print(f"  {label}: {len(items)} responses, think traces stripped: {sum(counts.values())} "
              f"(block {counts['block']}, orphan </think> {counts['orphan']}, "
              f"unclosed <think> {counts['unclosed']})")
        if n_errors:
            print(f"  [WARN] {label}: {n_errors} responses are generation errors (scored as-is).")
        if meta.get("generation_seed") != args.seed:
            print(f"  [WARN] {label}: generated with seed {meta.get('generation_seed', 'unknown (legacy file)')}, "
                  f"--seed is {args.seed}.")

    # per-sample lookup
    per_sample = {s["id"]: {"id": s["id"], "question": s["question"], "reference": s["reference"]}
                  for s in samples}

    # ── Phase 4: compute metrics ───────────────────────────────────────────────
    print(f"\n[Phase 4]  Computing metrics ({args.n_bootstrap} bootstrap resamples, "
          f"seed {args.bootstrap_seed}) …")
    indices = bootstrap_indices(len(samples), args.n_bootstrap, args.bootstrap_seed)
    results = {}

    for label in sorted(all_responses.keys()):
        resp_map = all_responses[label]
        hyps     = [resp_map.get(sid, "") for sid in ids]

        print(f"  {label} …")
        m = compute_metrics(hyps, references, indices)
        results[label] = m

        for i, sid in enumerate(ids):
            per_sample[sid][f"{label}_response"] = hyps[i]
            for name, values in m["per_item"].items():
                if values is not None:
                    per_sample[sid][f"{label}_{name}"] = round(values[i], 2)

    paired_rows  = paired_comparisons(results)
    summary_rows = []
    for label in sorted(results):
        m   = results[label]
        row = {"model_variant": label}
        for name in METRICS:
            row[name]              = _r2(m["score"][name])
            row[f"{name}_ci_low"]  = _r2(m["ci"][name][0])
            row[f"{name}_ci_high"] = _r2(m["ci"][name][1])
        row["think_traces_stripped"] = sum(think_stripped[label].values())
        summary_rows.append(row)
    rows_by_label = {row["model_variant"]: row for row in summary_rows}
    for pr in paired_rows:
        ctrl_mode = pr["control"].rsplit("_", 1)[1]
        rows_by_label[pr["treatment"]][f"{pr['metric']}_delta_vs_{ctrl_mode}"] = pr["delta"]
        rows_by_label[pr["treatment"]][f"{pr['metric']}_p_vs_{ctrl_mode}"]     = pr["p_value"]

    # ── Phase 5: save outputs ──────────────────────────────────────────────────
    print("\n[Phase 5]  Saving outputs …")

    raw_path = os.path.join(OUTPUT_DIR, "benchmark_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "generated":         datetime.now().isoformat(),
                "n_samples":         len(samples),
                "seed":              SAMPLE_SEED,      # test-set sampling seed
                "generation_seed":   args.seed,
                "test_set":          test_source,
                "overlap":           overlap,
                "base_model_ids":    {mk: BASE_MODEL_IDS[mk] for mk in model_keys},
                "vanilla_model_ids": {mk: vanilla_ids[mk]    for mk in model_keys},
                "generation":        generation_meta,
                "think_traces_stripped": think_stripped,
                "bootstrap": {
                    "n_resamples": args.n_bootstrap,
                    "seed":        args.bootstrap_seed,
                    "ci":          f"{int(CI_LEVEL * 100)}% percentile, items resampled with replacement",
                    "p_value":     "two-sided paired bootstrap, null-centred deltas",
                },
                "model_variants":    list(all_responses.keys()),
            },
            "samples":          list(per_sample.values()),
            "summary":          summary_rows,
            "paired_bootstrap": paired_rows,
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

        # Sheet 2 — aggregate metrics with 95% CIs
        df_sum = pd.DataFrame(summary_rows)

        # Sheet 3 — paired bootstrap tests
        df_paired = pd.DataFrame(paired_rows)

        # Sheet 4 — per-sample CMI (code-mixing quality)
        cmi_cols = {"ID": ids, "Question": [per_sample[sid]["question"] for sid in ids]}
        for label in sorted(all_responses.keys()):
            for name in ("cmi_standard", "cmi_heuristic"):
                cmi_cols[f"{label}_{name}"] = [per_sample[sid].get(f"{label}_{name}", "") for sid in ids]
        df_cmi = pd.DataFrame(cmi_cols)

        xlsx_path = os.path.join(OUTPUT_DIR, "benchmark_results.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df_resp.to_excel(writer,   sheet_name="Responses", index=False)
            df_sum.to_excel(writer,    sheet_name="Summary Metrics", index=False)
            df_paired.to_excel(writer, sheet_name="Paired Bootstrap", index=False)
            df_cmi.to_excel(writer,    sheet_name="CMI per Sample", index=False)
        print(f"  Saved: {xlsx_path}")

        csv_path = os.path.join(OUTPUT_DIR, "benchmark_summary.csv")
        df_sum.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

        paired_path = os.path.join(OUTPUT_DIR, "benchmark_paired.csv")
        df_paired.to_csv(paired_path, index=False)
        print(f"  Saved: {paired_path}")

    except ImportError as e:
        print(f"  [WARN] {e} — Excel output skipped.")

    # ── Summary table ──────────────────────────────────────────────────────────
    headers = {"bleu4": "BLEU-4", "chrf": "chrF", "rougeL": "ROUGE-L", "bertscore": "BERTScore",
               "cmi_standard": "CMI", "cmi_heuristic": "CMI-heur"}
    width = 27 + 22 * len(METRICS)
    print(f"\n{'='*width}")
    print(f"  Evaluation Summary  ({len(samples)} samples from {args.test_file}, sample seed={SAMPLE_SEED}, "
          f"generation seed={args.seed}; [{int(CI_LEVEL * 100)}% bootstrap CI])")
    print(f"{'='*width}")
    print(f"  {'Variant':<25}" + "".join(f"  {headers[n]:>20}" for n in METRICS))
    print("  " + "-" * (width - 2))
    for row in summary_rows:
        cells = []
        for name in METRICS:
            cell = "N/A" if row[name] is None else \
                f"{row[name]:.2f} [{row[name + '_ci_low']:.1f},{row[name + '_ci_high']:.1f}]"
            cells.append(f"  {cell:>20}")
        print(f"  {row['model_variant']:<25}" + "".join(cells))
    print(f"{'='*width}")

    if paired_rows:
        print(f"\n  Paired bootstrap (two-sided, {args.n_bootstrap} resamples)")
        print(f"  {'Comparison':<38}  {'Metric':<14}  {'Delta':>7}  {'95% CI':>16}  {'p':>7}")
        for pr in paired_rows:
            ci = f"[{pr['delta_ci_low']:.2f}, {pr['delta_ci_high']:.2f}]"
            print(f"  {pr['treatment'] + ' vs ' + pr['control']:<38}  {pr['metric']:<14}"
                  f"  {pr['delta']:>7.2f}  {ci:>16}  {pr['p_value']:>7.4f}")
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
    p.add_argument("--test-file",      type=str, default=TEST_FILE,
                   help="Held-out JSONL to sample the test set from (scripts/prepare_data.py eval split)")
    p.add_argument("--train-file",     type=str, default=TRAIN_FILE,
                   help="Training JSONL the adapters were fine-tuned on (for the overlap check)")
    p.add_argument("--resample",       action="store_true",
                   help="Rebuild eval_results/test_samples.json even if a cached test set exists")
    p.add_argument("--seed",           type=int, default=GEN_SEED,
                   help="Generation seed passed to transformers.set_seed")
    p.add_argument("--vanilla-model",  action="append", default=[], metavar="FAMILY=MODEL_ID",
                   help="Override the vanilla base checkpoint for a family (repeatable), "
                        "e.g. --vanilla-model qwen=<upstream base from the SEA-LION model card>")
    p.add_argument("--n-bootstrap",    type=int, default=N_BOOTSTRAP,
                   help="Bootstrap resamples for CIs and paired tests")
    p.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED,
                   help="Seed for the bootstrap resampling")
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
    p.add_argument("--child-source-id",     dest="child_source_id",   type=str,            help=argparse.SUPPRESS)
    p.add_argument("--child-seed",          dest="child_seed",        type=int,            help=argparse.SUPPRESS)
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()
    if args.child_mode:
        child_generate(args)
        return
    try:
        args.vanilla_overrides = parse_model_overrides(args.vanilla_model)
    except ValueError as e:
        parser.error(str(e))
    try:
        orchestrate(args)
    except EvalDataError as e:
        sys.stdout.flush()
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
