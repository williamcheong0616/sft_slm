#!/usr/bin/env python3
"""
Iterative QLoRA fine-tuning for ALL three SEA-LION models.

Memory strategy
───────────────
Each (model × n_rows) pair is trained in a **child subprocess**.
When the child exits, CUDA releases *all* memory unconditionally.
This is the only reliable way to avoid OOM between iterations when
using bitsandbytes 4-bit quantization with device_map="auto".

Models trained
──────────────
  gemma  →  aisingapore/Gemma-SEA-LION-v4-4B-VL  (ViT dropped via AutoModelForCausalLM)
  llama  →  aisingapore/Llama-SEA-LION-v3.5-8B-R
  qwen   →  aisingapore/Qwen-SEA-LION-v4-8B-VL

Iteration schedule
──────────────────
  5 000 → 10 000 → 15 000 → 20 000 → … → max rows
  Step and start are configurable (see --step / --start).

Output folder per run
──────────────────────
  ./output/{model_short}_{YYYYMMDD_HHMMSS}_{n_rows}rows/

Usage (orchestrator mode — default)
────────────────────────────────────
  python train_iterative.py                         # all models, all sizes
  python train_iterative.py --model gemma           # one model
  python train_iterative.py --step 10000            # custom step
  python train_iterative.py --no-wandb              # disable WandB
  python train_iterative.py --model llama --epochs 2 --lora-rank 32
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────
DATA_FILE        = "./curated_data.jsonl"
OUTPUT_BASE      = "./output"
MAX_SEQ_LENGTH   = 512
EPOCHS           = 1
LORA_RANK        = 16
LORA_DROPOUT     = 0.05
ITERATION_STEP   = 5_000
ITERATION_START  = 5_000

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

MODEL_CONFIGS = {
    "gemma": {
        "model_id":        "aisingapore/Gemma-SEA-LION-v4-4B-VL",
        "short_name":      "gemma-4b",
        "batch_size":      4,
        "grad_accum":      4,   # effective batch = 16
        "learning_rate":   2e-4,
        "run_name_prefix": "gemma-sealion-4b-qlora",
    },
    "llama": {
        "model_id":        "aisingapore/Llama-SEA-LION-v3.5-8B-R",
        "short_name":      "llama-8b",
        "batch_size":      2,
        "grad_accum":      8,   # effective batch = 16
        "learning_rate":   2e-4,
        "run_name_prefix": "llama-sealion-8b-qlora",
    },
    "qwen": {
        "model_id":        "aisingapore/Qwen-SEA-LION-v4-8B-VL",
        "short_name":      "qwen-8b",
        "batch_size":      2,
        "grad_accum":      8,   # effective batch = 16
        "learning_rate":   2e-4,
        "run_name_prefix": "qwen-sealion-8b-qlora",
    },
}

# Internal sentinel — child mode is triggered by passing this flag
_CHILD_FLAG = "--_child-run"


# ──────────────────────────────────────────────────────────────────────────────
# UTILITY
# ──────────────────────────────────────────────────────────────────────────────

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_jsonl(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_iteration_sizes(total: int, start: int, step: int) -> list:
    sizes = list(range(start, total, step))
    if not sizes or sizes[-1] != total:
        sizes.append(total)
    return sizes


# ──────────────────────────────────────────────────────────────────────────────
# CHILD MODE  ─  one subprocess = one training run
# ──────────────────────────────────────────────────────────────────────────────

def child_train(child_args):
    """
    This function runs inside a fresh subprocess.
    All GPU allocations are freed when this process exits.
    """
    # ── lazy imports (only needed in child) ──────────────────────────────────
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    model_key  = child_args.child_model_key
    n_rows     = child_args.child_n_rows
    run_dir    = child_args.child_run_dir
    data_file  = child_args.child_data_file
    eval_start = child_args.child_eval_start   # index where eval slice begins
    epochs     = child_args.child_epochs
    lora_rank  = child_args.child_lora_rank
    lora_alpha = lora_rank * 2
    max_seq    = child_args.child_max_seq
    no_wandb   = child_args.child_no_wandb
    cfg        = MODEL_CONFIGS[model_key]

    print(f"\n{'='*70}")
    print(f"  🚀  {cfg['short_name'].upper()} — {n_rows:,} rows")
    print(f"      Output → {run_dir}")
    print(f"      PID    → {os.getpid()}")
    print(f"{'='*70}\n")

    # ── Load dataset slice ────────────────────────────────────────────────────
    print("📖  Loading dataset …")
    all_records = load_jsonl(data_file)

    def to_messages(r):
        return {"messages": [{"role": c["role"], "content": c["content"]}
                              for c in r["conversations"]]}

    train_records = all_records[:n_rows]
    eval_records  = all_records[eval_start:]

    train_dataset = Dataset.from_list([to_messages(r) for r in train_records])
    eval_dataset  = Dataset.from_list([to_messages(r) for r in eval_records])

    print(f"📂  Train: {len(train_dataset):,} | Eval: {len(eval_dataset):,}")

    # ── Load model (text-only via AutoModelForCausalLM — ViT not loaded) ─────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"🔄  Loading model: {cfg['model_id']} (4-bit QLoRA, text-only)")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # ── QLoRA prep ────────────────────────────────────────────────────────────
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, peft_config)
    print("\n📊  Trainable parameters:")
    model.print_trainable_parameters()

    # ── Formatting function ───────────────────────────────────────────────────
    def make_fmt(tok, key):
        def fmt(example):
            msgs = example.get("messages", [])
            kw = dict(tokenize=False, add_generation_prompt=False)
            if key == "llama":
                kw["thinking_mode"] = "off"
            return tok.apply_chat_template(msgs, **kw)
        return fmt

    # ── SFT config ────────────────────────────────────────────────────────────
    report_to = "none" if no_wandb else "wandb"
    run_name  = f"{cfg['run_name_prefix']}-{n_rows}rows"

    sft_config = SFTConfig(
        output_dir=run_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["learning_rate"],
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        report_to=report_to,
        run_name=run_name,
        max_length=max_seq,
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=make_fmt(tokenizer, model_key),
    )

    print(f"\n   Epochs:          {epochs}")
    print(f"   Batch size:      {cfg['batch_size']}")
    print(f"   Grad accum:      {cfg['grad_accum']}")
    print(f"   Effective batch: {cfg['batch_size'] * cfg['grad_accum']}")
    print(f"   Learning rate:   {cfg['learning_rate']}")
    print(f"   LoRA rank/alpha: {lora_rank}/{lora_alpha}")
    print(f"   Max seq length:  {max_seq}")
    print(f"   WandB:           {'❌ disabled' if no_wandb else '✅ enabled'}\n")

    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    final_path = os.path.join(run_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n✅  Adapter saved → {final_path}")

    # ── Metadata ──────────────────────────────────────────────────────────────
    with open(os.path.join(run_dir, "training_meta.txt"), "w") as mf:
        mf.write(f"model_id:      {cfg['model_id']}\n")
        mf.write(f"n_rows:        {n_rows}\n")
        mf.write(f"eval_rows:     {len(eval_dataset)}\n")
        mf.write(f"timestamp:     {timestamp()}\n")
        mf.write(f"epochs:        {epochs}\n")
        mf.write(f"lora_rank:     {lora_rank}\n")
        mf.write(f"lora_alpha:    {lora_alpha}\n")
        mf.write(f"batch_size:    {cfg['batch_size']}\n")
        mf.write(f"grad_accum:    {cfg['grad_accum']}\n")
        mf.write(f"learning_rate: {cfg['learning_rate']}\n")
        mf.write(f"max_seq:       {max_seq}\n")

    print("\n🏁  Child process done — CUDA memory will be fully released on exit.")


# ──────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR MODE  ─  loops and spawns child subprocesses
# ──────────────────────────────────────────────────────────────────────────────

def orchestrate(args):
    # Which models to train
    if args.model:
        keys = [args.model.lower()]
        if keys[0] not in MODEL_CONFIGS:
            print(f"❌  Unknown model '{keys[0]}'. Choose from: {list(MODEL_CONFIGS)}")
            sys.exit(1)
    else:
        keys = list(MODEL_CONFIGS.keys())  # gemma → llama → qwen

    step       = args.step  or ITERATION_STEP
    start      = args.start or ITERATION_START
    data_file  = args.data_file or DATA_FILE
    epochs     = args.epochs or EPOCHS
    lora_rank  = args.lora_rank or LORA_RANK
    max_seq    = args.max_seq_length or MAX_SEQ_LENGTH

    # ── Figure out dataset size without loading everything in orchestrator ────
    print(f"\n📖  Counting rows in: {data_file}")
    with open(data_file, "r", encoding="utf-8") as f:
        total = sum(1 for line in f if line.strip())
    print(f"    Total records: {total:,}")

    # Reserve last 10 % as fixed eval
    eval_size   = max(500, int(total * 0.10))
    eval_start  = total - eval_size      # index passed to child
    train_total = total - eval_size

    sizes = build_iteration_sizes(train_total, start, step)
    # Clamp to actual train pool size and deduplicate
    sizes = sorted(set(min(s, train_total) for s in sizes))

    print(f"\n📋  Iteration schedule ({len(sizes)} run(s) per model):")
    for i, s in enumerate(sizes, 1):
        print(f"    [{i:2d}]  {s:>7,} rows")
    print(f"\n🗂️   Fixed eval set: {eval_size:,} rows (last 10% of file)")
    print(f"    Train pool:     {train_total:,} rows")

    grand_total = len(keys) * len(sizes)
    results     = []   # (label, status)
    run_idx     = 0

    for model_key in keys:
        cfg = MODEL_CONFIGS[model_key]
        print(f"\n\n{'#'*70}")
        print(f"#  Model: {cfg['model_id']}")
        print(f"#  Short: {cfg['short_name']}")
        print(f"{'#'*70}")

        for n_rows in sizes:
            run_idx += 1
            ts      = timestamp()
            run_dir = os.path.join(
                OUTPUT_BASE,
                f"{cfg['short_name']}_{ts}_{n_rows}rows"
            )
            os.makedirs(run_dir, exist_ok=True)
            label = f"[{run_idx}/{grand_total}] {cfg['short_name']} — {n_rows:,} rows"

            print(f"\n{label}")
            print(f"  Spawning child subprocess for isolated GPU memory …")

            # Build child argv
            child_cmd = [
                sys.executable, __file__,
                _CHILD_FLAG,
                "--child-model-key",  model_key,
                "--child-n-rows",     str(n_rows),
                "--child-run-dir",    run_dir,
                "--child-data-file",  data_file,
                "--child-eval-start", str(eval_start),
                "--child-epochs",     str(epochs),
                "--child-lora-rank",  str(lora_rank),
                "--child-max-seq",    str(max_seq),
            ]
            if args.no_wandb:
                child_cmd.append("--child-no-wandb")

            # Run child; stream its output live
            proc = subprocess.run(child_cmd)

            if proc.returncode == 0:
                print(f"  ✅  {label} — SUCCESS")
                results.append((label, "SUCCESS"))
            else:
                print(f"  ❌  {label} — FAILED (exit code {proc.returncode})")
                results.append((label, f"FAILED (exit {proc.returncode})"))
                # Write failure marker so we can resume later
                with open(os.path.join(run_dir, "FAILED.txt"), "w") as ff:
                    ff.write(f"exit_code: {proc.returncode}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  📊  Final summary ({run_idx} run(s)):")
    for lbl, status in results:
        icon = "✅" if status == "SUCCESS" else "❌"
        print(f"  {icon}  {lbl}  →  {status}")
    print(f"\n  Outputs in: {os.path.abspath(OUTPUT_BASE)}/")
    print(f"{'='*70}\n")


# ──────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Iterative QLoRA training — Gemma / Llama / Qwen SEA-LION",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Orchestrator flags ────────────────────────────────────────────────────
    p.add_argument("--model",          type=str,  default=None,
                   help="Train only: gemma | llama | qwen  (default: all)")
    p.add_argument("--data-file",      type=str,  default=None,
                   help=f"Path to JSONL dataset  (default: {DATA_FILE})")
    p.add_argument("--step",           type=int,  default=None,
                   help=f"Row increment between iterations  (default: {ITERATION_STEP})")
    p.add_argument("--start",          type=int,  default=None,
                   help=f"First slice size  (default: {ITERATION_START})")
    p.add_argument("--epochs",         type=int,  default=None,
                   help=f"Training epochs per run  (default: {EPOCHS})")
    p.add_argument("--lora-rank",      type=int,  default=None,
                   help=f"LoRA rank  (default: {LORA_RANK})")
    p.add_argument("--max-seq-length", type=int,  default=None,
                   help=f"Max sequence length  (default: {MAX_SEQ_LENGTH})")
    p.add_argument("--no-wandb",       action="store_true",
                   help="Disable WandB logging")

    # ── Child-only flags (internal, not for direct user use) ──────────────────
    p.add_argument(_CHILD_FLAG,          dest="child_mode",      action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--child-model-key",  dest="child_model_key", type=str,
                   help=argparse.SUPPRESS)
    p.add_argument("--child-n-rows",     dest="child_n_rows",    type=int,
                   help=argparse.SUPPRESS)
    p.add_argument("--child-run-dir",    dest="child_run_dir",   type=str,
                   help=argparse.SUPPRESS)
    p.add_argument("--child-data-file",  dest="child_data_file", type=str,
                   help=argparse.SUPPRESS)
    p.add_argument("--child-eval-start", dest="child_eval_start",type=int,
                   help=argparse.SUPPRESS)
    p.add_argument("--child-epochs",     dest="child_epochs",    type=int,
                   help=argparse.SUPPRESS)
    p.add_argument("--child-lora-rank",  dest="child_lora_rank", type=int,
                   help=argparse.SUPPRESS)
    p.add_argument("--child-max-seq",    dest="child_max_seq",   type=int,
                   help=argparse.SUPPRESS)
    p.add_argument("--child-no-wandb",   dest="child_no_wandb",  action="store_true",
                   help=argparse.SUPPRESS)

    return p


def main():
    args = build_parser().parse_args()

    if args.child_mode:
        # ── Running as a child subprocess ─────────────────────────────────────
        child_train(args)
    else:
        # ── Running as the orchestrator ───────────────────────────────────────
        orchestrate(args)


if __name__ == "__main__":
    main()
