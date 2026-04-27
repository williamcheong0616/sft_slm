#!/usr/bin/env python3
"""
Iterative QLoRA fine-tuning for ALL three SEA-LION models.

Models trained:
  1. aisingapore/Gemma-SEA-LION-v4-4B-VL   → text-only (ViT dropped to save VRAM)
  2. aisingapore/Llama-SEA-LION-v3.5-8B-R
  3. aisingapore/Qwen-SEA-LION-v4-8B-VL

Iteration schedule:
  Trains each model on 5k, 10k, 15k, 20k, ... rows, then the full dataset.
  Each iteration is saved to:
    ./output/{model_short}_{YYYYMMDD_HHMMSS}_{n_rows}rows/

Usage:
  # Train all models, all iterations
  python train_iterative.py

  # Train a specific model only
  python train_iterative.py --model gemma
  python train_iterative.py --model llama
  python train_iterative.py --model qwen

  # Override step size (default 5000) and starting point (default 5000)
  python train_iterative.py --step 10000 --start 10000

  # Use a different dataset file
  python train_iterative.py --data-file ./curated_data.jsonl

  # Skip WandB logging
  python train_iterative.py --no-wandb
"""

import argparse
import gc
import os
import sys
from datetime import datetime

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# =============================================================================
# GLOBAL DEFAULTS
# =============================================================================
DATA_FILE = "./curated_data.jsonl"
OUTPUT_BASE = "./output"
MAX_SEQ_LENGTH = 512
EPOCHS = 1
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
ITERATION_STEP = 5_000   # row increment
ITERATION_START = 5_000  # first slice size

# Per-model configs ─────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "gemma": {
        "model_id": "aisingapore/Gemma-SEA-LION-v4-4B-VL",
        "short_name": "gemma-4b",
        "batch_size": 4,
        "grad_accum": 4,      # effective batch = 16
        "learning_rate": 2e-4,
        "run_name_prefix": "gemma-sealion-4b-qlora",
        # AutoModelForCausalLM ignores the vision tower → big VRAM saving
        "model_class": "auto",
    },
    "llama": {
        "model_id": "aisingapore/Llama-SEA-LION-v3.5-8B-R",
        "short_name": "llama-8b",
        "batch_size": 2,
        "grad_accum": 8,      # effective batch = 16
        "learning_rate": 2e-4,
        "run_name_prefix": "llama-sealion-8b-qlora",
        "model_class": "auto",
    },
    "qwen": {
        "model_id": "aisingapore/Qwen-SEA-LION-v4-8B-VL",
        "short_name": "qwen-8b",
        "batch_size": 2,
        "grad_accum": 8,      # effective batch = 16
        "learning_rate": 2e-4,
        "run_name_prefix": "qwen-sealion-8b-qlora",
        "model_class": "auto",
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def free_gpu_memory():
    """Release GPU memory between iterations / models."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    import json
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def conversations_to_messages(record: dict) -> dict:
    """
    Convert a record with 'conversations' list
    (each item has 'role' and 'content') to a
    standard 'messages' field used by chat templates.
    """
    convs = record.get("conversations", [])
    messages = [{"role": c["role"], "content": c["content"]} for c in convs]
    return {"messages": messages}


def build_dataset_slice(all_records: list[dict], n: int) -> Dataset:
    """Return the first *n* records as a HuggingFace Dataset."""
    slice_records = all_records[:n]
    converted = [conversations_to_messages(r) for r in slice_records]
    return Dataset.from_list(converted)


def make_formatting_func(tokenizer, model_key: str):
    """Return a per-example formatter using the model's chat template."""
    def format_example(example):
        messages = example.get("messages", [])
        kwargs = dict(tokenize=False, add_generation_prompt=False)
        # Llama SEA-LION supports thinking_mode
        if model_key == "llama":
            kwargs["thinking_mode"] = "off"
        return tokenizer.apply_chat_template(messages, **kwargs)
    return format_example


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_and_tokenizer(cfg: dict):
    """Load model in 4-bit QLoRA via AutoModelForCausalLM (text-only path)."""
    model_id = cfg["model_id"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"\n🔄  Loading model: {model_id}")
    print(    "    Mode: 4-bit QLoRA | text-only (ViT skipped for Gemma/Qwen)")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        # trust_remote_code needed for Qwen VL architecture files
        trust_remote_code=True,
    )

    # Use AutoTokenizer for all models (works for Gemma, Llama, Qwen)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


# =============================================================================
# SINGLE TRAINING ITERATION
# =============================================================================

def train_one_iteration(
    model_key: str,
    cfg: dict,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    n_rows: int,
    args,
):
    """Train one model on one dataset slice and save the result."""
    ts = timestamp()
    short = cfg["short_name"]
    run_dir = os.path.join(OUTPUT_BASE, f"{short}_{ts}_{n_rows}rows")
    run_name = f"{cfg['run_name_prefix']}-{n_rows}rows"

    epochs = args.epochs or EPOCHS
    lora_rank = args.lora_rank or LORA_RANK
    lora_alpha = (lora_rank * 2)
    max_seq = args.max_seq_length or MAX_SEQ_LENGTH
    batch_size = cfg["batch_size"]
    grad_accum = cfg["grad_accum"]
    lr = cfg["learning_rate"]
    report_to = "none" if args.no_wandb else "wandb"

    print(f"\n{'='*70}")
    print(f"  🚀  {short.upper()} — {n_rows:,} rows")
    print(f"      Output → {run_dir}")
    print(f"{'='*70}")

    # ── Load model fresh for each iteration ──────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(cfg)
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
    print(f"📂  Train: {len(train_dataset):,} | Eval: {len(eval_dataset):,}")

    sft_config = SFTConfig(
        output_dir=run_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
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
        formatting_func=make_formatting_func(tokenizer, model_key),
    )

    print(f"\n   Epochs:          {epochs}")
    print(f"   Batch size:      {batch_size}")
    print(f"   Grad accum:      {grad_accum}")
    print(f"   Effective batch: {batch_size * grad_accum}")
    print(f"   Learning rate:   {lr}")
    print(f"   LoRA rank/alpha: {lora_rank}/{lora_alpha}")
    print(f"   Max seq length:  {max_seq}")
    print(f"   WandB:           {'❌ disabled' if args.no_wandb else '✅ enabled'}")
    print()

    trainer.train()

    # ── Save final adapter ────────────────────────────────────────────────────
    final_path = os.path.join(run_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n✅  Saved → {final_path}")

    # ── Write a small metadata file ───────────────────────────────────────────
    meta_path = os.path.join(run_dir, "training_meta.txt")
    with open(meta_path, "w") as mf:
        mf.write(f"model_id:     {cfg['model_id']}\n")
        mf.write(f"n_rows:       {n_rows}\n")
        mf.write(f"timestamp:    {ts}\n")
        mf.write(f"epochs:       {epochs}\n")
        mf.write(f"lora_rank:    {lora_rank}\n")
        mf.write(f"lora_alpha:   {lora_alpha}\n")
        mf.write(f"batch_size:   {batch_size}\n")
        mf.write(f"grad_accum:   {grad_accum}\n")
        mf.write(f"learning_rate:{lr}\n")
        mf.write(f"max_seq:      {max_seq}\n")

    # ── Release memory before next iteration ──────────────────────────────────
    del model, trainer
    free_gpu_memory()


# =============================================================================
# MAIN LOOP
# =============================================================================

def build_iteration_sizes(total: int, start: int, step: int) -> list[int]:
    """Build the list [start, start+step, ..., last_multiple, total]."""
    sizes = list(range(start, total, step))
    # Always include the full dataset at the end
    if not sizes or sizes[-1] != total:
        sizes.append(total)
    return sizes


def run_all(args):
    # ── Resolve which models to train ────────────────────────────────────────
    if args.model:
        keys = [args.model.lower()]
        if keys[0] not in MODEL_CONFIGS:
            print(f"❌  Unknown model '{keys[0]}'. Choose from: {list(MODEL_CONFIGS)}")
            sys.exit(1)
    else:
        keys = list(MODEL_CONFIGS.keys())   # gemma → llama → qwen

    step = args.step or ITERATION_STEP
    start = args.start or ITERATION_START
    data_file = args.data_file or DATA_FILE

    # ── Load entire dataset once ──────────────────────────────────────────────
    print(f"\n📖  Loading dataset from: {data_file}")
    all_records = load_jsonl(data_file)
    total = len(all_records)
    print(f"    Total records: {total:,}")

    # Build iteration schedule
    sizes = build_iteration_sizes(total, start, step)
    print(f"\n📋  Iteration schedule ({len(sizes)} runs per model):")
    for i, s in enumerate(sizes, 1):
        print(f"    [{i:2d}] {s:>7,} rows")

    # Reserve 10 % of the FULL dataset for eval (shared across all iterations)
    eval_size = max(500, int(total * 0.10))
    eval_records = all_records[-eval_size:]          # last 10% as eval
    train_pool = all_records[:-eval_size]            # first 90% as train pool
    eval_dataset = Dataset.from_list(
        [conversations_to_messages(r) for r in eval_records]
    )
    print(f"\n🗂️   Eval set: {len(eval_dataset):,} rows (last 10% of dataset)")
    print(f"    Train pool: {len(train_pool):,} rows (first 90%)")

    # Adjust sizes that exceed the train pool
    sizes = [min(s, len(train_pool)) for s in sizes]
    sizes = sorted(set(sizes))   # deduplicate if capping caused duplicates

    # ── Iterate over models ───────────────────────────────────────────────────
    grand_total = len(keys) * len(sizes)
    run_idx = 0

    for model_key in keys:
        cfg = MODEL_CONFIGS[model_key]
        print(f"\n\n{'#'*70}")
        print(f"#  Model: {cfg['model_id']}")
        print(f"#  Iterations: {len(sizes)}")
        print(f"{'#'*70}")

        for n_rows in sizes:
            run_idx += 1
            print(f"\n[Run {run_idx}/{grand_total}]")

            train_dataset = build_dataset_slice(train_pool, n_rows)

            try:
                train_one_iteration(
                    model_key=model_key,
                    cfg=cfg,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    n_rows=n_rows,
                    args=args,
                )
            except Exception as exc:
                print(f"\n⚠️   Run {run_idx} FAILED: {exc}")
                print("    Continuing to next iteration …\n")
                free_gpu_memory()
                continue

    print(f"\n\n{'='*70}")
    print(f"  🎉  All done! {run_idx} training run(s) completed.")
    print(f"      Outputs are in: {os.path.abspath(OUTPUT_BASE)}/")
    print(f"{'='*70}\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Iterative QLoRA training for Gemma / Llama / Qwen SEA-LION models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Train only this model: gemma | llama | qwen  (default: all)"
    )
    parser.add_argument(
        "--data-file", type=str, default=None,
        help=f"Path to JSONL dataset (default: {DATA_FILE})"
    )
    parser.add_argument(
        "--step", type=int, default=None,
        help=f"Row increment between iterations (default: {ITERATION_STEP})"
    )
    parser.add_argument(
        "--start", type=int, default=None,
        help=f"First slice size (default: {ITERATION_START})"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=f"Training epochs per run (default: {EPOCHS})"
    )
    parser.add_argument(
        "--lora-rank", type=int, default=None,
        help=f"LoRA rank (default: {LORA_RANK})"
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=None,
        help=f"Maximum sequence length (default: {MAX_SEQ_LENGTH})"
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable WandB logging (use report_to='none')"
    )
    args = parser.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
