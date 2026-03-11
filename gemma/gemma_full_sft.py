#!/usr/bin/env python3
"""
Full SFT (no QLoRA) for Gemma-SEA-LION-v4-4B-VL.

All parameters are trained in bf16. Requires ~32GB+ VRAM.

Usage:
  python gemma/gemma_full_sft.py
  python gemma/gemma_full_sft.py --epochs 2 --lr 2e-5
  accelerate launch gemma/gemma_full_sft.py
"""

import argparse
import torch
from datasets import load_dataset
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from trl import SFTTrainer, SFTConfig

# =============================================================================
# DEFAULTS
# =============================================================================
MODEL_ID = "aisingapore/Gemma-SEA-LION-v4-4B-VL"
OUTPUT_DIR = "./output/gemma-4b-full-sft"
DATA_DIR = "./data"
MAX_SEQ_LENGTH = 512
EPOCHS = 1
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16
LEARNING_RATE = 2e-5


def load_model_and_tokenizer():
    """Load model in bf16 — all parameters trainable."""
    print(f"🔄 Loading model: {MODEL_ID} (bf16, full parameters)")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params:     {total_params / 1e9:.2f}B")
    print(f"   Trainable params: {trainable / 1e9:.2f}B (100%)")

    return model, tokenizer


def make_formatting_func(tokenizer):
    """Format examples using Gemma's chat template."""
    def format_example(example):
        if "messages" in example:
            messages = example["messages"]
        else:
            messages = []
            user_content = example["instruction"]
            if example.get("input"):
                user_content += f"\n\n{example['input']}"
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": example["output"]})

        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

    return format_example


def train(args):
    epochs = args.epochs or EPOCHS
    lr = args.lr or LEARNING_RATE
    batch_size = args.batch_size or BATCH_SIZE
    output_dir = args.output_dir or OUTPUT_DIR
    data_dir = args.data_dir or DATA_DIR
    max_seq = args.max_seq_length or MAX_SEQ_LENGTH

    model, tokenizer = load_model_and_tokenizer()
    model.gradient_checkpointing_enable()

    dataset = load_dataset("json", data_files={
        "train": f"{data_dir}/train.jsonl",
        "eval": f"{data_dir}/eval.jsonl",
    })
    print(f"\n📂 Train: {len(dataset['train'])} | Eval: {len(dataset['eval'])}")

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=3,
        bf16=True,
        optim="adamw_torch",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to="wandb",
        run_name="gemma-sealion-4b-full-sft",
        max_length=max_seq,
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
        formatting_func=make_formatting_func(tokenizer),
    )

    print(f"\n🚀 Full SFT — Gemma-SEA-LION-v4-4B-VL")
    print(f"   Epochs:          {epochs}")
    print(f"   Batch size:      {batch_size}")
    print(f"   Grad accum:      {GRADIENT_ACCUMULATION}")
    print(f"   Effective batch: {batch_size * GRADIENT_ACCUMULATION}")
    print(f"   Learning rate:   {lr}")
    print(f"   Max seq length:  {max_seq}")
    print(f"   WandB:           ✅ enabled")
    print()

    trainer.train()

    final_path = f"{output_dir}/final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n✅ Model saved to: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Full SFT for Gemma-SEA-LION-v4-4B-VL")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
