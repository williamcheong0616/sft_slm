#!/usr/bin/env python3
"""
QLoRA fine-tuning for Gemma-SEA-LION-v4-4B-VL.

Model:  aisingapore/Gemma-SEA-LION-v4-4B-VL
Arch:   Gemma 3 (4B params)
VRAM:   ~6-8 GB with QLoRA (fits RTX 3090 24GB easily)

Usage:
  python gemma/train.py
  python gemma/train.py --epochs 2 --lr 1e-4 --batch-size 4
"""

import argparse
import torch
from datasets import load_dataset
from transformers import (
    Gemma3ForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# =============================================================================
# DEFAULTS
# =============================================================================
MODEL_ID = "aisingapore/Gemma-SEA-LION-v4-4B-VL"
OUTPUT_DIR = "./output/gemma-4b"
DATA_DIR = "./data"
MAX_SEQ_LENGTH = 512
EPOCHS = 1
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4     # Effective batch = 4 * 4 = 16
LEARNING_RATE = 2e-4
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_model_and_tokenizer():
    """Load model in 4-bit quantization for QLoRA."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"🔄 Loading model: {MODEL_ID} (4-bit QLoRA)")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

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
    lora_rank = args.lora_rank or LORA_RANK
    lora_alpha = (args.lora_rank * 2) if args.lora_rank else LORA_ALPHA
    output_dir = args.output_dir or OUTPUT_DIR
    data_dir = args.data_dir or DATA_DIR
    max_seq = args.max_seq_length or MAX_SEQ_LENGTH

    model, tokenizer = load_model_and_tokenizer()

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, peft_config)

    print("\n📊 Trainable parameters:")
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files={
        "train": f"{data_dir}/train.jsonl",
        "eval": f"{data_dir}/eval.jsonl",
    })
    print(f"📂 Train: {len(dataset['train'])} | Eval: {len(dataset['eval'])}")

    training_args = TrainingArguments(
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
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        report_to="wandb",
        run_name="gemma-sealion-4b-qlora",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        tokenizer=tokenizer,
        formatting_func=make_formatting_func(tokenizer),
        max_seq_length=max_seq,
        packing=True,
    )

    print(f"\n🚀 QLoRA Training — Gemma-SEA-LION-v4-4B-VL")
    print(f"   Epochs:          {epochs}")
    print(f"   Batch size:      {batch_size}")
    print(f"   Grad accum:      {GRADIENT_ACCUMULATION}")
    print(f"   Effective batch: {batch_size * GRADIENT_ACCUMULATION}")
    print(f"   Learning rate:   {lr}")
    print(f"   LoRA rank:       {lora_rank} (alpha: {lora_alpha})")
    print(f"   Max seq length:  {max_seq}")
    print(f"   WandB:           ✅ enabled")
    print()

    trainer.train()

    final_path = f"{output_dir}/final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n✅ Model saved to: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Gemma-SEA-LION-v4-4B-VL")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
