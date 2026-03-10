#!/usr/bin/env python3
"""
Merge LoRA weights into Llama-SEA-LION-v3.5-8B-R base model.

Usage:
  python llama/merge_weights.py
  python llama/merge_weights.py --adapter-path ./output/llama-8b/final --merged-path ./llama-merged
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "aisingapore/Llama-SEA-LION-v3.5-8B-R"
ADAPTER_PATH = "./output/llama-8b/final"
MERGED_PATH = "./output/llama-8b-merged"


def merge(adapter_path, merged_path):
    print(f"🔄 Loading base model: {MODEL_ID}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"🔄 Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("🔄 Merging...")
    merged = model.merge_and_unload()

    print(f"💾 Saving to: {merged_path}")
    merged.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print("✅ Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", default=ADAPTER_PATH)
    parser.add_argument("--merged-path", default=MERGED_PATH)
    args = parser.parse_args()
    merge(args.adapter_path, args.merged_path)


if __name__ == "__main__":
    main()
