#!/usr/bin/env python3
"""
Merge LoRA weights into Qwen-SEA-LION-v4-8B-VL base model.

Usage:
  python qwen/merge_weights.py
  python qwen/merge_weights.py --adapter-path ./output/qwen-8b/final --merged-path ./qwen-merged
"""

import argparse
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

MODEL_ID = "aisingapore/Qwen-SEA-LION-v4-8B-VL"
ADAPTER_PATH = "./output/qwen-8b/final"
MERGED_PATH = "./output/qwen-8b-merged"


def merge(adapter_path, merged_path):
    print(f"🔄 Loading base model: {MODEL_ID}")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoProcessor.from_pretrained(MODEL_ID).tokenizer

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
