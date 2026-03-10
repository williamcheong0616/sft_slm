#!/usr/bin/env python3
"""
Merge LoRA weights into Gemma-SEA-LION-v4-4B-VL base model.

Usage:
  python gemma/merge_weights.py
  python gemma/merge_weights.py --adapter-path ./output/gemma-4b/final --merged-path ./gemma-merged
"""

import argparse
import torch
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from peft import PeftModel

MODEL_ID = "aisingapore/Gemma-SEA-LION-v4-4B-VL"
ADAPTER_PATH = "./output/gemma-4b/final"
MERGED_PATH = "./output/gemma-4b-merged"


def merge(adapter_path, merged_path):
    print(f"🔄 Loading base model: {MODEL_ID}")
    base_model = Gemma3ForConditionalGeneration.from_pretrained(
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
