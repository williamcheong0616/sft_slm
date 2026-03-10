#!/usr/bin/env python3
"""Merge QLoRA adapter into base Llama-SEA-LION model."""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "aisingapore/Llama-SEA-LION-v3.5-8B-R"
ADAPTER_PATH = "./output/llama-8b/final"
MERGED_PATH = "./output/llama-8b/merged"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", default=ADAPTER_PATH)
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--output", "-o", default=MERGED_PATH)
    args = parser.parse_args()

    print(f"🔄 Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"🔗 Loading adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    print("🔀 Merging weights...")
    model = model.merge_and_unload()

    print(f"💾 Saving to: {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("✅ Done!")


if __name__ == "__main__":
    main()
