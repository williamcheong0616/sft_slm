#!/usr/bin/env python3
"""
Inference with fine-tuned Qwen-SEA-LION-v4-8B-VL.

Usage:
  python qwen/inference.py --prompt "Explain what SEALion is"
  python qwen/inference.py --interactive
"""

import argparse
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

MODEL_ID = "aisingapore/Qwen-SEA-LION-v4-8B-VL"
ADAPTER_PATH = "./output/qwen-8b/final"


def load_model(adapter_path):
    print(f"🔄 Loading {MODEL_ID} + adapter...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ Ready!\n")
    return model, tokenizer


def generate(model, tokenizer, instruction, input_ctx="",
             max_new_tokens=512, temperature=0.7):
    messages = [{"role": "user", "content": instruction + (
        f"\n\n{input_ctx}" if input_ctx else ""
    )}]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=0.9,
            repetition_penalty=1.1, do_sample=True,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def interactive(model, tokenizer):
    print("=" * 60)
    print("🤖 Qwen-SEA-LION Interactive — type 'quit' to stop")
    print("=" * 60)
    while True:
        try:
            instr = input("\n📝 Instruction: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if instr.lower() in ("quit", "exit", "q"):
            break
        if not instr:
            continue
        ctx = input("📎 Context (Enter to skip): ").strip()
        print(f"\n🤖 {generate(model, tokenizer, instr, ctx)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", "-p", type=str)
    parser.add_argument("--input", "-i", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--adapter-path", default=ADAPTER_PATH)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter_path)

    if args.interactive:
        interactive(model, tokenizer)
    elif args.prompt:
        print(f"🤖 {generate(model, tokenizer, args.prompt, args.input, args.max_tokens, args.temperature)}")
    else:
        print("Provide --prompt or --interactive")


if __name__ == "__main__":
    main()
