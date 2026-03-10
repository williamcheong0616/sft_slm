#!/usr/bin/env python3
"""
Inference with fine-tuned Gemma-SEA-LION-v4-4B-VL (full SFT).

Usage:
  python gemma/inference.py --prompt "Explain what SEALion is"
  python gemma/inference.py --interactive
"""

import argparse
import torch
from transformers import Gemma3ForConditionalGeneration, AutoProcessor

MODEL_PATH = "./output/gemma-4b/final"


def load_model(model_path):
    print(f"🔄 Loading fine-tuned model: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
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
    print("🤖 Gemma-SEA-LION Interactive — type 'quit' to stop")
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
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)

    if args.interactive:
        interactive(model, tokenizer)
    elif args.prompt:
        print(f"🤖 {generate(model, tokenizer, args.prompt, args.input, args.max_tokens, args.temperature)}")
    else:
        print("Provide --prompt or --interactive")


if __name__ == "__main__":
    main()
