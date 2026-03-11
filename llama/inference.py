#!/usr/bin/env python3
"""
Inference with QLoRA fine-tuned Llama-SEA-LION-v3.5-8B-R.

Usage:
  python llama/inference.py --prompt "Macam mana AI boleh bantu cikgu?"
  python llama/inference.py --interactive
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "aisingapore/Llama-SEA-LION-v3.5-8B-R"
ADAPTER_PATH = "./output/llama-8b/final"


def load_model(adapter_path, base_model_id):
    print(f"🔄 Loading base model: {base_model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb_config, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    print(f"🔗 Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
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
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=max(1e-5, temperature), top_p=0.9, top_k=50,
            repetition_penalty=1.1, do_sample=True,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def interactive(model, tokenizer):
    print("=" * 60)
    print("🤖 Llama-SEA-LION Interactive — type 'quit' to stop")
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
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter_path, args.base_model)

    if args.interactive:
        interactive(model, tokenizer)
    elif args.prompt:
        print(f"🤖 {generate(model, tokenizer, args.prompt, args.input, args.max_tokens, args.temperature)}")
    else:
        print("Provide --prompt or --interactive")


if __name__ == "__main__":
    main()
