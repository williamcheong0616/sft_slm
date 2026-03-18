#!/usr/bin/env python3
"""
Interactive inference for the original Qwen-SEA-LION-v4-8B-VL base model.

Usage:
  python scripts/inference_qwen_original.py
"""

import argparse
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL_ID = "aisingapore/Qwen-SEA-LION-v4-8B-VL"

def main():
    parser = argparse.ArgumentParser(description="Run inference on the original Qwen base model.")
    parser.add_argument("--model-id", type=str, default=MODEL_ID, help="Base model ID")
    parser.add_argument("--device", type=str, default="auto", help="Device map")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    
    args = parser.parse_args()

    print(f"🔄 Loading original model: {args.model_id}")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = processor.tokenizer
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model.eval()
    print("✅ Ready!\n")

    print("=" * 70)
    print("🤖 Original Qwen Model Interactive — type 'quit' to stop")
    print("=" * 70)
    
    while True:
        try:
            instruction = input("\n📝 Instruction: ").strip()
            if instruction.lower() in ("quit", "exit", "q"):
                break
            if not instruction:
                continue
            
            context = input("📎 Context (Enter to skip): ").strip()
            
            messages = [{"role": "user", "content": instruction + (f"\n\n{context}" if context else "")}]
            
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=args.max_new_tokens,
                    temperature=max(1e-5, args.temperature), 
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    do_sample=True,
                )
            
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            print(f"\n🤖 {response}")
            
        except (EOFError, KeyboardInterrupt):
            break

if __name__ == "__main__":
    main()
