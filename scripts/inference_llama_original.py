#!/usr/bin/env python3
"""
Interactive inference for the original Llama-SEA-LION-v3.5-8B-R base model.

Usage:
  python scripts/inference_llama_original.py
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "aisingapore/Llama-SEA-LION-v3.5-8B-R"

def main():
    parser = argparse.ArgumentParser(description="Run inference on the original Llama base model.")
    parser.add_argument("--model-id", type=str, default=MODEL_ID, help="Base model ID")
    parser.add_argument("--device", type=str, default="auto", help="Device map")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    
    args = parser.parse_args()

    print(f"🔄 Loading original model: {args.model_id}")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model.eval()
    print("✅ Ready!\n")

    print("=" * 70)
    print("🤖 Original Llama Model Interactive — type 'quit' to stop")
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
            
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, thinking_mode="off")
                
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
