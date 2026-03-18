#!/usr/bin/env python3
"""
Interactive inference for an explicitly fully-merged model.

Usage:
  python scripts/inference_merged.py --model-path llama-8b/merged --model-type llama
  python scripts/inference_merged.py --model-path gemma-4b/merged --model-type gemma
  python scripts/inference_merged.py --model-path qwen-8b/merged --model-type qwen
"""

import argparse
import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    AutoProcessor
)

def main():
    parser = argparse.ArgumentParser(description="Run inference on a merged Llama, Gemma, or Qwen model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the directory containing merged weights")
    parser.add_argument("--model-type", type=str, required=True, choices=["llama", "gemma", "qwen"], help="Which model to run")
    parser.add_argument("--device", type=str, default="auto", help="Device map (e.g. 'auto', 'cuda:0')")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    
    args = parser.parse_args()

    print(f"🔄 Loading {args.model_type} merged model from: {args.model_path}")
    
    try:
        if args.model_type == "llama":
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map=args.device,
                low_cpu_mem_usage=True
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            
        elif args.model_type == "gemma":
            model = Gemma3ForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map=args.device,
                low_cpu_mem_usage=True
            )
            processor = AutoProcessor.from_pretrained(args.model_path)
            tokenizer = processor.tokenizer
            
        elif args.model_type == "qwen":
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map=args.device,
                low_cpu_mem_usage=True
            )
            processor = AutoProcessor.from_pretrained(args.model_path)
            tokenizer = processor.tokenizer
            
    except Exception as e:
        print(f"❌ Error loading model! Are you sure {args.model_path} contains the fully merged weights for a {args.model_type} model?")
        print(f"Details: {e}")
        sys.exit(1)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model.eval()
    print("✅ Ready!\n")

    print("=" * 70)
    print(f"🤖 {args.model_type.capitalize()} Merged Model Interactive — type 'quit' to stop")
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
            
            # Apply chat template and turn off thinking mode for all models
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, thinking_mode="off")
                
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=args.max_new_tokens,
                    temperature=max(1e-5, args.temperature), 
                    top_p=0.9,
                    top_k=50,
                    do_sample=True,
                )
            
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            print(f"\n🤖 {response}")
            
        except (EOFError, KeyboardInterrupt):
            break

if __name__ == "__main__":
    main()
