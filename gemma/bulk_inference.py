#!/usr/bin/env python3
"""
Inference with QLoRA fine-tuned Gemma-SEA-LION-v4-4B-VL.

Usage:
  python inference.py --prompt "Explain what SEALion is"
  python inference.py --interactive
  python inference.py --bulk-generate 200 --output-file "tts_dataset.txt"
"""

import argparse
import torch
import random
import os
from tqdm import tqdm
from transformers import Gemma3ForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "aisingapore/Gemma-SEA-LION-v4-4B-VL"
ADAPTER_PATH = "./output/gemma-4b/final"

def load_model(adapter_path, base_model_id):
    print(f"🔄 Loading base model: {base_model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = Gemma3ForConditionalGeneration.from_pretrained(
        base_model_id, quantization_config=bnb_config, device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(base_model_id)
    tokenizer = processor.tokenizer

    print(f"🔗 Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ Ready!\n")
    return model, tokenizer


def generate(model, tokenizer, instruction, input_ctx="", max_new_tokens=512, temperature=0.7):
    messages = [{"role": "user", "content": instruction + (
        f"\n\n{input_ctx}" if input_ctx else ""
    )}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=max(1e-5, temperature), top_p=0.9, top_k=50,
            do_sample=True,
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


def bulk_generate_tts(model, tokenizer, num_scripts, output_file):
    print(f"🚀 Starting bulk generation of {num_scripts} TTS scripts...")
    
    # Diverse parameters to ensure variety across the 200 runs
    topics = [
        "Retail Fussy Customer", "POS System Crash", "Federal Highway Jam", 
        "LRT Delay Problem", "Endless Office Meeting", "Boss Deadline Overwork", 
        "Diet Fail Nasi Kandar", "Gadget Battery Kong", "Sudden Heavy Rain", 
        "Closing Shift Balance", "Stocktake Backroom", "Fitting Room Mess", 
        "Angry Caller Delay", "Refund Karen Escalation", "System Glitch Tech Support"
    ]
    emotions = ["Frustrated", "Annoyed", "Exhausted", "Panicked", "Stressed", "Resigned", "Anxious"]
    openers_variety = ["Serius ah", "Biar betik", "Korang tau tak", "Mak aih", "Macam ni ceritanya", "Faham tak", "Letih do"]

    system_persona = (
        "System Role & Objective:\n"
        "You are an expert Malaysian scriptwriter and linguist specializing in authentic colloquial "
        "'Bahasa Rojak' and 'Manglish'. Generate a high-quality Text-to-Speech (TTS) training script "
        "that perfectly captures the natural, unfiltered rhythm of street-level Malaysian conversations.\n\n"
        "Constraints:\n"
        "- Word Count: Exactly 70 to 80 words.\n"
        "- Perspective: First-person monologue.\n"
        "- Authentic Code-Switching: Seamlessly blend English vocabulary into Malay sentence structures. Use Malay prefixes/suffixes on English words (e.g., men-debug, revert back).\n"
        "- Dropped Vowels & Contractions: Spell words as they are spoken casually (e.g., macam mana -> camne, dekat -> kat, dia orang -> diorang).\n"
        "- Particles: Use do, weh, siot, baii, la, kan, eh naturally.\n"
        "- Minimal Profanity: Keep it gritty but exclude heavy slurs. Use milder expressions (aduh, gila, bapak ah, siot).\n"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Dataset: 200 Bahasa Rojak & Manglish TTS Scripts\n")
        f.write("Model: Gemma-SEA-LION-v4-4B-VL (Fine-tuned)\n")
        f.write("="*60 + "\n\n")

    for i in tqdm(range(1, num_scripts + 1), desc="Generating Scripts"):
        topic = random.choice(topics)
        emotion = random.choice(emotions)
        opener = random.choice(openers_variety)

        instruction = (
            f"{system_persona}\n"
            f"Task: Generate ONE script about '{topic}'. The tone should be '{emotion}'.\n"
            f"Please ensure it starts naturally with an expression like '{opener}' or similar variety, "
            f"and strict adherence to the 70-80 word count limit."
        )

        # Generate script using the model
        script_content = generate(model, tokenizer, instruction, max_new_tokens=150, temperature=0.85)

        # Format output
        output_block = f"Skrip {i}: {topic} ({emotion})\n{script_content}\n\n"
        
        # Save incrementally to avoid losing data if the script crashes
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(output_block)

    print(f"\n✅ Bulk generation complete. Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", "-p", type=str)
    parser.add_argument("--input", "-i", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--bulk-generate", type=int, help="Number of scripts to generate (e.g., 200)")
    parser.add_argument("--output-file", type=str, default="generated_tts_scripts.txt", help="File to save bulk generation")
    parser.add_argument("--adapter-path", default=ADAPTER_PATH)
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter_path, args.base_model)

    if args.interactive:
        interactive(model, tokenizer)
    elif args.bulk_generate:
        bulk_generate_tts(model, tokenizer, args.bulk_generate, args.output_file)
    elif args.prompt:
        print(f"🤖 {generate(model, tokenizer, args.prompt, args.input, args.max_tokens, args.temperature)}")
    else:
        print("Provide --prompt, --interactive, or --bulk-generate")

if __name__ == "__main__":
    main()