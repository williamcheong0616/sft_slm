#!/usr/bin/env python3
"""
Evaluate fine-tuned Qwen-SEA-LION-v4-8B-VL.

Usage:
  python qwen/evaluate.py
  python qwen/evaluate.py --eval-file data/eval.jsonl --num-samples 100
"""

import argparse
import json
import time
import torch
from pathlib import Path
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
    return model, tokenizer


def generate_response(model, tokenizer, instruction, input_ctx=""):
    messages = [{"role": "user", "content": instruction + (
        f"\n\n{input_ctx}" if input_ctx else ""
    )}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, top_p=0.9, do_sample=True)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def extract(example):
    if "messages" in example:
        instr, exp = "", ""
        for m in example["messages"]:
            c = m["content"]
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if x.get("type") == "text")
            if m["role"] == "user":
                instr = c
            elif m["role"] == "assistant":
                exp = c
        return instr, "", exp
    return example.get("instruction", ""), example.get("input", ""), example.get("output", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", default="./data/eval.jsonl")
    parser.add_argument("--adapter-path", default=ADAPTER_PATH)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output-file", "-o")
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter_path)
    path = Path(args.eval_file)
    if not path.exists():
        print(f"❌ Not found: {args.eval_file}")
        return

    with open(path, "r") as f:
        examples = [json.loads(l) for l in f if l.strip()][:args.num_samples]

    results = []
    t0 = time.time()
    for i, ex in enumerate(examples):
        instr, ctx, expected = extract(ex)
        if not instr:
            continue
        generated = generate_response(model, tokenizer, instr, ctx)
        results.append({"instruction": instr, "expected": expected, "generated": generated,
                        "exact_match": expected.strip().lower() == generated.strip().lower()})
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(examples)}]")

    elapsed = time.time() - t0
    matches = sum(1 for r in results if r["exact_match"])
    overlaps = []
    for r in results:
        ref = set(r["expected"].lower().split())
        if ref:
            overlaps.append(len(set(r["generated"].lower().split()) & ref) / len(ref))

    print(f"\n{'='*60}")
    print(f"📊 Qwen-SEA-LION-v4-8B-VL — {len(results)} examples, {elapsed:.1f}s")
    print(f"   Exact match: {matches}/{len(results)} ({matches/len(results):.1%})")
    print(f"   Token overlap: {sum(overlaps)/len(overlaps):.1%}" if overlaps else "")
    for r in results[:5]:
        icon = "✅" if r["exact_match"] else "❌"
        print(f"\n  {icon} {r['instruction'][:80]}")
        print(f"     Expected:  {r['expected'][:100]}")
        print(f"     Generated: {r['generated'][:100]}")

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump({"results": results}, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved to: {args.output_file}")


if __name__ == "__main__":
    main()
