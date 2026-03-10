#!/usr/bin/env python3
"""Evaluate QLoRA fine-tuned Llama-SEA-LION-v3.5-8B-R."""

import argparse, json, time, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "aisingapore/Llama-SEA-LION-v3.5-8B-R"
ADAPTER_PATH = "./output/llama-8b/final"


def load_model(adapter_path, base_model_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb_config, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def gen(model, tokenizer, instr, ctx=""):
    msgs = [{"role": "user", "content": instr + (f"\n\n{ctx}" if ctx else "")}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, temperature=0.1, top_p=0.9, do_sample=True)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def extract(ex):
    if "messages" in ex:
        instr, exp = "", ""
        for m in ex["messages"]:
            c = m["content"]
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if x.get("type") == "text")
            if m["role"] == "user": instr = c
            elif m["role"] == "assistant": exp = c
        return instr, "", exp
    return ex.get("instruction", ""), ex.get("input", ""), ex.get("output", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", default="./data/eval.jsonl")
    parser.add_argument("--adapter-path", default=ADAPTER_PATH)
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output-file", "-o")
    args = parser.parse_args()

    print(f"🔄 Loading model + adapter...")
    model, tokenizer = load_model(args.adapter_path, args.base_model)

    path = Path(args.eval_file)
    if not path.exists():
        print(f"❌ Not found: {args.eval_file}"); return

    with open(path) as f:
        examples = [json.loads(l) for l in f if l.strip()][:args.num_samples]

    results, t0 = [], time.time()
    for i, ex in enumerate(examples):
        instr, ctx, expected = extract(ex)
        if not instr: continue
        generated = gen(model, tokenizer, instr, ctx)
        results.append({"instruction": instr, "expected": expected, "generated": generated,
                        "exact_match": expected.strip().lower() == generated.strip().lower()})
        if (i + 1) % 10 == 0: print(f"  [{i+1}/{len(examples)}]")

    elapsed = time.time() - t0
    matches = sum(1 for r in results if r["exact_match"])
    overlaps = []
    for r in results:
        ref = set(r["expected"].lower().split())
        if ref: overlaps.append(len(set(r["generated"].lower().split()) & ref) / len(ref))

    print(f"\n{'='*60}")
    print(f"📊 Llama-SEA-LION — {len(results)} examples, {elapsed:.1f}s")
    print(f"   Exact match:   {matches}/{len(results)} ({matches/len(results):.1%})")
    if overlaps: print(f"   Token overlap:  {sum(overlaps)/len(overlaps):.1%}")
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
