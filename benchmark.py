#!/usr/bin/env python3
import json
import sqlite3
import pandas as pd
import argparse
import random
import torch
import gc
import os

# Import user's inference modules
import gemma.inference as gemma_inf
import qwen.inference as qwen_inf
import llama.inference as llama_inf

MODELS = {
    "gemma": {
        "module": gemma_inf,
        "base": "aisingapore/Gemma-SEA-LION-v4-4B-VL",
        "adapter": "./output/gemma-4b/final",
        "db": "benchmark_gemma.db",
        "excel": "benchmark_gemma.xlsx"
    },
    "qwen": {
        "module": qwen_inf,
        "base": "aisingapore/Qwen-SEA-LION-v4-8B-VL",
        "adapter": "./output/qwen-8b/final",
        "db": "benchmark_qwen.db",
        "excel": "benchmark_qwen.xlsx"
    },
    "llama": {
        "module": llama_inf,
        "base": "aisingapore/Llama-SEA-LION-v3.5-8B-R",
        "adapter": "./output/llama-8b/final",
        "db": "benchmark_llama.db",
        "excel": "benchmark_llama.xlsx"
    }
}

def load_data(file_path, num_samples=500):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # Use fixed seed for reproducibility across different model runs
    random.seed(42)
    sampled = random.sample(data, min(num_samples, len(data)))
    print(f"Sampled {len(sampled)} items.")
    return sampled

def run_benchmark(model_name, data):
    cfg = MODELS[model_name]
    print(f"\n=============================================")
    print(f"=== Starting benchmark for {model_name.upper()} ===")
    print(f"=============================================")
    
    # Try to load model
    try:
        model, tokenizer = cfg["module"].load_model(cfg["adapter"], cfg["base"])
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return
    
    results = []
    
    for i, item in enumerate(data):
        conversations = item.get("conversations", [])
        if len(conversations) < 2:
            continue
            
        question = conversations[0]["content"]
        original_comment = conversations[1]["content"]
        
        print(f"[{i+1}/{len(data)}] Q: {question[:50]}...")
        try:
            # We don't use input_ctx for curated_data.jsonl as it's not present
            model_response = cfg["module"].generate(model, tokenizer, question)
        except Exception as e:
            print(f"Error generating response: {e}")
            model_response = f"ERROR: {e}"
            
        results.append({
            "Question": question,
            "Original_Comment": original_comment,
            "Model_Response": model_response
        })
        
    df = pd.DataFrame(results)
    
    # Save to SQLite
    print(f"Saving to {cfg['db']}...")
    conn = sqlite3.connect(cfg['db'])
    df.to_sql('benchmark', conn, if_exists='replace', index=False)
    conn.close()
    
    # Save to Excel
    print(f"Saving to {cfg['excel']}...")
    # Openpyxl is required for DataFrame.to_excel
    try:
        df.to_excel(cfg['excel'], index=False)
    except ModuleNotFoundError:
        print("openpyxl not found. Please install via: pip install openpyxl")
        df.to_csv(cfg['excel'].replace('.xlsx', '.csv'), index=False)
        print(f"Saved as CSV instead: {cfg['excel'].replace('.xlsx', '.csv')}")
    
    print(f"=== Finished benchmark for {model_name} ===")
    
    # Cleanup memory
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Run benchmark on fine-tuned models")
    parser.add_argument("--data", default="curated_data.jsonl", help="Path to data file")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to evaluate")
    parser.add_argument("--model", choices=["gemma", "qwen", "llama", "all"], default="all", help="Which model to run")
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"Error: Data file {args.data} not found.")
        return
        
    data = load_data(args.data, args.samples)
    
    if args.model == "all":
        for model_name in MODELS.keys():
            run_benchmark(model_name, data)
    else:
        run_benchmark(args.model, data)

if __name__ == "__main__":
    main()
