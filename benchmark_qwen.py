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
import qwen.inference as qwen_inf

def load_data(file_path, num_samples=500):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # Use fixed seed for reproducibility
    random.seed(42)
    sampled = random.sample(data, min(num_samples, len(data)))
    print(f"Sampled {len(sampled)} items.")
    return sampled

def run_benchmark(data):
    print(f"\n=============================================")
    print(f"=== Starting benchmark for QWEN ===")
    print(f"=============================================")
    
    base_model = "aisingapore/Qwen-SEA-LION-v4-8B-VL"
    adapter_path = "./output/qwen-8b/final"
    db_path = "benchmark_qwen.db"
    excel_path = "benchmark_qwen.xlsx"
    
    # Try to load model
    try:
        model, tokenizer = qwen_inf.load_model(adapter_path, base_model)
    except Exception as e:
        print(f"Failed to load Qwen: {e}")
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
            model_response = qwen_inf.generate(model, tokenizer, question)
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
    print(f"Saving to {db_path}...")
    conn = sqlite3.connect(db_path)
    df.to_sql('benchmark', conn, if_exists='replace', index=False)
    conn.close()
    
    # Save to Excel
    print(f"Saving to {excel_path}...")
    try:
        df.to_excel(excel_path, index=False)
    except ModuleNotFoundError:
        print("openpyxl not found. Please install via: pip install openpyxl")
        csv_path = excel_path.replace('.xlsx', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved as CSV instead: {csv_path}")
    
    print(f"=== Finished benchmark for Qwen ===")
    
    # Cleanup memory
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Run benchmark for Qwen")
    parser.add_argument("--data", default="curated_data.jsonl", help="Path to data file")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to evaluate")
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"Error: Data file {args.data} not found.")
        return
        
    data = load_data(args.data, args.samples)
    run_benchmark(data)

if __name__ == "__main__":
    main()
