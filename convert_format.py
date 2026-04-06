import json
import os

files = ['random_prompts.json', 'personalized_prompts.json', 'long_test_prompts.json']
output_dir = '/Users/williamcheong/Desktop/sft_slm'

for fname in files:
    filepath = os.path.join(output_dir, fname)
    if not os.path.exists(filepath):
        continue
    
    # Read the current list of strings
    cwd_data = None
    with open(filepath, 'r') as f:
        cwd_data = json.load(f)
    
    # Convert and overwrite
    if isinstance(cwd_data, list):
        with open(filepath, 'w') as f:
            for prompt_str in cwd_data:
                obj = {
                    "conversations": [
                        {"role": "user", "content": prompt_str},
                        {"role": "assistant", "content": ""}
                    ]
                }
                f.write(json.dumps(obj) + '\n')
        print(f"Converted {fname} to JSONL format.")
