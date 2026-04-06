import json
import os

filepath = '/Users/williamcheong/Desktop/sft_slm/test.json'
output_dir = os.path.dirname(filepath)

with open(filepath, 'r') as f:
    data = json.load(f)

for key, value in data.items():
    output_path = os.path.join(output_dir, f"{key}.json")
    with open(output_path, 'w') as f:
        json.dump(value, f, indent=2)
    print(f"Created {output_path}")
