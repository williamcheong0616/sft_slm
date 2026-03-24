import json

def combine_files(file1_path, file2_path, output_path):
    current_id = 1
    
    print(f"Combining {file1_path} and {file2_path} into {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # Process first file
        print(f"Processing {file1_path}...")
        with open(file1_path, 'r', encoding='utf-8') as f1:
            for line in f1:
                if not line.strip(): 
                    continue
                data = json.loads(line)
                # Reassign ID to ensure no duplicates
                data['id'] = current_id
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                current_id += 1
                
        # Process second file
        print(f"Processing {file2_path}...")
        with open(file2_path, 'r', encoding='utf-8') as f2:
            for line in f2:
                if not line.strip(): 
                    continue
                data = json.loads(line)
                # Reassign ID to continue from the last ID
                data['id'] = current_id
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                current_id += 1
                
    print(f"Success! Combined {current_id - 1} records into {output_path}")

if __name__ == "__main__":
    # Define file paths
    FILE_1 = "curated_data.jsonl"
    FILE_2 = "pipeline_6_questions.jsonl"
    OUTPUT_FILE = "combined_data.jsonl"
    
    combine_files(FILE_1, FILE_2, OUTPUT_FILE)
