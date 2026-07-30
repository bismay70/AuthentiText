import pandas as pd
import os
import csv

# Define paths matching your workspace image
automated_file = 'data/processed/automated_responses.csv'
llama_patch = 'data/processed/llama_recovery.csv'
manual_bank_file = 'manual_bank.csv'
gemini_file = 'gemini.csv'

all_rows = []

def load_file_safely(filepath, default_model=None):
    if not os.path.exists(filepath):
        print(f"⚠️ File not found: {filepath}")
        return []
    
    rows_loaded = 0
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        # Use csv.DictReader to automatically handle tricky multi-line quotes
        reader = csv.DictReader(f)
        
        for idx, row in enumerate(reader):
            # Clean up keys and values from hidden spaces
            clean_row = {str(k).strip().lower(): str(v).strip() for k, v in row.items() if k is not None}
            
            # Extract essential tracking fields
            prompt_id = clean_row.get('prompt_id')
            category = clean_row.get('category')
            model = clean_row.get('model', default_model)
            response = clean_row.get('response')
            
            # Fallback if headers are slightly different
            if not model and default_model:
                model = default_model
                
            if prompt_id and response:
                # Remove actual hard line breaks inside the text so it sits on 1 physical line
                clean_response = " ".join(response.split())
                
                all_rows.append({
                    "prompt_id": int(float(prompt_id)),
                    "category": category,
                    "model": str(model).lower().strip(),
                    "response": clean_response
                })
                rows_loaded += 1
                
    print(f"✅ Successfully extracted {rows_loaded} clean rows from: {filepath}")

print("🚀 Starting Absolute Dataset Compilation Phase...")
print("-" * 60)

# Load every single data layer smoothly
load_file_safely(automated_file)
load_file_safely(llama_patch)
load_file_safely(manual_bank_file)
load_file_safely(gemini_file, default_model='gemini')

# --- Stitch and Deduplicate Matrix ---
if all_rows:
    print("-" * 60)
    print("Building unified Pandas DataFrame...")
    master_df = pd.DataFrame(all_rows)
    
    # Drop exact overlaps (e.g. if a prompt exists in both automated and patches)
    initial_count = len(master_df)
    master_df.drop_duplicates(subset=['prompt_id', 'model'], keep='first', inplace=True)
    final_count = len(master_df)
    
    # Save target file to processed metrics folder
    output_path = 'data/processed/master_dataset.csv'
    os.makedirs('data/processed', exist_ok=True)
    master_df.to_csv(output_path, index=False)
    
    print(f"\n🎉 Master matrix exported completely to: {output_path}")
    print("=" * 60)
    print("📊 WEEK 2 MASTER TEXT DISTRIBUTION MATRIX:")
    print(master_df['model'].value_counts())
    print("=" * 60)
    print(f"Total clean samples ready for feature extraction: {len(master_df)} rows")
else:
    print("❌ Critical Failure: Zero data points could be collected.")