import pandas as pd
import os

# Paths to your files
main_file = 'data/processed/automated_responses.csv'
recovery_file = 'data/processed/llama_recovery.csv'

if os.path.exists(main_file):
    df_main = pd.read_csv(main_file)
    print(f"Loaded main dataset with {len(df_main)} rows.")
    
    # Check if the recovery file caught rows 161 and 162 before failing
    if os.path.exists(recovery_file):
        df_recover = pd.read_csv(recovery_file)
        print(f"Loaded recovery dataset with {len(df_recover)} rows.")
        
        # Combine them on top of each other cleanly
        df_combined = pd.concat([df_main, df_recover], ignore_index=True)
        
        # Drop any accidental exact duplicates to keep the statistics pristine
        df_combined.drop_duplicates(subset=['prompt_id', 'model'], keep='first', inplace=True)
        
        # Overwrite into your master file
        df_combined.to_csv(main_file, index=False)
        print(f"🎉 Successfully merged! Master file updated to {len(df_combined)} rows.")
    else:
        print("No recovery file found yet. Your main file is secure!")
else:
    print("Could not locate data/processed/automated_responses.csv. Check your path strings.")