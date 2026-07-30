import pandas as pd
import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load raw prompts and slice from index 160 to 200 (Prompt IDs 161 to 200)
prompts_df = pd.read_csv('data/raw/prompts.csv')
missing_prompts = prompts_df.iloc[160:200] 

recovered_data = []

print(f"🚀 Shift 1: Capturing the final {len(missing_prompts)} LLaMA responses...")
for idx, row in missing_prompts.iterrows():
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": row['text']}],
            temperature=0.7, max_tokens=500
        )
        recovered_data.append({
            "prompt_id": row['prompt_id'],
            "category": row['category'],
            "model": "groq_llama",
            "response": res.choices[0].message.content
        })
        print(f"    Captured LLaMA Prompt ID {row['prompt_id']}")
    except Exception as e:
        print(f"    LLaMA Error at ID {row['prompt_id']}: {e}")
    time.sleep(8) # Safe delay

pd.DataFrame(recovered_data).to_csv('data/processed/llama_recovery.csv', index=False)
print(" Shift 1 Complete! Saved to data/processed/llama_recovery.csv")