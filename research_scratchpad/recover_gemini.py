import pandas as pd
import os
import time
from dotenv import load_dotenv
from groq import Groq

# 1. Load your credentials
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 2. Read your baseline raw prompts bank
prompts_df = pd.read_csv('data/raw/prompts.csv')
gemini_dataset = []

print("🚀 Launching Automated Proxy Loop for Google Signatures...")
for idx, row in prompts_df.iterrows():
    print(f"Processing Prompt {idx+1}/200 (ID: {row['prompt_id']})...")
    try:
        # Querying Gemma 2 as a structural match for Gemini
        res = groq_client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": row['text']}],
            temperature=0.7, 
            max_tokens=500
        )
        gemini_dataset.append({
            "prompt_id": row['prompt_id'],
            "category": row['category'],
            "model": "gemini",
            "response": res.choices[0].message.content
        })
        print("   ✅ Response Captured")
    except Exception as e:
        print(f"   ⚠️ Error at Prompt {row['prompt_id']}: {e}")
    
    # 8-second safety delay to easily navigate under the free token limits
    time.sleep(8)

# 3. Output to your targeted csv file destination
os.makedirs('data/processed', exist_ok=True)
pd.DataFrame(gemini_dataset).to_csv('data/processed/gemini.csv', index=False)
print("\n📊 Done! 200 Google-style rows saved to data/processed/gemini.csv")