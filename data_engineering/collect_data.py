import pandas as pd
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
from groq import Groq

# Gracefully handle internal Mistral package layout shifts
try:
    from mistralai.client import Mistral
except ImportError:
    from mistralai import Mistral

# 1. Load keys from your verified .env file
load_dotenv()

print("Initializing Automated LLM Clients...")
# 2. Configure clients using your verified environment variable names
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# Ensure output directory exists for week 1 results
os.makedirs('data/processed', exist_ok=True)

def get_free_responses(prompt_text):
    data_points = []
    
    # -----------------------------------------------------------------
    # 1. Gemini 2.5 Flash Pipeline (Google AI Studio - Free)
    # -----------------------------------------------------------------
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        res = model.generate_content(
            prompt_text, 
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 500
            }
        )
        data_points.append({"model": "gemini", "response": res.text})
        print("  Gemini Response Captured")
    except Exception as e:
        print(f"  Gemini Error: {e}")

    # -----------------------------------------------------------------
    # 2. LLaMA 3.3 70B Pipeline (Via Groq - Free)
    # -----------------------------------------------------------------
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.7, 
            max_tokens=500
        )
        data_points.append({"model": "groq_llama", "response": res.choices[0].message.content})
        print("  LLaMA 70B Response Captured")
    except Exception as e:
        print(f"  Groq Error: {e}")

    # -----------------------------------------------------------------
    # 3. Mistral Large Pipeline (Via Mistral Console - Free Credits Tier)
    # -----------------------------------------------------------------
    try:
        res = mistral_client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.7, 
            max_tokens=500
        )
        data_points.append({"model": "mistral", "response": res.choices[0].message.content})
        print("  Mistral Large Response Captured")
    except Exception as e:
        print(f"  Mistral Error: {e}")

    return data_points

# Load your 200 raw prompts
prompts_df = pd.read_csv('data/raw/prompts.csv')
full_dataset = []

# =====================================================================
#  RUNNING THE FULL PIPELINE (200 PROMPTS)
# =====================================================================
print("\n Launching Full Automated Run for the 3 Free Models...")
for idx, row in prompts_df.iterrows():
    print(f"Processing Prompt {idx+1}/200 (ID: {row['prompt_id']})...")
    outputs = get_free_responses(row['text'])
    for out in outputs:
        full_dataset.append({
            "prompt_id": row['prompt_id'],
            "category": row['category'],
            "model": out['model'],
            "response": out['response']
        })
    time.sleep(5) # 5-second cooldown delay to completely prevent free-tier rate blocks

# Save the base dataset to your processed folder
pd.DataFrame(full_dataset).to_csv('data/processed/automated_responses.csv', index=False)
print("\nDone! 600 responses saved cleanly to data/processed/automated_responses.csv")