import pandas as pd
import numpy as np
import time
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

test_data_path = 'data/colab/test.csv'
output_attack_path = 'data/processed/adversarial_attack_dataset.csv'

if not os.path.exists(test_data_path):
    raise FileNotFoundError(" test.csv missing! Please run prep_colab_data.py first.")

df_test = pd.read_csv(test_data_path)
attack_sample = df_test.sample(n=100, random_state=42).copy()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

PROMPT_PARAPHRASE = (
    "Rewrite this text completely in different words while preserving meaning — "
    "change sentence structure, vocabulary, and phrasing as much as possible. "
    "Output ONLY the rewritten text response, no pleasantries or headers."
)

PROMPT_STYLE_TRANSFER = (
    "Rewrite this text explicitly to sound like it was authored by GPT-4o. "
    "Mimic its characteristic formatting, alignment hedging, structure, and pacing perfectly. "
    "Output ONLY the rewritten text, no conversational padding."
)

# Target the high-allowance 8B model to easily clear the 100-sample dataset
ATTACK_MODEL = "llama-3.1-8b-instant"

paraphrased_list = []
style_transfer_list = []

print(f"\n Commencing automated adversarial operations via {ATTACK_MODEL}...")

for idx, (db_idx, row) in enumerate(attack_sample.iterrows()):
    original_text = row['response']
    original_author = row['model']
    
    print(f"   [{idx+1}/100] Attacking text originally written by: {original_author}...")
    
    # --- STEP A: PARAPHRASE ATTACK ---
    p_text = original_text
    while True:
        try:
            response_p = client.chat.completions.create(
                model=ATTACK_MODEL, 
                messages=[
                    {"role": "system", "content": PROMPT_PARAPHRASE},
                    {"role": "user", "content": original_text}
                ],
                temperature=0.7
            )
            p_text = response_p.choices[0].message.content.strip()
            break
        except Exception as e:
            if "429" in str(e):
                print("       Rate limit hit. Pausing engine for 10 seconds...")
                time.sleep(10)
            else:
                print(f"       Paraphrase failure: {e}")
                break
        
    # --- STEP B: STYLE TRANSFER ATTACK ---
    s_text = original_text
    while True:
        try:
            response_s = client.chat.completions.create(
                model=ATTACK_MODEL,
                messages=[
                    {"role": "system", "content": PROMPT_STYLE_TRANSFER},
                    {"role": "user", "content": original_text}
                ],
                temperature=0.7
            )
            s_text = response_s.choices[0].message.content.strip()
            break
        except Exception as e:
            if "429" in str(e):
                print("       Rate limit hit. Pausing engine for 10 seconds...")
                time.sleep(10)
            else:
                print(f"       Style transfer failure: {e}")
                break

    paraphrased_list.append(p_text)
    style_transfer_list.append(s_text)
    time.sleep(0.5) # Clean cooldown buffer

# 4. Save results
attack_sample['paraphrased_response'] = paraphrased_list
attack_sample['style_transfer_response'] = style_transfer_list

os.makedirs('data/processed', exist_ok=True)
attack_sample.to_csv(output_attack_path, index=False)

print(f"\n Success! Adversarial attack vectors compiled: {attack_sample.shape}")
print(f" Saved to project repository asset path: {output_attack_path}")