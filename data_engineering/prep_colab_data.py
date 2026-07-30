import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load master dataset
master_path = 'data/processed/master_dataset.csv'
df = pd.read_csv(master_path)

# Drop any potential missing data row
df = df.dropna(subset=['response', 'model'])

# Clean up any potential spelling mismatch in categories
df['category'] = df['category'].astype(str).str.replace('Factual explanaton', 'Factual explanation')

# Select only the columns needed for deep sequence learning
df_clean = df[['response', 'model', 'category']].copy()

# Map textual classes to numeric IDs for our 5-class sequence classifier loss function
model_mapping = {model_name: idx for idx, model_name in enumerate(sorted(df_clean['model'].unique()))}
df_clean['label'] = df_clean['model'].map(model_mapping)

print("Mapping scheme configuration locked:")
print(model_mapping)

# Implement the strict 70/15/15 stratified division
df_train, df_temp = train_test_split(df_clean, test_size=0.30, random_state=42, stratify=df_clean['category'])
df_val, df_test = train_test_split(df_temp, test_size=0.50, random_state=42, stratify=df_temp['category'])

# Save them out cleanly
os.makedirs('data/colab', exist_ok=True)
df_train.to_csv('data/colab/train.csv', index=False)
df_val.to_csv('data/colab/val.csv', index=False)
df_test.to_csv('data/colab/test.csv', index=False)

print(f"\n Datasets exported for Colab transfer! Shapes: Train {df_train.shape}, Val {df_val.shape}, Test {df_test.shape}")