import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import os

# Define our paths
master_text_file = 'data/processed/master_dataset.csv'
stylo_features_file = 'data/processed/stylometric_features.csv'
final_matrix_output = 'data/processed/final_ml_matrix.csv'

print(" Loading features and raw text datasets...")
df_text = pd.read_csv(master_text_file)
df_stylo = pd.read_csv(stylo_features_file)

# 1. Initialize TF-IDF Vectorizer
# We look for the top 100 most important words/bi-grams, removing standard English stop words
print(" Fitting TF-IDF Vectorizer on vocabulary strings...")
tfidf = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df_text['response'].astype(str))

# Convert sparse matrix to a structured DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{word}" for word in tfidf.get_feature_names_out()])

# 2. Separate numerical tracking metrics from structural anchors
anchor_cols = ['prompt_id', 'category', 'model']
numerical_stylo_cols = [col for col in df_stylo.columns if col not in anchor_cols]

X_stylo = df_stylo[numerical_stylo_cols]

# 3. Standardize Numerical Stylometrics (Crucial for distance models like SVM/kNN)
print(" Standardizing structural densities via StandardScaler...")
scaler = StandardScaler()
X_stylo_scaled = pd.DataFrame(scaler.fit_transform(X_stylo), columns=numerical_stylo_cols)

# 4. Horizontally Concatenate: Anchors + Scaled Stylometrics + TF-IDF Features
print(" Stitching structural and semantic matrices horizontally...")
final_df = pd.concat([df_stylo[anchor_cols], X_stylo_scaled, tfidf_df], axis=1)

# Save your final data engineering product
final_df.to_csv(final_matrix_output, index=False)
print(f"\n Success! Final Machine Learning Matrix exported to: {final_matrix_output}")
print(f" Final shape optimized for classification training models: {final_df.shape}")
print(f"   (Contains {len(numerical_stylo_cols)} structural dimensions and 100 vocabulary word dimensions)")