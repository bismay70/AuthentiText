import pandas as pd
import numpy as np
import spacy
import textstat
import os
import re
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import words

# Download the baseline English words dictionary if missing
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

COMMON_WORDS = set(w.lower() for w in words.words()[:10000])

# Define exact file paths
input_file = 'data/processed/master_dataset.csv'
output_file = 'data/processed/final_feature_matrix.csv'

print(" Loading master dataset...")
df = pd.read_csv(input_file)
responses = df['response'].astype(str).tolist()

print(" Loading spaCy parsing pipeline and SentenceTransformer...")
nlp = spacy.load("en_core_web_sm")
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Define specific lists for hedge and filler tracking based on the brief
HEDGE_WORDS = {'perhaps', 'might', 'could', 'possibly', 'arguably', 'seemingly', 'maybe'}
FILLER_PHRASES = {'certainly', 'absolutely', 'great question', 'of course', 'delve', 'interestingly'}

# --- EXTRACT SEMANTIC LDA TOPICS FIRST ---
print(" Computing Latent Dirichlet Allocation (LDA) Topic Distributions...")
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
tf_matrix = vectorizer.fit_transform(responses)
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda_features = lda.fit_transform(tf_matrix)

processed_features = []

print(" Starting row-by-row deep stylometric feature extraction...")
for idx, doc in enumerate(nlp.pipe(responses, batch_size=50)):
    text_str = doc.text
    tokens = [tok for tok in doc if not tok.is_space]
    words_only = [tok for tok in tokens if not tok.is_punct]
    raw_words = [tok.text.lower() for tok in words_only]
    
    word_count = len(words_only)
    token_count = len(tokens)
    
    if word_count == 0:
        processed_features.append({col: 0 for col in range(15)}) # Fallback empty row
        continue

    # === A. LEXICAL FEATURES ===
    ttr = len(set(raw_words)) / word_count
    avg_word_len = sum(len(w) for w in raw_words) / word_count
    rare_word_count = sum(1 for w in raw_words if w not in COMMON_WORDS)
    rare_word_freq = rare_word_count / word_count
    
    hedge_count = sum(1 for w in raw_words if w in HEDGE_WORDS)
    hedge_rate = hedge_count / word_count
    
    filler_count = sum(1 for phrase in FILLER_PHRASES if phrase in text_str.lower())
    filler_rate = filler_count / word_count
    
    contraction_count = len(re.findall(r"\b\w+['’][stmdre]ll\b|\bcan't\b|\bwon't\b", text_str.lower()))
    em_dash_count = text_str.count('—') + text_str.count('--')
    comma_count = text_str.count(',')
    dash_vs_comma = em_dash_count / (comma_count + 1)
    
    bullet_count = text_str.count('*') + text_str.count('•') + sum(1 for line in text_str.split('\n') if re.match(r'^\s*\d+\.', line))
    list_frequency = bullet_count / word_count

    # === B. SYNTACTIC FEATURES ===
    sentences = list(doc.sents)
    sentence_lengths = [len([t for t in sent if not t.is_punct]) for sent in sentences]
    avg_sentence_len = np.mean(sentence_lengths) if sentences else 0
    sentence_variance = np.var(sentence_lengths) if sentences else 0
    
    passive_markers = sum(1 for tok in doc if tok.dep_ in ["nsubjpass", "auxpass"])
    active_markers = sum(1 for tok in doc if tok.dep_ == "nsubj")
    total_voice = passive_markers + active_markers
    passive_active_ratio = passive_markers / total_voice if total_voice > 0 else 0
    
    max_clause_depth = 0
    for tok in doc:
        if tok.dep_ in ["ccomp", "xcomp", "advcl", "acl"]:
            depth = 0
            curr = tok
            while curr.head != curr:
                depth += 1
                curr = curr.head
            max_clause_depth = max(max_clause_depth, depth)
            
    paragraphs = [p for p in text_str.split('\n\n') if p.strip()]
    paragraph_count = len(paragraphs)
    avg_para_len = np.mean([len(p.split()) for p in paragraphs]) if paragraphs else 0

    # === C. SEMANTIC FEATURES ===
    flesch = textstat.flesch_reading_ease(text_str)
    
    # Formality baseline calculation proxy
    nouns = sum(1 for t in words_only if t.pos_ in ["NOUN", "PROPN"])
    adjectives = sum(1 for t in words_only if t.pos_ == "ADJ")
    pronouns = sum(1 for t in words_only if t.pos_ == "PRON")
    interjections = sum(1 for t in words_only if t.pos_ == "INTJ")
    formality_score = (nouns + adjectives) / (pronouns + interjections + 1)
    
    blob = TextBlob(text_str)

    # Collect all row dimensions
    processed_features.append({
        "type_token_ratio": ttr,
        "avg_word_length": avg_word_len,
        "rare_word_frequency": rare_word_freq,
        "hedge_word_rate": hedge_rate,
        "filler_phrase_rate": filler_rate,
        "contraction_usage": contraction_count / word_count,
        "em_dash_vs_comma": dash_vs_comma,
        "list_frequency": list_frequency,
        "avg_sentence_length": avg_sentence_len,
        "sentence_variance": sentence_variance,
        "passive_active_ratio": passive_active_ratio,
        "subordinate_clause_depth": max_clause_depth,
        "paragraph_count": paragraph_count,
        "avg_paragraph_length": avg_para_len,
        "flesch_formality": formality_score,
        "sentiment_polarity": blob.sentiment.polarity,
        "sentiment_subjectivity": blob.sentiment.subjectivity
    })

# Convert to DataFrame
features_df = pd.DataFrame(processed_features)

# Add LDA columns
for i in range(5):
    features_df[f'lda_topic_{i}'] = lda_features[:, i]

# --- EXTRACT DENSE TRANSFORMER EMBEDDINGS ---
print(" Vectorizing text using SentenceTransformers (Dense Semantic Embeddings)...")
embeddings = encoder.encode(responses, show_progress_bar=False)
# Convert the embedding dimensions into lightweight column abstractions
embedding_means = pd.DataFrame(embeddings[:, :20], columns=[f'emb_dim_{i}' for i in range(20)])

# Combine everything into the master classification matrix
anchor_cols = df[['prompt_id', 'category', 'model']].copy()
final_matrix = pd.concat([anchor_cols, features_df, embedding_means], axis=1)

os.makedirs('data/processed', exist_ok=True)
final_matrix.to_csv(output_file, index=False)
print(f"\n Feature Engineering Pipeline Complete! Final Matrix Shape: {final_matrix.shape}")