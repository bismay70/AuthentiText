import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import joblib
import re
from src.test_utils import clean_stylometric_text
from src.extract_features import extract_stylometrics


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Style Forensics",
    page_icon="🔍",
    layout="wide"
)

# ====================================================================
# STEP 1: LOAD LOCAL DEEP LEARNING TRANSFORMER ASSETS
# ====================================================================
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# --- LOAD LOCAL DEEP LEARNING TRANSFORMER ASSETS ---
@st.cache_resource
def load_forensic_transformer():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_weights_path = os.path.join(base_dir, "distilbert_weights")
    local_head_file = os.path.join(local_weights_path, "custom_classification_head.pt")

    # Safetensors check to make sure we don't try loading half-empty folders on the cloud
    has_config = os.path.exists(os.path.join(local_weights_path, "config.json"))
    has_weights = os.path.exists(os.path.join(local_weights_path, "model.safetensors"))

    # Only run local loading if BOTH structural files exist
    if has_config and has_weights:
        print(" Loading local fine-tuned base configuration...")
        tokenizer = AutoTokenizer.from_pretrained(local_weights_path)
        model = AutoModelForSequenceClassification.from_pretrained(local_weights_path, num_labels=5)
        
        if os.path.exists(local_head_file):
            st.sidebar.success(" Fine-Tuned Local Head Active")
            checkpoint = torch.load(local_head_file, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint)
        return tokenizer, model, True
    else:
        # Secure fallback for cloud sandbox environments
        print(" Local weights missing or untracked. Initializing Hugging Face Cloud Fallback...")
        st.sidebar.warning(" Cloud Sandbox Mode: Baseline Transformer Active")
        
        fallback_model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(fallback_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(fallback_model_name, num_labels=5)
        return tokenizer, model, True
tokenizer, model, transformer_loaded = load_forensic_transformer()

# --- BACKEND UTILITY: DYNAMIC FEATURE EXTRACTION ---
def extract_stylometric_features(text):
    """Calculates macro-stylometric structural tokens from a raw string input."""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if len(s.strip()) > 0]
    
    word_count = len(words) if len(words) > 0 else 1
    sent_count = len(sentences) if len(sentences) > 0 else 1
    
    avg_word_length = np.mean([len(w) for w in words]) if words else 0
    avg_sentence_len = word_count / sent_count
    
    # Structural count patterns
    comma_density = text.count(',') / word_count
    semicolon_density = text.count(';') / word_count
    exclamation_density = text.count('!') / word_count
    question_density = text.count('?') / word_count
    
    features = {
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_len,
        'comma_density': comma_density,
        'semicolon_density': semicolon_density,
        'exclamation_density': exclamation_density,
        'question_density': question_density,
        'response_length': len(text)
    }
    return pd.DataFrame([features])

# --- LOAD LOCAL TRAINED MODEL ASSETS ---
# --- LOAD LOCAL TRAINED MODEL ASSETS ---
@st.cache_resource
def load_forensic_models():
    model_path = "D:/Aiml/data/processed/gradient_boosting_model.pkl"
    vectorizer_path = "D:/Aiml/data/processed/tfidf_vectorizer.pkl"
    scaler_path = "D:/Aiml/data/processed/scaler.pkl" # Add your scaler path here
    
    # Load model and scaler if they exist
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler, True
    return None, None, False

# Update global variable naming assignment
gb_model, feature_scaler, models_loaded = load_forensic_models()

# --- APPLICATION HEADER ---
st.title(" AI Writing Style Forensics Dashboard")
st.markdown("""
This forensic platform analyzes natural language patterns to identify the specific LLM architecture 
responsible for generating a text segment. Trained on distinct sequence markers and macro stylometric features.
""")

st.sidebar.header(" Configuration")
st.sidebar.markdown("**Model Architecture:** `DistilBERT Ensemble` + `GBM Baseline`")
st.sidebar.markdown("**Target Classes:** 5 Large Language Models")

# --- INITIALIZE INTERACTIVE TABS ---
tab1, tab2, tab3 = st.tabs([" Live Model Fingerprinter", " Robustness Benchmark", " Feature Explorer"])

# ==========================================
# TAB 1: LIVE MODEL FINGERPRINTER
# ==========================================

with tab1:
    st.header("Analyze Text Signature")
    st.write("Paste a raw generation block below to evaluate its stylistic authorship matrix.")
    
    user_text = st.text_area(
        "Input Generation Text:", 
        height=200, 
        placeholder="Type or paste sample AI text here (minimum 20 words recommended)...",
        key="forensic_text_input"
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Forensic Action")
        run_analysis = st.button(" Extract Signature", use_container_width=True)
        
    with col2:
        if run_analysis and user_text.strip():
            with st.spinner("Analyzing deep token transitions via DistilBERT..."):
                
                if transformer_loaded:
                    # 1. Clean the text using your optimized utility function
                    cleaned_text = clean_stylometric_text(user_text)
                    
                    # 2. ROUTE THE CLEANED TEXT TO THE TOKENIZER
                    inputs = tokenizer(
                        cleaned_text,  # Fixed to process the sanitized string
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    )
                    
                    # 3. Execute transformer inference
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        # Convert transformer scores to raw probability arrays
                        transformer_probs = F.softmax(logits, dim=-1).squeeze().tolist()
                    
                    # 4. Extract explicit manual features for your baseline
                        manual_features = extract_stylometrics(cleaned_text)
                    
                    # 5. Extract Tabular Probabilities via your Gradient Boosting Asset
                    if models_loaded:
                        # Use gb_model directly as loaded at the top of your file
                        manual_features_scaled = feature_scaler.transform(manual_features)
                        baseline_probs = gb_model.predict_proba(manual_features_scaled)[0].tolist()
                        
                        # Hybrid Fusion Calculation: Blend weights to handle adversarial tone
                        t_weight = 0.4
                        b_weight = 0.6
                        
                        probabilities = [
                            (transformer_probs[i] * t_weight) + (baseline_probs[i] * b_weight)
                            for i in range(len(transformer_probs))
                        ]
                    else:
                        # Fallback to pure transformer output if local pkl files are missing
                        probabilities = transformer_probs
                    
                else:
                    st.error("Critical Error: Could not locate configuration files inside `D:/Aiml/distilbert_weights/`.")
                    st.stop()

                # --- CODE ESCAPE: Out of the else block, executing at successful run-time ---
                # 3. Explicitly map our 5 target LLM class labels uniformly
                model_names = ['chatgpt', 'claude', 'gemini', 'groq_llama', 'mistral']

                # --- GENERATE DATA FRAME OUTPUT ---
                pred_df = pd.DataFrame({
                    'LLM Engine': model_names,
                    'Probability Match': probabilities
                }).sort_values(by='Probability Match', ascending=False)
                
                st.success(" Analysis complete!")
                
                top_model = pred_df.iloc[0]['LLM Engine'].upper()
                st.metric(label=" Primary Suspect Author Match", value=top_model)
                
                # Render your optimized Plotly performance graph
                fig = px.bar(
                    pred_df, 
                    x='Probability Match', 
                    y='LLM Engine', 
                    orientation='h',
                    color='Probability Match',
                    color_continuous_scale='Blues',
                    text_auto='.2%'
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)
                
        elif run_analysis:
            st.warning(" Please paste a valid text sequence before initiating forensic extraction.")

# ==========================================
# TAB 2: ROBUSTNESS BENCHMARK
# ==========================================
with tab2:
    st.header(" Adversarial Attack Vulnerability Logs")
    st.write("Performance analysis under deliberate structural rewriting and style-spoofing vectors (Week 3 Milestone).")
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric(" Clean Baseline Accuracy", "93.28%")
    metric_col2.metric(" Paraphrase Robustness", "44.00%", "-49.28% Degradation")
    metric_col3.metric(" Style Transfer Robustness", "36.00%", "-57.28% Degradation")
    
    st.markdown("---")
    st.markdown("""
    ###  Strategic Takeaways for the Research Report
    * **Surface-Level Volatility:** The massive drop from **93.28% down to 44.00%** confirms that transformer-based classifications depend significantly on sequential token proximity and exact phrasing rhythms.
    * **The Mimicry Deficit:** Style transfer proved to be the most lethal vector (**36.00%** remaining accuracy). When an external model deliberately structure-hedges to spoof OpenAI parameters (headers, list arrays), it shifts attention distributions heavily, demonstrating that sequence classifiers can be effectively blinded by adversarial presentation wrappers.
    """)

# ==========================================
# TAB 3: FEATURE EXPLORATION
# ==========================================
with tab3:
    st.header(" Dataset Boundary Constraints")
    st.write("A metadata overview of the evaluation boundaries and sample file structures.")
    
    if os.path.exists('data/colab/test.csv'):
        st.info(" Verified: Local data workspace slices detected.")
        df_preview = pd.read_csv('data/colab/test.csv').head(5)
        st.markdown("###  Sample Input Pool Preview")
        st.dataframe(df_preview[['category', 'model', 'response']])
    else:
        st.warning(" Local CSV workspace paths isolated. Running app framework in standalone test mode.")