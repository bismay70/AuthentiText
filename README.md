# 🔍 AI Model Fingerprinting & Forensic Stylometry

Every Large Language Model has an invisible writing signature — shaped by its training data, alignment tuning, and RLHF process. GPT-4 hedges differently than Claude. Gemini structures arguments differently than LLaMA. These patterns are subtle, consistent, and machine-detectable.

This repository implements an advanced Natural Language Processing (NLP) forensics pipeline using a fine-tuned **DistilBERT** sequence classifier to decode and capture these hidden linguistic footprints. It decouples superficial presentation layers from core stylistic syntax to ensure highly robust authorship classification.

---

## 🛠️ Complete Project Architecture

```text
├── analysis_plots/             # Visual evaluation and model explainability assets
│   ├── confusion_matrix.png     # Evaluation breakdown across target model categories
│   ├── shap_adversarial_gpt4o_spoof.png # SHAP feature attributions for spoofing attempts
│   └── shap_original_claude.png # Baseline SHAP linguistic attribution maps for Claude
├── data_engineering/           # Specialized pipelines for handling data curation
│   ├── check_csv.py             # Integrity validation framework for runtime inputs
│   ├── collect_data.py          # Automated generation and crawling API collectors
│   ├── merge_data.py            # Dataset aggregation and consolidation module
│   ├── prep_colab_data.py       # Serialization layer for high-compute environments
│   └── repair_csv_structure.py  # Structural recovery script for broken CSV boundaries
├── distilbert_weights/         # Local model calibration parameters (Git ignored binary layers)
│   ├── config.json              # Model configuration specifications
│   ├── tokenizer.json           # Sub-word token mappings and vocabulary arrays
│   └── tokenizer_config.json    # Behavioral padding/truncation runtime targets
├── research_scratchpad/        # Experimental sandboxes and validation baselines
│   ├── check_data.py            # Ad-hoc distribution verification checks
│   ├── distilbert_report.txt    # Extracted validation performance logs
│   ├── find_pkl.py              # Feature object tracking lookup utility
│   ├── inspect_raw.py           # Raw text string sample diagnostic suite
│   ├── pr.ipy / pr.ipynb        # Interactive exploratory data analysis notebooks
│   ├── recover.py / recover_gemini.py # Fallback procedures for interrupted runs
│   └── test2.py / test_env.py   # Local runtime package and hardware checks
├── src/                        # Core production pipeline components
│   ├── adversarial_attacks.py   # Evaluates boundary vulnerabilities against adversarial text
│   ├── compile_final_dataset.py # Builds tokenization-ready training tensors
│   ├── evaluate_matrix.py       # Calculates validation metrics across model slices
│   ├── extract_features.py      # Computes semantic density profiles
│   ├── feature_pipeline.py      # Streamlined data transformation pipeline execution
│   ├── local_tune.py            # PyTorch Trainer constructor for transformer headers
│   └── test_utils.py            # Stylometric cleaner (filters out cheater tokens & markdown masks)
├── .gitignore                  # Keeps heavy weights out of cloud commits
├── README.md                   # Repository documentation and overview
├── app.py                      # Multi-tab interactive Streamlit web dashboard
├── master_matrix.py            # Combines stylometric embeddings with tf-idf weights
├── requirements.txt            # System dependencies (PyTorch, Transformers, Streamlit)
└── train_baselines.py          # Benchmark scripts for traditional estimators


```

---

## 🚀 Engineering Breakthroughs & Optimization

During initial baseline tracking, the classifier encountered an optimization bottleneck (stuck at ~13% accuracy) due to categorical target index mismatches and structural formatting biases where the network relied heavily on raw markdown elements (like tables and hashes) rather than linguistic stylometry.

### Core Architecture Upgrades:

1. **Dynamic Token Augmentation:** Implemented a robust pre-processing pipeline that strips volatile structural patterns to force the transformer to evaluate underlying syntactic and lexical habits.
2. **Layer Liberation & Tuning:** Unfroze the base encoder blocks of the DistilBERT architecture and applied an optimized learning rate ($2\times 10^{-5}$) over 5 epochs to break out of the low-accuracy local minimum.
3. **Robust Environment Routing:** Architected the deployment pipeline to dynamically recognize its execution context—loading fine-tuned weights locally while falling back to a clean sandbox wrapper in restricted cloud environments.

As a direct result of these modifications, overall validation accuracy shot up to a stellar **90.0%**.

---

## 📊 Performance & Validation Metrics

The model was rigorously validated against an untouched verification partition. Below is the validated classification report and true positive boundary mapping:

### Forensic Metrics Breakdown

* **Claude (F1: 0.92):** High separation accuracy; successfully maps complex semantic structures and empathetic pacing.
* **Mistral (F1: 0.95):** Achieved the highest individual style resolution performance.
* **Gemini (Precision: 0.97):** Exceptional precision marker—minimal false positive generation matches.
* **Overall Metrics Accuracy:** `90.0%`

### Confirmed Confusion Matrix Heatmap

The true positive vs. false positive classification boundaries across all 5 LLM targets:

---

## 🛠️ Advanced Features & Diagnostics

Beyond pure classification, this repository integrates cutting-edge diagnostics to stress-test and interpret the model's performance:

* **Explainable AI (XAI):** Integrated SHAP (SHapley Additive exPlanations) visualizers (`analysis_plots/shap_original_claude.png`) to extract and analyze exactly which tokens heavily influence a specific model's fingerprint calculation.
* **Adversarial Attack Simulation:** Features a dedicated testing module (`src/adversarial_attacks.py`) designed to simulate stylistic camouflage and evaluate the classifier's defensive boundaries against prompt-spoofing techniques.

---

## 💻 Installation & Local Deployment

### 1. Environment Setup

Clone this repository to your local drive and install the required deep learning and visualization dependencies:

```bash
pip install torch transformers pandas scikit-learn matplotlib seaborn streamlit plotly

```

### 2. Launching the Desktop UI Dashboard

To run the fully optimized, 90% accurate fine-tuned model weights locally, execute the app from your project root folder:

```bash
python -m streamlit run app.py

```

*The local environment will automatically identify your saved weights and light up a green **🎯 Fine-Tuned Local Head Active** badge on the sidebar.*

### 3. Cloud Mode Deployment

This project is fully ready for deployment to **Streamlit Community Cloud**. Because large model weights are safely omitted via `.gitignore` to keep the repository lightweight, the cloud server automatically enters a secure fallback sandbox mode, pulling standard base parameters directly from the Hugging Face Hub to showcase software architecture fluidly.
