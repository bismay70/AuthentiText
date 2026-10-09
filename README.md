# 🔍 AuthentiText: AI Model Fingerprinting & Forensic Stylometry

<p align="center">
  <img src="ss/ss1.png" width="48%" alt="Live Model Fingerprinter"/>
  <img src="ss/ss2.png" width="48%" alt="Adversarial Attack Vulnerability Logs"/>
  <br>
  <img src="ss/ss3.png" width="48%" alt="Dataset Boundary Constraints"/>
  <img src="ss/ss6.png" width="48%" alt="New Project Screenshot — Placeholder"/>
  <br>
  <img src="ss/ss5.png" width="48%" alt="Confusion Matrix"/>
  <img src="ss/ss7.png" width="48%" alt="Confusion Matrix"/>
</p>

Every Large Language Model has a distinctive writing signature, influenced by its training data, alignment, and response-generation behavior. GPT models may differ from Claude in how they qualify claims, while Gemini and LLaMA may exhibit different structural and lexical tendencies. These patterns can provide useful signals for model attribution.

**AuthentiText** is an NLP forensics project that investigates these signals using a fine-tuned **DistilBERT sequence classifier**, supported by text preprocessing, stylometric analysis, and model explainability. Its preprocessing pipeline aims to reduce reliance on superficial formatting cues so the classifier can focus more on linguistic patterns.

---

## 🛠️ Project Architecture

The repository is organized into data engineering, model training, evaluation, explainability, and interactive demonstration components.

```text
├── analysis_plots/              # Evaluation and explainability visualizations
│   ├── confusion_matrix.png     # Classification performance across target models
│   ├── shap_adversarial_gpt4o_spoof.png
│   └── shap_original_claude.png
├── data_engineering/            # Dataset collection, validation, and preparation
│   ├── check_csv.py
│   ├── collect_data.py
│   ├── merge_data.py
│   ├── prep_colab_data.py
│   └── repair_csv_structure.py
├── distilbert_weights/          # Local model configuration and tokenizer assets
├── research_scratchpad/         # Experiments and diagnostic utilities
├── src/                         # Training, preprocessing, and evaluation modules
│   ├── adversarial_attacks.py
│   ├── compile_final_dataset.py
│   ├── evaluate_matrix.py
│   ├── extract_features.py
│   ├── feature_pipeline.py
│   ├── local_tune.py
│   └── test_utils.py
├── .gitignore
├── README.md
├── app.py                       # Interactive Streamlit dashboard
├── master_matrix.py             # TF-IDF and stylometric feature analysis
├── requirements.txt
└── train_baselines.py            # Traditional ML baselines
```

---

## 🚀 Engineering Improvements

During early experiments, the classifier reached an optimization bottleneck of approximately **13% accuracy**. Investigation identified target-label indexing mismatches and formatting-related shortcuts, including potential over-reliance on Markdown elements such as tables and headings.

The subsequent improvements focused on:

1. **Formatting-aware preprocessing:** Reducing superficial structural cues to encourage learning from lexical and syntactic patterns.
2. **Transformer fine-tuning:** Unfreezing the DistilBERT encoder and using a learning rate of \(2 \times 10^{-5}\) over five epochs.
3. **Flexible model loading:** Supporting local fine-tuned weights while allowing a fallback mode when those weights are unavailable in a deployment environment.

The current README reports **90.0% validation accuracy**. The evaluation setup and dataset split should be checked before interpreting this as a reliable estimate of performance on unseen prompts.

---

## 📊 Performance & Evaluation

The reported results indicate promising separation between several target model classes.

* **Claude:** F1-score of 0.92.
* **Mistral:** F1-score of 0.95.
* **Gemini:** Precision of 0.97.
* **Reported overall validation accuracy:** 90.0%.

### Confusion Matrix

![Confusion Matrix](analysis_plots/confusion_matrix.png)

The confusion matrix provides a class-by-class view of correct predictions and misclassifications across the five target model categories.

---

## 🧪 Explainability & Adversarial Diagnostics

* **Explainable AI:** SHAP visualizations help investigate which text features contribute to a prediction.
* **Adversarial evaluation:** `src/adversarial_attacks.py` supports experiments that test whether stylistic rewriting or formatting changes affect model attribution.
* **Feature analysis:** `master_matrix.py` combines TF-IDF and stylometric signals for further investigation.

These components help examine both the classifier's behavior and its potential weaknesses; their presence alone does not establish robustness against all adversarial inputs.

---

## 💻 Installation & Local Deployment

### 1. Install dependencies

```bash
pip install torch transformers pandas scikit-learn matplotlib seaborn streamlit plotly
```

### 2. Launch the Streamlit dashboard

```bash
python -m streamlit run app.py
```

When compatible fine-tuned weights are available locally, the application can load them for inference. Check the dashboard's status indicator to confirm which model-loading path is active.

### 3. Streamlit Community Cloud

Large model-weight files may be excluded from Git using `.gitignore`. If the fine-tuned weights are not included in the deployment, ensure the fallback path is configured correctly and clearly distinguish its behavior from inference using the locally fine-tuned checkpoint.

---

*Note: The accuracy and class-level metrics above preserve the figures in the supplied README. They should be described as verified results only if they are supported by the corresponding evaluation logs and test protocol.*
