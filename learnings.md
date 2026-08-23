# Core Learnings & Theoretical Foundations

This document acts as a knowledge base containing the core theoretical concepts and foundational learnings applied throughout the AI Model Fingerprinting project.

## 1. Deep Learning & NLP Fundamentals

### Transformers & Self-Attention
Unlike traditional RNNs or LSTMs that process sequences sequentially, the Transformer architecture processes entire sequences simultaneously. The core of this is the **Self-Attention Mechanism**, which allows the model to calculate the relevance (or "attention weight") of every word in a sentence relative to every other word. This allows the model to capture deep, long-range dependencies and stylistic pacing that simple bag-of-words models miss.

### Knowledge Distillation (DistilBERT)
DistilBERT is created through a process called knowledge distillation, employing a "Teacher-Student" framework. The smaller student model (DistilBERT) is trained to mimic the softmax output distribution (the "soft targets") of the larger teacher model (BERT). This results in a model that is 40% smaller and 60% faster, yet retains 97% of BERT's language understanding capabilities—ideal for local training and rapid classification.

### Sub-word Tokenization (WordPiece)
Language models do not process whole words. They use sub-word tokenizers like WordPiece. This handles out-of-vocabulary (OOV) words by breaking them down into known sub-components (e.g., "playing" -> "play", "##ing"). This is crucial for stylistic fingerprinting, as it captures the morphological structure of how different AIs generate language.

## 2. Machine Learning Algorithms & Ensembles

### Gradient Boosting Machines (GBM)
While Random Forests use "bagging" (building independent trees in parallel), GBM uses "boosting" (building trees sequentially, where each new tree specifically tries to correct the errors of the previous ones). In this project, GBM is used to process tabular, macro-stylometric features (e.g., average sentence length, punctuation frequency) because tree-based models excel at finding non-linear thresholds in structured numeric data.

### TF-IDF (Term Frequency - Inverse Document Frequency)
TF-IDF calculates how important a word is to a document within a larger corpus. It penalizes highly frequent words (like "the" or "and") and boosts the weight of rare, specific words. Combining sparse TF-IDF matrices with dense Transformer embeddings allows the ensemble to capture both explicit vocabulary choices and implicit semantic meaning.

### Hybrid Ensemble Robustness
A core learning from adversarial testing is that pure sequence classifiers (Transformers) are highly vulnerable to style-mimicry (adversarial spoofing). By creating a Hybrid Ensemble that blends Deep Learning (sequence probabilities) with Traditional ML (hard macro-features), the model anchors its predictions. An LLM can easily spoof tone, but it struggles to spoof mathematical consistencies like em-dash frequency and subordinate clause depth.

## 3. Model Evaluation & MLOps

### Precision vs. Recall
In forensic applications, understanding the difference between Precision and Recall is vital:
*   **Precision:** Of all the texts the model flagged as "ChatGPT", how many actually were? (Crucial for minimizing false positives, avoiding false accusations of AI cheating).
*   **Recall:** Of all the actual "ChatGPT" texts in the dataset, how many did the model successfully catch?
*   **F1-Score:** The harmonic mean of both, representing overall balance.

### Prevention of Catastrophic Forgetting
When fine-tuning a pre-trained LLM, updating the weights too aggressively will overwrite the foundational language rules the model learned during pre-training. Applying a very small learning rate (e.g., $2 \times 10^{-5}$) ensures the model adapts its classification head without corrupting the base encoder's contextual representations.

### Explainable AI (SHAP)
SHAP (SHapley Additive exPlanations) bridges the gap between accuracy and trust. Based on cooperative game theory, it calculates the marginal contribution of each token toward the final prediction. This turns a "black box" neural network into a transparent tool, allowing users to see exactly which words or syntactic structures pushed the model toward a specific classification.
