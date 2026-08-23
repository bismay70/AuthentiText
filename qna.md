# Interview Q&A and Technical Probing

If you are interviewing for a Machine Learning Engineer, Data Scientist, or Python Backend role using this project, the interviewer will focus heavily on NLP, Transformers, and MLOps. Here are the expected questions and domains.

## 1. Deep Learning & Transformer Architecture (DistilBERT)
*   **Explain the architecture of a Transformer. How does the Self-Attention mechanism work?**
    *   *Focus on how attention allows the model to look at the entire sequence simultaneously, calculating weighted importance between every token in the sentence.*
*   **What is the difference between BERT and DistilBERT? How does knowledge distillation work?**
    *   *Discuss the teacher-student framework, where DistilBERT learns to mimic the output distribution of the larger BERT model, resulting in a lighter, faster architecture retaining 97% of the language understanding.*
*   **Why did you choose DistilBERT over standard BERT, RoBERTa, or a generative model like LLaMA?**
    *   *Classification efficiency. DistilBERT is optimized for sequence classification and can easily be fine-tuned locally without requiring massive A100 GPU clusters.*
*   **Explain the role of the `[CLS]` token.**
    *   *The `[CLS]` token aggregates the entire sequence's representation. Its final hidden state is passed to the classification head.*
*   **What does "unfreezing the base encoder blocks" mean? Why a learning rate of $2 \times 10^{-5}$?**
    *   *Unfreezing allows the pre-trained weights to update during backpropagation. A low learning rate prevents catastrophic forgetting.*

## 2. Natural Language Processing (NLP) & Feature Engineering
*   **How did you approach text preprocessing? What specific "markdown masks" did you filter out?**
    *   *Filtered out bullet points, bolding markers, and header hashes so the model learns grammar, not formatting.*
*   **How does TF-IDF work? Why combine it with dense embeddings?**
    *   *TF-IDF captures rare but highly specific vocabulary choices (sparse features), complementing the semantic context captured by dense transformer embeddings.*
*   **How do you handle variable-length sequences when feeding them into DistilBERT?**
    *   *Using Padding (to match the tensor size of a batch) and Truncation (cutting off at 512 tokens max length).*

## 3. Model Training & Optimization
*   **How did you diagnose the issue when your model was stuck at 13% accuracy?**
    *   *Identified categorical target index mismatch and analyzed that the model was memorizing formatting rules rather than stylistic fingerprints.*
*   **Explain the loss function used for multi-class classification.**
    *   *Cross-Entropy Loss. It penalizes predictions that are confident but wrong, pushing the softmax probabilities toward the true class.*
*   **Is accuracy the best metric? What is the difference between Precision and Recall?**
    *   *No, especially if classes are imbalanced. Precision measures how many predicted positives were actually true (important for avoiding false accusations of AI usage). Recall measures how many actual positives were caught.*

## 4. Explainable AI (XAI) & Security
*   **What is SHAP? How does it help explain black-box NLP models?**
    *   *SHapley Additive exPlanations uses game theory to calculate the marginal contribution of each feature (or sub-word token) toward the final prediction, making the neural network transparent.*
*   **What is an "Adversarial Attack" in the context of NLP?**
    *   *Techniques like prompt spoofing (asking an LLM to write like another) or synonym replacement designed to bypass the classifier.*

## 5. Python, PyTorch & MLOps Architecture
*   **How do you handle memory management in PyTorch?**
    *   *Using `with torch.no_grad():` during inference to prevent the graph from storing gradients, saving massive amounts of VRAM.*
*   **How did you implement "Dynamic Environment Routing"?**
    *   *Using `os.path.exists()` checks on the local filesystem to detect the presence of heavy `.safetensors` files, gracefully falling back to Hugging Face API requests if they are absent.*
*   **Why exclude large model weights using `.gitignore`? Where should they be stored?**
    *   *Git is not designed for massive binary blobs. Weights should be stored in S3 buckets, MLflow, or the Hugging Face Model Hub.*

## 6. Behavioral & Project-Specific
*   **What was the most challenging technical hurdle?**
    *   *Overcoming the 13% bottleneck and defending against adversarial spoofing via the Hybrid Ensemble.*
*   **Walk me through your system architecture from raw text to the final Streamlit output.**
    *   *Data Collection -> Text Stripping -> Dual Pipeline (DistilBERT + Stylometric GBM) -> Probability Ensemble Blending -> UI Rendering & SHAP Visualization.*
