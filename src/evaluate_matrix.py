import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

print(" Initializing Forensic Evaluation Script...")

# 1. Paths & Setup
WEIGHTS_DIR = "D:/Aiml/distilbert_weights"
VAL_CSV = "D:/Aiml/data/colab/val.csv"
HEAD_FILE = os.path.join(WEIGHTS_DIR, "custom_classification_head.pt")
SAVE_IMAGE_PATH = "D:/Aiml/confusion_matrix.png"

label_map = {'chatgpt': 0, 'claude': 1, 'gemini': 2, 'groq_llama': 3, 'mistral': 4}
inv_label_map = {v: k for k, v in label_map.items()}
model_names = [inv_label_map[i] for i in range(5)]

# 2. Dataset Parser
class ValidationDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=128):
        df = pd.read_csv(csv_path)
        
        # Pull text blocks and string label columns safely
        text_col = [col for col in df.columns if col.lower() in ['text', 'response']][0]
        label_col = [col for col in df.columns if col.lower() in ['model', 'category']][0]
        
        df = df.rename(columns={text_col: 'text', label_col: 'label'}).dropna(subset=['text', 'label'])
        
        text_series = df['text'].iloc[:, 0] if isinstance(df['text'], pd.DataFrame) else df['text']
        label_series = df['label'].iloc[:, 0] if isinstance(df['label'], pd.DataFrame) else df['label']
        
        raw_texts = text_series.astype(str).tolist()
        raw_labels = label_series.astype(str).tolist()
        
        # Match names into our clean 0-4 index maps
        self.texts = []
        self.labels = []
        for t, l in zip(raw_texts, raw_labels):
            clean_key = l.strip().lower()
            if clean_key in label_map:
                self.texts.append(t)
                self.labels.append(label_map[clean_key])
                
        print(f" Loaded validation set from {os.path.basename(csv_path)}! Found {len(self.labels)} records.")
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(self.labels[idx], dtype=torch.long)}

# 3. Load Model and Stitch Fine-Tuned Weights
tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
model = AutoModelForSequenceClassification.from_pretrained(WEIGHTS_DIR, num_labels=5)

if os.path.exists(HEAD_FILE):
    print(" Loading calibrated full state dictionary...")
    # Load the synchronized state directly into the architecture
    model.load_state_dict(torch.load(HEAD_FILE, map_location=torch.device('cpu')))
else:
    print(" Warning: No custom head found! Evaluating baseline.")

val_dataset = ValidationDataset(VAL_CSV, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 4. Run Inference Loop over Validation Set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

all_preds = []
all_true = []

print(" Crushing validation records...")
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)
        
        all_preds.extend(preds.cpu().tolist())
        all_true.extend(labels.cpu().tolist())

# 5. Generate Report Statistics
print("\n --- FORENSIC CLASSIFICATION REPORT ---")
print(classification_report(all_true, all_preds, target_names=model_names))

# 6. Plot & Save Confusion Matrix
cm = confusion_matrix(all_true, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model_names, yticklabels=model_names)
plt.title('AI Style Forensics: Confusion Matrix')
plt.ylabel('Actual LLM Source')
plt.xlabel('Predicted LLM Source')
plt.tight_layout()
plt.savefig(SAVE_IMAGE_PATH)
print(f" SUCCESS! Confusion matrix saved safely to: {SAVE_IMAGE_PATH}")