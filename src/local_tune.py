import os
import torch
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from test_utils import clean_stylometric_text

print(" Initializing Upgraded Deep Variation Fine-Tuning Pipeline...")

# 1. Define Paths & Configurations
WEIGHTS_DIR = "D:/Aiml/distilbert_weights"
TRAIN_CSV = "D:/Aiml/data/colab/train.csv"  
VAL_CSV = "D:/Aiml/data/colab/val.csv"

# Label mapping to numerical channels
label_map = {'chatgpt': 0, 'claude': 1, 'gemini': 2, 'groq_llama': 3, 'mistral': 4}

# --- CRITICAL FILE PATH VALIDATION ---
if not os.path.exists(TRAIN_CSV):
    print(f" Error: Cannot find train.csv at: {TRAIN_CSV}")
    exit()

if not os.path.exists(WEIGHTS_DIR):
    print(f" Error: Weights directory missing at: {WEIGHTS_DIR}")
    exit()

# 2. Custom Dataset Loader with Safe 1D Slicing and Data Augmentation
# ====================================================================

class StylometricDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=128, augment=False):
        df = pd.read_csv(csv_path)
        
        # 1. Force extraction of text blocks safely
        text_col = [col for col in df.columns if col.lower() in ['text', 'response']][0]
        
        # 2. TARGET THE TEXT STRINGS (model/category) INSTEAD OF THE SCRAMBLED INTEGERS
        # This points the loader to names like 'groq_llama', 'chatgpt', 'gemini'
        label_col = [col for col in df.columns if col.lower() in ['model', 'category']][0]
        
        df = df.rename(columns={text_col: 'text', label_col: 'label'}).dropna(subset=['text', 'label'])
        
        text_series = df['text'].iloc[:, 0] if isinstance(df['text'], pd.DataFrame) else df['text']
        label_series = df['label'].iloc[:, 0] if isinstance(df['label'], pd.DataFrame) else df['label']
        
        raw_texts = text_series.astype(str).tolist()
        raw_texts = [clean_stylometric_text(text) for text in raw_texts]
        raw_labels = label_series.astype(str).tolist()
        
        # 3. Match the text names cleanly into our stable 0-4 numerical index bounds
        self.texts = []
        self.labels = []
        for t, l in zip(raw_texts, raw_labels):
            clean_key = l.strip().lower()
            if clean_key in label_map:
                self.texts.append(t)
                self.labels.append(label_map[clean_key])
                
        print(f"✅ Successfully initialized dataset from {os.path.basename(csv_path)}! Found {len(self.labels)} valid matching records.")
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Data Augmentation: Randomly strip markdown tables 30% of the time 
        if self.augment and random.random() < 0.30:
            lines = text.split('\n')
            clean_lines = [l for l in lines if '|' not in l and '---' not in l]
            text = '\n'.join(clean_lines) if clean_lines else text

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
# 3. Load Model Components Safely
tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
model = AutoModelForSequenceClassification.from_pretrained(WEIGHTS_DIR, num_labels=5)

# --- UNFREEZE TOP LAYERS FOR REAL LEARNING ---
# Instead of freezing everything, we unfreeze the encoder layers so the model can learn text styling
for param in model.parameters():
    param.requires_grad = True

# 4. Prepare Data Utilities
train_dataset = StylometricDataset(TRAIN_CSV, tokenizer, augment=True)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 5. Set Up Optimizer with an Optimal Learning Rate
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f" Running Unfrozen Deep Fine-Tuning on: {device.type.upper()}...")
model.train()

# 6. Training Loop Execution
for epoch in range(5):  # 5 epochs is plenty when the base model is unfrozen!
    running_loss = 0.0
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    avg_loss = running_loss / len(train_loader)
    print(f" Epoch [{epoch + 1}/5] Complete | Avg Calibration Loss: {avg_loss:.4f}")

print("\n Exporting complete optimized state dictionary...")
# Save the full state dictionary to completely eliminate script loading mismatches
head_weights_path = os.path.join(WEIGHTS_DIR, "custom_classification_head.pt")
torch.save(model.state_dict(), head_weights_path)
tokenizer.save_pretrained(WEIGHTS_DIR)
print(" SUCCESS! Full forensic state parameters calibrated and saved!")