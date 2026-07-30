import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# Define paths
matrix_path = 'data/processed/final_feature_matrix.csv'

print(" Loading final feature matrix...")
if not os.path.exists(matrix_path):
    raise FileNotFoundError(f" Matrix not found at {matrix_path}. Run feature_pipeline.py first!")

df = pd.read_csv(matrix_path)
print(f" Dataset successfully loaded: {df.shape[0]} rows across {df.shape[1]} features.")

# 1. Isolate target tracking labels from pure numeric features
anchor_cols = ['prompt_id', 'category', 'model']
X = df.drop(columns=anchor_cols)
y = df['model']

# We preserve the 'category' series to run our per-category accuracy checks later
categories = df['category']
# Isolate target tracking labels from pure numeric features
anchor_cols = ['prompt_id', 'category', 'model']
X = df.drop(columns=anchor_cols)
y = df['model']

# Fetch the category column AND fix the hidden spelling typo automatically
df['category'] = df['category'].astype(str).str.replace('Factual explanaton', 'Factual explanation')
categories = df['category']

# 2. Implement the strict 70 / 15 / 15 Stratified Split
print("\n Partitioning dataset (70% Train / 15% Val / 15% Test) stratified by category...")

# First split: Separate 70% training data from the remaining 30%
X_train, X_temp, y_train, y_temp, cat_train, cat_temp = train_test_split(
    X, y, categories, test_size=0.30, random_state=42, stratify=categories
)

# Second split: Divide the remaining 30% equally into Validation (15%) and Testing (15%)
X_val, X_test, y_val, y_test, cat_val, cat_test = train_test_split(
    X_temp, y_temp, cat_temp, test_size=0.50, random_state=42, stratify=cat_temp
)

print(f"    Train Matrix Vector Space      : {X_train.shape}")
print(f"    Validation Matrix Vector Space : {X_val.shape}")
print(f"    Test Matrix Vector Space       : {X_test.shape}")

# 3. Scale Features (Crucial for SVM with RBF and Gradient Boosting distance calculations)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# EVALUATION PIPELINE FOR CLASSICAL MODELS
# ==============================================================================
models = {
    " Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    " Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1, max_depth=3),
    " SVM (RBF Kernel)": SVC(kernel='rbf', C=1.0, random_state=42, class_weight='balanced')
}

best_accuracy = 0
best_model_name = ""
best_preds = None

print("\n" + "="*60 + "\n STARTING TRAINING LOGS FOR BASELINE CLASSIFIERS\n" + "="*60)

for name, clf in models.items():
    print(f"\nTraining {name}...")
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate on the validation set to check generalization health
    val_preds = clf.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"    Validation Set Accuracy: {val_acc * 100:.2f}%")
    
    # Evaluate on the final unseen test set
    test_preds = clf.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_preds)
    
    print(f"\n {name.upper()} FINAL TEST SET REPORT:")
    print(classification_report(y_test, test_preds))
    
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        best_model_name = name
        best_preds = test_preds
        best_clf = clf

# ==============================================================================
# REPORT REQUIREMENTS BREAKDOWN
# ==============================================================================
print("\n" + "="*60 + "\n FINAL EVALUATION REPORT FOR THE WINNING MODEL\n" + "="*60)
print(f"Winning Architecture: {best_model_name} with {best_accuracy * 100:.2f}% accuracy\n")

# A. Confusion Matrix Report
print(" 1. CONFUSION MATRIX (Which models get mixed up the most?):")
labels = sorted(y.unique())
cm = confusion_matrix(y_test, best_preds, labels=labels)
cm_df = pd.DataFrame(cm, index=[f"True {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
print(cm_df)

# B. Per-Category Accuracy Evaluation
print("\n 2. ACCURACY BROKEN DOWN BY PROMPT CATEGORY:")
test_df = pd.DataFrame({'True_Label': y_test, 'Predicted': best_preds, 'Category': cat_test})
for category_name, group in test_df.groupby('Category'):
    cat_acc = accuracy_score(group['True_Label'], group['Predicted'])
    print(f"    {category_name:<35}: {cat_acc * 100:.2f}% Accuracy")

import joblib
import os

# Create the absolute folder destination
os.makedirs("D:/Aiml/data/processed", exist_ok=True)

print("\n💾 Serializing Winning Model and Feature Scaler Assets...")

try:
    # 1. Save the ACTUAL winning model tracked by your loop condition
    model_save_path = "D:/Aiml/data/processed/gradient_boosting_model.pkl"
    joblib.dump(best_clf, model_save_path)
    print(f"🎯 SUCCESS! Serialized Winner ({best_model_name.strip()}) to: {model_save_path}")
    
    # 2. Save the fitted scaler so your Streamlit app can normalize inference data perfectly!
    scaler_save_path = "D:/Aiml/data/processed/scaler.pkl"
    joblib.dump(scaler, scaler_save_path)
    print(f"🎯 SUCCESS! Serialized Feature Scaler to: {scaler_save_path}")

except NameError:
    print("❌ Fatal: Could not resolve the 'best_clf' or 'scaler' variable state in memory.")