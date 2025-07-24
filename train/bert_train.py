# train_transformer_regression.py
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np
import os
from pathlib import Path
import shutil
print(transformers.__version__)
# 1. Load your data
df = pd.read_csv("validation/validation.csv")  # Make sure this contains 'post_text' and 'engagement_score'
OUTPUT_DIR = Path("./models/distilbert-regression")
# 2. Preprocess
df = df.dropna(subset=["combined_text", "score"])  # Adjust if your label column is named differently
df = df.rename(columns={"combined_text": "text", "score": "label"})  # 'label' is required by Trainer
df["label"] = df["label"].astype(float)
# 3. Split and Tokenize
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

model_name = "distilbert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=256)

train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
val_dataset = Dataset.from_pandas(val_df).map(tokenize, batched=True)

# 4. Load Model for Regression
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, problem_type="regression")
model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR)

# 5. Trainer
training_args = TrainingArguments(
    output_dir="./models/distilbert-base-uncased",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=6,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    save_safetensors=False,
)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.squeeze(preds)
    mse = np.mean((preds - labels)**2)
    return {"mse": mse, "rmse": np.sqrt(mse)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 6. Train
trainer.train()


# 7. Save
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Model and tokenizer saved to {OUTPUT_DIR}")
