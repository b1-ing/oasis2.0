import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from transformers.trainer_callback import EarlyStoppingCallback
import datasets

# ===== CONFIG =====
MODEL_NAME = "roberta-base"  # can switch to distilroberta-base for lighter
subreddit="NationalServiceSG"
INPUT_CSV = f"validation/validation_{subreddit}_labelled.csv"  # adjust path if needed
OUTPUT_DIR = "models/roberta_viral_classifier"
RANDOM_SEED = 42
BATCH_SIZE = 16
NUM_EPOCHS = 15
MAX_LENGTH = 256  # tokenization truncation length

# ===== UTILITIES =====
def load_dataframe():
    df = pd.read_csv(INPUT_CSV)
    if "viral" not in df.columns:
        raise KeyError("CSV must contain 'viral' column (0/1 label).")
    # Determine text column
    if "combined_text" in df.columns:
        df["text"] = df["combined_text"]
    elif "post_text" in df.columns:
        df["text"] = df["post_text"]
    elif "text" in df.columns:
        pass
    else:
        raise KeyError("CSV must contain one of 'combined_text', 'post_text', or 'text' columns.")
    df = df.dropna(subset=["text", "viral"])
    df["viral"] = df["viral"].astype(int)
    return df

def compute_class_weights(labels):
    # balanced weights for binary
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    weight_map = {c: w for c, w in zip(classes, weights)}
    return weight_map  # e.g. {0: w0, 1: w1}

# Custom Trainer to inject sample weights into loss
# Custom Trainer to inject sample weights into loss
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        # Forward pass (exclude labels if model expects them separately)
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits  # shape (batch, num_labels)
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.label_weights.to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    confidences = probs[:, 1]  # probability of positive 'viral' class

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    try:
        roc_auc = roc_auc_score(labels, confidences)
    except ValueError:
        roc_auc = float("nan")
    precision, recall, _, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

    return {
        "accuracy": acc,
        "f1": f1,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
    }

def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    df = load_dataframe()
    train_df, val_df = train_test_split(df, test_size=0.3, stratify=df["viral"], random_state=RANDOM_SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,  # viral vs nonviral
        problem_type="single_label_classification"
    )

    # Compute class weights and attach to trainer later
    class_weights = compute_class_weights(train_df["viral"].values)
    weight_tensor = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float)

    # Prepare HuggingFace datasets
    train_dataset = datasets.Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = datasets.Dataset.from_pandas(val_df.reset_index(drop=True))

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=MAX_LENGTH)

    train_dataset = train_dataset.map(tokenize_batch, batched=False)
    val_dataset = val_dataset.map(tokenize_batch, batched=False)

    # Rename label column
    train_dataset = train_dataset.rename_column("viral", "labels")
    val_dataset = val_dataset.rename_column("viral", "labels")

    # Set format
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        logging_strategy="epoch",
        push_to_hub=False,
        fp16=torch.cuda.is_available(),
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # attach class weight tensor so compute_loss can use it
    trainer.label_weights = weight_tensor

    # Train
    trainer.train()

    # Save
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Evaluate
    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)

    # Save predictions with confidence
    preds_output = trainer.predict(val_dataset)
    logits = preds_output.predictions
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pred_labels = np.argmax(probs, axis=1)
    confidences = probs[:, 1]
    out_df = val_df.copy()
    out_df["predicted_label"] = pred_labels
    out_df["predicted_proba"] = confidences
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_df.to_csv(os.path.join(OUTPUT_DIR, "val_predictions.csv"), index=False)
    print(f"Saved validation predictions to {os.path.join(OUTPUT_DIR, 'val_predictions.csv')}")

if __name__ == "__main__":
    main()
