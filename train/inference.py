import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# CONFIG - adjust these paths
MODEL_DIR = "models/roberta_viral_classifier"
subreddit="NationalServiceSG"
INPUT_CSV = f"validation/validation_{subreddit}.csv"  # adjust path if needed
OUTPUT_CSV = f"validation/validation_{subreddit}_inference.csv"
MAX_LENGTH = 256

def run_inference():
    # Load tokenizer and model from saved directory
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    # Load your new data CSV
    df = pd.read_csv(INPUT_CSV)

    # Ensure your CSV has a column with text, e.g., 'text', or 'post_text'
    if "combined_text" in df.columns:
        texts = df["combined_text"].tolist()
    elif "post_text" in df.columns:
        texts = df["post_text"].tolist()
    else:
        raise KeyError("Input CSV must contain a 'text' or 'post_text' column for inference.")


# Check what types are inside texts
    print("Sample entries and their types:")
    for i, t in enumerate(texts[:10]):
        print(f"{i}: {repr(t)} (type: {type(t)})")

    # Check if any element is not a str or empty
    bad_items = [(i, t) for i, t in enumerate(texts) if not isinstance(t, str)]
    print(f"Number of non-str items: {len(bad_items)}")
    if bad_items:
        print("Some bad items:", bad_items[:5])
    # Tokenize texts (batch tokenization)
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")

    # Move tensors to device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    # Forward pass (no gradients needed)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

    # Get predicted labels and confidence of positive class (viral=1)
    pred_labels = torch.argmax(probs, dim=1).cpu().numpy()
    pred_probs = probs[:, 1].cpu().numpy()  # probability for class 1 (viral)

    # Append predictions to dataframe
    df["predicted_label"] = pred_labels
    df["predicted_probability"] = pred_probs

    # Save results
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Inference complete. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_inference()
