# make_viral_label.py
import pandas as pd
subreddit="NationalServiceSG"
INPUT_CSV = f"validation/validation_{subreddit}_trimmed.csv"  # adjust path if needed
OUTPUT_CSV = f"validation/validation_{subreddit}_labelled.csv"

def main():
    df = pd.read_csv(INPUT_CSV)
    if "score" not in df.columns:
        raise KeyError("Input CSV must have a 'score' column to base virality on.")

    # compute 70th percentile threshold
    threshold = df["score"].quantile(0.70)
    print(f"70th percentile of score is {threshold:.3f}")

    # create binary label
    df["viral"] = (df["score"] >= threshold).astype(int)

    # optional: inspect balance
    counts = df["viral"].value_counts()
    pct = df["viral"].value_counts(normalize=True) * 100
    print("Viral label distribution:")
    print(pd.DataFrame({"count": counts, "percent": pct.round(2)}))

    # save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved labeled CSV to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
