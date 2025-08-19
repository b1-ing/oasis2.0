import json
import os
from datetime import datetime, timedelta
import google.generativeai as genai

# ---------- CONFIG ----------
# Better practice: put your API key in an environment variable instead of hardcoding
# Example: export GEMINI_API_KEY="..." in shell, then:
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBMDhVRQ3zBrfGSDiVoz16ELCwsFGoZ1Eo")
genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash"

SUBREDDIT = "NationalServiceSG"
POSTS_FILE = f"posts/posts_{SUBREDDIT}.json"
OUTPUT_DIR = "output/virality_predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- UTILITIES ----------

def load_posts(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- MAIN ----------

def main():
    try:
        all_posts = load_posts(POSTS_FILE)
    except FileNotFoundError:
        print(f"Posts file not found: {POSTS_FILE}")
        return

    model = genai.GenerativeModel(model_name=MODEL_NAME)

    # Convert all posts to JSON string (truncate if very large)
    posts_str = json.dumps(all_posts, indent=2, ensure_ascii=False)

    prompt = f"""
Here are posts from the reddit community r/{SUBREDDIT}:

{posts_str}

Some posts (e.g., 91, 92, 94, 95) were viral.
Based on these, identify the features that make posts go viral in this community.
"""

    try:
        response = model.generate_content(prompt)
        answer = response.text if hasattr(response, "text") else response.get("output", "")
    except Exception as e:
        print(f"Error querying model: {e}")
        answer = f"ERROR: {e}"

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "num_posts": len(all_posts),
        "model_response": answer,
        "post_ids": [p.get("post_id") for p in all_posts],
    }

    out_path = os.path.join(OUTPUT_DIR, f"virality_predictions_{SUBREDDIT}.jsonl")
    with open(out_path, "a", encoding="utf-8") as outf:
        outf.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Queried {len(all_posts)} posts → response length {len(answer):,}")
    print(answer)

if __name__ == "__main__":
    main()