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
TIMESTEP_HOURS = 6  # window size
START_TIME = datetime(2025, 7, 19, 10, 0, 0)  # adjust if needed
NUM_TIMESTEPS = 40  # how many sequential windows to process

# ---------- UTILITIES ----------

def parse_iso_utc(s):
    # Expecting something like "2025-08-01T13:45:00Z" or without Z
    if not isinstance(s, str):
        return None
    if s.endswith("Z"):
        s = s[:-1]
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None

def load_posts(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def group_posts_by_timestep(posts, start_time, num_steps, step_hours):
    buckets = []
    for t in range(num_steps):
        window_start = start_time + timedelta(hours=t * step_hours)
        window_end = window_start + timedelta(hours=step_hours)
        bucket = [
            p for p in posts
            if (dt := parse_iso_utc(p.get("created_utc", ""))) is not None
               and window_start <= dt < window_end
        ]
        buckets.append((window_start, window_end, bucket))
    return buckets

# ---------- MAIN ----------

def main():
    try:
        all_posts = load_posts(POSTS_FILE)
    except FileNotFoundError:
        print(f"Posts file not found: {POSTS_FILE}")
        return

    # Group into timesteps
    timesteps = group_posts_by_timestep(all_posts, START_TIME, NUM_TIMESTEPS, TIMESTEP_HOURS)

    model = genai.GenerativeModel(model_name=MODEL_NAME)

    results = []
    for idx, (win_start, win_end, posts_in_window) in enumerate(timesteps):
        if not posts_in_window:
            # skip empty windows or optionally log them
            continue

        # Prepare the posts string (truncate if too large)
        posts_str = json.dumps(posts_in_window, indent=2, ensure_ascii=False)
        prompt = f"""
These are the posts on the reddit community r/{SUBREDDIT} between {win_start.isoformat()} and {win_end.isoformat()}:

{posts_str}

Which of these posts do you think will go viral?
Please list post IDs you expect to go viral and briefly justify each choice.
"""

        try:
            response = model.generate_content(
                prompt,
                # optional: control length / temperature etc via additional params if needed
            )
            answer = response.text if hasattr(response, "text") else response.get("output", "")
        except Exception as e:
            print(f"[Timestep {idx}] Error querying model: {e}")
            answer = f"ERROR: {e}"

        entry = {
            "timestep_index": idx,
            "window_start": win_start.isoformat(),
            "window_end": win_end.isoformat(),
            "num_posts": len(posts_in_window),
            "model_response": answer,
            "post_ids": [p.get("post_id") for p in posts_in_window],
        }
        results.append(entry)

        # Save incremental results so you don't lose progress
        out_path = os.path.join(OUTPUT_DIR, f"virality_predictions_{SUBREDDIT}.jsonl")
        with open(out_path, "a", encoding="utf-8") as outf:
            outf.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"[Timestep {idx}] Queried {len(posts_in_window)} posts → response length {len(answer):,}")
        print(answer)

    print("Done. Predictions written to", os.path.join(OUTPUT_DIR, f"virality_predictions_{SUBREDDIT}.jsonl"))

if __name__ == "__main__":
    main()
