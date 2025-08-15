import json
import os
import time
from json_repair import repair_json
from tqdm import tqdm
import google.generativeai as genai
subreddit="NationalServiceSG"
# --- CONFIGURATION ---
JSON_FILE_PATH = f"posts/posts_{subreddit}_2.json"
BATCH_SIZE = 20
SLEEP_BETWEEN_BATCHES = 1  # seconds, avoid hitting rate limits

# --- INIT GEMINI ---
API_KEY = "AIzaSyBMDhVRQ3zBrfGSDiVoz16ELCwsFGoZ1Eo"  # or hardcode if needed
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-lite")  # or gemini-pro

# --- VIRAL DETECTION PROMPT ---
PROMPT_TEMPLATE = """You're an expert in social media analysis.

Given the following Reddit posts, for each post, determine whether it's likely to be *viral* or *not viral* based on features like title, body, likes, comments, and topic.

Respond with a JSON list. For each post, include:
- "id": post id
- "prediction": "very viral","viral", "not viral", "very not viral"
- "reason": short explanation


Posts:
{posts_json}
"""

# --- LOAD POSTS ---
with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
    posts = json.load(f)

# --- FILTER UNPROCESSED POSTS ---
unprocessed = [p for p in posts]

# --- PROCESS IN BATCHES ---
for i in tqdm   (range(0, len(unprocessed), BATCH_SIZE), desc="Processing batches"):
    batch = unprocessed[i:i + BATCH_SIZE]

    # Use only relevant fields to avoid long prompt
    minimal_batch = [
        {
            "id": p.get("id") or p.get("post_id"),
            "title": p.get("title"),
            "text": p.get("body") or p.get("post_text") or "",
        }
        for p in batch
    ]

    prompt = PROMPT_TEMPLATE.format(
        posts_json=json.dumps(minimal_batch, indent=2, ensure_ascii=False)
    )

    # ✅ Print prompt
    print(f"\n--- Prompt for batch {i//BATCH_SIZE + 1} ---")
    print(prompt)

    # ✅ Save prompt to a file
    prompt_filename = f"prompt_batch_{i//BATCH_SIZE + 1:03d}.txt"
    with open(prompt_filename, "w", encoding="utf-8") as pf:
        pf.write(prompt)

    # ❌ Skip Gemini API call — use prompt manually instead
    # You can later manually paste this prompt into Gemini

    time.sleep(SLEEP_BETWEEN_BATCHES)
# #
#     try:
#         response = model.generate_content(PROMPT_TEMPLATE.format(
#             posts_json=json.dumps(minimal_batch, indent=2)
#         ))
#
#         print("Raw response text:", response.text)
#
#         fixed_json=repair_json(response.text)
#         output = json.loads(fixed_json)
#         result_map = {item["id"]: item for item in output}
#         print(result_map)
#
#         # Update original posts with prediction
#         for post in posts:
#             pid = post.get("id") or post.get("post_id")
#             if pid in result_map:
#                 post["virality_prediction"] = result_map[pid]
#
#     except Exception as e:
#         print(f"❌ Error processing batch {i//BATCH_SIZE + 1}: {e}")
#         continue
#
#     time.sleep(SLEEP_BETWEEN_BATCHES)
#
# # --- SAVE BACK TO JSON ---
# with open(JSON_FILE_PATH, "w", encoding="utf-8") as f:
#     json.dump(posts, f, indent=2, ensure_ascii=False)
#
# print("✅ Finished processing and saved back to file.")
