import json
import os
import random
from datetime import datetime, timedelta
import time
import pandas as pd
from openai import OpenAI  # OpenRouter uses OpenAI-compatible API

from posts.analyse_posts import load_posts, apply_action_to_post, update_posts_csv_from_llm_output
from recommendation.fyp import recommend_posts

# ---------- CONFIG ----------
# Prefer setting OPENROUTER_API_KEY in env: export OPENROUTER_API_KEY="..."
API_KEY = "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"
MODEL_NAME = "meta-llama/llama-3.3-70b-instruct:free"  # free tier llama3 model
model="llama3.3"
START_TIME = datetime(2025, 7, 19, 10, 0, 0)
TIMESTEP_DAYS = 1
NUM_TIMESTEPS = 10
ONLINE_RATE = 0.0075
SUBREDDIT_SIZE = 43000

subreddit = "NationalServiceSG"
POSTS_FILE = f"posts/posts_{subreddit}.json"
AGENTS_FILE = "agents/agents.json"
OUTPUT_DIR = "output"
LOG_FILE = os.path.join(OUTPUT_DIR, "logs", f"{subreddit}/simulation_log.csv")
POSTS_OUT_FILE = os.path.join(OUTPUT_DIR, "posts", f"{subreddit}/posts_{model}.csv")

# ---------- SETUP ----------
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(POSTS_OUT_FILE), exist_ok=True)

with open(POSTS_FILE, "r", encoding="utf-8") as f:
    post_queue = json.load(f)

with open(AGENTS_FILE, "r", encoding="utf-8") as f:
    agents = json.load(f)

# Initialize post and log state
posts = []
logs = []

def get_online_agents(agent_data, rate=ONLINE_RATE):
    n_total = len(agent_data)
    n_online = max(1, int(rate * n_total))
    return random.sample(agent_data, n_online)

# OpenRouter client
client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# ---------- MAIN SIMULATION LOOP ----------
for t in range(NUM_TIMESTEPS):
    current_time = START_TIME + timedelta(days=t * TIMESTEP_DAYS)
    print(f"\n⏰ Timestep {t} — {current_time}")

    # 1. Post new content scheduled at this time
    target_prefix = current_time.strftime("%Y-%m-%d")
    new_posts = [p for p in post_queue if p.get("created_utc", "").startswith(target_prefix)]
    for p in new_posts:
        print(f"📢 New post {p['post_id']} published.")
        posts.append(p)

    # Keep only newest 20
    recommended_posts = sorted(posts, key=lambda x: x.get("created_utc", ""), reverse=True)[:20]
    posts_str = json.dumps(recommended_posts, indent=2)

    prompt = f"""
{current_time.isoformat()} -
You are acting as a decision making body of {SUBREDDIT_SIZE*ONLINE_RATE} agents browsing the r/{subreddit} subreddit.

Your goals are to recognize which posts are likely to go viral, have each of the agents respond realistically, and engage in ways typical of this community.

You will simulate 4 timesteps, each being 6 hours.

1. Don't feel pressured to have the agents comment or like on everything. They can ignore posts.
2. Agents' behavior:
- Be realistic, not overly enthusiastic
- You may ask clarifying questions
- Don’t overreact

3. The posts provided already have likes (num_likes) and comments (num_comments). USE THIS COUNT AS YOUR STARTING POINT. DO NOT START FROM 0

Posts:
{posts_str}

Respond in this format:

Timestep 1 (0-6 hours)
Post 96: 10 likes, 8 comments
Post 98: 8 likes, 6 comments
...

ONLY use this format. Do not add anything else.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an AI that simulates Reddit community activity realistically."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error in LLaMA call: {e}")
        continue

    print(reply)
    posts = update_posts_csv_from_llm_output(reply, posts)
    pd.DataFrame(posts).to_csv(POSTS_OUT_FILE, index=False)
    time.sleep(30)

print("✅ Simulation complete. Logs and posts saved.")
