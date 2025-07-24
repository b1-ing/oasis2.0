import json
from datetime import datetime, timedelta
from ollama import chat
from posts.analyse_posts import load_posts, apply_action_to_post
import pandas as pd
import os

# ---------- CONFIG ----------
START_TIME = datetime(2025, 7, 15, 8, 0, 0)
TIMESTEP_DAYS = 1
NUM_TIMESTEPS = 10

POSTS_FILE = "posts/posts.json"
AGENTS_FILE = "agents/agents.json"
OUTPUT_DIR = "output"
LOG_FILE = os.path.join(OUTPUT_DIR, "logs", "simulation_log.csv")
POSTS_OUT_FILE = os.path.join(OUTPUT_DIR, "posts", "posts.csv")

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

# ---------- MAIN SIMULATION LOOP ----------
for t in range(NUM_TIMESTEPS):
    current_time = START_TIME + timedelta(days=t * TIMESTEP_DAYS)
    print(f"\n⏰ Timestep {t} — {current_time}")

    # 1. Post new content scheduled at this time
    new_posts = [p for p in post_queue if p.get("created_utc", "").startswith(current_time.strftime("%Y-%m-%d"))]
    print(new_posts)
    for p in new_posts:
        print(f"📢 New post {p['post_id']} published.")
        posts.append(p)

    # 2. Each agent observes and acts
    for agent in agents:
        persona = agent.get("persona", "You are a curious social media user.")
        agent_id = agent["id"]
        username = agent["username"]

        # Format prompt
        posts_str = json.dumps(posts, indent=2)
        prompt = f"""
{current_time.isoformat()} - Agent observing environment:

You MUST perform at most 3 actions. Do not exceed this number under any circumstance.
ALWAYS state which post_id your action applies to.
After refreshing, you see some posts: {posts_str}



You have the following available actions:
- Refresh
- Like
- Dislike
- Comment (state content right afterwards here)
- Share
- Create new post

Always explain your reasoning behind the action.

Respond in the following format:
- Action: [action type] on Post [id]
- Reason: [short explanation based on post content and virality signals]
STRICTLY adhere to the format for Action.

"""

        messages = [
            {"role": "system", "content": persona},
            {"role": "user", "content": prompt}
        ]

        try:
            response = chat(model="gemma3:1b", messages=messages)
            reply = response["message"]["content"]
        except Exception as e:
            print(f"❌ Error for agent {agent_id}: {e}")
            continue

        posts = apply_action_to_post(posts, reply)

        # Save to log
        logs.append({
            "timestep": t,
            "timestamp": current_time.isoformat(),
            "agent_id": agent_id,
            "username": username,
            "persona": persona,
            "action_text": reply.strip()
        })

        print(f"🧠 Agent {agent_id} says:\n{reply.strip()}\n")
        pd.DataFrame(logs).to_csv(LOG_FILE, index=False)
        pd.DataFrame(posts).to_csv(POSTS_OUT_FILE, index=False)

# ---------- SAVE OUTPUT ----------


print("✅ Simulation complete. Logs and posts saved.")


# 🧠 Community Virality Insight:
# In this community, posts that go viral typically:
# - contain questions in the post text
# - express emotions like *anger*, *frustration*, or *sadness*
# - use keywords such as *booking*, *complaint*, *training*, *weekend*, *saf*
# - have strong emotional sentiment (positive or negative)
# - are longer than 50 words