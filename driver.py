import json
import os
import random
import asyncio
from datetime import datetime, timedelta

import pandas as pd
import ollama  # <-- new import
from posts.analyse_posts import load_posts, apply_action_to_post
from recommendation.fyp import recommend_posts

# ---------- CONFIG ----------
MODEL_NAME = "llama3"  # <-- Your local Ollama model
subreddit = "SecurityCamera"
POSTS_FILE = f"posts/posts.json"
AGENTS_FILE = "agents/agents.json"
OUTPUT_DIR = "output"
LOG_FILE = os.path.join(OUTPUT_DIR, "logs", f"{subreddit}/{MODEL_NAME}simulation_log.csv")
POSTS_OUT_FILE = os.path.join(OUTPUT_DIR, "posts", f"{subreddit}/{MODEL_NAME}/posts.csv")

START_TIME = datetime(2025, 7, 9, 15, 0, 0)
TIMESTEP_HOURS = 6
NUM_TIMESTEPS = 60
ONLINE_RATE = 0.0075  # ~0.75% of users online per timestep


# ---------- SETUP ----------
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(POSTS_OUT_FILE), exist_ok=True)

with open(POSTS_FILE, "r", encoding="utf-8") as f:
    post_queue = json.load(f)

with open(AGENTS_FILE, "r", encoding="utf-8") as f:
    agents = json.load(f)

posts = []
logs = []

# ---------- LOCAL CHAT FUNCTION ----------
async def chat_ollama(prompt, model=MODEL_NAME):
    response = ollama.chat(model=model, messages=[
        {"role": "system", "content": "You are a helpful Reddit user agent."},
        {"role": "user", "content": prompt}
    ])
    return response["message"]["content"].strip()

async def process_agent(agent, model, posts, current_time, t):
    try:
        persona = agent.get("persona", "You are a curious social media user.")
        agent_profile = {
            "topics_of_interest": agent.get("topics", []),
            "comment_style": agent.get("comment_style", "neutral and concise"),
            "posting_frequency": agent.get("posting_frequency", "occasional"),
            "daily_activity_rate": agent.get("daily_activity_rate", 0.02),
        }
        agent_id = agent["id"]
        username = agent["username"]

        recommended_posts = recommend_posts(agent, posts, current_time)
        posts_str = json.dumps(recommended_posts, indent=2)

        prompt = f"""
{current_time.isoformat()} - Agent browsing environment:

Your user profile:
- Topics of interest: {', '.join(agent_profile['topics_of_interest'])}
- Commenting style: {agent_profile['comment_style']}
- Posting frequency: {agent_profile['posting_frequency']}
- Daily activity rate: {agent_profile["daily_activity_rate"]}
You are a Reddit user browsing the r/{subreddit} subreddit.

Your goals are to recognize which posts are likely to go viral, respond realistically, and engage in ways typical of this community.

1. Each post has a field called virality_prediction, with a label (either viral or non-viral). YOUR RESPONSES SHOULD USE THIS AS THE GROUND TRUTH.

2. Don't feel pressured to comment or like on everything. You can ignore posts.

Your probability of reacting is defined by your activity rate (e.g. 0.02 = 2% chance of action per post). Ignore posts that are irrelevant or uninteresting.

3. Your behavior:
- Be realistic, not overly enthusiastic
- You may ask clarifying questions
- Don’t overreact

Posts:
{posts_str}

🎯 Respond only to posts you decide to act on (optional):
- Action: [like | comment | ignore]
- Post_ID: [post id]
- Reason: [brief explanation]
- (If comment) Comment: [realistic Reddit-style reply]

ONLY use this format. Do not add anything else.
"""

        reply = await chat_ollama(prompt, model)
        updated_posts = apply_action_to_post(posts, reply)

        log_entry = {
            "timestep": t,
            "timestamp": current_time.isoformat(),
            "agent_id": agent_id,
            "username": username,
            "persona": persona,
            "action_text": reply.strip()
        }

        print(f"🧠 Agent {agent_id} says:\n{reply.strip()}\n")
        return updated_posts, log_entry

    except Exception as e:
        print(f"❌ Error for agent {agent_id}: {e}")
        return posts, None

def get_online_agents(agent_data, rate=ONLINE_RATE):
    n_total = len(agent_data)
    n_online = max(1, int(rate * n_total))
    return random.sample(agent_data, n_online)

# ---------- MAIN SIMULATION LOOP ----------
async def run_simulation(posts, logs):
    for t in range(NUM_TIMESTEPS):
        current_time = START_TIME + timedelta(hours=t * TIMESTEP_HOURS)
        print(f"\n⏰ Timestep {t} — {current_time}")

        # 1. Post new content scheduled at this time
        target_prefix = current_time.strftime("%Y-%m-%d")
        new_posts = [p for p in post_queue if p.get("created_utc", "").startswith(target_prefix)]
        for p in new_posts:
            print(f"📢 New post {p['post_id']} published.")
            posts.append(p)

        # 2. Get online agents and process in parallel
        online_agents = get_online_agents(agents, ONLINE_RATE)
        tasks = [process_agent(agent, MODEL_NAME, posts, current_time, t) for agent in online_agents]
        results = await asyncio.gather(*tasks)

        for updated_posts, log_entry in results:
            if log_entry:
                logs.append(log_entry)
                posts = updated_posts

        pd.DataFrame(logs).to_csv(LOG_FILE, index=False)
        pd.DataFrame(posts).to_csv(POSTS_OUT_FILE, index=False)

# ---------- RUN ----------
asyncio.run(run_simulation(posts, logs))
print("✅ Simulation complete. Logs and posts saved.")
