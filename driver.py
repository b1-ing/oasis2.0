import json
import os
import random
import asyncio
from datetime import datetime, timedelta

import pandas as pd
import google.generativeai as genai

from posts.analyse_posts import load_posts, apply_action_to_post
from recommendation.fyp import recommend_posts

# ---------- CONFIG ----------
# Prefer setting GEMINI_API_KEY in env: export GEMINI_API_KEY="..."
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBMDhVRQ3zBrfGSDiVoz16ELCwsFGoZ1Eo")
genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash"

START_TIME = datetime(2025, 7, 9, 15, 0, 0)
TIMESTEP_HOURS = 12
NUM_TIMESTEPS = 40
ONLINE_RATE = 0.0075  # ~0.75% of users online per timestep
GROUP_SIZE=20

subreddit = "SecurityCamera"
POSTS_FILE = f"posts/posts.json"
AGENTS_FILE = "agents/agents.json"
OUTPUT_DIR = "output"
LOG_FILE = os.path.join(OUTPUT_DIR, "logs", f"{subreddit}/simulation_log.csv")
POSTS_OUT_FILE = os.path.join(OUTPUT_DIR, "posts", f"{subreddit}/posts.csv")

# ---------- SETUP ----------

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
model = genai.GenerativeModel(MODEL_NAME)

# ---------- MAIN SIMULATION LOOP ----------
for t in range(NUM_TIMESTEPS):
    current_time = START_TIME + timedelta(hours=t * TIMESTEP_HOURS)
    print(f"\n⏰ Timestep {t} — {current_time}")

    # 1. Post new content scheduled at this time
    target_prefix = current_time.strftime("%Y-%m-%d")  # matches date and hour (minute can vary)
    new_posts = [p for p in post_queue if p.get("created_utc", "").startswith(target_prefix)]
    print(new_posts)
    for p in new_posts:
        print(f"📢 New post {p['post_id']} published.")
        posts.append(p)

    # 2. Each agent observes and acts
    online_agents = get_online_agents(agents, ONLINE_RATE)
    for agent in online_agents:
        persona = agent.get("persona", "You are a curious social media user.")
        agent_profile = {
            "topics_of_interest": agent.get("topics", []),
            "comment_style": agent.get("comment_style", "neutral and concise"),
            "posting_frequency": agent.get("posting_frequency", "occasional"),
            "daily_activity_rate": agent.get("daily_activity_rate", 0.02),
        }
        agent_id = agent["id"]
        username = agent["username"]

        # Format prompt
        recommended_posts= recommend_posts(agent, posts, current_time)
        # print("recommended posts for agent", agent_id, recommended_posts)
        posts_str = json.dumps(recommended_posts, indent=2)
        prompt = f"""
        {current_time.isoformat()} - Agent browsing environment:

        Your user profile:
        - Topics of interest: {', '.join(agent_profile['topics_of_interest'])}
        - Commenting style: {agent_profile['comment_style']}
        - Posting frequency: {agent_profile['posting_frequency']}
        - Daily activity rate: {agent_profile["daily_activity_rate"]}
        You are a Reddit user browsing the r/{subreddit} subreddit, where people share personal experiences, ask for advice, and discuss concerns related to National Service in Singapore.

Your goals are to recognize which posts are likely to go viral, respond realistically, and engage in ways typical of this community.




2. Don't feel pressured to comment or like on everything. You have the choice to ignore posts.

The probability that you like/comment is decided by your daily activity rate in your profile.

THIS IS A SCALE OF 0 TO 1. IF YOUR DAILY ACTIVITY RATE IS 0.02, THE PROBABILITY THAT YOU WILL REACT TO A POST IS 0.02.
YOU SHOULD IGNORE/REMAIN INACTIVE FOR THE REMAINING 0.98.

Only like/comment what you truly are interested in.


3. Your Behavior as a Redditor


Don’t overreact; most users are informative or chill.

Ask clarifying questions if something’s unclear, especially in comment replies.




You see the following posts: {posts_str}



        🎯 Respond in this format for POSTS YOU DECIDE TO ACT ON (you do not need to act on everything you see, only what best aligns with your profile!):

        - Action: [like | comment | ignore]
        - Post_ID: [post id]
        - Reason: [brief explanation — use your profile and post content to justify]
        - (If comment) Comment: [realistic Reddit-style reply in your voice]


        FOLLOW STRICTLY THIS RESPONSE FORMAT. DO NOT ADD ADDITIONAL CHARACTERS BEFORE/AFTER eg. *.



        """



        try:
            response = model.generate_content(prompt)
            reply = response.text
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

#
#  ---
#
#         🔍 VIRAL POST RECOGNITION
#         A post may be viral if:
#         - It has 5+ comments
#         - It asks for help (technical/setup/product advice)
#         - It asks for identification (e.g. a camera or person)
#         - It tells a personal story
#         - It has brand names, camera types, or install details
#         - It invites discussion or shared experience
#
#         💬 INTERACTION RULES
#         1. Like if: clear, helpful, matches your interests or is well-written.
#         2. Comment if:
#            - It matches your topics
#            - The author is asking for feedback or identification
#            - You have a strong opinion on the setup or method shown
#
#         3. Ignore if:
#            - It's vague, spammy, off-topic, or very low quality
#            - It's about sales or single events
#
#         🛑 But: **every post must receive at least one action (like, comment)** across all agents. If none of the above fit, you can still react with a neutral like or comment for realism.
#
#         ---
#
#         🧠 EXAMPLES
#         Comment: “That’s a Reolink NVR — I use the same one, works well for rural.”
#         Comment: “Try moving the camera 2 feet higher — better angle.”
#
#         ---