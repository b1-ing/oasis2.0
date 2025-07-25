import json
from datetime import datetime, timedelta
from ollama import chat
from posts.analyse_posts import load_posts, apply_action_to_post
import pandas as pd
import os
from recommendation.fyp import recommend_posts
import random
# ---------- CONFIG ----------
START_TIME = datetime(2025, 7, 8, 8, 0, 0)
TIMESTEP_DAYS = 1
NUM_TIMESTEPS = 20
ONLINE_RATE = 0.01  # 10% of users online per timestep


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

def get_online_agents(agent_data, rate=ONLINE_RATE):
    n_total = len(agent_data)
    n_online = max(1, int(rate * n_total))
    return random.sample(agent_data, n_online)


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
    online_agents = get_online_agents(agents, ONLINE_RATE)
    for agent in online_agents:
        persona = agent.get("persona", "You are a curious social media user.")
        agent_id = agent["id"]
        username = agent["username"]

        # Format prompt
        recommended_posts= recommend_posts(agent, posts, current_time)
        # print("recommended posts for agent", agent_id, recommended_posts)
        posts_str = json.dumps(recommended_posts, indent=2)
        prompt = f"""
{current_time.isoformat()} - Agent observing environment:
You are simulating a Reddit user browsing r/SecurityCamera. Your goal is to read posts, decide whether to interact (like, comment, ignore), and generate realistic user behavior based on Reddit norms.

You have an individual profile detailing topics you're interested in, comment style, and activity rate.

You see the following posts: {posts_str}
Follow these instructions:

🔍 Viral Post Recognition
You should consider a post likely to be viral if it shows any of the following:

Has many comments (5+ is a signal)

Asks for help with product recommendations, technical troubleshooting, or setup validation

Involves identifying a camera, person, or vehicle

Has a personal story or describes a real-life issue (e.g. farm security, package theft)

Includes specific details: brand names, environment (e.g. “rural”, “business”), photos, or install scenarios

Tone invites discussion, feedback, or shared experience

💬 Interaction Rules
1. Like a post if:

It is helpful, clearly written, or resonates with your simulated persona

It presents an interesting or common scenario (e.g. "camera not recording", "wifi not reaching barn")

2. Comment on a post if:

You have knowledge related to the problem or question

The post asks for feedback or identification

You want to agree/disagree with install practices

Avoid low-effort responses—offer relevant opinions, questions, or links to known brands/products

3. Ignore a post if:

It’s vague, uninformative, or lacks a clear question

It’s poorly written or spammy

It has very low engagement and offers no new angle or topic

It has to do with one-off events, eg. online sales; this isnt a shopping subreddit!

For this case, don't need to provide any output; just skip.

DON'T ACT ON EVERY POST.

🧠 Examples of Good Agent Comments:
“That looks like a Reolink RLC-810A — same form factor I use for my setup.”

“I'd avoid wireless for barn setups. PoE works much better in the long run.”

“This install looks sketchy — where’s the junction box?”

“I’ve had the same problem. Turned out the NVR wasn’t getting power from the switch.”

Always explain your reasoning behind the action.

Respond in the following format:
- Action: [action type]
- Post_ID: [post id which your action applies to]
- Reason: [short explanation based on post content and virality signals]



"""

        messages = [
            {"role": "user", "content": prompt}
        ]

        try:
            response = chat(model="gemma3:4b", messages=messages)
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