import os
import json
from datetime import datetime, timedelta

subreddit="NationalServiceSG"
START_TIME = datetime(2025, 7, 19, 10, 0, 0)
TIMESTEP_DAYS=1
POSTS_FILE = f"posts/posts_{subreddit}.json"
ONLINE_RATE = 0.0075  # ~0.75% of users online per timestep
SUBREDDIT_SIZE=43000
PROMPTS_DIR = os.path.join("prompts", subreddit)
NUM_TIMESTEPS=10
os.makedirs(PROMPTS_DIR, exist_ok=True)

posts=[]
with open(POSTS_FILE, "r", encoding="utf-8") as f:
    post_queue = json.load(f)

for t in range(NUM_TIMESTEPS):
    current_time = START_TIME + timedelta(days=t * TIMESTEP_DAYS)
    print(f"\n⏰ Timestep {t} — {current_time}")

    # New posts at this timestep
    target_prefix = current_time.strftime("%Y-%m-%d")
    new_posts = [p for p in post_queue if p.get("created_utc", "").startswith(target_prefix)]
    for p in new_posts:
        posts.append(p)

    # Select recent posts for prompt

    posts_str = json.dumps(new_posts, indent=2)

    # Build prompt string
    prompt = f"""
{current_time.isoformat()} -
You are acting as a decision making body of {SUBREDDIT_SIZE*ONLINE_RATE} agents browsing the r/{subreddit} subreddit.

Your goals are to recognize which posts are likely to go viral, have each of the agents respond realistically, and engage in ways typical of this community.

You will simulate 4 timesteps, each being 6 hours. 4 timesteps will add up to be a day.

1. Don't feel pressured to have the agents comment or like on everything. They can ignore posts.

2. Agents' behavior:
- Be realistic, not overly enthusiastic
- You may ask clarifying questions
- Don’t overreact

3. The posts provided already have likes (num_likes) and comments (num_comments). USE THIS COUNT AS YOUR STARTING POINT. DO NOT START FROM 0


Here are the existing posts:
[]

Here are the newly posted posts:
{posts_str}

Respond in this format:

Timestep 1 (0-6 hours)
Post 96: 10 likes, 8 comments
Post 98: 8 likes, 6 comments
Post 99: 15 likes, 12 comments
Post 95: 1 comment
Post 97: 1 comment
Post 100: 2 comments

Timestep 2 (6-12 hours)
Post 96: 25 likes, 20 comments
Post 98: 18 likes, 15 comments
...
ONLY use this format. Do not add anything else.
"""

    # Save prompt to file
    prompt_filename = os.path.join(PROMPTS_DIR, f"timestep_{t}.txt")
    with open(prompt_filename, "w", encoding="utf-8") as f:
        f.write(prompt.strip())

print(f"✅ Saved {NUM_TIMESTEPS} prompts in {PROMPTS_DIR}")
