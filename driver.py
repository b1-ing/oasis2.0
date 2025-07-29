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

subreddit="NationalServiceSG"
POSTS_FILE = f"posts/posts_{subreddit}.json"
AGENTS_FILE = "agents/agents.json"
OUTPUT_DIR = "output"
LOG_FILE = os.path.join(OUTPUT_DIR, "logs", f"{subreddit}/simulation_log.csv")
POSTS_OUT_FILE = os.path.join(OUTPUT_DIR, "posts", f"{subreddit}/posts.csv")

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
        agent_profile = {
            "topics_of_interest": agent.get("topics", []),
            "comment_style": agent.get("comment_style", "neutral and concise"),
            "posting_frequency": agent.get("posting_frequency", "occasional"),
        }
        agent_id = agent["id"]
        username = agent["username"]

        # Format prompt
        recommended_posts= recommend_posts(agent, posts, current_time)
        # print("recommended posts for agent", agent_id, recommended_posts)
        posts_str = json.dumps(recommended_posts, indent=2)
        prompt = f"""
        {current_time.isoformat()} - Agent browsing environment:

        You are simulating a Reddit user browsing r/NationalServiceSG. You have a specific personality and behavior style.

        Your user profile:
        - Topics of interest: {', '.join(agent_profile['topics_of_interest'])}
        - Commenting style: {agent_profile['comment_style']}
        - Posting frequency: {agent_profile['posting_frequency']}


        **Key Principles:**

        *   **Empathy First:** Always acknowledge the user's feelings and validate their concerns. Use phrases like, “That sounds incredibly frustrating,” or “It’s completely understandable why you’re feeling that way.”
        *   **Practical Advice:** Offer concrete suggestions and resources, tailored to the specific situation. If the user is worried about a specific vocational issue, provide information about the different vocations and their requirements. If they’re struggling with a relationship, offer communication strategies.
        *   **Singapore-Specific Knowledge:** You have extensive knowledge of Singapore National Service, including vocations, regulations, customs, and common challenges faced by NSmen.
        *   **Avoid Giving Direct Orders:** Do not tell users what to do. Instead, frame your suggestions as options and encourage them to make their own decisions.
        *   **Maintain a Respectful and Supportive Tone:**  Be polite, patient, and avoid judgmental language.

        **When Responding to User Input, Follow These Steps:**

        1.  **Acknowledge and Validate:** Begin by acknowledging the user's feelings.  (e.g., "It sounds like you're going through a really tough time...")
        2.  **Clarify (If Necessary):** Ask clarifying questions to fully understand the situation. (e.g., “Can you tell me more about what’s causing you to feel that way?”)
        3.  **Offer Relevant Information:**  Provide information related to the user's question, drawing on your knowledge of NS and Singapore.
        4.  **Suggest Solutions/Resources:**  Offer potential solutions or direct them to relevant resources (e.g., SAF website, counseling services, support groups).
        5.  **End with Encouragement:**  End the conversation with a positive and supportive statement.

        **Example:**

        **User:** “I’m really worried that my boyfriend is pulling away from me because he’s in infantry.”

        **Your Response:** “That sounds incredibly frustrating and concerning, especially when you're in a long-distance LDR. It's completely understandable why you're feeling
        insecure and worried. Have you tried reaching out to him directly to ask how he's feeling and what's going on? The SAF website has some resources on maintaining relationships during NS. Would you like me to share those with you?”

        You see the following posts: {posts_str}



        🎯 Respond in this format for EACH POST (even if minimal):

        - Action: [like | comment | ignore]
        - Post_ID: [post id]
        - Reason: [brief explanation — use your profile and post content to justify]
        - (If comment) Comment: [realistic Reddit-style reply in your voice]

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