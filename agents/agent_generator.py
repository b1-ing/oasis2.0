import json
import random

# Load scraped Reddit user profiles
with open("user_profiles.json", "r", encoding="utf-8") as f:
    user_profiles = json.load(f)  # Expecting list of 100 profiles

# Example fallback usernames if not in profiles
usernames = [
    "kingpamela", "cybermonk", "techsensei", "paranoidpi", "viralvigilante",
    "justscrollin", "calm_mod", "curiouscat", "angstyteen", "policywatch"
]

# Number of agents to generate
NUM_AGENTS = 1000

# Generate 1000 agents and assign profiles in round-robin
def generate_agents(num_agents, user_profiles):
    agents = []
    for i in range(num_agents):
        profile = user_profiles[i % len(user_profiles)]
        agent = {
            "id": i,
            "username": profile.get("username", usernames[i % len(usernames)] + str(i)),
            "role": "user",
            "post_freq": profile.get("post_freq"),
            "topics": profile.get("topics"),
            "comment_style": profile.get("comment_style"),
            "daily_activity_rate": profile.get("daily_activity_rate"),
            "comment_post_ratio": profile.get("comment_post_ratio"),
        }
        agents.append(agent)
    return agents

# Generate and write to agents.json
agents = generate_agents(NUM_AGENTS, user_profiles)
with open("agents/agents.json", "w", encoding="utf-8") as f:
    json.dump(agents, f, indent=4)
print(f"✅ Generated agents.json with {NUM_AGENTS} agents and assigned 100 Reddit user profiles.")
