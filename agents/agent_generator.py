import json
import random

# Example personas you can customize or expand
personas = [
    "You are a tech-focused security enthusiast.",
    "You are a privacy-concerned, emotionally driven person.",
    "You are a casual user who reacts impulsively to emotional posts.",
    "You are a social media influencer who looks for viral content to share.",
    "You are a skeptical user who frequently questions information.",
    "You are a bored student looking for controversial content.",
    "You are a disciplined researcher who only comments on factual posts.",
    "You are a helpful community moderator with a positive attitude."
]

# Optional usernames for realism
usernames = [
    "kingpamela", "cybermonk", "techsensei", "paranoidpi", "viralvigilante",
    "justscrollin", "calm_mod", "curiouscat", "angstyteen", "policywatch"
]

# Number of agents to generate
NUM_AGENTS = 15


def generate_agents(num_agents):
    agents = []
    for i in range(num_agents):
        agent = {
            "id": i,
            "username": usernames[i % len(usernames)],
            "role": "user",
        }
        agents.append(agent)
    return agents

# Generate and write to agents.json
agents = generate_agents(NUM_AGENTS)
with open("agents/agents.json", "w", encoding="utf-8") as f:
    json.dump(agents, f, indent=4)
print(f"✅ Generated agents.json with {NUM_AGENTS} agents.")
