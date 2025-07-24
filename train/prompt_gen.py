import pandas as pd
from ollama import chat

# Load the CSV file
df = pd.read_csv("validation/validation.csv")

# Convert DataFrame to a string with only the relevant fields
post_summaries = []
for _, row in df.iterrows():
    post = f"""
Post ID: {row['post_id']}
Title: {row['title']}
Score: {row['score']}
Likes: {row['num_likes']}, Dislikes: {row['num_dislikes']}, Comments: {row['num_comments']}, Shares: {row['num_shares']}
Posted at: {row['created_utc']}
Flair: {row['flair']}
Post Text:
{row['combined_text']}
---"""
    post_summaries.append(post)

csv_summary = "\n".join(post_summaries)

# LLM prompt for Gemma
messages = [

    {
        "role": "user",
        "content": f"""
Your main role is to identify which posts are viral, why they are viral, and to produce a prompt for LLM social media agents to mimic the actual behaviour as closely possible.

Below are real Reddit posts scraped from the r/SecurityCamera subreddit, including metadata such as title, author, timestamps, likes, dislikes, comment counts, shares, and the full post text.

---

**Your task:**

1. **Identify which posts are “viral”** based on available metrics (likes, comments, shares). For this, provide responses as such:
Post: [Post id]
Score: [Post score]
Content: [Post content]
Reason: [Why you think it is viral]
2. **Compare viral and non-viral posts**, and **summarize key patterns** that tend to make posts more engaging.
   - Consider tone, timing, length, clarity of problem, specificity of request, or technical detail.
3. Based on this, **generate a prompt for large language model (LLM) agents** participating in a social media simulation. These agents will be browsing and reacting to posts.
   - The prompt should teach the agents how to:
     - Recognize which posts are likely to be viral.
     - Choose whether to comment, like, or ignore posts based on these viral cues.
     - Mimic realistic Reddit behavior in the context of the subreddit’s interests and tone.
4. Return me the best possible prompt ONLY.

---

Here are the posts:
{csv_summary}
"""
    }
]

print(messages)

# Run it through Ollama's Gemma
response = chat(model="gemma3:4b", messages=messages)
print(response["message"]["content"])
