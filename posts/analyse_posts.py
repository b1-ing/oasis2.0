import json
import re

def apply_action_to_post(posts, response_text):
    """Parse LLM response and apply the action to the relevant post."""
    pattern = r"Action:\s*(\w+).*?Post[_ ]ID:\s*(\d+)"
    matches = re.findall(pattern, response_text, re.IGNORECASE | re.DOTALL)

    if not matches:
        print("[!] No valid action found.")
        return posts

    for action, post_id in matches:
        action = action.lower()
        post_id = int(post_id)

        # Find and update the post
        updated = False
        for post in posts:
            if post.get("post_id") == post_id:
                if action == "like":
                    post["num_likes"] += 1
                elif action == "dislike":
                    post["num_dislikes"] += 1
                elif action == "share":
                    post["num_shares"] += 1
                elif action == "comment":
                    post["num_comments"] += 1
                    # Just append a dummy comment for now
                updated = True
                print(f"✅ Applied action '{action}' to post {post_id}")
                print(post)
                break

        if not updated:
            print(f"[!] Post {post_id} not found.")

    return posts

def load_posts(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def analyse_posts(posts):
    stats = {
        "total_posts": len(posts),
        "total_likes": 0,
        "total_dislikes": 0,
        "total_shares": 0,
        "total_reports": 0,
        "total_comments": 0,
    }

    for post in posts:
        stats["total_likes"] += post.get("num_likes", 0)
        stats["total_dislikes"] += post.get("num_dislikes", 0)
        stats["total_shares"] += post.get("num_shares", 0)
        stats["total_reports"] += post.get("num_reports", 0)
        stats["total_comments"] += len(post.get("comments", []))

    return stats

def format_posts_for_prompt(posts, limit=3):
    formatted = []
    for post in posts[:limit]:
        snippet = post["content"].split("\n")[0][:120]  # First line, max 120 chars
        formatted.append(
            f"- Post ID {post['post_id']}: \"{snippet.strip()}...\" "
            f"(Likes: {post['num_likes']}, Comments: {len(post['comments'])})"
        )
    return "\n".join(formatted)

# if __name__ == "__main__":
#     posts = load_posts("posts.json")  # Replace with your actual file path
#     stats = analyze_posts(posts)
#     print("📊 Post Summary Stats:")
#     for k, v in stats.items():
#         print(f"{k}: {v}")
#
#     print("\n📝 Sample Post Snippets:")
#     print(format_posts_for_prompt(posts))
