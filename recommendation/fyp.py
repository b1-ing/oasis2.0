from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_posts(user, posts,  timestep_time, top_k=5):
    now = timestep_time
    recommendations = []

    for post in posts:
        score = 0

        # --- Popularity Boost ---
        score += post["num_likes"] * 2 + post["num_comments"] * 1.5 + post["num_shares"] * 2.5

        # --- Recency Boost ---
        post_time = datetime.fromisoformat(post["created_utc"].replace("Z", ""))
        hours_old = (now - post_time).total_seconds() / 3600
        # print(post_time, now)
        recency_score = max(0, 10 - hours_old)  # decay after 10h

        score += recency_score


        # --- Keyword Match Boost ---
        for interest in user.get("interests", []):
            if interest.lower() in post["content"].lower():
                score += 3  # custom boost

        recommendations.append((post, score))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return [p for p, s in recommendations[:top_k]]
