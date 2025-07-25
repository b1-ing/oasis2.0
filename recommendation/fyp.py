from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_posts(user, posts, timestep_time, top_k=5):
    now = timestep_time
    recommendations = []

    popularity_scores = []
    recency_scores = []

    for post in posts:
        # --- Calculate Popularity ---
        popularity = post["num_likes"] * 2 + post["num_comments"] * 1.5 + post["num_shares"] * 2.5
        popularity_scores.append(popularity)

        # --- Calculate Recency ---
        post_time = datetime.fromisoformat(post["created_utc"].replace("Z", ""))
        hours_old = (now - post_time).total_seconds() / 3600
        recency = max(0, 10 - hours_old)  # decay after 10 hours
        recency_scores.append(recency)

    # Normalize scores (avoid division by zero)
    def normalize(scores):
        max_score = max(scores) if scores else 1
        if max_score > 0:
            return [s / max_score for s in scores]
        else: return [0 for _ in scores]

    norm_popularity = normalize(popularity_scores)
    norm_recency = normalize(recency_scores)

    print(norm_recency,norm_popularity)

    for i, post in enumerate(posts):
        score = 0

        # --- Weighted Score: 60% Recency, 40% Popularity ---
        score += 0.6 * norm_recency[i]
        score += 0.4 * norm_popularity[i]

        # --- Keyword Match Boost ---
        for interest in user.get("interests", []):
            if interest.lower() in post["content"].lower():
                score += 0.1  # small bonus, not overpowering

        recommendations.append((post, score))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return [p for p, s in recommendations[:top_k]]
