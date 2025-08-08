from datetime import datetime
import numpy as np
import string
def recommend_posts(
        posts,
        timestep_time,
        top_k=20,
        weight_recency=0.7,
        weight_popularity=0.3,
        recency_half_life_hours=5.0,
        epsilon=1e-8
):
    now = timestep_time
    n = len(posts)
    recommendations = []

    raw_popularity = []
    raw_recency = []

    for post in posts:
        # Popularity score with weighted components
        pop_score = (
            post.get("num_likes", 0) * 2 +
            post.get("num_comments", 0) * 1.5 +
            post.get("num_shares", 0) * 2.5
        )
        raw_popularity.append(np.log1p(pop_score))  # log(1 + pop_score)

        # Recency score with exponential decay
        post_time = datetime.fromisoformat((post["created_utc"]).replace("Z", ""))
        hours_old = (now - post_time).total_seconds() / 3600.0
        decay = np.exp(-np.log(2) * hours_old / recency_half_life_hours)
        raw_recency.append(decay)

    raw_popularity = np.array(raw_popularity, dtype=float)
    raw_recency = np.array(raw_recency, dtype=float)

    # Normalize arrays to [0, 1]
    def normalize(arr):
        min_v = arr.min()
        max_v = arr.max()
        if max_v - min_v < epsilon:
            return np.zeros_like(arr)
        return (arr - min_v) / (max_v - min_v)

    norm_popularity = normalize(raw_popularity)
    norm_recency = normalize(raw_recency)

    # Combine scores
    for i, post in enumerate(posts):
        score = (
            weight_recency * norm_recency[i] +
            weight_popularity * norm_popularity[i]
        )
        recommendations.append((post, score))

    # Sort and return top_k posts
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return [p for p, s in recommendations[:top_k]]
