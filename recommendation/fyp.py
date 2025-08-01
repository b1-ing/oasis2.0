from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_posts(
        user,
        posts,
        timestep_time,
        top_k=10,
        weight_recency=0.6,
        weight_popularity=0.4,
        recency_half_life_hours=5.0,  # controls exponential decay speed
        interest_boost_factor=1.0,     # scale of interest similarity contribution
        epsilon=1e-8
):
    now = timestep_time
    n = len(posts)
    recommendations = []

    # --- Prepare content corpus for interest matching ---
    # If user has interests, create a pseudo-document from them.
    interests = user.get("interests", [])
    use_interest_matching = len(interests) > 0

    # Build TF-IDF over post contents (and optionally the interest "doc")
    corpus = [post["title"] for post in posts]
    if use_interest_matching:
        interest_doc = " ".join(interests)
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(corpus + [interest_doc])  # last row is interest
        post_vecs = tfidf_matrix[:n]
        interest_vec = tfidf_matrix[n]
        # Cosine similarity of each post to the interest vector (shape: (n, ))
        interest_sim = cosine_similarity(post_vecs, interest_vec.reshape(1, -1)).flatten()
    else:
        interest_sim = np.zeros(n)

    # --- Compute raw popularity and recency scores ---
    raw_popularity = []
    raw_recency = []

    for post in posts:
        # popularity: apply log(1 + x) smoothing to avoid domination
        pop_score = (
                post.get("num_likes", 0) * 2
                + post.get("num_comments", 0) * 1.5
                + post.get("num_shares", 0) * 2.5
        )
        raw_popularity.append(np.log1p(pop_score))  # log(1 + pop_score)

        # recency: exponential decay based on hours since creation
        post_time = datetime.fromisoformat(post["created_utc"].replace("Z", ""))
        hours_old = (now - post_time).total_seconds() / 3600.0
        # decay: exp(-ln(2) * hours_old / half_life) so that at half_life it halves
        decay = np.exp(-np.log(2) * hours_old / recency_half_life_hours)
        raw_recency.append(decay)

    raw_popularity = np.array(raw_popularity, dtype=float)
    raw_recency = np.array(raw_recency, dtype=float)

    # --- Normalize popularity and recency to [0,1] ---
    def normalize(arr):
        min_v = arr.min()
        max_v = arr.max()
        if max_v - min_v < epsilon:
            return np.zeros_like(arr)
        return (arr - min_v) / (max_v - min_v)

    norm_popularity = normalize(raw_popularity)
    norm_recency = normalize(raw_recency)
    norm_interest = normalize(interest_sim)  # already in [0,1] but normalized for safety

    # --- Compose final score ---
    for i, post in enumerate(posts):
        score = 0.0
        score += weight_recency * norm_recency[i]
        score += weight_popularity * norm_popularity[i]
        if use_interest_matching:
            # interest similarity adds a boost (can be tuned)
            score += interest_boost_factor * norm_interest[i]

        recommendations.append((post, score))

    # sort and return top_k posts
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return [p for p, s in recommendations[:top_k]]
