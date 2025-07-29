import praw
import time
from collections import Counter, defaultdict
from datetime import datetime
import json
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat

# ---- Setup PRAW ----
reddit = praw.Reddit(
    client_id="G575eRPtPUvq2FLs5HoG-w",
    client_secret="NWyLu_mRq-9hhVWPlSSmCgQFibkFIQ",
    user_agent="oasis2.0",
)

analyzer = SentimentIntensityAnalyzer()

# ---- Step 1: Collect 100 commenters from r/securitycameras ----
def collect_users(subreddit_name="NationalServiceSG", post_limit=50, user_limit=100):
    users = set()
    subreddit = reddit.subreddit(subreddit_name)

    for submission in subreddit.new(limit=post_limit):
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            if comment.author and comment.author.name not in users:
                users.add(comment.author.name)
                if len(users) >= user_limit:
                    return list(users)
        time.sleep(0.5)

    return list(users)


# Define simple tokenizer with stopword removal


# ---- Step 2: Scrape and Profile Each User ----
def profile_user(username, max_items=25):
    try:
        user = reddit.redditor(username)
        comments = list(user.comments.new(limit=max_items))
        posts = list(user.submissions.new(limit=max_items))
    except Exception as e:
        print(f"Error scraping {username}: {e}")
        return None

    if not comments and not posts:
        return None

    # --- Frequency ---
    timestamps = [datetime.utcfromtimestamp(x.created_utc) for x in comments + posts]
    days_active = (max(timestamps) - min(timestamps)).days or 1
    daily_freq = (len(comments) + len(posts)) / days_active

    # --- Topics (TF-IDF) ---
    texts = [c.body for c in comments] + [p.title + " " + p.selftext for p in posts if p.selftext]

# Step 1: Extract TF-IDF without filtering stopwords
    vectorizer = TfidfVectorizer(max_features=50)  # get more features first
    try:
        tfidf = vectorizer.fit_transform(texts)
        keywords = vectorizer.get_feature_names_out()
        weights = tfidf.mean(axis=0).tolist()[0]

        # Step 2: Filter the keywords manually
        filtered = [
            (k, round(float(w), 4))
            for k, w in zip(keywords, weights)
            if k.lower() not in ENGLISH_STOP_WORDS
        ]

        # Step 3: Optionally keep only top 20 filtered keywords
        filtered = sorted(filtered, key=lambda x: -x[1])[:20]

        # Convert to dict
        topics = dict(filtered)

    except Exception as e:
        topics = {}
    # --- Comment Style ---
    avg_len = sum(len(c.body) for c in comments) / len(comments) if comments else 0
    avg_post_len = sum(len(p.title + p.selftext) for p in posts) / len(posts) if posts else 0

    q_marks = sum(c.body.count("?") for c in comments)
    e_marks = sum(c.body.count("!") for c in comments)

    sentiments = [analyzer.polarity_scores(c.body)["compound"] for c in comments]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

    readability_scores = [textstat.flesch_reading_ease(c.body) for c in comments]
    avg_readability = sum(readability_scores) / len(readability_scores) if readability_scores else 0

    profile = {
        "username": username,
        "num_comments": len(comments),
        "num_posts": len(posts),
        "comment_post_ratio": round(len(comments) / (len(posts) or 1), 2),
        "daily_activity_rate": round(daily_freq, 2),
        "comment_subreddits": dict(Counter([c.subreddit.display_name for c in comments])),
        "post_subreddits": dict(Counter([p.subreddit.display_name for p in posts])),
        "topics": topics,
        "comment_style": {
            "avg_length": round(avg_len, 1),
            "questions_per_comment": round(q_marks / len(comments), 2) if comments else 0,
            "exclamations_per_comment": round(e_marks / len(comments), 2) if comments else 0,
            "avg_sentiment": round(avg_sentiment, 3),
            "readability": round(avg_readability, 1),
        },
        "active_hours": dict(Counter([
            datetime.utcfromtimestamp(item.created_utc).hour for item in comments + posts
        ])),
    }

    return profile

# ---- Main Runner ----
if __name__ == "__main__":
    users = collect_users()
    print(f"Collected {len(users)} users.")

    profiles = []

    for i, username in enumerate(users):
        print(f"Profiling {i+1}/{len(users)}: u/{username}")
        prof = profile_user(username)
        if prof:
            profiles.append(prof)
        time.sleep(1)  # avoid rate-limiting

    with open(f"user_profiles_{subreddit}.json", "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)

    print(f"✅ Done! Saved to user_profiles_{subreddit}.json")
