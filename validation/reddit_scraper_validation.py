import requests
import json
import pandas as pd
from datetime import datetime

def fetch_pushshift_posts(subreddit, after_timestamp, size=100):
    url = f"https://api.pushshift.io/reddit/search/submission/"
    params = {
        "subreddit": subreddit,
        "after": after_timestamp,
        "size": size,
        "sort": "asc",  # oldest first
    }
    response = requests.get(url, params=params)
    data = response.json().get("data", [])

    print(data)

    for idx, post in enumerate(posts, start=1):
        data = post['data']
        ups = data.get("ups", 0)
        score = data.get("score", 0)
        downs = max(0, ups - score)

        results.append({
            "post_id": idx,
            "title": data.get("title"),
            "author": data.get("author"),
            "url": "https://www.reddit.com" + data.get("permalink"),
            "num_likes": ups,
            "num_dislikes": downs,
            "num_comments": data.get("num_comments"),
            "num_shares": 0,
            "created_utc": datetime.utcfromtimestamp(data.get("created_utc")).isoformat() + "Z",
            "flair": data.get("link_flair_text"),
            "post_text": data.get("selftext") or "",
            "combined_text":data.get("title") + data.get("selftext") or "",
            "score": ups+downs+data.get("num_comments"),
        })

    return results

def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding="utf-8")

if __name__ == "__main__":
    subreddit = "NationalServiceSG"
    after_dt = datetime(2025, 7, 1)
    after_ts = int(after_dt.timestamp())

    posts = fetch_pushshift_posts(subreddit, after_ts, size=200)
    csv_file = f"validation/validation_{subreddit}.csv"
    save_to_csv(posts, csv_file)
    print(f"✅ Extracted {len(posts)} posts from r/{subreddit} and saved to {csv_file}")
