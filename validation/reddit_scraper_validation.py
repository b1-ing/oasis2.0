import requests
import json
import pandas as pd
from datetime import datetime

def fetch_reddit_json(subreddit, limit=25, sort="new"):
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={limit}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; scraping-bot/1.0)"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")

    posts = response.json()['data']['children']
    results = []

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
    subreddit = "SecurityCamera"
    posts = fetch_reddit_json(subreddit, limit=200)
    csv_file = "validation/validation.csv"
    save_to_csv(posts, csv_file)
    print(f"✅ Extracted {len(posts)} posts from r/{subreddit} and saved to {csv_file}")
