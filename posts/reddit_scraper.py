import requests
import json
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
            "num_likes": 0,
            "num_dislikes": 0,
            "num_comments": 0,
            "num_shares": 0,

#             "upvotes": ups,
#             "downvotes": downs,
#             "score": score,
#             "num_comments": data.get("num_comments"),
            "created_utc": datetime.utcfromtimestamp(data.get("created_utc")).isoformat() + "Z",
            "flair": data.get("link_flair_text"),
            "post_text": data.get("selftext") or None
        })

    return results

def save_to_json(data, subreddit):
    filename = f"posts/posts_{subreddit}_h1.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    subreddit = "NationalServiceSG"
    posts = fetch_reddit_json(subreddit, limit=125)
    save_to_json(posts, subreddit)
    print(f"✅ Extracted {len(posts)} posts from r/{subreddit} using the Reddit JSON API")
