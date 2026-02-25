"""
Bluesky Bitcoin Historical Data Pipeline
Target: 60k–100k posts
Date Range: May 1, 2022 – Jan 31, 2023
Output: Clean JSON dataset (no images, no embeds)
"""

import json
import time
from datetime import datetime
from atproto import Client

# CONFIG
QUERY = "bitcoin OR btc"
TARGET_POSTS = 100000
START_DATE = datetime(2022, 5, 1)
END_DATE = datetime(2023, 1, 31)
SAVE_EVERY = 1000
OUTPUT_FILE = "bitcoin_bluesky_may2022_jan2023.json"
BATCH_LIMIT = 100  # Max posts per API call

def get_credentials():
    handle = input("Enter Bluesky handle (without @): ").strip()
    password = input("Enter Bluesky App Password: ").strip()
    return handle, password

def is_within_date(created_at: str):
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        return START_DATE <= dt <= END_DATE
    except:
        return False

def save_json(data):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def clean_post(post: dict):
    """
    Takes a plain dict (not Record object) and returns cleaned post.
    """
    try:
        record = post.get("record", {})  # now a dict
        text = record.get("text", "")
        created_at = record.get("created_at", "")
        if not text or not is_within_date(created_at):
            return None

        cleaned = {
            "text": text,
            "created_at": created_at,
            "like_count": post.get("like_count", 0),
            "repost_count": post.get("repost_count", 0),
            "reply_count": post.get("reply_count", 0),
            "uri": post.get("uri", ""),
            "word_count": len(text.split()),
            "char_count": len(text),
        }
        return cleaned
    except Exception as e:
        print(f"[!] Skipping post due to error: {e}")
        return None

def fetch_posts():
    handle, password = get_credentials()
    client = Client()
    client.login(handle, password)
    print(f"[✓] Logged in as {handle}")

    collected = []
    cursor = None

    while len(collected) < TARGET_POSTS:
        try:
            response = client.app.bsky.feed.search_posts(
                params={
                    "q": QUERY,
                    "limit": BATCH_LIMIT,
                    "cursor": cursor,
                    "sort": "latest",
                }
            )

            # Convert all posts to plain dicts immediately
            posts = [p.__dict__ for p in getattr(response, "posts", [])]
            if not posts:
                print("[i] No more posts found.")
                break

            for post in posts:
                cleaned = clean_post(post)
                if cleaned:
                    collected.append(cleaned)

            cursor = getattr(response, "cursor", None)
            print(f"[+] Collected {len(collected)} posts")

            if len(collected) % SAVE_EVERY < BATCH_LIMIT:
                save_json(collected)
                print(f"[✓] Auto-saved at {len(collected)} posts")

            if not cursor:
                print("[i] Reached end of available posts.")
                break

            time.sleep(0.7)

        except Exception as e:
            print(f"[!] Error: {e}")
            time.sleep(5)
            continue

    save_json(collected)
    print(f"\n[✓] Finished. Total posts saved: {len(collected)}")
    print(f"[✓] Output file: {OUTPUT_FILE}")

if __name__ == "__main__":
    fetch_posts()
