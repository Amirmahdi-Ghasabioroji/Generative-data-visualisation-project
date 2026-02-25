"""
Bluesky Bitcoin Historical Data Pipeline
Target: 60k–100k posts
Date Range: Jan 1, 2024 – Sep 30, 2024
Output: Clean JSON dataset (no images, no embeds)
"""

import json
import time
import getpass
import os
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from atproto import Client

# CONFIG
QUERY = "bitcoin OR btc"
TARGET_POSTS = 100000
START_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2024, 9, 30, 23, 59, 59, 999999, tzinfo=timezone.utc)
SAVE_EVERY = 1000
OUTPUT_FILE = "bitcoin_bluesky_jan2024_sep2024.json"
BATCH_LIMIT = 100  # Max posts per API call

def get_credentials():
    handle = (os.getenv("BLUESKY_HANDLE") or "").strip()
    password = (os.getenv("BLUESKY_APP_PASSWORD") or "").strip()

    if handle and password:
        return handle, password

    if handle and not password:
        print("[i] BLUESKY_HANDLE is set, but BLUESKY_APP_PASSWORD is missing. Prompting for password.")
        password = getpass.getpass("Enter Bluesky App Password: ").strip()
        return handle, password

    print("[i] Environment credentials not fully set. Falling back to interactive prompt.")
    handle = input("Enter Bluesky handle (without @): ").strip()
    password = getpass.getpass("Enter Bluesky App Password: ").strip()
    return handle, password


def _candidate_handles(handle: str) -> list[str]:
    value = (handle or "").strip()
    if value.startswith("@"):
        value = value[1:]

    if not value:
        return []

    candidates = [value]
    if "." not in value:
        candidates.append(f"{value}.bsky.social")

    # Preserve order, remove duplicates
    unique: list[str] = []
    seen = set()
    for item in candidates:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _login_with_retries(client: Client, handle: str, password: str) -> str:
    last_error = None
    for candidate in _candidate_handles(handle):
        try:
            client.login(candidate, password)
            return candidate
        except Exception as e:
            last_error = e

    raise RuntimeError(
        "Authentication failed. Use your full handle (for example: yourname.bsky.social) "
        "and an App Password from Bluesky Settings > App Passwords. "
        f"Last error: {last_error}"
    )


def _get_any(obj, names, default=None):
    if obj is None:
        return default

    if isinstance(obj, dict):
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]

    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value

    return default

def _normalize_timestamp(created_at) -> datetime | None:
    if not created_at:
        return None

    try:
        if isinstance(created_at, datetime):
            dt = created_at
        else:
            ts = str(created_at)
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)

        return dt
    except Exception:
        return None


def is_within_date(created_at):
    dt = _normalize_timestamp(created_at)
    return dt is not None and START_DATE <= dt <= END_DATE


def _format_timestamp(created_at) -> str:
    dt = _normalize_timestamp(created_at)
    if dt is None:
        return ""
    return dt.isoformat().replace("+00:00", "Z")


def _build_query(base_query: str) -> str:
    since = START_DATE.date().isoformat()
    # day-after END_DATE for inclusive upper bound in query language
    until = (END_DATE.date() + timedelta(days=1)).isoformat()
    return f"({base_query}) since:{since} until:{until}"


def _build_query_terms(base_query: str) -> list[str]:
    text = (base_query or "").strip()
    if not text:
        return []

    separators = [" OR ", " or ", "|"]
    terms = [text]
    for separator in separators:
        if separator in text:
            terms = [part.strip() for part in text.split(separator) if part.strip()]
            break

    unique: list[str] = []
    seen = set()
    for term in terms:
        if term not in seen:
            seen.add(term)
            unique.append(term)

    return unique


def _init_stats() -> dict[str, int]:
    return {
        "seen": 0,
        "kept": 0,
        "dropped_duplicate": 0,
        "dropped_missing_text": 0,
        "dropped_date": 0,
    }


def _print_stats(stats: dict[str, int]):
    print(
        "[i] Stats: "
        f"seen={stats['seen']} kept={stats['kept']} "
        f"drop_duplicate={stats['dropped_duplicate']} "
        f"drop_missing_text={stats['dropped_missing_text']} "
        f"drop_date={stats['dropped_date']}"
    )

def save_json(data):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _extract_access_jwt(client: Client) -> str | None:
    session = getattr(client, "_session", None)
    token = _get_any(session, ["access_jwt", "accessJwt"], None)
    if token:
        return token

    get_session = getattr(client, "get_session", None)
    if callable(get_session):
        try:
            session_obj = get_session()
            token = _get_any(session_obj, ["access_jwt", "accessJwt"], None)
            if token:
                return token
        except Exception:
            pass

    return None


def _search_posts_raw(client: Client, query: str, limit: int, cursor: str | None, sort: str = "latest") -> dict:
    params = {
        "q": query,
        "limit": str(limit),
        "sort": sort,
    }
    if cursor:
        params["cursor"] = cursor

    query_string = urllib.parse.urlencode(params)
    token = _extract_access_jwt(client)

    endpoints = [
        "https://bsky.social/xrpc/app.bsky.feed.searchPosts",
        "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts",
    ]

    last_error = None

    for base_url in endpoints:
        url = f"{base_url}?{query_string}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python/urllib",
            "Accept": "application/json",
        }
        if token and "bsky.social" in base_url:
            headers["Authorization"] = f"Bearer {token}"

        request = urllib.request.Request(url=url, headers=headers, method="GET")

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read().decode("utf-8")
                return json.loads(payload)
        except urllib.error.HTTPError as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"Raw search failed across endpoints: {last_error}")


def _get_follower_count(client: Client, actor: str, cache: dict[str, int]) -> int | None:
    if not actor:
        return None

    if actor in cache:
        return cache[actor]

    try:
        profile = client.app.bsky.actor.get_profile(params={"actor": actor})
        followers = _get_any(profile, ["followers_count", "followersCount"], None)
        if followers is None:
            cache[actor] = None
            return None

        followers = int(followers)
        cache[actor] = followers
        return followers
    except Exception:
        cache[actor] = None
        return None

def clean_post(post, client: Client, follower_cache: dict[str, int], stats: dict[str, int]):
    """
    Takes a post object/dict and returns cleaned post.
    """
    try:
        record = _get_any(post, ["record", "value", "post", "payload"], {}) or {}

        text = _get_any(record, ["text", "content", "body"], "") or ""
        created_at_raw = _get_any(record, ["created_at", "createdAt", "time"], "") or ""

        stats["seen"] += 1

        if not text:
            stats["dropped_missing_text"] += 1
            return None

        if not is_within_date(created_at_raw):
            stats["dropped_date"] += 1
            return None

        created_at = _format_timestamp(created_at_raw)

        author = _get_any(post, ["author"], {}) or {}
        author_handle = _get_any(author, ["handle"], "") or ""
        author_did = _get_any(author, ["did"], "") or ""
        actor = author_did or author_handle

        follower_count = _get_follower_count(client, actor, follower_cache)

        cleaned = {
            "text": text,
            "created_at": created_at,
            "like_count": _get_any(post, ["like_count", "likeCount", "likes"], 0) or 0,
            "repost_count": _get_any(post, ["repost_count", "repostCount", "reposts"], 0) or 0,
            "reply_count": _get_any(post, ["reply_count", "replyCount", "replies"], 0) or 0,
            "uri": _get_any(post, ["uri", "post_uri", "id"], "") or "",
            "author_handle": author_handle,
            "author_did": author_did,
            "author_follower_count": follower_count,
            "word_count": len(text.split()),
            "char_count": len(text),
        }
        stats["kept"] += 1
        return cleaned
    except Exception as e:
        print(f"[!] Skipping post due to error: {e}")
        return None

def fetch_posts():
    handle, password = get_credentials()
    client = Client()

    try:
        logged_in_as = _login_with_retries(client, handle, password)
        print(f"[✓] Logged in as {logged_in_as}")
    except Exception as e:
        print(f"[!] {e}")
        return

    collected = []
    seen_uris: set[str] = set()
    follower_cache: dict[str, int] = {}
    next_save_at = SAVE_EVERY
    stats = _init_stats()
    query_terms = _build_query_terms(QUERY)
    if not query_terms:
        print("[!] Query is empty. Nothing to fetch.")
        return

    print(f"[i] Query terms: {query_terms}")
    print(f"[i] Date filter window: {START_DATE.isoformat()} to {END_DATE.isoformat()}")

    for term in query_terms:
        if len(collected) >= TARGET_POSTS:
            break

        search_query = _build_query(term)
        cursor = None
        use_raw_search = False
        print(f"[i] Starting term query: {search_query}")

        while len(collected) < TARGET_POSTS:
            try:
                if use_raw_search:
                    response = _search_posts_raw(
                        client=client,
                        query=search_query,
                        limit=BATCH_LIMIT,
                        cursor=cursor,
                        sort="latest",
                    )
                else:
                    response = client.app.bsky.feed.search_posts(
                        params={
                            "q": search_query,
                            "limit": BATCH_LIMIT,
                            "cursor": cursor,
                            "sort": "latest",
                        }
                    )

                posts = _get_any(response, ["posts", "data", "results"], []) or []
                if not posts:
                    print(f"[i] No more posts for term '{term}'.")
                    break

                for post in posts:
                    cleaned = clean_post(post, client=client, follower_cache=follower_cache, stats=stats)
                    if not cleaned:
                        continue

                    uri = cleaned.get("uri", "")
                    if uri and uri in seen_uris:
                        stats["dropped_duplicate"] += 1
                        continue

                    if uri:
                        seen_uris.add(uri)
                    collected.append(cleaned)

                cursor = _get_any(response, ["cursor", "next", "cursor_str"], None)
                print(f"[+] Collected {len(collected)} posts")
                _print_stats(stats)

                if len(collected) >= next_save_at:
                    save_json(collected)
                    print(f"[✓] Auto-saved at {len(collected)} posts")
                    next_save_at += SAVE_EVERY

                if not cursor:
                    print(f"[i] Reached end of available posts for term '{term}'.")
                    break

                time.sleep(0.7)

            except Exception as e:
                print(f"[!] Error: {e}")
                if not use_raw_search and "validation error for Response" in str(e):
                    print("[i] Switching to raw API mode due to atproto response validation mismatch.")
                    use_raw_search = True
                time.sleep(5)
                continue

    save_json(collected)
    _print_stats(stats)
    print(f"\n[✓] Finished. Total posts saved: {len(collected)}")
    print(f"[✓] Output file: {OUTPUT_FILE}")

if __name__ == "__main__":
    fetch_posts()
