"""
Bluesky Data Pipeline
Outputs: X_combined (np.ndarray) ready to feed into your PCA / VAE

Dependencies:
    pip install atproto numpy sentence-transformers scikit-learn
"""

import re
import time
import unicodedata
from datetime import datetime

import numpy as np
from atproto import Client
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler


# ═══════════════════════════════════════════════════════════════════════════════
# 1. INGESTION
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_bluesky_posts(
    query: str,
    limit: int = 300,
    handle: str = None,
    password: str = None,
) -> list[dict]:
    client = Client()

    if handle and password:
        client.login(handle, password)
        print(f"[✓] Authenticated as {handle}")
    else:
        print("[i] Public (unauthenticated) access")

    posts, cursor, fetched = [], None, 0

    while fetched < limit:
        try:
            response = client.app.bsky.feed.search_posts(
                params={"q": query, "limit": min(100, limit), "cursor": cursor}
            )
            batch = response.posts
            if not batch:
                break

            for post in batch:
                record = post.record
                text = getattr(record, "text", "") or ""
                posts.append({
                    "text":         text,
                    "created_at":   getattr(record, "created_at", ""),
                    "like_count":   getattr(post, "like_count", 0) or 0,
                    "repost_count": getattr(post, "repost_count", 0) or 0,
                    "reply_count":  getattr(post, "reply_count", 0) or 0,
                    "has_image":    _has_image(record),
                    "has_link":     _has_link(record),
                    "char_count":   len(text),
                    "word_count":   len(text.split()),
                })

            fetched += len(batch)
            cursor = getattr(response, "cursor", None)
            print(f"  [{fetched}/{limit}] fetched …")

            if not cursor or fetched >= limit:
                break

            time.sleep(0.5)

        except Exception as e:
            print(f"[!] {e}")
            break

    return posts[:limit]


def _has_image(record) -> bool:
    embed = getattr(record, "embed", None)
    if embed is None:
        return False
    embed_type = getattr(embed, "$type", "") or getattr(type(embed), "__name__", "")
    return "image" in str(embed_type).lower()


def _has_link(record) -> bool:
    for facet in (getattr(record, "facets", None) or []):
        for feature in getattr(facet, "features", []):
            if any(k in getattr(type(feature), "__name__", "").lower() for k in ("link", "uri")):
                return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(
    posts: list[dict],
    embed_model: str = "all-MiniLM-L6-v2",
    embedding_weight: float = 0.8,
    structured_weight: float = 0.2,
) -> np.ndarray:
    """
    Returns X : np.ndarray, shape (n_posts, embedding_dim + 15)
    Ready to pass directly into your PCA / VAE.
    """
    if not posts:
        print("[i] No posts found. Returning empty feature matrix.")
        return np.empty((0, 15), dtype=np.float64)

    texts = [p["text"] for p in posts]

    # ── Semantic embeddings ───────────────────────────────────────────────────
    print(f"[→] Encoding with {embed_model} …")
    embeddings = SentenceTransformer(embed_model).encode(
        texts, show_progress_bar=True, batch_size=64
    )  # (n, embedding_dim)

    # ── Engagement ────────────────────────────────────────────────────────────
    log_likes   = np.log1p([p["like_count"]   for p in posts])
    log_reposts = np.log1p([p["repost_count"] for p in posts])
    log_replies = np.log1p([p["reply_count"]  for p in posts])
    engagement  = log_likes + log_reposts * 2 + log_replies

    # ── Structural ────────────────────────────────────────────────────────────
    char_counts    = np.array([p["char_count"]  for p in posts], dtype=float)
    word_counts    = np.array([p["word_count"]  for p in posts], dtype=float)
    has_image      = np.array([p["has_image"]   for p in posts], dtype=float)
    has_link       = np.array([p["has_link"]    for p in posts], dtype=float)
    hashtag_counts = np.array([len(re.findall(r"#\w+", t)) for t in texts], dtype=float)
    mention_counts = np.array([len(re.findall(r"@\w+", t)) for t in texts], dtype=float)
    exclamations   = np.array([t.count("!")    for t in texts], dtype=float)
    questions      = np.array([t.count("?")    for t in texts], dtype=float)
    emoji_counts   = np.array([_count_emojis(t) for t in texts], dtype=float)

    # ── Temporal (cyclical encoding) ──────────────────────────────────────────
    hours    = np.array([_parse_hour(p["created_at"]) for p in posts])
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)

    # ── Assemble + standardise ────────────────────────────────────────────────
    X_structured = np.column_stack([
        log_likes, log_reposts, log_replies, engagement,
        char_counts, word_counts, has_image, has_link,
        hashtag_counts, mention_counts, exclamations, questions, emoji_counts,
        hour_sin, hour_cos,
    ])  # (n, 15)

    X_structured = StandardScaler().fit_transform(X_structured)

    X = np.hstack([embeddings * embedding_weight, X_structured * structured_weight])
    print(f"[✓] Feature matrix ready: {X.shape}")
    return X


def _count_emojis(text: str) -> int:
    return sum(1 for ch in text if unicodedata.category(ch) in ("So", "Sm"))


def _parse_hour(ts: str) -> float:
    try:
        return float(datetime.fromisoformat(ts.replace("Z", "+00:00")).hour)
    except Exception:
        return 12.0


# ═══════════════════════════════════════════════════════════════════════════════
# USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    posts = fetch_bluesky_posts(query="generative art", limit=300)
    if not posts:
        print("[i] No Bluesky posts were returned for this query. Exiting cleanly.")
    else:
        X = build_feature_matrix(posts)