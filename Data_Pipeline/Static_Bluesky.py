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
import os
import sys

# Ensure workspace root is on sys.path so AI_systems can be imported when
# running this script directly.
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# Attempt to force UTF-8 stdout on Windows to avoid UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    try:
        sys.setdefaultencoding  # type: ignore
    except Exception:
        pass

from AI_systems.pca_model import PCA


def save_pca(pca: PCA, path: str):
    """Save PCA state to a .npz file."""
    np.savez(path,
             mean=pca.mean_,
             components=pca.components_,
             explained_variance=pca.explained_variance_)


def load_pca(path: str) -> PCA:
    """Load PCA state from a .npz file and return a PCA instance."""
    data = np.load(path)
    components = data["components"]
    n_components = components.shape[1]
    pca = PCA(n_components=n_components)
    pca.mean_ = data["mean"]
    pca.components_ = components
    pca.explained_variance_ = data["explained_variance"]
    total = np.sum(pca.explained_variance_)
    pca.explained_variance_ratio_ = pca.explained_variance_ / total if total != 0 else None
    return pca


def fit_pca_model(X: np.ndarray, n_components: int = 3, model_path: str | None = None):
    """Fit PCA on X, optionally save, and return reduced coords + PCA instance."""
    pca = PCA(n_components=n_components)
    X = np.asarray(X)
    pca.fit(X)
    X_reduced = pca.transform(X)
    if model_path:
        save_pca(pca, model_path)
    return X_reduced, pca


def transform_with_pca(X: np.ndarray, pca: PCA) -> np.ndarray:
    """Project X using a fitted PCA instance."""
    return pca.transform(np.asarray(X))


def apply_pca(X: np.ndarray, n_components: int = 3, fit: bool = True, model_path: str | None = None):
    """Compatibility wrapper: fit or load then transform, returning (coords, pca)."""
    if fit:
        return fit_pca_model(X, n_components=n_components, model_path=model_path)
    if not model_path:
        raise ValueError("model_path is required when fit=False")
    pca = load_pca(model_path)
    return transform_with_pca(X, pca), pca


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

    def _get_any(obj, names, default=None):
        if obj is None:
            return default
        # dict-style
        if isinstance(obj, dict):
            for n in names:
                if n in obj:
                    return obj[n]
        # attribute-style
        for n in names:
            val = getattr(obj, n, None)
            if val is not None:
                return val
        return default

    posts, cursor, fetched = [], None, 0

    while fetched < limit:
        try:
            response = client.app.bsky.feed.search_posts(
                params={"q": query, "limit": min(100, limit - fetched), "cursor": cursor}
            )

            # If the API returned an error-like response (e.g. success=False), handle it now
            resp_success = _get_any(response, ["success", "ok"], None)
            resp_status = _get_any(response, ["status_code", "status"], None)
            if resp_success is False:
                if resp_status == 401 or "AuthMissing" in str(response):
                    print("[!] Bluesky API returned 401 (AuthMissing). This endpoint requires authentication.")
                    print("    Call fetch_bluesky_posts(handle=..., password=...) or provide valid credentials.")
                    return []
                print(f"[!] Bluesky API error: status={resp_status} response={response}")
                return []

            # response may be an object or a dict depending on client version
            batch = _get_any(response, ["posts", "data", "results"], []) or []

            if not batch:
                # debug-print the raw response to help diagnose API shape issues
                print("[i] Empty batch returned from API; response repr:")
                try:
                    print(response)
                except Exception:
                    pass
                break

            for post in batch:
                record = _get_any(post, ["record", "value", "post", "payload"], {}) or {}
                text = _get_any(record, ["text", "content", "body"], "") or ""
                created_at = _get_any(record, ["created_at", "createdAt", "time"], "") or ""

                like_count = _get_any(post, ["like_count", "likeCount", "likes"], 0) or 0
                repost_count = _get_any(post, ["repost_count", "repostCount", "reposts"], 0) or 0
                reply_count = _get_any(post, ["reply_count", "replyCount", "replies"], 0) or 0

                posts.append({
                    "text":         text,
                    "created_at":   created_at,
                    "like_count":   like_count,
                    "repost_count": repost_count,
                    "reply_count":  reply_count,
                    "has_image":    _has_image(record),
                    "has_link":     _has_link(record),
                    "char_count":   len(text),
                    "word_count":   len(text.split()),
                })

            fetched += len(batch)
            cursor = _get_any(response, ["cursor", "next", "cursor_str"], None)
            print(f"  [{fetched}/{limit}] fetched …")

            if not cursor or fetched >= limit:
                break

            time.sleep(0.5)

        except Exception as e:
            print(f"[!] fetch_bluesky_posts error: {e}")
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
    print(f"[->] Encoding with {embed_model} ...")
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
