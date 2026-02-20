"""
Bluesky Data Pipeline
Outputs: X_combined (np.ndarray) ready to feed into your PCA / VAE

Dependencies:
    pip install atproto numpy sentence-transformers scikit-learn
"""

import re
import time
import unicodedata
import csv
import json
import getpass
from datetime import datetime
from pathlib import Path

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
    pass

from AI_systems.pca_model import PCA


# ═══════════════════════════════════════════════════════════════════════════════
# 1. INGESTION
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_bluesky_posts(
    query: str,
    limit: int = 300,
    handle: str = None,
    password: str = None,
    sort: str = "latest",
    exclude_uris: set[str] | None = None,
) -> list[dict]:
    client = Client()

    # Allow credentials to be supplied via environment variables so the
    # pipeline can run non-interactively.
    handle = (handle or os.getenv("BLUESKY_HANDLE") or "").strip()
    password = (password or os.getenv("BLUESKY_APP_PASSWORD") or "").strip()

    # Support users passing handles like @name.bsky.social
    if handle.startswith("@"):
        handle = handle[1:]

    if handle and password:
        try:
            client.login(handle, password)
            print(f"[✓] Authenticated as {handle}")
        except Exception as e:
            print(f"[!] Login failed for {handle}: {e}")
            return []
    elif handle and not password:
        print("[!] BLUESKY_HANDLE is set but BLUESKY_APP_PASSWORD is missing.")
        print("    Set both variables in the same terminal session.")
        return []
    elif password and not handle:
        print("[!] BLUESKY_APP_PASSWORD is set but BLUESKY_HANDLE is missing.")
        print("    Set both variables in the same terminal session.")
        return []
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
    skipped_existing = 0

    while fetched < limit:
        try:
            response = client.app.bsky.feed.search_posts(
                params={
                    "q": query,
                    "limit": min(100, limit - fetched),
                    "cursor": cursor,
                    "sort": sort,
                }
            )

            # If the API returned an error-like response (e.g. success=False), handle it now
            resp_success = _get_any(response, ["success", "ok"], None)
            resp_status = _get_any(response, ["status_code", "status"], None)
            if resp_success is False:
                if resp_status == 401 or "AuthMissing" in str(response):
                    print("[!] Bluesky API returned 401 (AuthMissing). This endpoint requires authentication.")
                    print("    Provide BLUESKY_HANDLE and BLUESKY_APP_PASSWORD (app password),")
                    print("    or call fetch_bluesky_posts(handle=..., password=...).")
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

                uri = _get_any(post, ["uri", "post_uri", "id"], "") or ""
                if exclude_uris and uri and uri in exclude_uris:
                    skipped_existing += 1
                    continue

                posts.append({
                    "text":         text,
                    "created_at":   created_at,
                    "like_count":   like_count,
                    "repost_count": repost_count,
                    "reply_count":  reply_count,
                    "uri":          uri,
                    "has_image":    _has_image(record),
                    "has_link":     _has_link(record),
                    "char_count":   len(text),
                    "word_count":   len(text.split()),
                })

            fetched = len(posts)
            cursor = _get_any(response, ["cursor", "next", "cursor_str"], None)
            print(f"  [{fetched}/{limit}] fetched …")

            if not cursor or fetched >= limit:
                break

            time.sleep(0.5)

        except Exception as e:
            print(f"[!] fetch_bluesky_posts error: {e}")
            break

    if exclude_uris:
        print(f"[i] Skipped {skipped_existing} posts already present in previous dataset.")

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


def _prompt_bluesky_credentials() -> tuple[str | None, str | None]:
    """
    Prompt for Bluesky credentials in interactive runs.
    Returns (handle, app_password). If input is not provided, returns (None, None).
    """
    print("[i] Enter Bluesky credentials to fetch public posts.")
    print("    Use your handle without @ (example: yourname.bsky.social)")

    try:
        handle = input("Bluesky handle: ").strip()
    except EOFError:
        return None, None

    if not handle:
        return None, None

    if handle.startswith("@"):
        handle = handle[1:]

    print("[i] Password input is hidden; just type and press Enter.")
    try:
        password = getpass.getpass("Bluesky app password: ").strip()
    except (EOFError, Exception):
        print("[i] Hidden input is not available in this terminal. Falling back to visible input.")
        try:
            password = input("Bluesky app password (visible): ").strip()
        except EOFError:
            return None, None

    if not password:
        return None, None

    return handle, password


def _load_previous_uris(output_dir: str) -> set[str]:
    posts_json_path = Path(output_dir) / "posts.json"
    if not posts_json_path.exists():
        return set()

    try:
        with posts_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return set()
        return {
            (item.get("uri") or "").strip()
            for item in data
            if isinstance(item, dict) and (item.get("uri") or "").strip()
        }
    except Exception:
        return set()


def save_static_dataset(
    posts: list[dict],
    X: np.ndarray,
    X_reduced: np.ndarray,
    query: str,
    output_dir: str,
):
    """
    Save one static Bluesky dataset snapshot to disk.

    Artifacts:
            - posts.json        : raw post dictionaries
            - posts.csv         : table view for quick inspection
            - features.npy      : feature matrix for AI systems
            - pca_3d.npy        : 3D coordinates for visualization
            - summary.json      : run metadata and artifact paths
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    posts_json_path = out / "posts.json"
    posts_csv_path = out / "posts.csv"
    features_path = out / "features.npy"
    pca_coords_path = out / "pca_3d.npy"
    summary_path = out / "summary.json"

    with posts_json_path.open("w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "text",
        "created_at",
        "like_count",
        "repost_count",
        "reply_count",
        "has_image",
        "has_link",
        "char_count",
        "word_count",
    ]
    with posts_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(posts)

    np.save(features_path, X)
    np.save(pca_coords_path, X_reduced)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "num_posts": len(posts),
        "feature_shape": list(X.shape),
        "pca_shape": list(X_reduced.shape),
        "artifacts": {
            "posts_json": str(posts_json_path),
            "posts_csv": str(posts_csv_path),
            "features_npy": str(features_path),
            "pca_3d_npy": str(pca_coords_path),
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def run_static_bluesky_pipeline(
    query: str = "generative art",
    limit: int = 300,
    n_components: int = 3,
    output_dir: str | None = None,
    show_plot: bool = True,
):
    """
    Static pipeline: fetch once, build features, fit PCA, save artifacts.
    """
    output_dir = output_dir or os.path.join(ROOT, "Data_Pipeline", "datasets", "bluesky_static")

    env_handle = (os.getenv("BLUESKY_HANDLE") or "").strip()
    env_password = (os.getenv("BLUESKY_APP_PASSWORD") or "").strip()

    handle = env_handle or None
    password = env_password or None

    if not (handle and password):
        print("[i] BLUESKY_HANDLE / BLUESKY_APP_PASSWORD not fully set in environment.")
        prompt_handle, prompt_password = _prompt_bluesky_credentials()
        if prompt_handle and prompt_password:
            handle, password = prompt_handle, prompt_password
            print(f"[i] Using prompted credentials for {handle}")
        else:
            print("[i] No credentials entered. Attempting unauthenticated mode.")

    print(f"[i] Fetch request timestamp: {datetime.now().isoformat(timespec='seconds')}")

    previous_uris = _load_previous_uris(output_dir)
    if previous_uris:
        print(f"[i] Previous dataset has {len(previous_uris)} post URIs; fetching unseen posts first.")

    fetch_pool_limit = max(limit * 3, limit)
    posts_pool = fetch_bluesky_posts(
        query=query,
        limit=fetch_pool_limit,
        handle=handle,
        password=password,
        sort="latest",
        exclude_uris=previous_uris,
    )

    if not posts_pool and previous_uris:
        print("[i] No unseen posts found in latest pool; falling back to latest posts regardless of prior run.")
        posts_pool = fetch_bluesky_posts(
            query=query,
            limit=fetch_pool_limit,
            handle=handle,
            password=password,
            sort="latest",
            exclude_uris=None,
        )

    if len(posts_pool) > limit:
        rng = np.random.default_rng()
        sample_idx = rng.choice(len(posts_pool), size=limit, replace=False)
        posts = [posts_pool[int(i)] for i in sample_idx]
        print(f"[i] Sampled {limit} posts from fresh pool of {len(posts_pool)}.")
    else:
        posts = posts_pool

    if not posts:
        print("[i] No Bluesky posts were returned for this query. Exiting cleanly.")
        print("[i] If you are seeing AuthMissing, provide credentials when prompted or set env vars.")
        return None

    if len(posts) < limit:
        print(f"[i] Retrieved {len(posts)} posts (requested {limit}). Continuing with available data.")

    if posts:
        first_created_at = posts[0].get("created_at", "")
        last_created_at = posts[-1].get("created_at", "")
        print(f"[i] Created_at range in fetched posts: newest={first_created_at} oldest={last_created_at}")

    X = build_feature_matrix(posts)
    if X.shape[0] == 0:
        print("[i] Empty feature matrix after processing. Exiting cleanly.")
        return None

    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    print(f"[✓] PCA reduced matrix ready: {X_reduced.shape}")
    if getattr(pca, "explained_variance_ratio_", None) is not None:
        print(f"[i] Explained variance ratio: {np.array2string(pca.explained_variance_ratio_, precision=4)}")
    preview_rows = min(5, X_reduced.shape[0])
    print(f"[i] PCA preview (first {preview_rows} rows):")
    print(X_reduced[:preview_rows])

    if show_plot and X_reduced.shape[1] >= 3:
        try:
            import matplotlib.pyplot as plt

            pca.plot_3d_scatter(X_reduced, title="Bluesky Static Dataset PCA (3D)")
            print("[i] Showing PCA 3D plot (close the plot window to continue)...")
            plt.show(block=True)
        except Exception as e:
            print(f"[i] Could not display PCA plot in this environment: {e}")

    summary = save_static_dataset(
        posts=posts,
        X=X,
        X_reduced=X_reduced,
        query=query,
        output_dir=output_dir,
    )

    print("[✓] Static dataset snapshot saved")
    print(f"    query: {summary['query']}")
    print(f"    posts: {summary['num_posts']}")
    print(f"    features: {summary['feature_shape']}")
    print(f"    pca_3d: {summary['pca_shape']}")
    print(f"    summary file: {summary['artifacts']}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_static_bluesky_pipeline(query="generative art", limit=300, n_components=3)
