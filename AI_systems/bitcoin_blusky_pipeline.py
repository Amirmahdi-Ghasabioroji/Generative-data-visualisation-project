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
import re
import uuid
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path
from atproto import Client

try:
    import joblib
except Exception:
    joblib = None

try:
    import tensorflow as tf
except Exception:
    tf = None

# ================== EDIT THESE SETTINGS ==================
QUERY = "bitcoin OR btc"
EXTRA_QUERY_TERMS = ["crypto"]
SEARCH_LANGUAGE = "en"
TARGET_POSTS = 150000
START_DATE = datetime(2024, 6, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 1, 1, 23, 59, 59, 999999, tzinfo=timezone.utc)
OUTPUT_FILE = "bitcoin_bluesky_june2024_jan2025.json"
RUN_REPORT_FILE = "bitcoin_bluesky_run_report_2024_june_jan2025.json"
MIN_RELEVANCE_SCORE = 0.25
SENTIMENT_NEUTRAL_BAND = 0.15
ML_SENTIMENT_BLEND = 0.7 # 70% lexical (post-level) + 30% cluster profile

# AI Model Settings
USE_AI_MODEL = True  # Set to True to use trained Keras model + KMeans
MODEL_DIR = "AI_systems/scraper_model_artifacts"  # Path to model artifacts
# =========================================================

# Internal config (do not edit)
SAVE_EVERY = 1000
BATCH_LIMIT = 100  # Max posts per API call
WINDOW_DAYS = 30
MAX_CONSECUTIVE_ERRORS_PER_TERM = 3
DEFAULT_MODEL_DIR = MODEL_DIR

# Lightweight AI-style enrichment configuration (rule + lexicon based)
BITCOIN_TERMS = {
    "bitcoin", "btc", "satoshi", "hodl", "halving", "mining", "onchain",
    "lightning", "utxo", "whale", "cold wallet", "hot wallet", "etf", "spot etf",
}
MARKET_TERMS = {
    "price", "volatility", "bull", "bear", "breakout", "support", "resistance",
    "rally", "dump", "pump", "liquidation", "volume", "market cap", "trend",
    "ath", "drawdown", "correction", "momentum", "return",
}
SPAM_TERMS = {
    "airdrop", "giveaway", "referral", "promo", "dm me", "guaranteed profit",
    "100x", "moonshot signal", "join my group", "vip signals",
}
BULLISH_TERMS = {
    "bullish", "buy", "long", "rally", "breakout", "green", "uptrend",
    "strength", "accumulate", "recovery", "optimism", "greed",
    "hodl", "strong", "surge", "soar", "moon", "pump", "explosion",
    "bull run", "upside", "relief", "gain", "gains", "profit", "momentum",
    "advance", "buyers", "buying",
}
BEARISH_TERMS = {
    "bearish", "sell", "short", "crash", "dump", "red", "downtrend",
    "weakness", "panic", "fear", "drawdown", "recession",
    "decline", "drop", "plunge", "collapse", "bear", "sellers", "selling",
    "liquidation", "rekt", "bleed", "correction",
}

TOPIC_KEYWORDS = {
    "price_action": {"price", "breakout", "support", "resistance", "trend", "ath", "return"},
    "volatility": {"volatility", "liquidation", "whipsaw", "squeeze", "drawdown"},
    "macro": {"fed", "rates", "inflation", "cpi", "macro", "dollar", "treasury"},
    "regulation": {"sec", "regulation", "compliance", "ban", "legal", "lawsuit"},
    "etf": {"etf", "spot etf", "approval", "inflow", "outflow"},
    "adoption": {"adoption", "merchant", "payments", "lightning", "wallet", "onchain"},
}

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


def _build_query(base_query: str, since_date, until_date_exclusive) -> str:
    since = since_date.isoformat()
    until = until_date_exclusive.isoformat()
    return f"({base_query}) lang:{SEARCH_LANGUAGE} since:{since} until:{until}"


def _build_time_slices(start_dt: datetime, end_dt: datetime, window_days: int = WINDOW_DAYS) -> list[tuple]:
    slices: list[tuple] = []

    start_date = start_dt.date()
    end_exclusive = end_dt.date() + timedelta(days=1)

    current_end = end_exclusive
    while current_end > start_date:
        current_start = max(start_date, current_end - timedelta(days=window_days))
        slices.append((current_start, current_end))
        current_end = current_start

    return slices


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

    for term in EXTRA_QUERY_TERMS:
        term = (term or "").strip()
        if term and term not in seen:
            seen.add(term)
            unique.append(term)

    return unique


def _is_english_post(record) -> bool:
    langs = _get_any(record, ["langs", "languages"], None)
    if not langs:
        return False

    if isinstance(langs, str):
        langs = [langs]

    normalized = [str(lang).strip().lower() for lang in langs if str(lang).strip()]
    return any(lang == "en" or lang.startswith("en-") for lang in normalized)


def _init_stats() -> dict[str, int]:
    return {
        "seen": 0,
        "kept": 0,
        "classified_ml": 0,
        "classified_rules": 0,
        "dropped_duplicate": 0,
        "dropped_missing_text": 0,
        "dropped_non_english": 0,
        "dropped_date": 0,
        "dropped_low_relevance": 0,
        "dropped_spam": 0,
    }


def _print_stats(stats: dict[str, int]):
    print(
        "[i] Stats: "
        f"seen={stats['seen']} kept={stats['kept']} "
        f"ml_classified={stats['classified_ml']} rules_classified={stats['classified_rules']} "
        f"drop_duplicate={stats['dropped_duplicate']} "
        f"drop_missing_text={stats['dropped_missing_text']} "
        f"drop_non_english={stats['dropped_non_english']} "
        f"drop_date={stats['dropped_date']} "
        f"drop_low_relevance={stats['dropped_low_relevance']} "
        f"drop_spam={stats['dropped_spam']}"
    )

def save_json(data):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_run_report(report: dict):
    with open(RUN_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _contains_any(text: str, terms: set[str]) -> int:
    count = 0
    for term in terms:
        if term in text:
            count += 1
    return count


def _is_probable_spam(text: str) -> bool:
    url_count = len(re.findall(r"https?://", text))
    spam_hits = _contains_any(text, SPAM_TERMS)
    return spam_hits > 0 or url_count >= 3


def _relevance_score(text: str, query_term: str) -> tuple[float, str]:
    normalized = _normalize_text(text)
    if not normalized:
        return 0.0, "irrelevant"

    btc_hits = _contains_any(normalized, BITCOIN_TERMS)
    market_hits = _contains_any(normalized, MARKET_TERMS)
    query_hits = 1 if _normalize_text(query_term) in normalized else 0
    has_ticker = 1 if "$btc" in normalized or " btc " in f" {normalized} " else 0

    raw = (
        0.35 * min(btc_hits, 3)
        + 0.25 * min(market_hits, 3)
        + 0.25 * query_hits
        + 0.15 * has_ticker
    )
    score = max(0.0, min(1.0, raw))

    if score >= 0.75:
        label = "high"
    elif score >= 0.45:
        label = "medium"
    elif score >= MIN_RELEVANCE_SCORE:
        label = "low"
    else:
        label = "irrelevant"
    return score, label


def _fear_greed_score(text: str) -> tuple[float, str]:
    normalized = _normalize_text(text)
    bullish = _contains_any(normalized, BULLISH_TERMS)
    bearish = _contains_any(normalized, BEARISH_TERMS)
    total = bullish + bearish
    if total == 0:
        return 0.0, "neutral"

    # Range: [-1, 1] where -1=fear/bearish and +1=greed/bullish
    score = (bullish - bearish) / total
    return round(float(score), 4), _sentiment_from_score(score)


def _sentiment_from_score(score: float) -> str:
    if score >= SENTIMENT_NEUTRAL_BAND:
        return "greed"
    if score <= -SENTIMENT_NEUTRAL_BAND:
        return "fear"
    return "neutral"


def _extract_topic_tags(text: str) -> list[str]:
    normalized = _normalize_text(text)
    tags: list[str] = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            tags.append(topic)
    return tags


def _build_run_report(
    run_id: str,
    stats: dict[str, int],
    collected_count: int,
    query_terms: list[str],
    ai_mode: str,
    model_dir: str,
) -> dict:
    now = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    return {
        "run_id": run_id,
        "generated_at": now,
        "query": QUERY,
        "query_terms": query_terms,
        "language": SEARCH_LANGUAGE,
        "target_posts": TARGET_POSTS,
        "date_range": {
            "start": START_DATE.isoformat(),
            "end": END_DATE.isoformat(),
        },
        "output_file": OUTPUT_FILE,
        "posts_saved": collected_count,
        "filtering_stats": stats,
        "classification_counts": {
            "ml": stats.get("classified_ml", 0),
            "rules": stats.get("classified_rules", 0),
        },
        "relevance_threshold": MIN_RELEVANCE_SCORE,
        "ai_mode": ai_mode,
        "model_dir": model_dir,
    }


def _parse_iso_date(value: str, end_of_day: bool = False) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    if end_of_day:
        return parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
    return parsed.replace(hour=0, minute=0, second=0, microsecond=0)



class AIInferenceEngine:
    """
        AI inference engine for the scraper.

        Preferred unsupervised artifacts in model_dir:
            - model.keras
            - kmeans.joblib
            - cluster_profiles.json
            - thresholds.json

        If artifacts (or dependencies) are unavailable, falls back to rule scoring.
    """

    def __init__(self, use_ai_model: bool, model_dir: str):
        self.enabled = False
        self.model_dir = Path(model_dir)
        self.model_source = "rules"
        self.mode = "rules"
        self.prediction_errors = 0

        # Unsupervised Keras + KMeans artifacts
        self.keras_model = None
        self.embedding_extractor = None
        self.kmeans_model = None
        self.cluster_profiles = {}
        self.thresholds = {}

        if not use_ai_model:
            return

        if self._try_enable_keras_unsupervised():
            return

        print("[i] AI model artifacts not found or invalid. Falling back to rule-based scoring.")

    def _try_enable_keras_unsupervised(self) -> bool:
        if tf is None or joblib is None:
            return False

        keras_path = self.model_dir / "model.keras"
        kmeans_path = self.model_dir / "kmeans.joblib"
        profiles_path = self.model_dir / "cluster_profiles.json"
        thresholds_path = self.model_dir / "thresholds.json"

        required = [keras_path, kmeans_path, profiles_path, thresholds_path]
        if not all(path.exists() for path in required):
            return False

        try:
            self.keras_model = tf.keras.models.load_model(keras_path)
            self.embedding_extractor = tf.keras.Model(
                self.keras_model.input,
                self.keras_model.get_layer("latent").output,
            )
            self.kmeans_model = joblib.load(kmeans_path)

            with profiles_path.open("r", encoding="utf-8") as f:
                self.cluster_profiles = json.load(f)

            with thresholds_path.open("r", encoding="utf-8") as f:
                self.thresholds = json.load(f)

            self.enabled = True
            self.model_source = "ml-unsupervised"
            self.mode = "keras-unsupervised"
            print(f"[✓] Unsupervised Keras+KMeans mode enabled from {self.model_dir}")
            return True
        except Exception as e:
            print(f"[i] Could not load Keras unsupervised artifacts: {e}")
            return False

    def predict(self, text: str) -> dict:
        if not self.enabled:
            return {"source": "rules"}
        try:
            return self._predict_unsupervised(text)
        except Exception as e:
            self.prediction_errors += 1
            if self.prediction_errors <= 3:
                print(f"[i] ML prediction failed for a post, using rule fallback: {e}")
            return {"source": "rules"}

    def _predict_unsupervised(self, text: str) -> dict:
        model_input = tf.constant([text], dtype=tf.string)
        vector = self.embedding_extractor.predict(model_input, verbose=0)
        cluster_id = int(self.kmeans_model.predict(vector)[0])

        profile = self.cluster_profiles.get(str(cluster_id), {})
        relevance_score = float(profile.get("relevance_score", 0.0))
        spam_score = float(profile.get("spam_score", 0.0))
        fear_greed_score = float(profile.get("fear_greed_score", 0.0))
        sentiment_label = str(profile.get("sentiment_label", "neutral"))
        topic_scores = profile.get("topic_scores", {}) or {}

        relevance_high = float(self.thresholds.get("relevance_high", 0.75))
        relevance_medium = float(self.thresholds.get("relevance_medium", 0.45))
        topic_threshold = float(self.thresholds.get("topic_threshold", 0.5))

        if relevance_score >= relevance_high:
            relevance_label = "high"
        elif relevance_score >= relevance_medium:
            relevance_label = "medium"
        elif relevance_score >= MIN_RELEVANCE_SCORE:
            relevance_label = "low"
        else:
            relevance_label = "irrelevant"

        topic_tags = [tag for tag, score in topic_scores.items() if float(score) >= topic_threshold]

        return {
            "source": "ml",
            "mode": "keras-unsupervised",
            "cluster_id": cluster_id,
            "relevance_score": round(relevance_score, 4),
            "relevance_label": relevance_label,
            "spam_score": round(spam_score, 4),
            "fear_greed_score": round(fear_greed_score, 4),
            "sentiment_label": sentiment_label,
            "topic_tags": topic_tags,
            "topic_confidence": {k: round(float(v), 4) for k, v in topic_scores.items()},
        }


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

def clean_post(
    post,
    client: Client,
    follower_cache: dict[str, int],
    stats: dict[str, int],
    query_term: str,
    run_id: str,
    ai_engine: AIInferenceEngine,
):
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

        if not _is_english_post(record):
            stats["dropped_non_english"] += 1
            return None

        if not is_within_date(created_at_raw):
            stats["dropped_date"] += 1
            return None

        normalized_text = _normalize_text(text)

        model_result = ai_engine.predict(normalized_text)
        model_source = model_result.get("source", "rules")
        model_mode = str(model_result.get("mode", ai_engine.mode))

        if model_source == "ml":
            stats["classified_ml"] += 1
        else:
            stats["classified_rules"] += 1

        if model_source == "ml":
            spam_score = float(model_result.get("spam_score", 0.0))
            if spam_score >= 0.5:
                stats["dropped_spam"] += 1
                return None

            relevance_score = float(model_result.get("relevance_score", 0.0))
            relevance_label = str(model_result.get("relevance_label", "irrelevant"))
            if relevance_score < MIN_RELEVANCE_SCORE:
                stats["dropped_low_relevance"] += 1
                return None

            cluster_fg = float(model_result.get("fear_greed_score", 0.0))
            lexical_fg, _ = _fear_greed_score(normalized_text)
            fear_greed_score = round(
                (ML_SENTIMENT_BLEND * float(lexical_fg))
                + ((1.0 - ML_SENTIMENT_BLEND) * float(cluster_fg)),
                4,
            )
            sentiment_label = _sentiment_from_score(fear_greed_score)
            topic_tags = model_result.get("topic_tags", []) or []
            topic_confidence = model_result.get("topic_confidence", {}) or {}
        else:
            if _is_probable_spam(normalized_text):
                stats["dropped_spam"] += 1
                return None

            relevance_score, relevance_label = _relevance_score(normalized_text, query_term)
            if relevance_score < MIN_RELEVANCE_SCORE:
                stats["dropped_low_relevance"] += 1
                return None

            fear_greed_score, sentiment_label = _fear_greed_score(normalized_text)
            topic_tags = _extract_topic_tags(normalized_text)
            topic_confidence = {tag: 1.0 for tag in topic_tags}

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
            "query_term": query_term,
            "relevance_score": round(float(relevance_score), 4),
            "relevance_label": relevance_label,
            "topic_tags": topic_tags,
            "topic_confidence": topic_confidence,
            "fear_greed_score": fear_greed_score,
            "sentiment_label": sentiment_label,
            "classification_source": model_source,
            "classification_mode": model_mode,
            "cluster_id": model_result.get("cluster_id", None),
            "ingestion_run_id": run_id,
            "ingested_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        }
        stats["kept"] += 1
        return cleaned
    except Exception as e:
        print(f"[!] Skipping post due to error: {e}")
        return None

def fetch_posts(use_ai_model: bool = False, model_dir: str = DEFAULT_MODEL_DIR):
    handle, password = get_credentials()
    client = Client()
    run_id = str(uuid.uuid4())
    ai_engine = AIInferenceEngine(use_ai_model=use_ai_model, model_dir=model_dir)

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
    print(f"[i] Run ID: {run_id}")
    print(f"[i] Classification mode: {ai_engine.model_source}")
    if use_ai_model and ai_engine.enabled:
        print("[✓] AI model is active. Posts will use Keras+KMeans classification.")
    elif use_ai_model and not ai_engine.enabled:
        print("[!] AI model was requested but not loaded. Using rule-based fallback.")
    else:
        print("[i] AI model disabled in config. Using rule-based classification.")
    time_slices = _build_time_slices(START_DATE, END_DATE)
    print(f"[i] Time slices: {len(time_slices)} windows of ~{WINDOW_DAYS} days")

    for since_date, until_date_exclusive in time_slices:
        if len(collected) >= TARGET_POSTS:
            break

        print(f"[i] Window: since:{since_date.isoformat()} until:{until_date_exclusive.isoformat()}")
        for term in query_terms:
            if len(collected) >= TARGET_POSTS:
                break

            search_term = term
            search_query = _build_query(search_term, since_date, until_date_exclusive)
            cursor = None
            use_raw_search = False
            consecutive_errors = 0
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

                    # Only count back-to-back failures for skip logic.
                    consecutive_errors = 0

                    posts = _get_any(response, ["posts", "data", "results"], []) or []
                    if not posts:
                        print(f"[i] No more posts for term '{term}' in this window.")
                        break

                    for post in posts:
                        cleaned = clean_post(
                            post,
                            client=client,
                            follower_cache=follower_cache,
                            stats=stats,
                            query_term=search_term,
                            run_id=run_id,
                            ai_engine=ai_engine,
                        )
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
                        print(f"[i] Reached end of available posts for term '{term}' in this window.")
                        break

                    time.sleep(0.7)

                except Exception as e:
                    print(f"[!] Error: {e}")
                    consecutive_errors += 1
                    if not use_raw_search and "validation error for Response" in str(e):
                        print("[i] Switching to raw API mode due to atproto response validation mismatch.")
                        use_raw_search = True
                        consecutive_errors = 0

                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS_PER_TERM:
                        print(
                            f"[i] Skipping term '{search_term}' in this window after "
                            f"{MAX_CONSECUTIVE_ERRORS_PER_TERM} consecutive errors."
                        )
                        break

                    time.sleep(5)
                    continue

    save_json(collected)
    run_report = _build_run_report(
        run_id=run_id,
        stats=stats,
        collected_count=len(collected),
        query_terms=query_terms,
        ai_mode=ai_engine.model_source,
        model_dir=str(Path(model_dir)),
    )
    save_run_report(run_report)
    _print_stats(stats)
    print(f"\n[✓] Finished. Total posts saved: {len(collected)}")
    print(f"[✓] Output file: {OUTPUT_FILE}")
    print(f"[✓] Run report: {RUN_REPORT_FILE}")

if __name__ == "__main__":
    print("=" * 70)
    print("BLUESKY BITCOIN SCRAPER - CONFIGURATION")
    print("=" * 70)
    print(f"[✓] Query: {QUERY}")
    print(f"[✓] Date range: {START_DATE.date()} to {END_DATE.date()}")
    print(f"[✓] Target posts: {TARGET_POSTS:,}")
    print(f"[✓] Min relevance: {MIN_RELEVANCE_SCORE}")
    print(f"[✓] AI model enabled: {USE_AI_MODEL}")
    if USE_AI_MODEL:
        print(f"[✓] Model directory: {MODEL_DIR}")
    print(f"[✓] Output file: {OUTPUT_FILE}")
    print(f"[✓] Run report: {RUN_REPORT_FILE}")
    print("=" * 70)
    print()
    
    fetch_posts(use_ai_model=USE_AI_MODEL, model_dir=MODEL_DIR)
