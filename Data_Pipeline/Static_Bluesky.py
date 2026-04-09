"""
Live Bluesky social polling utilities.

This module contains the live social ingestion path used by
Generative_visualisation/live_btc_visual_pipeline.py.

"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Optional

import numpy as np
from atproto import Client


# Ensure workspace root is on sys.path so AI_systems can be imported when
# running this module directly.
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)


# Attempt to force UTF-8 stdout on Windows to avoid UnicodeEncodeError.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


try:
    from AI_systems.bitcoin_blusky_pipeline import (
        AIInferenceEngine,
        _extract_topic_tags as _ai_extract_topic_tags,
        _fear_greed_score as _ai_fear_greed_score,
        _is_probable_spam as _ai_is_probable_spam,
        _normalize_text as _ai_normalize_text,
        _relevance_score as _ai_relevance_score,
        _sentiment_from_score as _ai_sentiment_from_score,
    )
except Exception:
    AIInferenceEngine = None

    def _ai_normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    def _ai_sentiment_from_score(score: float) -> str:
        if score >= 0.15:
            return "greed"
        if score <= -0.15:
            return "fear"
        return "neutral"

    def _ai_fear_greed_score(text: str) -> tuple[float, str]:
        return 0.0, "neutral"

    def _ai_extract_topic_tags(text: str) -> list[str]:
        return []

    def _ai_relevance_score(text: str, query_term: str) -> tuple[float, str]:
        return 0.5, "medium"

    def _ai_is_probable_spam(text: str) -> bool:
        return False


def _clip01(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))


def _get_any(obj, names, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def fetch_bluesky_posts(
    query: str,
    limit: int = 300,
    handle: str | None = None,
    password: str | None = None,
    sort: str = "latest",
    exclude_uris: set[str] | None = None,
) -> list[dict]:
    """Fetch posts from Bluesky and return normalized post dictionaries."""

    client = Client()

    handle = (handle or os.getenv("BLUESKY_HANDLE") or "").strip()
    password = (password or os.getenv("BLUESKY_APP_PASSWORD") or "").strip()

    if handle.startswith("@"):
        handle = handle[1:]

    if handle and password:
        try:
            client.login(handle, password)
        except Exception as e:
            print(f"[SOCIAL] login failed for {handle}: {e}")
            return []

    posts: list[dict] = []
    cursor = None

    while len(posts) < limit:
        try:
            response = client.app.bsky.feed.search_posts(
                params={
                    "q": query,
                    "limit": min(100, limit - len(posts)),
                    "cursor": cursor,
                    "sort": sort,
                }
            )

            resp_success = _get_any(response, ["success", "ok"], None)
            if resp_success is False:
                return posts

            batch = _get_any(response, ["posts", "data", "results"], []) or []
            if not batch:
                break

            for post in batch:
                record = _get_any(post, ["record", "value", "post", "payload"], {}) or {}
                text = _get_any(record, ["text", "content", "body"], "") or ""
                if not text.strip():
                    continue

                created_at = _get_any(record, ["created_at", "createdAt", "time"], "") or ""
                like_count = _get_any(post, ["like_count", "likeCount", "likes"], 0) or 0
                repost_count = _get_any(post, ["repost_count", "repostCount", "reposts"], 0) or 0
                reply_count = _get_any(post, ["reply_count", "replyCount", "replies"], 0) or 0
                uri = _get_any(post, ["uri", "post_uri", "id"], "") or ""

                if exclude_uris and uri and uri in exclude_uris:
                    continue

                posts.append(
                    {
                        "text": text,
                        "created_at": created_at,
                        "like_count": int(float(like_count or 0)),
                        "repost_count": int(float(repost_count or 0)),
                        "reply_count": int(float(reply_count or 0)),
                        "uri": uri,
                    }
                )

            cursor = _get_any(response, ["cursor", "next", "cursor_str"], None)
            if not cursor:
                break

            # Light pacing to reduce API pressure.
            time.sleep(0.25)

        except Exception as e:
            print(f"[SOCIAL] fetch error: {e}")
            break

    return posts[:limit]


class LiveSocialSentimentPoller:
    """Poll Bluesky on an interval and maintain rolling social factors for live blending."""

    MAX_PERSISTED_POSTS = 300

    def __init__(
        self,
        query: str = "bitcoin OR btc",
        fetch_limit: int = 100,
        rolling_posts: int = 300,
        use_ai_model: bool = True,
        model_dir: str | None = None,
        handle: str | None = None,
        password: str | None = None,
        debug: bool = False,
    ):
        self.query = query
        self.fetch_limit = int(max(10, fetch_limit))
        # Enforce a strict fixed-size ring buffer for persisted social posts.
        self.rolling_posts = int(self.MAX_PERSISTED_POSTS)
        self.handle = handle
        self.password = password
        self.debug = bool(debug)
        self._seen_capacity = int(max(2500, self.rolling_posts * 10))

        self._seen_fifo: Deque[str] = deque()
        self._seen_uris: set[str] = set()
        self._rolling_scored: Deque[dict] = deque(maxlen=self.rolling_posts)
        self._rolling_posts_raw: Deque[dict] = deque(maxlen=self.rolling_posts)
        self.posts_output_path = Path(ROOT) / "binance_realtime" / "bluesky_live_posts.json"
        self.total_scored_posts: int = 0

        self.last_factors = {
            "turbulence": 0.5,
            "trend_bias": 0.5,
            "distortion": 0.5,
            "fragmentation": 0.5,
            "velocity": 0.5,
            "quality": 0.0,
        }
        self.last_update_ts: float = 0.0
        self.last_new_posts: int = 0
        self.error_streak: int = 0

        self.ai_engine = None
        if AIInferenceEngine is not None:
            try:
                self.ai_engine = AIInferenceEngine(
                    use_ai_model=bool(use_ai_model),
                    model_dir=model_dir or "AI_systems/scraper_model_artifacts",
                )
            except Exception as e:
                if self.debug:
                    print(f"[SOCIAL] AI engine init failed, falling back to lexical mode: {e}")

    def _mark_seen(self, uri: str) -> None:
        if not uri:
            return
        if uri in self._seen_uris:
            return
        self._seen_fifo.append(uri)
        self._seen_uris.add(uri)
        while len(self._seen_fifo) > self._seen_capacity:
            old = self._seen_fifo.popleft()
            self._seen_uris.discard(old)

    def _score_post(self, post: dict) -> Optional[dict]:
        text = str(post.get("text", "") or "")
        if not text.strip():
            return None

        normalized = _ai_normalize_text(text)
        model_result = self.ai_engine.predict(normalized) if self.ai_engine is not None else {"source": "rules"}
        source = str(model_result.get("source", "rules"))

        if source == "ml":
            relevance_score = float(model_result.get("relevance_score", 0.0))
            spam_score = float(model_result.get("spam_score", 0.0))
            cluster_fg = float(model_result.get("fear_greed_score", 0.0))
            lexical_fg, _ = _ai_fear_greed_score(normalized)
            fear_greed_score = float(np.clip(0.70 * float(lexical_fg) + 0.30 * cluster_fg, -1.0, 1.0))
            sentiment_label = _ai_sentiment_from_score(fear_greed_score)
            topic_confidence = model_result.get("topic_confidence", {}) or {}
        else:
            relevance_score, _ = _ai_relevance_score(normalized, "bitcoin")
            spam_score = 1.0 if _ai_is_probable_spam(normalized) else 0.0
            fear_greed_score, sentiment_label = _ai_fear_greed_score(normalized)
            topic_confidence = {tag: 1.0 for tag in _ai_extract_topic_tags(normalized)}

        if spam_score >= 0.9 or relevance_score < 0.05:
            return None

        like_count = float(post.get("like_count", 0) or 0.0)
        repost_count = float(post.get("repost_count", 0) or 0.0)
        reply_count = float(post.get("reply_count", 0) or 0.0)
        engagement = float(np.log1p(like_count + 1.4 * repost_count + reply_count))

        created_at = str(post.get("created_at", "") or "")
        ts = time.time()
        if created_at:
            try:
                ts = datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
            except Exception:
                pass

        return {
            "timestamp": float(ts),
            "fear_greed_score": float(np.clip(fear_greed_score, -1.0, 1.0)),
            "sentiment_label": str(sentiment_label or "neutral"),
            "relevance_score": _clip01(relevance_score),
            "spam_score": _clip01(spam_score),
            "engagement": float(max(0.0, engagement)),
            "topic_confidence": {k: float(v) for k, v in topic_confidence.items()},
            "source": source,
        }

    def _aggregate_factors(self) -> dict:
        if not self._rolling_scored:
            return dict(self.last_factors)

        rows = list(self._rolling_scored)
        fg = np.asarray([float(r["fear_greed_score"]) for r in rows], dtype=np.float64)
        rel = np.asarray([float(r["relevance_score"]) for r in rows], dtype=np.float64)
        spam = np.asarray([float(r["spam_score"]) for r in rows], dtype=np.float64)
        engage = np.asarray([float(r["engagement"]) for r in rows], dtype=np.float64)
        ts = np.asarray([float(r["timestamp"]) for r in rows], dtype=np.float64)

        trend_bias = _clip01(0.5 + 0.5 * float(np.mean(fg)))

        fg_diff = np.diff(fg) if fg.size > 1 else np.asarray([0.0], dtype=np.float64)
        fg_std = float(np.std(fg))
        fg_speed = float(np.mean(np.abs(fg_diff)))

        if engage.size > 1:
            engage_norm = (engage - np.mean(engage)) / (np.std(engage) + 1e-8)
            engage_var = float(np.std(engage_norm))
        else:
            engage_var = 0.0

        sentiment_counts = Counter([str(r.get("sentiment_label", "neutral")) for r in rows])
        total_sent = float(max(1, sum(sentiment_counts.values())))
        sent_probs = np.asarray([c / total_sent for c in sentiment_counts.values()], dtype=np.float64)
        sent_entropy = float(-np.sum(sent_probs * np.log2(np.maximum(sent_probs, 1e-8))))
        sent_entropy_norm = _clip01(sent_entropy / np.log2(3.0))

        topic_accumulator: Dict[str, float] = {}
        for row in rows:
            topic_conf = row.get("topic_confidence", {}) or {}
            for key, value in topic_conf.items():
                topic_accumulator[key] = topic_accumulator.get(key, 0.0) + float(value)

        if topic_accumulator:
            topic_vals = np.asarray(list(topic_accumulator.values()), dtype=np.float64)
            topic_probs = topic_vals / (np.sum(topic_vals) + 1e-8)
            topic_entropy = float(-np.sum(topic_probs * np.log2(np.maximum(topic_probs, 1e-8))))
            topic_entropy_norm = _clip01(topic_entropy / np.log2(max(2, len(topic_vals))))
            topic_volatility = float(topic_accumulator.get("volatility", 0.0) / max(1, len(rows)))
            topic_regime = float(
                (
                    topic_accumulator.get("macro", 0.0)
                    + topic_accumulator.get("regulation", 0.0)
                    + topic_accumulator.get("etf", 0.0)
                )
                / max(1, len(rows))
            )
        else:
            topic_entropy_norm = 0.0
            topic_volatility = 0.0
            topic_regime = 0.0

        now_ts = time.time()
        recent_posts = int(np.sum(ts >= (now_ts - 300.0))) if ts.size else 0
        post_rate_norm = _clip01(recent_posts / 20.0)

        turbulence = _clip01(0.42 * fg_std + 0.36 * fg_speed + 0.22 * topic_volatility)
        distortion = _clip01(0.45 * fg_speed + 0.30 * engage_var + 0.25 * topic_regime)
        fragmentation = _clip01(0.55 * sent_entropy_norm + 0.45 * topic_entropy_norm)
        velocity = _clip01(0.62 * fg_speed + 0.38 * post_rate_norm)

        coverage = _clip01(len(rows) / float(self.rolling_posts))
        quality = _clip01(
            coverage
            * (
                0.55 * float(np.mean(rel))
                + 0.25 * post_rate_norm
                + 0.20 * (1.0 - float(np.mean(spam)))
            )
        )

        return {
            "turbulence": turbulence,
            "trend_bias": trend_bias,
            "distortion": distortion,
            "fragmentation": fragmentation,
            "velocity": velocity,
            "quality": quality,
        }

    def poll_once(self) -> dict:
        try:
            posts = fetch_bluesky_posts(
                query=self.query,
                limit=self.fetch_limit,
                handle=self.handle,
                password=self.password,
                sort="latest",
                exclude_uris=self._seen_uris,
            )

            new_scored = 0
            for post in posts:
                uri = str(post.get("uri", "") or "")
                if uri and uri in self._seen_uris:
                    continue

                self._mark_seen(uri)
                scored = self._score_post(post)
                if scored is None:
                    continue

                self._rolling_scored.append(scored)
                self._rolling_posts_raw.append(
                    {
                        "uri": str(post.get("uri", "") or ""),
                        "created_at": str(post.get("created_at", "") or ""),
                        "text": str(post.get("text", "") or ""),
                        "like_count": int(float(post.get("like_count", 0) or 0)),
                        "repost_count": int(float(post.get("repost_count", 0) or 0)),
                        "reply_count": int(float(post.get("reply_count", 0) or 0)),
                        "fear_greed_score": float(scored.get("fear_greed_score", 0.0)),
                        "sentiment_label": str(scored.get("sentiment_label", "neutral")),
                        "relevance_score": float(scored.get("relevance_score", 0.0)),
                        "spam_score": float(scored.get("spam_score", 0.0)),
                        "source": str(scored.get("source", "rules")),
                        "topic_confidence": dict(scored.get("topic_confidence", {})),
                        "scored_ts": float(scored.get("timestamp", time.time())),
                    }
                )
                new_scored += 1
                self.total_scored_posts += 1

            self.last_new_posts = new_scored
            if new_scored > 0:
                self.last_factors = self._aggregate_factors()
                self.last_update_ts = time.time()
                self.error_streak = 0

            if self.debug:
                print(
                    "[SOCIAL] "
                    f"fetched={len(posts)} "
                    f"new_scored={new_scored} "
                    f"rolling={len(self._rolling_scored)} "
                    f"social_quality={self.last_factors['quality']:.3f}"
                )

            self._persist_posts_json()
            return dict(self.last_factors)

        except Exception as e:
            self.error_streak += 1
            if self.debug:
                print(f"[SOCIAL] poll error (streak={self.error_streak}): {e}")
            self._persist_posts_json()
            # Freeze-on-failure behavior: keep last factors unchanged.
            return dict(self.last_factors)

    def _persist_posts_json(self) -> None:
        try:
            self.posts_output_path.parent.mkdir(parents=True, exist_ok=True)
            posts_window = list(self._rolling_posts_raw)[-self.rolling_posts :]
            payload = {
                "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "query": self.query,
                "fetch_limit": int(self.fetch_limit),
                "rolling_posts_target": int(self.rolling_posts),
                "rolling_posts_current": int(len(posts_window)),
                "new_posts_last_poll": int(self.last_new_posts),
                "total_scored_posts": int(self.total_scored_posts),
                "error_streak": int(self.error_streak),
                "factors": dict(self.last_factors),
                "posts": posts_window,
            }
            self.posts_output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            if self.debug:
                print(f"[SOCIAL] persist error: {e}")

    def get_current_factors(self) -> dict:
        return dict(self.last_factors)

    def get_snapshot(self) -> dict:
        return {
            "factors": dict(self.last_factors),
            "last_update_ts": float(self.last_update_ts),
            "rolling_posts": int(len(self._rolling_scored)),
            "new_posts_last_poll": int(self.last_new_posts),
            "error_streak": int(self.error_streak),
            "posts_output_path": str(self.posts_output_path),
        }
