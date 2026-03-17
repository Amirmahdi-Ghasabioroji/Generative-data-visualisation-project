"""
Train an unsupervised Keras + KMeans model for the Bluesky scraper.

This script builds:
1) A CNN text autoencoder-like model with a latent bottleneck.
2) KMeans clusters over latent embeddings.
3) Cluster profiles (relevance/spam/sentiment/topic scores) from lexical signals.

Saved artifacts (in --output-dir):
- model.keras
- kmeans.joblib
- cluster_profiles.json
- thresholds.json
- label_maps.json
- model_metadata.json

Usage example:
python AI_systems/train_unsupervised_scraper_model.py \
  --input-json Data_Pipeline/datasets/bitcoin_bluesky_jan2024_sep2024.json \
  --output-dir AI_systems/model_artifacts
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans


# Force UTF-8 console streams on Windows to avoid UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


DEFAULT_INPUT_JSON = Path("Data_Pipeline/datasets/bitcoin_bluesky_jan2024_sep2024.json")
DEFAULT_OUTPUT_DIR = Path("AI_systems/scraper_model_artifacts")


BITCOIN_TERMS = {
    "bitcoin", "btc", "satoshi", "hodl", "halving", "mining", "onchain",
    "lightning", "utxo", "whale", "cold wallet", "hot wallet", "etf", "spot etf",
}
MARKET_TERMS = {
    "price", "volatility", "bull", "bear", "breakout", "support", "resistance",
    "rally", "dump", "pump", "liquidation", "volume", "trend", "ath", "drawdown",
}
SPAM_TERMS = {
    "airdrop", "giveaway", "referral", "promo", "dm me", "guaranteed profit",
    "100x", "moonshot signal", "join my group", "vip signals",
}
BULLISH_TERMS = {
    "bullish", "buy", "long", "rally", "breakout", "green", "uptrend", "greed",
}
BEARISH_TERMS = {
    "bearish", "sell", "short", "crash", "dump", "red", "downtrend", "fear", "panic",
}
TOPIC_KEYWORDS = {
    "price_action": {"price", "breakout", "support", "resistance", "trend", "ath", "return"},
    "volatility": {"volatility", "liquidation", "drawdown", "squeeze"},
    "macro": {"fed", "rates", "inflation", "cpi", "macro", "dollar"},
    "regulation": {"sec", "regulation", "compliance", "ban", "legal", "lawsuit"},
    "etf": {"etf", "spot etf", "approval", "inflow", "outflow"},
    "adoption": {"adoption", "merchant", "payments", "lightning", "wallet", "onchain"},
}


def _normalize_text(text: str) -> str:
    value = " ".join((text or "").strip().lower().split())
    # Keep training vocabulary ASCII-only to avoid Windows cp1252 save issues.
    return value.encode("ascii", errors="ignore").decode("ascii")


def _contains_any(text: str, terms: set[str]) -> int:
    return sum(1 for term in terms if term in text)


def _relevance_score(text: str) -> float:
    btc_hits = _contains_any(text, BITCOIN_TERMS)
    market_hits = _contains_any(text, MARKET_TERMS)
    has_ticker = 1 if "$btc" in text or " btc " in f" {text} " else 0
    raw = 0.45 * min(btc_hits, 3) + 0.40 * min(market_hits, 3) + 0.15 * has_ticker
    return float(max(0.0, min(1.0, raw)))


def _spam_score(text: str) -> float:
    spam_hits = _contains_any(text, SPAM_TERMS)
    url_hits = text.count("http://") + text.count("https://")
    if spam_hits > 0 or url_hits >= 3:
        return 1.0
    return 0.0


def _fear_greed_score(text: str) -> float:
    bullish = _contains_any(text, BULLISH_TERMS)
    bearish = _contains_any(text, BEARISH_TERMS)
    total = bullish + bearish
    if total == 0:
        return 0.0
    return float((bullish - bearish) / total)


def _sentiment_label(score: float) -> str:
    if score >= 0.25:
        return "greed"
    if score <= -0.25:
        return "fear"
    return "neutral"


def _topic_scores(texts: list[str]) -> dict[str, float]:
    if not texts:
        return {topic: 0.0 for topic in TOPIC_KEYWORDS}

    scores = {}
    n = float(len(texts))
    for topic, keywords in TOPIC_KEYWORDS.items():
        hit_count = 0
        for text in texts:
            if any(keyword in text for keyword in keywords):
                hit_count += 1
        scores[topic] = round(hit_count / n, 4)
    return scores


def _load_texts(input_json: Path, min_chars: int = 5) -> list[str]:
    data = json.loads(input_json.read_text(encoding="utf-8"))
    texts = []
    for item in data:
        text = _normalize_text(str(item.get("text", "")))
        if len(text) >= min_chars:
            texts.append(text)
    return texts


def _build_cnn_autoencoder_model(max_tokens: int, sequence_length: int, embed_dim: int, latent_dim: int):
    text_input = tf.keras.Input(shape=(), dtype=tf.string, name="text")

    seq_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=sequence_length,
        name="text_vectorizer",
    )

    x = seq_vectorizer(text_input)
    # Conv1D does not consume sequence masks, so keep mask_zero disabled.
    x = tf.keras.layers.Embedding(max_tokens, embed_dim, mask_zero=False, name="embedding")(x)
    x = tf.keras.layers.Conv1D(128, 5, activation="relu", padding="same", name="conv_1")(x)
    x = tf.keras.layers.Conv1D(64, 3, activation="relu", padding="same", name="conv_2")(x)
    x = tf.keras.layers.GlobalMaxPooling1D(name="global_max_pool")(x)
    latent = tf.keras.layers.Dense(latent_dim, activation="relu", name="latent")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(latent)
    bow_reconstruction = tf.keras.layers.Dense(max_tokens, activation="sigmoid", name="bow_reconstruction")(x)

    model = tf.keras.Model(text_input, bow_reconstruction, name="unsupervised_text_cnn")
    return model, seq_vectorizer


def train_unsupervised_model(
    input_json: Path,
    output_dir: Path,
    max_tokens: int,
    sequence_length: int,
    embed_dim: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    num_clusters: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = _load_texts(input_json)
    if len(texts) < 50:
        raise ValueError("Need at least 50 valid text posts for unsupervised training.")

    model, seq_vectorizer = _build_cnn_autoencoder_model(
        max_tokens=max_tokens,
        sequence_length=sequence_length,
        embed_dim=embed_dim,
        latent_dim=latent_dim,
    )

    # Adapt vectorizer vocabulary on full corpus
    seq_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(texts).batch(256))

    # Multi-hot targets for reconstruction objective.
    # Use adapt() rather than passing vocabulary explicitly to avoid
    # reserved-token ordering issues in Keras TextVectorization.
    # pad_to_max_tokens=True ensures output width == max_tokens,
    # matching the decoder output dimension.
    bow_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="multi_hot",
        pad_to_max_tokens=True,
    )
    bow_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(texts).batch(256))
    y_targets = bow_vectorizer(tf.constant(texts)).numpy()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
    )

    model.fit(
        x=np.array(texts, dtype=object),
        y=y_targets,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    # Build embedding extractor from latent layer
    embedding_extractor = tf.keras.Model(model.input, model.get_layer("latent").output)
    embeddings = embedding_extractor.predict(np.array(texts, dtype=object), batch_size=batch_size, verbose=0)

    k = min(max(2, num_clusters), len(texts))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(embeddings)

    # Create per-cluster lexical profiles used by pipeline inference
    cluster_profiles = {}
    for cluster_id in range(k):
        cluster_texts = [texts[i] for i, c in enumerate(clusters) if c == cluster_id]
        if not cluster_texts:
            continue

        relevance = float(np.mean([_relevance_score(t) for t in cluster_texts]))
        spam = float(np.mean([_spam_score(t) for t in cluster_texts]))
        fg_scores = [_fear_greed_score(t) for t in cluster_texts]
        fg = float(np.mean(fg_scores))
        sentiment = _sentiment_label(fg)
        topics = _topic_scores(cluster_texts)

        cluster_profiles[str(cluster_id)] = {
            "size": len(cluster_texts),
            "relevance_score": round(relevance, 4),
            "spam_score": round(spam, 4),
            "fear_greed_score": round(fg, 4),
            "sentiment_label": sentiment,
            "topic_scores": topics,
        }

    thresholds = {
        "relevance_high": 0.75,
        "relevance_medium": 0.45,
        "spam_threshold": 0.5,
        "topic_threshold": 0.5,
    }

    label_maps = {
        "sentiment_classes": ["fear", "neutral", "greed"],
        "topic_labels": list(TOPIC_KEYWORDS.keys()),
    }

    metadata = {
        "model_type": "unsupervised_text_cnn_kmeans",
        "trained_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "input_json": str(input_json),
        "num_posts": len(texts),
        "max_tokens": max_tokens,
        "sequence_length": sequence_length,
        "embedding_dim": embed_dim,
        "latent_dim": latent_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_clusters": k,
        "artifacts": {
            "model": "model.keras",
            "kmeans": "kmeans.joblib",
            "cluster_profiles": "cluster_profiles.json",
            "thresholds": "thresholds.json",
            "label_maps": "label_maps.json",
            "metadata": "model_metadata.json",
        },
    }

    model.save(output_dir / "model.keras")
    joblib.dump(kmeans, output_dir / "kmeans.joblib")
    (output_dir / "cluster_profiles.json").write_text(json.dumps(cluster_profiles, indent=2), encoding="utf-8")
    (output_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    (output_dir / "label_maps.json").write_text(json.dumps(label_maps, indent=2), encoding="utf-8")
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("[ok] Saved unsupervised artifacts to", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Train unsupervised Keras+KMeans model for Bluesky scraper.")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=DEFAULT_INPUT_JSON,
        help=f"Path to scraped JSON with text field (default: {DEFAULT_INPUT_JSON})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Artifact output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--max-tokens", type=int, default=30000)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-clusters", type=int, default=8)
    args = parser.parse_args()

    if not args.input_json.exists():
        raise FileNotFoundError(
            f"Input JSON not found at: {args.input_json}. "
            "Pass --input-json <path> or update DEFAULT_INPUT_JSON in this script."
        )

    train_unsupervised_model(
        input_json=args.input_json,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        sequence_length=args.sequence_length,
        embed_dim=args.embed_dim,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_clusters=args.num_clusters,
    )


if __name__ == "__main__":
    main()
