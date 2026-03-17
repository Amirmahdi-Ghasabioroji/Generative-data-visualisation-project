"""

Align market (BTC/ETH) and social (Bluesky) data into a unified {[market_features], [social_features]} matrix for VAE input.
Stack: numpy only (no pandas)

"""

import csv
import json
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path



BTC_CSV    = Path("Data_Pipeline/datasets/btcusdt_30m_20240101_20240930.csv")
POSTS_JSON = Path("Data_Pipeline/datasets/bitcoin_bluesky_jan2024_sep2024.json")
OUTPUT_DIR = Path("vae_model/data")

WINDOW_SECONDS = 30 * 60  # 30 minutes in seconds


#1
# RESAMPLE MARKET DATA TO 30m BINS


def _parse_ts(ts_str: str) -> int:
    """Convert ISO timestamp string to UTC unix seconds (int)."""
    ts_str = ts_str.strip()
    ts_str = ts_str.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(ts_str)
    except ValueError:
        dt = datetime.strptime(ts_str[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _floor_to_window(unix_ts: int, window: int = WINDOW_SECONDS) -> int:
    return (unix_ts // window) * window


def load_and_resample_market(csv_path: Path) -> dict[int, dict]:

    if not csv_path.exists():
        print(f"[!] CSV not found: {csv_path}")
        return {}

    bins: dict[int, dict] = {}

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # File is already 30m bars — use open_time_utc as the window key
                window_start = _floor_to_window(_parse_ts(row["open_time_utc"]))

                total_quote = float(row["quote_asset_volume"])
                total_taker = float(row["taker_buy_base_asset_volume"])
                taker_ratio = float(row["taker_ratio"])

                bins[window_start] = {
                    "window_start": window_start,
                    "open":         float(row["open"]),
                    "high":         float(row["high"]),
                    "low":          float(row["low"]),
                    "close":        float(row["close"]),
                    "volume":       float(row["volume"]),
                    "quote_volume": total_quote,
                    "num_trades":   int(float(row["number_of_trades"])),
                    "taker_volume": total_taker,
                    "taker_ratio":  taker_ratio,
                    "n_raw_rows":   1,
                }
            except (ValueError, KeyError):
                continue  # Skip malformed rows

    if not bins:
        print(f"[!] No valid rows loaded from {csv_path.name}")
        return {}

    print(f"[✓] BTCUSDT: {len(bins)} x 30m bins loaded from {csv_path.name}")
    return bins



# 2
# Reads posts JSON from the static Bluesky pipeline,
# converts created_at to UTC unix seconds, floors each post to its
# 30m window, then computes the minimum viable social features
# per bin (activity, engagement, sentiment).


# Simple crypto-domain lexicons for bullish/bearish sentiment
BULLISH_WORDS = {
    "moon", "bullish", "buy", "long", "pump", "up", "gain",
    "rally", "breakout", "ath", "green", "hodl", "hold", "rise",
    "surge", "soar", "rocket", "explode", "profit", "growth"
}
BEARISH_WORDS = {
    "bear", "bearish", "sell", "short", "dump", "down", "loss",
    "crash", "drop", "dip", "red", "fear", "panic", "collapse",
    "fall", "rekt", "correction", "plunge", "decline", "bleed"
}


def _lexicon_sentiment(text: str) -> tuple[int, int]:
    """
    Count bullish and bearish word hits in a post.
    Returns (bullish_hits, bearish_hits).
    """
    words = set(text.lower().split())
    bullish = sum(1 for w in words if w in BULLISH_WORDS)
    bearish = sum(1 for w in words if w in BEARISH_WORDS)
    return bullish, bearish


def load_and_bin_social(posts_json: Path) -> dict[int, dict]:

    if not posts_json.exists():
        print(f"[!] JSON not found: {posts_json}")
        return {}

    with posts_json.open("r", encoding="utf-8") as f:
        raw_posts = json.load(f)

    if not isinstance(raw_posts, list):
        print(f"[!] Expected a JSON array in {posts_json.name}")
        return {}

    # Group raw posts by 30m window
    raw_bins: dict[int, list[dict]] = {}

    for post in raw_posts:
        try:
            unix_ts = _parse_ts(post["created_at"])
            window  = _floor_to_window(unix_ts)
        except Exception:
            continue  # Skip posts with unparseable timestamps

        try:
            like_count     = float(post.get("like_count",            0) or 0)
            repost_count   = float(post.get("repost_count",          0) or 0)
            reply_count    = float(post.get("reply_count",           0) or 0)
            word_count     = int(float(post.get("word_count",        0) or 0))
            follower_count = float(post.get("author_follower_count", 0) or 0)
            text           = post.get("text",       "") or ""
            author_did     = post.get("author_did", "") or ""
        except (ValueError, TypeError):
            continue

        bullish_hits, bearish_hits = _lexicon_sentiment(text)

        raw_bins.setdefault(window, []).append({
            "like_count":     like_count,
            "repost_count":   repost_count,
            "reply_count":    reply_count,
            "word_count":     word_count,
            "follower_count": follower_count,
            "bullish_hits":   bullish_hits,
            "bearish_hits":   bearish_hits,
            "author_did":     author_did,
        })

    if not raw_bins:
        print(f"[!] No valid posts loaded from {posts_json.name}")
        return {}

    # Aggregate each window into social features
    bins: dict[int, dict] = {}
    for window_start, posts in sorted(raw_bins.items()):
        n = len(posts)

        like_counts     = [p["like_count"]     for p in posts]
        repost_counts   = [p["repost_count"]   for p in posts]
        reply_counts    = [p["reply_count"]    for p in posts]
        word_counts     = [p["word_count"]     for p in posts]
        follower_counts = [p["follower_count"] for p in posts]
        bullish_hits    = [p["bullish_hits"]   for p in posts]
        bearish_hits    = [p["bearish_hits"]   for p in posts]
        unique_authors  = len(set(p["author_did"] for p in posts if p["author_did"]))

        total_bullish = sum(bullish_hits)
        total_bearish = sum(bearish_hits)
        sentiment_net = total_bullish - total_bearish
        bull_bear_ratio = (
            total_bullish / total_bearish if total_bearish > 0
            else float(total_bullish)     # If no bearish hits, ratio = bullish count
        )

        engagement_sum = sum(
            l + r * 2 + rep
            for l, r, rep in zip(like_counts, repost_counts, reply_counts)
        )

        bins[window_start] = {
            "window_start":          window_start,
            # Activity
            "post_count":            n,
            "unique_authors":        unique_authors,
            # Engagement
            "engagement_sum":        engagement_sum,
            "like_mean":             sum(like_counts)     / n,
            "repost_mean":           sum(repost_counts)   / n,
            "reply_mean":            sum(reply_counts)    / n,
            # Sentiment
            "bullish_hits_sum":      total_bullish,
            "bearish_hits_sum":      total_bearish,
            "sentiment_net":         sentiment_net,
            "bull_bear_ratio":       bull_bear_ratio,
            # Text
            "word_count_mean":       sum(word_counts)     / n,
            # Author reach
            "author_followers_mean": sum(follower_counts) / n,
            # Flag (always 0 here — set to 1 during join for empty windows)
            "social_empty_flag":     0,
        }

    print(f"[✓] Social: {len(bins)} x 30m bins from {sum(len(v) for v in raw_bins.values())} posts")
    return bins


# 3
# LEFT-JOIN SOCIAL BINS ONTO MARKET TIMELINE
# Uses the market timeline as the master index.
# For each 30m market window, looks up the matching social bin.
# If no social data exists for that window, fills all social
# features with 0 and sets social_empty_flag = 1.

# Zero-filled social row used when no posts exist for a window
_EMPTY_SOCIAL = {
    "post_count":            0,
    "unique_authors":        0,
    "engagement_sum":        0.0,
    "like_mean":             0.0,
    "repost_mean":           0.0,
    "reply_mean":            0.0,
    "bullish_hits_sum":      0,
    "bearish_hits_sum":      0,
    "sentiment_net":         0,
    "bull_bear_ratio":       0.0,
    "word_count_mean":       0.0,
    "author_followers_mean": 0.0,
    "social_empty_flag":     1,
}


def join_market_social(
    market_bins: dict[int, dict],
    social_bins: dict[int, dict],
) -> list[dict]:

    aligned = []

    for window_start in sorted(market_bins.keys()):
        market_row = market_bins[window_start]

        # Look up social bin for this exact 30m window
        social_row = social_bins.get(window_start, _EMPTY_SOCIAL)

        merged = {
            "window_start":          window_start,
            # ── Market fields ──
            "open":                  market_row["open"],
            "high":                  market_row["high"],
            "low":                   market_row["low"],
            "close":                 market_row["close"],
            "volume":                market_row["volume"],
            "quote_volume":          market_row["quote_volume"],
            "num_trades":            market_row["num_trades"],
            "taker_volume":          market_row["taker_volume"],
            "taker_ratio":           market_row["taker_ratio"],
            # ── Social fields ──
            "post_count":            social_row["post_count"],
            "unique_authors":        social_row["unique_authors"],
            "engagement_sum":        social_row["engagement_sum"],
            "like_mean":             social_row["like_mean"],
            "repost_mean":           social_row["repost_mean"],
            "reply_mean":            social_row["reply_mean"],
            "bullish_hits_sum":      social_row["bullish_hits_sum"],
            "bearish_hits_sum":      social_row["bearish_hits_sum"],
            "sentiment_net":         social_row["sentiment_net"],
            "bull_bear_ratio":       social_row["bull_bear_ratio"],
            "word_count_mean":       social_row["word_count_mean"],
            "author_followers_mean": social_row["author_followers_mean"],
            "social_empty_flag":     social_row["social_empty_flag"],
        }
        aligned.append(merged)

    matched = sum(1 for r in aligned if r["social_empty_flag"] == 0)
    empty   = sum(1 for r in aligned if r["social_empty_flag"] == 1)
    print(f"[✓] Aligned: {len(aligned)} rows | {matched} with social data | {empty} empty (zero-filled)")
    return aligned


#4
# DERIVED MARKET FEATURES
#Takes the aligned rows and computes the 12
# minimum viable market features the VAE needs:
# log_return_close, high_low_range_pct, rv_6, rv_12,
# log_volume, volume_z_48, number_of_trades,
# volume_per_trade, taker_ratio, delta_taker_ratio,
# ema_gap_6_24, rsi_14

def _ema(values: list[float], span: int) -> list[float]:
    """Exponential moving average over a list. Returns same-length list."""
    k = 2 / (span + 1)
    result = [0.0] * len(values)
    if not values:
        return result
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = values[i] * k + result[i - 1] * (1 - k)
    return result


def _rsi(closes: list[float], period: int = 14) -> list[float]:
    """RSI over a list of closes. Returns same-length list (NaN → 50.0)."""
    result = [50.0] * len(closes)
    if len(closes) < 2:
        return result
    for i in range(1, len(closes)):
        if i < period:
            result[i] = 50.0
            continue
        window = closes[i - period: i]
        gains  = [max(window[j] - window[j-1], 0) for j in range(1, len(window))]
        losses = [max(window[j-1] - window[j], 0) for j in range(1, len(window))]
        avg_gain = sum(gains)  / len(gains)  if gains  else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100 - (100 / (1 + rs))
    return result


def add_derived_market_features(aligned: list[dict]) -> list[dict]:

    n = len(aligned)
    if n == 0:
        return aligned

    closes       = [r["close"]      for r in aligned]
    volumes      = [r["volume"]     for r in aligned]
    num_trades   = [r["num_trades"] for r in aligned]
    taker_ratios = [r["taker_ratio"] for r in aligned]

    log_returns = [0.0] * n
    for i in range(1, n):
        if closes[i - 1] > 0:
            log_returns[i] = float(np.log(closes[i] / closes[i - 1]))

    hl_range_pct = [
        (r["high"] - r["low"]) / r["open"] if r["open"] > 0 else 0.0
        for r in aligned
    ]

    def rolling_std(series: list[float], window: int) -> list[float]:
        out = [0.0] * len(series)
        for i in range(len(series)):
            sl = series[max(0, i - window + 1): i + 1]
            out[i] = float(np.std(sl)) if len(sl) > 1 else 0.0
        return out

    rv_6  = rolling_std(log_returns, 6)
    rv_12 = rolling_std(log_returns, 12)

    log_volume = [float(np.log1p(v)) for v in volumes]

    def rolling_zscore(series: list[float], window: int) -> list[float]:
        out = [0.0] * len(series)
        for i in range(len(series)):
            sl = series[max(0, i - window + 1): i + 1]
            mu = sum(sl) / len(sl)
            sd = float(np.std(sl)) if len(sl) > 1 else 1.0
            out[i] = (series[i] - mu) / sd if sd > 0 else 0.0
        return out

    volume_z_48 = rolling_zscore(log_volume, 48)

    volume_per_trade = [
        v / t if t > 0 else 0.0
        for v, t in zip(volumes, num_trades)
    ]

    delta_taker = [0.0] * n
    for i in range(1, n):
        delta_taker[i] = taker_ratios[i] - taker_ratios[i - 1]

    ema6  = _ema(closes, 6)
    ema24 = _ema(closes, 24)
    ema_gap = [
        (e6 - e24) / e24 if e24 > 0 else 0.0
        for e6, e24 in zip(ema6, ema24)
    ]

    rsi14 = _rsi(closes, 14)

    for i, row in enumerate(aligned):
        row["log_return_close"]   = log_returns[i]
        row["high_low_range_pct"] = hl_range_pct[i]
        row["rv_6"]               = rv_6[i]
        row["rv_12"]              = rv_12[i]
        row["log_volume"]         = log_volume[i]
        row["volume_z_48"]        = volume_z_48[i]
        row["number_of_trades"]   = num_trades[i]
        row["volume_per_trade"]   = volume_per_trade[i]
        row["taker_ratio"]        = taker_ratios[i]
        row["delta_taker_ratio"]  = delta_taker[i]
        row["ema_gap_6_24"]       = ema_gap[i]
        row["rsi_14"]             = rsi14[i]

    print(f"[✓] Derived market features added to {n} rows")
    return aligned


# 5
# CROSS-MODAL FEATURES
# Multiplies social signals by market signals to
# create interaction terms the VAE can use to detect when
# social activity diverges from or confirms price action.


def add_cross_modal_features(aligned: list[dict]) -> list[dict]:

    for row in aligned:
        row["sentiment_net_x_return"]  = (
            row["sentiment_net"] * row["log_return_close"]
        )
        row["post_count_x_volatility"] = (
            row["post_count"] * row["rv_12"]
        )
        row["engagement_x_volume"]     = (
            row["engagement_sum"] * row["log_volume"]
        )

    print(f"[✓] Cross-modal features added to {len(aligned)} rows")
    return aligned


# 6
# SPLIT, SCALE, AND EXPORT
# Splits aligned rows chronologically into
# train / val / test, fits a min-max scaler on train only,
# applies it to all splits, then exports:
#   - market_features.npy   (n, 12)
#   - social_features.npy   (n, 13)  includes author_followers_mean
#   - cross_features.npy    (n, 3)
#   - timestamps.npy        (n,)   unix seconds
#   - summary.json          run metadata

# Feature column definitions — order matters for numpy arrays
MARKET_COLS = [
    "log_return_close", "high_low_range_pct", "rv_6", "rv_12",
    "log_volume", "volume_z_48", "number_of_trades", "volume_per_trade",
    "taker_ratio", "delta_taker_ratio", "ema_gap_6_24", "rsi_14",
]
SOCIAL_COLS = [
    "post_count", "unique_authors", "engagement_sum", "like_mean",
    "repost_mean", "reply_mean", "bullish_hits_sum", "bearish_hits_sum",
    "sentiment_net", "bull_bear_ratio", "word_count_mean",
    "author_followers_mean", "social_empty_flag",
]
CROSS_COLS = [
    "sentiment_net_x_return", "post_count_x_volatility", "engagement_x_volume",
]


def _minmax_scale(
    train: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """
    Fit min-max scaler on train, apply to train + all others.
    Columns with zero range are left as-is (already constant).
    Returns scaled arrays in the same order as inputs.
    """
    col_min = train.min(axis=0)
    col_max = train.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0  # Avoid division by zero

    def scale(arr: np.ndarray) -> np.ndarray:
        return (arr - col_min) / col_range

    return (scale(train),) + tuple(scale(o) for o in others)


def split_scale_export(
    aligned: list[dict],
    output_dir: Path = OUTPUT_DIR,
    train_frac: float = 0.7,
    val_frac:   float = 0.15,
) -> dict:

    n = len(aligned)
    if n == 0:
        print("[!] No aligned rows to export.")
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamps = np.array([r["window_start"] for r in aligned], dtype=np.int64)

    market_raw = np.array(
        [[r[c] for c in MARKET_COLS] for r in aligned], dtype=np.float64
    )
    social_raw = np.array(
        [[r[c] for c in SOCIAL_COLS] for r in aligned], dtype=np.float64
    )
    cross_raw = np.array(
        [[r[c] for c in CROSS_COLS] for r in aligned], dtype=np.float64
    )

    n_train = max(1, int(n * train_frac))
    n_val   = max(1, int(n * val_frac))
    n_test  = max(1, n - n_train - n_val)

    if n_train + n_val + n_test > n:
        n_test = n - n_train - n_val

    i_val  = n_train
    i_test = n_train + n_val

    def split(arr):
        return arr[:i_val], arr[i_val:i_test], arr[i_test:]

    m_train, m_val, m_test = split(market_raw)
    s_train, s_val, s_test = split(social_raw)
    c_train, c_val, c_test = split(cross_raw)
    ts_train, ts_val, ts_test = split(timestamps)

    m_train, m_val, m_test = _minmax_scale(m_train, m_val, m_test)
    s_train, s_val, s_test = _minmax_scale(s_train, s_val, s_test)
    c_train, c_val, c_test = _minmax_scale(c_train, c_val, c_test)

    market_scaled = np.vstack([m_train, m_val, m_test])
    social_scaled = np.vstack([s_train, s_val, s_test])
    cross_scaled  = np.vstack([c_train, c_val, c_test])

    paths = {
        "market_features": output_dir / "market_features.npy",
        "social_features": output_dir / "social_features.npy",
        "cross_features":  output_dir / "cross_features.npy",
        "timestamps":      output_dir / "timestamps.npy",
    }
    np.save(paths["market_features"], market_scaled)
    np.save(paths["social_features"], social_scaled)
    np.save(paths["cross_features"],  cross_scaled)
    np.save(paths["timestamps"],      timestamps)

    split_meta = {
        "n_total": n,
        "n_train": int(i_val),
        "n_val":   int(i_test - i_val),
        "n_test":  int(n - i_test),
        "i_val":   int(i_val),
        "i_test":  int(i_test),
    }

    summary = {
        "timestamp":   datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "market_cols": MARKET_COLS,
        "social_cols": SOCIAL_COLS,
        "cross_cols":  CROSS_COLS,
        "shapes": {
            "market_features": list(market_scaled.shape),
            "social_features": list(social_scaled.shape),
            "cross_features":  list(cross_scaled.shape),
        },
        "splits":    split_meta,
        "artifacts": {k: str(v) for k, v in paths.items()},
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[✓] Exported to {output_dir}/")
    print(f"    market_features : {market_scaled.shape}")
    print(f"    social_features : {social_scaled.shape}")
    print(f"    cross_features  : {cross_scaled.shape}")
    print(f"    split           : train={split_meta['n_train']} | val={split_meta['n_val']} | test={split_meta['n_test']}")
    print(f"    summary         : {summary_path}")
    return summary


# FULL PIPELINE ENTRY POINT
# Runs all 6 steps end-to-end in one call.
# Use this from other scripts or notebooks to get the feature
# matrices ready for VAE training.

def run_feature_pipeline(
    btc_csv:    Path = BTC_CSV,
    posts_json: Path = POSTS_JSON,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """
    Run the full feature matrix pipeline (Steps 1–6).

    Returns the summary dict from split_scale_export().
    """
    market_bins = load_and_resample_market(btc_csv)
    social_bins = load_and_bin_social(posts_json)
    aligned     = join_market_social(market_bins, social_bins)
    aligned     = add_derived_market_features(aligned)
    aligned     = add_cross_modal_features(aligned)
    summary     = split_scale_export(aligned, output_dir)
    return summary


# Test

if __name__ == "__main__":

    summary = run_feature_pipeline()

    if summary:
        print("\n  Final feature shapes ")
        for name, shape in summary["shapes"].items():
            print(f"  {name}: {shape}")

        print("\n  Split breakdown ")
        s = summary["splits"]
        print(f"  train: rows 0   → {s['i_val']-1}  ({s['n_train']} rows)")
        print(f"  val:   rows {s['i_val']} → {s['i_test']-1}  ({s['n_val']} rows)")
        print(f"  test:  rows {s['i_test']} → end ({s['n_test']} rows)")

        print("\n Market columns ")
        print(" ", summary["market_cols"])
        print("\n  Social columns ")
        print(" ", summary["social_cols"])
        print("\n  Cross-modal columns ")
        print(" ", summary["cross_cols"])