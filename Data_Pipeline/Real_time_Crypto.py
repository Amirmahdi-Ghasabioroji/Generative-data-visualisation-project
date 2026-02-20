"""
Binance Real-Time Data Pipeline — Group J
Symbols: BTCUSDT + ETHUSDT
Stack: numpy only (no pandas)
Covers: FR1 (live ingestion), FR4 (real-time updates), NFR1 (<500ms latency)
"""

import asyncio
import csv
import os
import numpy as np
from collections import deque
from datetime import datetime
from binance import AsyncClient, BinanceSocketManager

# ─────────────────────────────────────────────────────────────
# STAGE 1: CONFIGURATION
# What it does: Defines what you're collecting and how much.
# Why it matters: Centralising config here means any team member
# can change symbols or buffer size without touching the logic.
# ─────────────────────────────────────────────────────────────

# Symbols to track
SYMBOLS = ['BTCUSDT', 'ETHUSDT']
INTERVAL = "1m"
OUTPUT_DIR = "binance_realtime"
# Buffer size for real-time data 
BUFFER_SIZE = 500

# Defining the structure of the final numpy array that the PCA/VAE recieve 
FEATURE_COLS = [
    "timestamp", "symbol",
    "open", "high", "low", "close", "volume",
    "num_trades", "quote_volume", "taker_volume",
    # Derived features (computed below)
    "price_range", "price_change", "volume_per_trade", "taker_ratio"
]


# ─────────────────────────────────────────────────────────────
# STAGE 2: SHARED STATE
# What it does: A thread-safe rolling buffer that both symbol
# streams write into concurrently.
# Why it matters: asyncio.gather() runs both streams at the same
# time — they share this buffer so the ML layer always sees a
# combined BTC+ETH snapshot, not just one coin.
# ─────────────────────────────────────────────────────────────

# One deque per symbol — stores raw row dicts of completed candles
buffers = {symbol: deque(maxlen=BUFFER_SIZE) for symbol in SYMBOLS}


# ─────────────────────────────────────────────────────────────
# STAGE 3: FEATURE EXTRACTION
# What it does: Converts one raw Binance kline message into a
# numpy-compatible feature row with derived metrics added.
# Why it matters: FR2 requires numerical input for PCA/VAE.
# Derived features (price_range, taker_ratio etc.) add signal
# that raw OHLCV alone doesn't capture.
#
# Key: we only process candles where kline['x'] == True,
# meaning the 1-minute candle is CLOSED (final values).
# Binance streams partial updates every ~250ms — filtering
# to closed candles only gives you clean, stable data points.
# ─────────────────────────────────────────────────────────────

def extract_features(kline: dict, symbol: str) -> dict | None:
    """
    Extracts features from a Binance kline message. (parsing the kline input)
    Returns a dict with both raw and derived features, or None if the candle isn't closed.
    """
    if not kline['x']:  # Only process closed candles
        return None

    try:
        open_price = float(kline['o'])
        high_price = float(kline['h'])
        low_price = float(kline['l'])
        close_price = float(kline['c'])
        volume = float(kline['v'])
        num_trades = int(kline['n'])
        quote_volume = float(kline['q'])
        taker_volume = float(kline['V'])

        # Derived features
        price_range = high_price - low_price
        price_change = close_price - open_price
        volume_per_trade = volume / num_trades if num_trades > 0 else 0
        taker_ratio = taker_volume / quote_volume if quote_volume > 0 else 0

        return {
            "timestamp": datetime.fromtimestamp(kline['t'] / 1000).isoformat(),
            "symbol": symbol,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "num_trades": num_trades,
            "quote_volume": quote_volume,
            "taker_volume": taker_volume,
            "price_range": price_range,
            "price_change": price_change,
            "volume_per_trade": volume_per_trade,
            "taker_ratio": taker_ratio
        }
    except (ValueError, KeyError) as e:
        print(f"Error extracting features: {e}")
        return None
    

# ─────────────────────────────────────────────────────────────
# STAGE 4: NUMPY FEATURE MATRIX
# What it does: Converts the rolling buffer into a pure numpy
# array of shape (N, 10) — the numerical features only, 
# normalised to [0, 1] for stable PCA/VAE training.
# Why it matters: This is the direct input to FR2 models.
# Normalisation prevents high-value features (close price
# at ~$90,000 for BTC) from dominating PCA components.
# ─────────────────────────────────────────────────────────────

NUMERICAL_KEYS = [
    "open", "high", "low", "close", "volume",
    "num_trades", "price_range", "price_change",
    "volume_per_trade", "taker_ratio"
]

def build_feature_matrix(symbol: str) -> np.ndarray:
    """
    Build a normalised (N, 10) numpy feature matrix from the buffer.
    Called after each new closed candle arrives.
    Returns empty array if buffer has fewer than 2 rows.
    """
    buf = list(buffers[symbol])
    if len(buf) < 2:
        return np.empty((0, len(NUMERICAL_KEYS)))

    # Stack rows into (N, 10) matrix
    matrix = np.array(
        [[row[k] for k in NUMERICAL_KEYS] for row in buf],
        dtype=np.float64
    )

    # Min-max normalise each column to [0, 1]
    col_min = matrix.min(axis=0)
    col_max = matrix.max(axis=0)
    col_range = col_max - col_min

    # Identify constant columns (zero range) to avoid division by zero
    constant_mask = (col_range == 0)
    col_range[constant_mask] = 1.0

    matrix = (matrix - col_min) / col_range

    # For constant columns, assign a neutral mid-range value (0.5)
    if np.any(constant_mask):
        matrix[:, constant_mask] = 0.5
    return matrix  # Shape: (N, 10) 


# ─────────────────────────────────────────────────────────────
# STAGE 5: CSV SAVING
# What it does: Appends each closed candle row to a per-symbol
# CSV file. Creates the folder and file automatically on first run.
# The output_dir parameter lets you specify exactly where files
# are saved at runtime rather than relying on a hardcoded path.
# Why it matters: Lets you collect hours of live data and replay
# it offline during development — no live connection needed
# when testing your PCA/VAE models. Separating output_dir as a
# parameter also makes this safe to call from Flask/React later.
# ─────────────────────────────────────────────────────────────

def save_row_to_csv(row: dict, symbol: str, output_dir: str = OUTPUT_DIR) -> None:
    """
    Append one feature row to the symbol's CSV file.
    Creates the folder and file with headers automatically if they
    don't exist yet — no manual setup required.

    Args:
        row        : Feature dict from extract_features()
        symbol     : e.g. "BTCUSDT" — used to name the file
        output_dir : Directory to save CSVs into.
                     Defaults to data/binance_realtime but can
                     be overridden at runtime.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{symbol.lower()}_klines.csv")
    file_exists = os.path.isfile(filepath)

    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLS)

        if not file_exists:
            writer.writeheader()
            print(f"[CSV] Created: {filepath}")  # Confirms exact save location

        writer.writerow(row)

# ─────────────────────────────────────────────────────────────
# STAGE 6: STREAM LISTENER (one per symbol)
# What it does: Opens a persistent WebSocket to Binance for one
# symbol and processes every incoming kline message.
# Why it matters: asyncio.gather() in Stage 7 runs one of these
# for BTC and one for ETH simultaneously — they're completely
# independent but share the same buffers and CSV files.
#
# Important Binance behaviour: the stream sends an update every
# ~250ms for the CURRENT candle. We filter with kline['x']
# so we only process a candle once it's fully closed.
# ─────────────────────────────────────────────────────────────

async def stream_symbol(bm: BinanceSocketManager, symbol: str) -> None:
    """
    Listen to kline stream for one symbol indefinitely.
    On each closed candle: extract features → buffer → save CSV → build matrix.
    """
    print(f"[STREAM] Starting {symbol} @ {INTERVAL}")

    async with bm.kline_socket(symbol=symbol, interval=INTERVAL) as stream:
        while True:
            msg = await stream.recv()

            # Handle stream errors gracefully (NFR4)
            if msg.get('e') == 'error':
                print(f"[ERROR] {symbol}: {msg.get('m', 'unknown error')}")
                break

            kline = msg['k']
            row = extract_features(kline, symbol)

            if row is None:
                continue  # Candle still open, skip

            # Add to rolling buffer
            buffers[symbol].append(row)

            # Save to CSV
            save_row_to_csv(row, symbol, OUTPUT_DIR)

            # Build feature matrix (this is what your ML model will consume)
            matrix = build_feature_matrix(symbol)

            print(
                f"[{symbol}] Candle closed | "
                f"Close: {row['close']:.2f} | "
                f"Trades: {row['num_trades']} | "
                f"Buffer: {len(buffers[symbol])}/{BUFFER_SIZE} | "
                f"Matrix: {matrix.shape}"
            )

            # ── PLUG YOUR MODEL IN HERE ──────────────────────────
            # When Weeks 5-6 arrive, replace this comment with:
            #   pca_result = pca_model.transform(matrix)
            # or in Weeks 7-8:
            #   latent_z = vae_encoder(torch.tensor(matrix))
            # ─────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# STAGE 7: CONCURRENT STREAM MANAGER
# What it does: Creates one Binance client, then uses
# asyncio.gather() to run both symbol streams in parallel.
# Why it matters: gather() is the correct way to run multiple
# async streams — they run concurrently in the same event loop
# without blocking each other. This satisfies NFR1 (<500ms)
# because neither stream waits for the other.
# ─────────────────────────────────────────────────────────────

async def run_pipeline() -> None:
    """
    Entry point: initialise client and launch both streams concurrently.
    No API key required — Binance public kline streams are unauthenticated.
    """
    print("=" * 50)
    print("  Binance Real-Time Pipeline — Group J")
    print(f"  Symbols : {', '.join(SYMBOLS)}")
    print(f"  Interval: {INTERVAL}")
    print(f"  Output  : {OUTPUT_DIR}/")
    print("=" * 50)

    # No API key needed for public market streams
    client = await AsyncClient.create()
    bm = BinanceSocketManager(client)

    try:
        # Run BTC and ETH streams concurrently
        await asyncio.gather(
            stream_symbol(bm, "BTCUSDT"),
            stream_symbol(bm, "ETHUSDT"),
        )
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user. Closing connection...")
    finally:
        await client.close_connection()
        print("[DONE] Connection closed. CSVs saved to:", OUTPUT_DIR)


# ─────────────────────────────────────────────────────────────
# STAGE 8: RECONNECTION WRAPPER (NFR4)
# What it does: Wraps the entire pipeline in a retry loop so
# that if the WebSocket drops (network blip, Binance restart),
# it automatically reconnects after 5 seconds.
# Why it matters: NFR4 requires error handling and graceful
# degradation. Without this, one dropped connection kills
# your entire data collection session.
# ─────────────────────────────────────────────────────────────

async def main() -> None:
    while True:
        try:
            await run_pipeline()
        except Exception as e:
            print(f"[WARN] Pipeline error: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
