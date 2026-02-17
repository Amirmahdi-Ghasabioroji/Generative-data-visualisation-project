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
# Buffer size for real-time data 
BUFFER_SIZE = 500
OUTPUT_DIR = "Binance_realtime"

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