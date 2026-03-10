"""
Optional live BTC PCA runner.

Run this file only when you want PCA visual output.
The base Data_Pipeline/Real_time_Crypto.py remains data-ingestion only.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
from binance import AsyncClient, BinanceSocketManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Data_Pipeline import Real_time_Crypto as rtc
from AI_systems import pca_model


PCA_N_COMPONENTS = 3
ENABLE_PCA_PLOT = True


pca_models = {
    symbol: pca_model.PCA(n_components=PCA_N_COMPONENTS)
    for symbol in rtc.SYMBOLS
}


def run_pca(symbol: str, matrix: np.ndarray) -> np.ndarray | None:
    min_rows = max(PCA_N_COMPONENTS + 1, 2)
    if matrix.shape[0] < min_rows:
        return None

    model = pca_models[symbol]
    plot_title = f"{symbol} PCA 3D (rolling {matrix.shape[0]} rows)"

    if ENABLE_PCA_PLOT:
        reduced = model.fit_transform_plot(matrix, title=plot_title)
    else:
        model.fit(matrix)
        reduced = model.transform(matrix)

    return reduced[-1]


async def stream_symbol_with_pca(bm: BinanceSocketManager, symbol: str) -> None:
    print(f"[PCA-STREAM] Starting {symbol} @ {rtc.INTERVAL}")

    async with bm.kline_socket(symbol=symbol, interval=rtc.INTERVAL) as stream:
        while True:
            msg = await stream.recv()

            if msg.get("e") == "error":
                print(f"[ERROR] {symbol}: {msg.get('m', 'unknown error')}")
                break

            kline = msg["k"]
            row = rtc.extract_features(kline, symbol)
            if row is None:
                continue

            rtc.buffers[symbol].append(row)
            rtc.save_row_to_csv(row, symbol, rtc.OUTPUT_DIR)

            matrix = rtc.build_feature_matrix(symbol)
            print(
                f"[{symbol}] Candle closed | "
                f"Close: {row['close']:.2f} | "
                f"Trades: {row['num_trades']} | "
                f"Buffer: {len(rtc.buffers[symbol])}/{rtc.BUFFER_SIZE} | "
                f"Matrix: {matrix.shape}"
            )

            pca_latest = run_pca(symbol, matrix)
            if pca_latest is None:
                min_rows = max(PCA_N_COMPONENTS + 1, 2)
                print(f"[{symbol}] PCA waiting: need >= {min_rows} rows, have {matrix.shape[0]}")
            else:
                print(f"[{symbol}] PCA latest ({PCA_N_COMPONENTS}D): {np.round(pca_latest, 4).tolist()}")


async def run_pipeline() -> None:
    print("=" * 58)
    print("  Live BTC Data + Optional PCA Visual Runner")
    print(f"  Symbols : {', '.join(rtc.SYMBOLS)}")
    print(f"  Interval: {rtc.INTERVAL}")
    print(f"  Output  : {rtc.OUTPUT_DIR}/")
    print(f"  PCA Plot: {'ON' if ENABLE_PCA_PLOT else 'OFF'}")
    print("=" * 58)

    rtc.preload_buffers_from_csv(rtc.OUTPUT_DIR, rtc.BUFFER_SIZE)

    client = await AsyncClient.create()
    bm = BinanceSocketManager(client)

    try:
        await asyncio.gather(*(stream_symbol_with_pca(bm, symbol) for symbol in rtc.SYMBOLS))
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user. Closing connection...")
    finally:
        await client.close_connection()
        print("[DONE] Connection closed. CSVs saved to:", rtc.OUTPUT_DIR)


async def main() -> None:
    while True:
        try:
            await run_pipeline()
            break
        except Exception as exc:
            print(f"[WARN] PCA runner error: {exc}. Reconnecting in 5s...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
