"""
BTC historical data pipeline (Binance Spot)

Range: 2024-01-01 00:00:00 UTC -> 2024-09-30 23:30:00 UTC (30m candles)
Output: Data_Pipeline/datasets/btcusdt_30m_20240101_20240930.csv

Features:
- OHLC (+ volume, quote volume, trades, taker buy volume)
- price_range = high - low
- price_change = close - open
- price_change_pct = (close - open) / open
- volume_per_trade = volume / number_of_trades
- taker_ratio = taker_buy_base_asset_volume / volume
"""

from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "30m"
INTERVAL_MS = 30 * 60 * 1000
LIMIT = 1000

# End is exclusive so we capture all candles up to 2024-6-1 23:00 UTC
START_UTC = datetime(2023, 9, 1, 0, 0, 0, tzinfo=timezone.utc)
END_EXCLUSIVE_UTC = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

OUTPUT_DIR = Path(__file__).resolve().parent / "datasets"
OUTPUT_FILE = OUTPUT_DIR / "btcusdt_30m_20230901_20240601.csv"

CSV_COLUMNS = [
	"open_time_utc",
	"close_time_utc",
	"symbol",
	"interval",
	"open",
	"high",
	"low",
	"close",
	"volume",
	"quote_asset_volume",
	"number_of_trades",
	"taker_buy_base_asset_volume",
	"taker_buy_quote_asset_volume",
	"price_range",
	"price_change",
	"price_change_pct",
	"volume_per_trade",
	"taker_ratio",
]


def to_ms(dt: datetime) -> int:
	return int(dt.timestamp() * 1000)


def ms_to_utc_iso(ms: int) -> str:
	return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_klines(start_ms: int, end_ms: int) -> list[list]:
	params = {
		"symbol": SYMBOL,
		"interval": INTERVAL,
		"startTime": start_ms,
		"endTime": end_ms,
		"limit": LIMIT,
	}
	url = f"{BINANCE_KLINES_URL}?{urllib.parse.urlencode(params)}"

	with urllib.request.urlopen(url, timeout=30) as response:
		payload = response.read().decode("utf-8")
		data = json.loads(payload)

	if not isinstance(data, list):
		raise RuntimeError(f"Unexpected Binance response: {data}")

	return data


def build_row(kline: list) -> dict:
	open_time_ms = int(kline[0])
	open_price = float(kline[1])
	high_price = float(kline[2])
	low_price = float(kline[3])
	close_price = float(kline[4])
	volume = float(kline[5])
	close_time_ms = int(kline[6])
	quote_asset_volume = float(kline[7])
	number_of_trades = int(kline[8])
	taker_buy_base_asset_volume = float(kline[9])
	taker_buy_quote_asset_volume = float(kline[10])

	price_range = high_price - low_price
	price_change = close_price - open_price
	price_change_pct = (price_change / open_price) if open_price != 0 else 0.0
	volume_per_trade = (volume / number_of_trades) if number_of_trades > 0 else 0.0
	taker_ratio = (taker_buy_base_asset_volume / volume) if volume > 0 else 0.0

	return {
		"open_time_utc": ms_to_utc_iso(open_time_ms),
		"close_time_utc": ms_to_utc_iso(close_time_ms),
		"symbol": SYMBOL,
		"interval": INTERVAL,
		"open": open_price,
		"high": high_price,
		"low": low_price,
		"close": close_price,
		"volume": volume,
		"quote_asset_volume": quote_asset_volume,
		"number_of_trades": number_of_trades,
		"taker_buy_base_asset_volume": taker_buy_base_asset_volume,
		"taker_buy_quote_asset_volume": taker_buy_quote_asset_volume,
		"price_range": price_range,
		"price_change": price_change,
		"price_change_pct": price_change_pct,
		"volume_per_trade": volume_per_trade,
		"taker_ratio": taker_ratio,
	}


def download_all_klines() -> list[list]:
	start_ms = to_ms(START_UTC)
	end_ms = to_ms(END_EXCLUSIVE_UTC) - 1
	all_klines: list[list] = []

	while start_ms <= end_ms:
		batch = fetch_klines(start_ms, end_ms)
		if not batch:
			break

		all_klines.extend(batch)
		last_open_time = int(batch[-1][0])
		next_start = last_open_time + INTERVAL_MS

		if next_start <= start_ms:
			next_start = start_ms + INTERVAL_MS

		start_ms = next_start
		time.sleep(0.15)

	return all_klines


def save_csv(rows: list[dict]) -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
		writer.writeheader()
		writer.writerows(rows)


def main() -> None:
	klines = download_all_klines()

	start_ms = to_ms(START_UTC)
	end_ms = to_ms(END_EXCLUSIVE_UTC)
	filtered_klines = [k for k in klines if start_ms <= int(k[0]) < end_ms]

	rows = [build_row(k) for k in filtered_klines]
	save_csv(rows)

	expected_rows = int((END_EXCLUSIVE_UTC - START_UTC).total_seconds() // (30 * 60))
	print(f"Saved: {OUTPUT_FILE}")
	print(f"Rows: {len(rows)} (expected: {expected_rows})")
	if rows:
		print(f"First candle: {rows[0]['open_time_utc']}")
		print(f"Last candle : {rows[-1]['open_time_utc']}")


if __name__ == "__main__":
	main()

