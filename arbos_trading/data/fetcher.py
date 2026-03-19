"""
Data fetching module for BTC 15-minute OHLCV from Binance public API.
"""

import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import os
from pathlib import Path

from ..config import DataConfig


class BinanceDataFetcher:
    """Fetches BTC/USDT 15-minute OHLCV data from Binance."""

    BASE_URL = "https://api.binance.com/api/v3"

    def __init__(self, symbol: str = DataConfig.SYMBOL, interval: str = DataConfig.INTERVAL):
        self.symbol = symbol.upper()
        self.interval = interval

    def fetch_klines(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = DataConfig.LIMIT_PER_REQUEST
    ) -> List[List]:
        """Fetch klines (candles) from Binance API."""
        url = f"{self.BASE_URL}/klines"
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": limit,
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        for attempt in range(DataConfig.RETRY_ATTEMPTS):
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt == DataConfig.RETRY_ATTEMPTS - 1:
                    raise
                time.sleep(DataConfig.RETRY_DELAY * (attempt + 1))

        return []

    def fetch_historical(
        self,
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical data from start_date to end_date.
        Uses pagination to fetch all available data.
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date) if end_date else datetime.utcnow()

        all_klines = []
        current_start = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        while current_start < end_ms:
            klines = self.fetch_klines(start_time=current_start, end_time=end_ms)
            if not klines:
                break

            all_klines.extend(klines)

            # Next start is the open time of the last candle + 1ms
            if klines:
                last_timestamp = klines[-1][0]
                current_start = last_timestamp + 1

            # Be nice to the API
            time.sleep(0.1)

        # Convert to DataFrame
        if not all_klines:
            return pd.DataFrame()

        df = pd.DataFrame(all_klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades_count",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])

        # Convert types
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
            df[col] = pd.to_numeric(df[col])
        df["trades_count"] = pd.to_numeric(df["trades_count"])

        df = df.sort_values("open_time").reset_index(drop=True)
        df = df[["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades_count", "taker_buy_base", "taker_buy_quote"]]

        return df

    def save_data(self, df: pd.DataFrame, filepath: str):
        """Save DataFrame to CSV."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load DataFrame from CSV."""
        return pd.read_csv(filepath, parse_dates=["open_time"])


def fetch_and_save_historical(
    symbol: str = DataConfig.SYMBOL,
    interval: str = DataConfig.INTERVAL,
    start_date: str = DataConfig.START_DATE,
    end_date: Optional[str] = None,
    filepath: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to fetch and save historical data.
    Returns the DataFrame.
    """
    if filepath is None:
        filepath = Path(DataConfig.DATA_DIR) / f"{symbol.lower()}_{interval}.csv"

    fetcher = BinanceDataFetcher(symbol, interval)
    df = fetcher.fetch_historical(start_date, end_date)

    if not df.empty:
        fetcher.save_data(df, filepath)
        print(f"Fetched {len(df)} candles, saved to {filepath}")
    else:
        print("No data fetched.")

    return df
