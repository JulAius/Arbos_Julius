"""
Data processing and cleaning utilities.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data: remove duplicates, handle missing values, ensure proper sorting.
    """
    df = df.copy()

    # Ensure sorted by time
    df = df.sort_values("open_time").reset_index(drop=True)

    # Remove duplicates
    df = df.drop_duplicates(subset=["open_time"])

    # Check for gaps in 15-minute intervals
    expected_diff = pd.Timedelta(minutes=15)
    time_diffs = df["open_time"].diff()
    gaps = time_diffs > expected_diff

    if gaps.any():
        print(f"Warning: Found {gaps.sum()} gaps in data.")

    # Forward fill missing OHLC if any (but not volume)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].ffill()

    # Drop rows where close is still NaN
    df = df.dropna(subset=["close"])

    return df


def resample_to_horizon(
    df: pd.DataFrame,
    horizon_minutes: int,
    factor: int  # e.g., 4 for 1h (4 * 15m)
) -> pd.DataFrame:
    """
    Resample 15-minute data to a longer horizon by aggregating.
    For horizons > 15m, we aggregate 'factor' number of 15m candles.
    """
    if factor <= 1:
        return df.copy()

    # Set index to open_time for resampling
    df_idx = df.set_index("open_time")

    # OHLC: open=first, high=max, low=min, close=last, volume=sum
    resampled = df_idx[["open", "high", "low", "close", "volume"]].resample(f"{horizon_minutes}T").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })

    resampled = resampled.dropna().reset_index()

    return resampled


def create_target(df: pd.DataFrame, forward_periods: int = 1) -> pd.DataFrame:
    """
    Create binary target: 1 if price goes UP in next 15-minute candle, 0 if DOWN.
    """
    df = df.copy()

    # Future close price (next 15-minute candle)
    df["future_close"] = df["close"].shift(-forward_periods)

    # Binary direction: 1 if close > current close, else 0
    df["target"] = (df["future_close"] > df["close"]).astype(int)

    return df


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate various return metrics."""
    df = df.copy()

    # Simple returns
    df["ret_1"] = df["close"].pct_change(1)

    # Log returns
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

    return df
