"""
Technical indicator features.
"""

import pandas as pd
import numpy as np
from typing import List


def add_rsi(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.DataFrame:
    """Add Relative Strength Index."""
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "close"
) -> pd.DataFrame:
    """Add MACD and signal line."""
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std: float = 2,
    column: str = "close"
) -> pd.DataFrame:
    """Add Bollinger Bands."""
    rolling_mean = df[column].rolling(window=period).mean()
    rolling_std = df[column].rolling(window=period).std()
    df["bb_middle"] = rolling_mean
    df["bb_upper"] = rolling_mean + (rolling_std * std)
    df["bb_lower"] = rolling_mean - (rolling_std * std)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / rolling_mean
    df["bb_position"] = (df[column] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
    return df


def add_atr(
    df: pd.DataFrame,
    period: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close"
) -> pd.DataFrame:
    """Add Average True Range."""
    high_low = df[high_col] - df[low_col]
    high_close = abs(df[high_col] - df[close_col].shift())
    low_close = abs(df[low_col] - df[close_col].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df["atr"] = true_range.rolling(window=period).mean()
    df["atr_ratio"] = df["atr"] / df[close_col]
    return df


def add_momentum(df: pd.DataFrame, periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """Add momentum features (rate of change)."""
    for period in periods:
        df[f"mom_{period}"] = df["close"].pct_change(period)
    return df


def add_volume_features(
    df: pd.DataFrame,
    ma_periods: List[int] = [5, 10, 20]
) -> pd.DataFrame:
    """Add volume-based features."""
    for period in ma_periods:
        df[f"volume_ma_{period}"] = df["volume"].rolling(window=period).mean()
        df[f"volume_ratio_{period}"] = df["volume"] / (df[f"volume_ma_{period}"] + 1e-10)

    # Volume spike detection
    df["volume_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / (df["volume"].rolling(20).std() + 1e-10)

    return df


def add_technical_indicators(df: pd.DataFrame, config=None) -> pd.DataFrame:
    """
    Add all technical indicators to DataFrame.

    Args:
        df: DataFrame with OHLCV columns
        config: FeatureConfig (optional, uses defaults if None)
    """
    df = df.copy()

    # Add all indicators
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_momentum(df, periods=[1, 3, 5, 10, 15, 20])
    df = add_volume_features(df, ma_periods=[5, 10, 20])

    # Additional price-derived features
    df["price_range"] = (df["high"] - df["low"]) / df["close"]
    df["close_open_ratio"] = df["close"] / df["open"]
    df["high_low_ratio"] = df["high"] / df["low"]

    return df
