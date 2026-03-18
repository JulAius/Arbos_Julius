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


def add_stochastic(df: pd.DataFrame, period: int = 14, smooth: int = 3) -> pd.DataFrame:
    """Add Stochastic Oscillator %K and %D."""
    low_min = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()
    df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
    df["stoch_d"] = df["stoch_k"].rolling(smooth).mean()
    return df


def add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Williams %R."""
    high_max = df["high"].rolling(period).max()
    low_min = df["low"].rolling(period).min()
    df["williams_r"] = -100 * (high_max - df["close"]) / (high_max - low_min + 1e-10)
    return df


def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add Commodity Channel Index."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    sma = typical_price.rolling(period).mean()
    mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"] = (typical_price - sma) / (0.015 * mad + 1e-10)
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized On-Balance Volume."""
    direction = np.sign(df["close"].diff())
    obv = (df["volume"] * direction).cumsum()
    df["obv_norm"] = (obv - obv.rolling(20).mean()) / (obv.rolling(20).std() + 1e-10)
    return df


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add candle body and shadow ratio features."""
    body = abs(df["close"] - df["open"])
    total_range = df["high"] - df["low"] + 1e-10
    df["body_ratio"] = body / total_range
    df["upper_shadow"] = (df["high"] - df[["close", "open"]].max(axis=1)) / total_range
    df["lower_shadow"] = (df[["close", "open"]].min(axis=1) - df["low"]) / total_range
    return df


def add_price_lags(df: pd.DataFrame, n_lags: int = 10) -> pd.DataFrame:
    """Add lagged close price returns for last N periods."""
    returns = df["close"].pct_change()
    for i in range(1, n_lags + 1):
        df[f"ret_lag_{i}"] = returns.shift(i - 1)
    return df


def add_volume_lags(df: pd.DataFrame, n_lags: int = 5) -> pd.DataFrame:
    """Add lagged volume ratio features."""
    vol_ratio = df["volume"] / (df["volume"].rolling(20).mean() + 1e-10)
    for i in range(1, n_lags + 1):
        df[f"vol_lag_{i}"] = vol_ratio.shift(i - 1)
    return df


def add_technical_indicators(df: pd.DataFrame, config=None) -> pd.DataFrame:
    """
    Add all technical indicators to DataFrame.

    Args:
        df: DataFrame with OHLCV columns
        config: FeatureConfig (optional, uses defaults if None)
    """
    df = df.copy()

    # Core indicators
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_momentum(df, periods=[1, 2, 3, 5, 10, 15, 20, 30])
    df = add_volume_features(df, ma_periods=[5, 10, 20])

    # Additional indicators (restored: doubles feature count → ~60/horizon, matching step 72)
    df = add_stochastic(df)
    df = add_williams_r(df)
    df = add_cci(df)
    df = add_obv(df)
    df = add_candle_features(df)
    df = add_price_lags(df, n_lags=10)
    df = add_volume_lags(df, n_lags=5)

    # Additional price-derived features
    df["price_range"] = (df["high"] - df["low"]) / df["close"]
    df["close_open_ratio"] = df["close"] / df["open"]
    df["high_low_ratio"] = df["high"] / df["low"]

    return df
