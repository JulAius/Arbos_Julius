"""
Market regime detection features.
"""

import pandas as pd
import numpy as np


def detect_volatility_regime(
    df: pd.DataFrame,
    atr_period: int = 20,
    atr_quantile: float = 0.7
) -> pd.DataFrame:
    """
    Detect if market is in high or low volatility regime.
    Classifies based on ATR percentile.
    """
    df = df.copy()
    atr_rolling = df["atr"] / df["close"]  # ATR as percentage of price
    threshold = atr_rolling.rolling(atr_period * 4).quantile(atr_quantile)
    df["vol_regime"] = (atr_rolling > threshold).astype(int)  # 1 = high vol, 0 = low vol
    return df


def detect_trend_regime(
    df: pd.DataFrame,
    fast_ma: int = 10,
    slow_ma: int = 30
) -> pd.DataFrame:
    """
    Detect trend regime using moving average cross.
    """
    df = df.copy()
    fast = df["close"].rolling(fast_ma).mean()
    slow = df["close"].rolling(slow_ma).mean()
    df["trend_regime"] = (fast > slow).astype(int)  # 1 = uptrend, 0 = downtrend/range
    return df


def detect_volume_regime(
    df: pd.DataFrame,
    lookback: int = 20
) -> pd.DataFrame:
    """
    Detect unusual volume activity.
    """
    df = df.copy()
    volume_ma = df["volume"].rolling(lookback).mean()
    volume_std = df["volume"].rolling(lookback).std()
    df["volume_regime"] = ((df["volume"] - volume_ma) / (volume_std + 1e-10) > 2).astype(int)
    return df


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all regime-detection features to DataFrame.
    """
    df = df.copy()

    # Ensure technical indicators exist
    if "atr" not in df.columns:
        from .technical import add_atr
        df = add_atr(df)

    df = detect_volatility_regime(df)
    df = detect_trend_regime(df)
    df = detect_volume_regime(df)

    # Combined regime indicator
    # count number of active regimes (high vol + uptrend + high volume)
    df["regime_strength"] = (
        df["vol_regime"] + df["trend_regime"] + df["volume_regime"]
    ) / 3.0

    return df
