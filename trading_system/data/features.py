"""
Feature Engineering - Compute technical indicators and statistical features.

Provides common features used in trading models: returns, momentum, volatility,
volume spikes, etc.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def calculate_returns(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    Calculate log returns and simple returns for price columns.

    Args:
        df: DataFrame with OHLCV data
        columns: List of columns to compute returns for (default: ['close'])

    Returns:
        DataFrame with additional return columns
    """
    result = df.copy()
    if columns is None:
        columns = ['close']

    for col in columns:
        if col in df.columns:
            result[f'{col}_log_return'] = np.log(df[col] / df[col].shift(1))
            result[f'{col}_return'] = df[col].pct_change()

    return result


def calculate_momentum(df: pd.DataFrame, periods: List[int] = [1, 6, 24, 168]) -> pd.DataFrame:
    """
    Calculate momentum features over various lookback periods.

    Args:
        df: DataFrame with price data
        periods: List of periods (in hours if timeframe is 1h) for momentum

    Returns:
        DataFrame with momentum columns
    """
    result = df.copy()

    for period in periods:
        # Price momentum (percent change over period)
        result[f'momentum_{period}h'] = df['close'].pct_change(periods=period)

        # Rank momentum within recent history (percentile rank)
        rolling_window = max(period * 4, 168)  # At least 4x or 1 week
        result[f'momentum_rank_{period}h'] = result[f'momentum_{period}h'].rolling(
            window=rolling_window, min_periods=1
        ).rank(pct=True)

    return result


def calculate_volatility(df: pd.DataFrame, periods: List[int] = [24, 168]) -> pd.DataFrame:
    """
    Calculate rolling volatility (standard deviation of returns).

    Args:
        df: DataFrame with return data
        periods: List of lookback periods

    Returns:
        DataFrame with volatility columns
    """
    result = df.copy()

    # Ensure returns exist
    if 'close_return' not in result.columns:
        result = calculate_returns(result)

    for period in periods:
        vol = result['close_return'].rolling(window=period).std() * np.sqrt(365 * 24)
        result[f'volatility_{period}h'] = vol

    return result


def calculate_volume_features(df: pd.DataFrame, volume_periods: List[int] = [24, 168]) -> pd.DataFrame:
    """
    Calculate volume-based features: volume z-score, volume spike detection.

    Args:
        df: DataFrame with volume data
        volume_periods: Lookback periods for volume statistics

    Returns:
        DataFrame with volume feature columns
    """
    result = df.copy()

    for period in volume_periods:
        rolling_mean = df['volume'].rolling(window=period).mean()
        rolling_std = df['volume'].rolling(window=period).std()

        # Volume z-score
        result[f'volume_zscore_{period}h'] = (df['volume'] - rolling_mean) / (rolling_std + 1e-8)

        # Volume spike indicator (2 std above mean)
        result[f'volume_spike_{period}h'] = (result[f'volume_zscore_{period}h'] > 2.0).astype(int)

    return result


def calculate_bbands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.

    Args:
        df: DataFrame with close prices
        period: SMA period
        std_dev: Number of standard deviations for bands

    Returns:
        DataFrame with bb_upper, bb_lower, bb_width, bb_position columns
    """
    result = df.copy()

    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()

    result['bb_upper'] = sma + (std * std_dev)
    result['bb_lower'] = sma - (std * std_dev)
    result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / sma
    result['bb_position'] = (df['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'] + 1e-8)

    return result


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Relative Strength Index.

    Args:
        df: DataFrame with close prices
        period: RSI lookback period

    Returns:
        DataFrame with rsi column
    """
    result = df.copy()
    delta = df['close'].diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / (loss + 1e-8)
    result['rsi'] = 100 - (100 / (1 + rs))

    return result


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average True Range.

    Args:
        df: DataFrame with OHLC data
        period: ATR lookback period

    Returns:
        DataFrame with atr and atr_pct columns
    """
    result = df.copy()

    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    result['atr'] = tr.rolling(window=period).mean()
    result['atr_pct'] = result['atr'] / df['close']

    return result


def calculate_adx(df: pd.DataFrame, period: int = 24) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX) for trend strength.

    ADX values above 25 indicate a strong trend (up or down).
    Values below 20 suggest a ranging market.

    Args:
        df: DataFrame with OHLC data
        period: Lookback period for ADX (default 24, ~6 hours for 15m data)

    Returns:
        DataFrame with adx, plus_di, minus_di columns
    """
    result = df.copy()

    # Compute +DM and -DM
    high = df['high']
    low = df['low']
    close = df['close']

    prev_high = high.shift(1)
    prev_low = low.shift(1)

    plus_dm = high - prev_high
    minus_dm = prev_low - low

    # Positive directional movement: up-move greater than down-move and positive
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    # Negative directional movement: down-move greater than up-move and positive
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    # True Range (TR)
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Smooth TR and DM over the period (using simple moving average)
    atr = tr.rolling(window=period).mean()
    plus_dm_smooth = plus_dm.rolling(window=period).mean()
    minus_dm_smooth = minus_dm.rolling(window=period).mean()

    # Directional Indicators
    plus_di = 100 * (plus_dm_smooth / (atr + 1e-8))
    minus_di = 100 * (minus_dm_smooth / (atr + 1e-8))

    # Directional Index (DX)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)

    # ADX is the moving average of DX
    result['adx'] = dx.rolling(window=period).mean()
    result['plus_di'] = plus_di
    result['minus_di'] = minus_di

    # Optional: categorical trend strength
    result['trend_strength'] = pd.cut(result['adx'], bins=[0, 20, 50, 100], labels=['weak', 'moderate', 'strong'])

    return result


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all available features in one pass.

    Args:
        df: Raw OHLCV DataFrame

    Returns:
        DataFrame with all computed features
    """
    result = df.copy()

    # Basic returns
    result = calculate_returns(result)

    # Momentum features
    result = calculate_momentum(result)

    # Volatility
    result = calculate_volatility(result)

    # Volume features
    result = calculate_volume_features(result)

    # Technical indicators
    result = calculate_bbands(result)
    result = calculate_rsi(result)
    result = calculate_atr(result)
    result = calculate_adx(result)  # Regime detection: trend strength

    # Additional derived features
    result['high_low_range'] = (result['high'] - result['low']) / result['close']
    result['close_open_ratio'] = result['close'] / result['open']

    # Time-based features
    result['hour'] = result.index.hour if hasattr(result.index, 'hour') else 0
    result['day_of_week'] = result.index.dayofweek if hasattr(result.index, 'dayofweek') else 0
    result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)

    return result


def prepare_xy(
    df: pd.DataFrame,
    target_column: str = 'close_return',
    feature_columns: Optional[List[str]] = None,
    lookback: int = 168,
    horizon: int = 1
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Prepare supervised learning dataset with lookback windows.

    Args:
        df: Feature DataFrame
        target_column: Column to use as target
        feature_columns: List of feature columns to use (if None, uses all numeric)
        lookback: Number of past timesteps to use as input sequence
        horizon: Steps ahead to predict

    Returns:
        X: Array of shape (n_samples, lookback, n_features)
        y: Array of shape (n_samples,)
        timestamps: Timestamps for each sample
    """
    if feature_columns is None:
        # Use all numeric columns except target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [c for c in numeric_cols if c != target_column and c in df.columns]

    # Drop rows with NaN in features or target
    df_clean = df.dropna(subset=feature_columns + [target_column]).copy()

    X_list, y_list, ts_list = [], [], []

    for i in range(lookback, len(df_clean) - horizon + 1):
        X_seq = df_clean[feature_columns].iloc[i - lookback:i].values
        y_val = df_clean[target_column].iloc[i + horizon - 1]

        X_list.append(X_seq)
        y_list.append(y_val)
        ts_list.append(df_clean.index[i + horizon - 1])

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y, pd.DatetimeIndex(ts_list)
