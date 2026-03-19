"""
Multi-horizon feature generation.
"""

import pandas as pd
import numpy as np
from typing import Dict
from .technical import add_technical_indicators
from .regime import add_regime_features


def resample_df(df: pd.DataFrame, factor: int) -> pd.DataFrame:
    """
    Resample 15-minute data to a longer timeframe by time-based aggregation.

    Args:
        df: DataFrame with 15m data, must have 'open_time' column (datetime)
        factor: number of 15-minute intervals to aggregate (e.g., 4 for 1h = 60min)

    Returns:
        Resampled DataFrame with open_time at the start of each aggregated period.
    """
    if factor <= 1:
        return df.copy()

    df_idx = df.set_index("open_time")
    horizon_minutes = factor * 15
    resampled = (
        df_idx[["open", "high", "low", "close", "volume"]]
        .resample(f"{horizon_minutes}T")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
    )
    resampled = resampled.dropna().reset_index()
    return resampled


def generate_horizon_features(
    df_15m: pd.DataFrame, horizons: Dict[str, int] = None
) -> pd.DataFrame:
    """
    Generate features for multiple horizons.

    Args:
        df_15m: 15-minute DataFrame with OHLCV
        horizons: dict of {name: factor} where factor is number of 15m candles

    Returns:
        DataFrame with horizon-specific features (suffixed with _30m, _1h, _4h, etc.)
    """
    if horizons is None:
        from ..config import FeatureConfig

        horizons = FeatureConfig.HORIZONS

    print(
        f"[horizon] Starting feature generation: {len(df_15m)} rows, horizons={list(horizons.keys())}",
        flush=True,
    )

    print("[horizon] Computing base 15m features...", flush=True)
    df_features = add_technical_indicators(df_15m)
    print("[horizon]   Technical indicators computed", flush=True)
    df_features = add_regime_features(df_features)
    print("[horizon]   Regime features computed", flush=True)

    result_df = df_15m[["open_time"]].copy()
    print("[horizon] Base features ready", flush=True)

    for name, factor in horizons.items():
        if name == "15m" or factor == 1:
            print(f"[horizon] Horizon {name}: using base features", flush=True)
            source_df = df_features.copy()
            feature_cols = [
                col
                for col in source_df.columns
                if col
                not in (
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "quote_volume",
                    "trades_count",
                    "target",
                    "future_close",
                    "taker_buy_base",
                    "taker_buy_quote",
                )
            ]
            to_merge = source_df[["open_time"] + feature_cols].copy()
            result_df = result_df.merge(to_merge, on="open_time", how="left")
            print(
                f"[horizon]   Merged (result now has {len(result_df)} rows, {len(result_df.columns)} cols)",
                flush=True,
            )
        else:
            print(
                f"[horizon] Horizon {name}: resampling by factor {factor}...",
                flush=True,
            )
            df_resampled = resample_df(df_15m, factor)
            print(f"[horizon]   Resampled to {len(df_resampled)} rows", flush=True)

            bar_duration = factor * 15
            bar_close_offset = bar_duration - 15
            merge_key = df_resampled["open_time"] + pd.Timedelta(
                minutes=bar_close_offset
            )

            print("[horizon]   Adding technical indicators...", flush=True)
            df_h = add_technical_indicators(df_resampled)
            print("[horizon]   Adding regime features...", flush=True)
            df_h = add_regime_features(df_h)
            df_h["merge_key"] = merge_key
            print(
                f"[horizon]   Horizon {name} features prepared ({len(df_h)} rows)",
                flush=True,
            )

            exclude = {
                "open_time",
                "merge_key",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "trades_count",
                "target",
                "future_close",
                "taker_buy_base",
                "taker_buy_quote",
            }
            feature_cols = [col for col in df_h.columns if col not in exclude]
            renamed = {col: f"{col}_{name}" for col in feature_cols}
            df_h = df_h.rename(columns=renamed)
            feature_cols = [f"{col}_{name}" for col in feature_cols]

            print(
                f"[horizon] Merging {len(feature_cols)} features for horizon {name}...",
                flush=True,
            )
            to_merge = df_h[["merge_key"] + feature_cols].copy()
            result_df = result_df.merge(
                to_merge, left_on="open_time", right_on="merge_key", how="left"
            )
            result_df = result_df.drop(columns=["merge_key"])
            print(
                f"[horizon]   Merged (result now has {len(result_df)} rows, {len(result_df.columns)} cols)",
                flush=True,
            )

    print("[horizon] Forward filling missing values...", flush=True)
    result_df = result_df.ffill()
    print(
        f"[horizon] Feature generation complete: {len(result_df)} rows, {len(result_df.columns)} columns",
        flush=True,
    )

    return result_df
