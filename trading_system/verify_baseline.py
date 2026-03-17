#!/usr/bin/env python3
"""
Manual test: evaluate BaselineMomentumModel with config (24,24) using
EvolutionEngine's exact data split (last 500 for validation, up to 2000 for training).
"""

import asyncio
import numpy as np
from data.connector import DataConnector
from data.features import compute_all_features, prepare_xy
from models.base import BaselineMomentumModel

async def test_config_on_evo_split(momentum_periods, volume_period):
    """Test a specific configuration using EvolutionEngine's data split."""
    # Fetch data exactly as EvolutionEngine does
    exchange = 'binance'
    symbol = 'BTC/USDT'
    timeframe = '15m'
    lookback_days = 30
    candles_per_day = 24 * 4  # 15m -> 96 per day
    limit = lookback_days * candles_per_day  # 30*96 = 2880 but we fetch maybe more

    async with DataConnector(exchange) as dc:
        raw_data = await dc.fetch_ohlcv(symbol, timeframe, limit=limit)

    print(f"Fetched {len(raw_data)} candles")
    features = compute_all_features(raw_data)
    print(f"Features shape: {features.shape}")

    # Use EvolutionEngine._get_datasets logic
    total_len = len(features)
    val_end = total_len
    val_start = max(0, total_len - 500)
    train_end = val_start
    train_start = max(0, train_end - 2000)

    train_df = features.iloc[train_start:train_end]
    val_df = features.iloc[val_start:val_end]

    print(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")

    lookback = 168
    horizon = 1

    X_train, y_train, _ = prepare_xy(train_df, target_column='close_return', lookback=lookback, horizon=horizon)
    X_val, y_val, _ = prepare_xy(val_df, target_column='close_return', lookback=lookback, horizon=horizon)

    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

    # Train model
    model = BaselineMomentumModel(
        name=f"test_{momentum_periods}_{volume_period}",
        horizon=horizon
    )
    model.feature_columns = features.columns.tolist()
    model.momentum_periods = momentum_periods if isinstance(momentum_periods, list) else [momentum_periods]
    model.volume_period = volume_period

    train_metrics = model.train(X_train, y_train)
    print(f"Train metrics: {train_metrics}")

    # Predict on validation
    y_pred = model.predict(X_val)

    # Directional accuracy
    y_val_dir = np.sign(y_val)
    y_pred_dir = np.sign(y_pred)
    accuracy = (y_val_dir == y_pred_dir).mean()

    # Check prediction distribution
    unique, counts = np.unique(y_pred_dir, return_counts=True)
    dist = dict(zip([int(u) for u in unique], counts))

    print(f"Validation directional accuracy: {accuracy:.4f}")
    print(f"Prediction distribution: {dist}")

    return accuracy

async def main():
    print("Testing configs using EvolutionEngine's data split")
    print("=" * 70)

    # Known good from manual test
    print("\nConfig A: momentum=24, volume=24 (expected ~57% on diff split)")
    accA = await test_config_on_evo_split(24, 24)
    print(f"Result: {accA:.2%}")

    # Config that evolution found (48,12)
    print("\nConfig B: momentum=48, volume=12 (evolution best)")
    accB = await test_config_on_evo_split(48, 12)
    print(f"Result: {accB:.2%}")

    # Config from initial run (64,16)
    print("\nConfig C: momentum=64, volume=16")
    accC = await test_config_on_evo_split(64, 16)
    print(f"Result: {accC:.2%}")

    print("\n" + "=" * 70)

if __name__ == '__main__':
    asyncio.run(main())
