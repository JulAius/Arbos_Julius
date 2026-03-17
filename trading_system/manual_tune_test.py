#!/usr/bin/env python3
"""
Quick manual tuning test: try various BaselineMomentumModel hyperparams
to see if >55% directional accuracy is achievable on 15m data.
"""

import asyncio
import pandas as pd
import numpy as np
from data.connector import fetch_data_sync
from data.features import compute_all_features, prepare_xy
from models.base import BaselineMomentumModel

async def test_configuration(momentum_periods, volume_period, threshold, lookback=168):
    """Test one model configuration."""
    # Fetch data
    df = await asyncio.to_thread(
        fetch_data_sync,
        exchange_name="binance",
        symbol="BTC/USDT",
        timeframe="15m",
        limit=2000  # ~20 days
    )
    features = compute_all_features(df)

    # Prepare dataset
    X, y, timestamps = prepare_xy(
        features,
        target_column='close_return',
        lookback=lookback,
        horizon=1
    )

    # Split train/test (70/30)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train model with given hyperparams
    model = BaselineMomentumModel(
        name=f"test_mom{momentum_periods}_vol{volume_period}",
        horizon=1
    )
    model.feature_columns = features.columns.tolist()
    model.momentum_periods = momentum_periods if isinstance(momentum_periods, list) else [momentum_periods]
    model.volume_period = volume_period

    train_metrics = model.train(X_train, y_train)
    print(f"Train accuracy: {train_metrics.get('accuracy', 0):.2%}")

    # Predict on test
    y_pred = model.predict(X_test)

    # Directional accuracy
    y_test_dir = np.sign(y_test)
    y_pred_dir = np.sign(y_pred)
    accuracy = (y_test_dir == y_pred_dir).mean()

    # Also compute signal distribution
    unique, counts = np.unique(y_pred_dir, return_counts=True)
    dist = dict(zip(unique, counts))

    return accuracy, train_metrics.get('accuracy', 0), dist

async def main():
    print("Manual Tuning Test for BaselineMomentumModel on 15m BTC")
    print("=" * 70)

    test_cases = [
        (6, 12, 0.3),     # short-term
        (24, 24, 0.3),    # 6h, 6h
        (64, 16, 0.3),    # 16h, 4h
        (96, 24, 0.3),    # 24h, 6h
        (256, 64, 0.3),   # 64h, 16h
        ([6, 24, 168], 24, 0.3),  # multi-scale like original
    ]

    results = []
    for mom, vol, thresh in test_cases:
        print(f"\nTesting: momentum_periods={mom}, volume_period={vol}")
        try:
            acc, train_acc, dist = await test_configuration(mom, vol, thresh)
            print(f"  Test directional accuracy: {acc:.2%}")
            print(f"  Train accuracy: {train_acc:.2%}")
            print(f"  Prediction distribution: {dist}")
            results.append((mom, vol, acc))
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for mom, vol, acc in results:
        print(f"mom={mom}, vol={vol} -> acc={acc:.2%}")

    # Best?
    if results:
        best = max(results, key=lambda x: x[2])
        print(f"\nBest: mom={best[0]}, vol={best[1]} -> acc={best[2]:.2%}")

if __name__ == '__main__':
    asyncio.run(main())
