#!/usr/bin/env python3
"""
Grid search over momentum_periods and volume_period to find all combos
that achieve >50% accuracy on EvolutionEngine data split.
"""

import asyncio
import numpy as np
from data.connector import DataConnector
from data.features import compute_all_features, prepare_xy
from models.base import BaselineMomentumModel

async def evaluate_config(momentum_periods, volume_period, threshold=0.3):
    """Evaluate a single configuration."""
    # Fetch exactly 1000 candles (as EvolutionEngine)
    exchange = 'binance'
    symbol = 'BTC/USDT'
    timeframe = '15m'
    limit = 1000

    async with DataConnector(exchange) as dc:
        raw_data = await dc.fetch_ohlcv(symbol, timeframe, limit=limit)

    features = compute_all_features(raw_data)

    # EvolutionEngine train/val split
    total_len = len(features)
    val_end = total_len
    val_start = max(0, total_len - 500)
    train_end = val_start
    train_start = max(0, train_end - 2000)

    train_df = features.iloc[train_start:train_end]
    val_df = features.iloc[val_start:val_end]

    lookback = 168
    horizon = 1
    X_train, y_train, _ = prepare_xy(train_df, target_column='close_return', lookback=lookback, horizon=horizon)
    X_val, y_val, _ = prepare_xy(val_df, target_column='close_return', lookback=lookback, horizon=horizon)

    # Train model
    model = BaselineMomentumModel(name=f"grid_{momentum_periods}_{volume_period}", horizon=horizon)
    model.feature_columns = features.columns.tolist()
    model.momentum_periods = momentum_periods if isinstance(momentum_periods, list) else [momentum_periods]
    model.volume_period = volume_period

    # Train
    train_metrics = model.train(X_train, y_train)

    # Predict on val
    y_pred = model.predict(X_val)
    y_val_dir = np.sign(y_val)
    y_pred_dir = np.sign(y_pred)
    accuracy = (y_val_dir == y_pred_dir).mean()

    # Signal distribution
    unique, counts = np.unique(y_pred_dir, return_counts=True)
    dist = dict(zip([int(u) for u in unique], counts))

    return {
        'momentum': momentum_periods,
        'volume': volume_period,
        'train_acc': train_metrics.get('accuracy', 0),
        'val_acc': accuracy,
        'dist': dist
    }

async def main():
    print("GRID SEARCH: momentum vs volume on 15m BTC (1000 candles)")
    print("=" * 70)

    momentum_options = [12, 24, 48, 96]
    volume_options = [12, 24, 48]

    results = []
    for mom in momentum_options:
        for vol in volume_options:
            print(f"Testing mom={mom}, vol={vol}...")
            try:
                res = await evaluate_config(mom, vol)
                results.append(res)
                print(f"  -> val_acc={res['val_acc']:.2%}, dist={res['dist']}")
            except Exception as e:
                print(f"  ERROR: {e}")

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"mom={r['momentum']:3d}, vol={r['volume']:3d} -> val_acc={r['val_acc']:6.2%}, train_acc={r['train_acc']:6.2%}, dist={r['dist']}")

    # Find best
    best = max(results, key=lambda x: x['val_acc'])
    print(f"\nBEST: mom={best['momentum']}, vol={best['volume']} => val_acc={best['val_acc']:.2%}")

    # Count how many exceed 55%
    good = [r for r in results if r['val_acc'] >= 0.55]
    print(f"\nConfigurations with acc >= 55%: {len(good)}/{len(results)}")
    for r in good:
        print(f"  - mom={r['momentum']}, vol={r['volume']} => {r['val_acc']:.2%}")

if __name__ == '__main__':
    asyncio.run(main())
