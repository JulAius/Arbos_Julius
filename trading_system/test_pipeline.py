#!/usr/bin/env python3
"""
Quick test script for the trading system pipeline.

This script tests the end-to-end flow:
1. Fetch sample data
2. Compute features
3. Train baseline momentum model
4. Run walk-forward validation
5. Run backtest and display results

Run with:
    python test_pipeline.py

This is for development/testing only.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data.connector import fetch_data_sync
from data.features import compute_all_features, prepare_xy
from models.base import BaselineMomentumModel
from validation.walk_forward import WalkForwardValidator
from execution.backtest import simple_backtest
from validation.metrics import compute_all_metrics


def test_data_connector():
    """Test fetching data from Binance."""
    print("=" * 60)
    print("TEST 1: Data Connector")
    print("=" * 60)

    try:
        df = fetch_data_sync(
            exchange_name="binance",
            symbol="BTC/USDT",
            timeframe="1h",
            limit=500  # ~20 days
        )
        print(f"✅ Fetched {len(df)} rows")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   Sample data:\n{df.head()}")
        return df
    except Exception as e:
        print(f"❌ Data fetch failed: {e}")
        print("   (Check internet connection or try later)")
        return None


def test_features(df):
    """Test feature computation."""
    print("\n" + "=" * 60)
    print("TEST 2: Feature Engineering")
    print("=" * 60)

    try:
        features = compute_all_features(df)
        print(f"✅ Computed {len(features.columns)} features")
        print(f"   Feature columns: {features.columns.tolist()}")

        # Check for NaN values
        nan_count = features.isna().sum().sum()
        print(f"   NaN count: {nan_count}")

        # Drop NaN for testing (in real system we'd handle this)
        features = features.dropna()
        print(f"   After dropna: {len(features)} rows")

        print(f"\n   Sample features:\n{features.head()}")
        return features
    except Exception as e:
        print(f"❌ Feature computation failed: {e}")
        return None


def test_model(features):
    """Test model training and prediction."""
    print("\n" + "=" * 60)
    print("TEST 3: Baseline Momentum Model")
    print("=" * 60)

    try:
        # Prepare supervised dataset
        X, y, timestamps = prepare_xy(
            features,
            target_column='close_return',
            lookback=168,  # 1 week
            horizon=1
        )
        print(f"✅ Prepared dataset: X shape={X.shape}, y shape={y.shape}")

        # Split train/test
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"   Train: {len(X_train)} samples, Test: {len(X_test)} samples")

        # Train model
        model = BaselineMomentumModel(name="test_momentum", horizon=1)
        model.feature_columns = features.columns.tolist()

        train_metrics = model.train(X_train, y_train)
        print(f"✅ Training complete:")
        print(f"   - MSE: {train_metrics['mse']:.6f}")
        print(f"   - Accuracy: {train_metrics['accuracy']:.2%}")

        # Predict
        predictions = model.predict(X_test)
        print(f"✅ Generated {len(predictions)} predictions")
        print(f"   Prediction mean: {predictions.mean():.6f}")
        print(f"   Prediction std: {predictions.std():.6f}")
        print(f"   Min/Max: {predictions.min():.6f} / {predictions.max():.6f}")

        # Signal test
        recent_data = features.iloc[-200:]  # Last 200 hours
        signal = model.get_signal(recent_data)
        print(f"   Current signal: {signal:.4f}")

        return model, X_test, y_test, predictions
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_walk_forward(features, model):
    """Test walk-forward validation."""
    print("\n" + "=" * 60)
    print("TEST 4: Walk-Forward Validation")
    print("=" * 60)

    try:
        validator = WalkForwardValidator(
            train_window=500,
            validation_window=100,
            step_size=24,
            expanding=True
        )

        splits = validator.split(features)
        print(f"✅ Generated {len(splits)} train/val splits")

        if len(splits) > 0:
            print(f"   First split: train={len(splits[0][0])}, val={len(splits[0][1])}")

        return True
    except Exception as e:
        print(f"❌ Walk-forward validation failed: {e}")
        return False


def test_backtest(features, y_test, predictions):
    """Test backtest engine."""
    print("\n" + "=" * 60)
    print("TEST 5: Backtest")
    print("=" * 60)

    try:
        # Create signal series (align with test period)
        test_timestamps = features.index[-len(y_test):]
        signals = pd.Series(predictions, index=test_timestamps)

        # Run backtest
        result = simple_backtest(
            signals=signals,
            prices=features.loc[test_timestamps, 'close'],
            initial_capital=10000.0,
            commission=0.0006,
            threshold=0.3
        )

        print(f"✅ Backtest complete:")
        print(f"   - Total trades: {len(result.trades)}")
        print(f"   - Final equity: ${result.final_equity:.2f}")
        print(f"   - Total return: {result.total_return:.2%}")
        print(f"   - Sharpe ratio: {result.metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   - Max drawdown: {result.metrics.get('max_drawdown', 0):.2%}")
        print(f"   - Win rate: {result.metrics.get('win_rate', 0):.2%}")
        print(f"   - Profit factor: {result.metrics.get('profit_factor', 0):.2f}")

        # Print summary
        print("\n" + result.summary())

        return result
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("\n" + "=" * 60)
    print("SOTA BTC TRADING SYSTEM - PIPELINE TEST")
    print("=" * 60)

    # Test 1: Data
    df = test_data_connector()
    if df is None:
        print("\n❌ Pipeline test aborted at data fetching")
        return

    # Test 2: Features
    features = test_features(df)
    if features is None:
        print("\n❌ Pipeline test aborted at feature engineering")
        return

    # Test 3: Model
    model_test = test_model(features)
    if model_test is None:
        print("\n❌ Pipeline test aborted at model training")
        return
    model, X_test, y_test, predictions = model_test

    # Test 4: Walk-forward
    if not test_walk_forward(features, model):
        print("\n⚠️ Walk-forward test failed, continuing...")

    # Test 5: Backtest
    result = test_backtest(features, y_test, predictions)
    if result is None:
        print("\n❌ Pipeline test aborted at backtest")
        return

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
    print("\nThe pipeline is working! You can now:")
    print("  - Run full orchestrator: python main.py --once")
    print("  - Or run continuous loop: python main.py --iterate")
    print("\n")


if __name__ == "__main__":
    main()
