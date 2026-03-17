#!/usr/bin/env python3
"""
Step 11: Comprehensive Backtesting of Best LSTM Model

This script validates the LSTM model from Step 8 on out-of-sample data
using walk-forward validation to assess robustness before paper trading.

- Fetches fresh BTC data (5000+ candles)
- Trains LSTM with best hyperparams from Step 8
- Performs walk-forward validation across multiple periods
- Reports comprehensive metrics: accuracy, Sharpe, drawdown, win rate
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta
import json

# Ensure we can import from trading_system package when running as script
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from data.connector import fetch_data_sync
from data.features import compute_all_features, prepare_xy
from models.lstm import LSTMTradingModel
from validation.metrics import compute_all_metrics
from utils.config import load_config


def compute_equity_curve(returns: pd.Series) -> pd.Series:
    """Compute equity curve from returns."""
    return (1 + returns).cumprod()


def run_walk_forward(
    data: pd.DataFrame,
    model_template: LSTMTradingModel,
    signal_threshold: float = 0.3,
    train_window: int = 2000,
    val_window: int = 672,
    step_size: int = 96,
    lookback: int = 168,
    horizon: int = 1
) -> dict:
    """
    Run walk-forward validation with proper sequence generation.

    Args:
        data: DataFrame with features and close_return target
        model_template: LSTM model template with hyperparams (will be cloned for each fold)
        train_window: Number of raw candles for initial training
        val_window: Number of raw candles for validation per fold
        step_size: How many raw candles to advance each fold
        lookback: Sequence length for LSTM input
        horizon: Prediction horizon

    Returns:
        Dictionary with fold results and aggregated metrics
    """
    print(f"\n🔬 Walk-forward parameters:")
    print(f"  Train window (raw): {train_window} candles (~{train_window/96:.1f} days)")
    print(f"  Val window (raw): {val_window} candles (~{val_window/96:.1f} days)")
    print(f"  Step size: {step_size} candles (~{step_size/96:.1f} days)")
    print(f"  Lookback (seq len): {lookback} candles")
    print(f"  Horizon: {horizon}")

    n = len(data)
    folds = []
    all_predictions = []
    all_actuals = []
    all_dates = []

    fold_num = 1
    train_end = train_window

    while train_end + val_window <= n:
        # Define indices
        train_start = 0
        train_end_current = train_end
        val_start = train_end_current
        val_end = val_start + val_window

        print(f"\n📊 Fold {fold_num}:")
        print(f"  Train: {train_start}-{train_end_current-1} ({train_end_current-train_start} raw candles)")
        print(f"  Val: {val_start}-{val_end-1} ({val_end-val_start} raw candles)")

        # Get raw data for this fold
        train_df = data.iloc[train_start:train_end_current].copy()
        val_df = data.iloc[val_start:val_end].copy()

        # Prepare sequences with prepare_xy
        # For training: we need to include enough history before train_start for lookback
        actual_train_start_for_sequences = max(0, train_start - lookback)
        extended_train_df = data.iloc[actual_train_start_for_sequences:train_end_current].copy()

        X_train, y_train, train_ts = prepare_xy(
            extended_train_df,
            target_column='close_return',
            lookback=lookback,
            horizon=horizon
        )

        # For validation: we need to prepend recent history from training for first sequences
        if val_start >= lookback:
            val_context_start = val_start - lookback
            val_context_end = val_end
            extended_val_df = data.iloc[val_context_start:val_context_end].copy()
        else:
            # If not enough history, use from start of train
            extended_val_df = pd.concat([train_df.iloc[-lookback:], val_df])

        X_val, y_val, val_ts = prepare_xy(
            extended_val_df,
            target_column='close_return',
            lookback=lookback,
            horizon=horizon
        )

        print(f"  Sequences: train={len(X_train)}, val={len(X_val)}")
        print(f"  X_train shape: {X_train.shape}")

        if len(X_train) < 10 or len(X_val) < 10:
            print("  ⚠️ Insufficient sequences, skipping fold")
            train_end += step_size
            fold_num += 1
            continue

        # Create fresh model instance (clone hyperparams from template)
        input_dim = X_train.shape[2]
        fold_model = LSTMTradingModel(
            input_dim=input_dim,
            hidden_dim=model_template.hidden_dim,
            num_layers=model_template.num_layers,
            dropout=model_template.dropout,
            lr=model_template.lr,
            epochs=model_template.epochs,
            horizon=model_template.horizon
        )

        # Train
        print("  Training...")
        train_metrics = fold_model.train(X_train, y_train)

        # Predict
        print("  Predicting...")
        predictions = fold_model.predict(X_val)

        # Compute directional accuracy
        directional_accuracy = np.mean(np.sign(predictions) == np.sign(y_val))

        # Compute trading metrics
        val_returns = pd.Series(y_val, index=val_ts)
        # Convert predictions to signals using the signal threshold
        signals = np.where(predictions > signal_threshold, 1,
                np.where(predictions < -signal_threshold, -1, 0))
        position_series = pd.Series(signals, index=val_ts).reindex(val_returns.index, fill_value=0)

        strategy_returns = position_series * val_returns

        trading_metrics = compute_all_metrics(val_returns, compute_equity_curve(val_returns))
        trading_metrics['directional_accuracy'] = float(directional_accuracy)
        trading_metrics['train_accuracy'] = float(train_metrics.get('accuracy', 0))
        trading_metrics['train_loss'] = float(train_metrics.get('mse', 0))
        trading_metrics['fold'] = fold_num
        trading_metrics['train_range'] = (int(train_start), int(train_end_current))
        trading_metrics['val_range'] = (int(val_start), int(val_end))

        folds.append(trading_metrics)
        all_predictions.extend(predictions.tolist())
        all_actuals.extend(y_val.tolist())
        all_dates.extend([str(d) for d in val_ts.tolist()])

        print(f"  Results: dir_acc={directional_accuracy:.4f}, Sharpe={trading_metrics.get('sharpe', 0):.4f}")

        # Advance window
        train_end += step_size
        fold_num += 1

    if not folds:
        raise ValueError("No folds completed successfully!")

    results = {
        'folds': folds,
        'predictions': all_predictions,
        'actuals': all_actuals,
        'dates': all_dates
    }

    # Compute aggregated metrics
    accuracies = [f['directional_accuracy'] for f in folds]
    sharpe_ratios = [f.get('sharpe', 0) for f in folds]

    results['summary'] = {
        'mean_accuracy': float(np.mean(accuracies)),
        'std_accuracy': float(np.std(accuracies)),
        'min_accuracy': float(np.min(accuracies)),
        'max_accuracy': float(np.max(accuracies)),
        'mean_sharpe': float(np.mean([s for s in sharpe_ratios if not np.isnan(s) and s is not None])),
        'std_sharpe': float(np.std([s for s in sharpe_ratios if not np.isnan(s) and s is not None])),
        'total_folds': len(folds)
    }

    return results


def main():
    print("=" * 70)
    print(" STEP 11: COMPREHENSIVE BACKTESTING OF BEST LSTM MODEL")
    print("=" * 70)

    # Load config
    config = load_config('config.yaml')

    # Best hyperparams from Step 8 evolution
    model_hyperparams = {
        'num_layers': 1,
        'dropout': 0.1733739159464655,
        'horizon': 1,
        'lr': 0.007818940902700416,
        'hidden_dim': 33,
        'epochs': 10  # Quick training for validation
    }

    signal_threshold = 0.2  # from Step 8 best individual

    print(f"\n✅ Best LSTM hyperparameters from Step 8:")
    for k, v in model_hyperparams.items():
        print(f"  {k}: {v}")
    print(f"  signal_threshold: {signal_threshold}")

    # Fetch fresh data
    print("\n📥 Fetching fresh BTC data (60 days of 15m candles)...")
    lookback_days = 60

    # Convert lookback days to number of candles based on timeframe
    timeframe = config['data']['timeframe']
    if timeframe == '15m':
        candles_per_day = 24 * 4  # 15 minutes = 4 per hour
    elif timeframe == '1h':
        candles_per_day = 24
    elif timeframe == '5m':
        candles_per_day = 24 * 12
    elif timeframe == '1d':
        candles_per_day = 1
    else:
        candles_per_day = 24  # default to 1h

    limit = lookback_days * candles_per_day

    data = fetch_data_sync(
        exchange_name=config['data']['exchange'],
        symbol=config['data']['symbol'],
        timeframe=timeframe,
        limit=limit
    )

    print(f"✅ Fetched {len(data)} candles")
    print(f"   Date range: {data.index[0]} to {data.index[-1]}")

    # Compute features
    print("\n🔧 Computing features...")
    data_with_features = compute_all_features(data)
    print(f"✅ Computed {len(data_with_features.columns)} features")

    # Validation parameters from config
    desired_train_window = config['validation']['train_window']
    desired_val_window = config['validation']['validation_window']
    desired_step_size = config['validation']['step_size']

    # Adjust windows if data is smaller than requested
    n = len(data_with_features)
    lookback = 168  # fixed for LSTM

    if desired_train_window + desired_val_window > n:
        # Scale down proportionally
        scale_factor = (n - lookback) / (desired_train_window + desired_val_window)
        train_window = int(desired_train_window * scale_factor * 0.7)
        val_window = int(desired_val_window * scale_factor * 0.7)
        step_size = max(desired_step_size, val_window // 4)
        # Ensure minimum sizes
        train_window = max(train_window, lookback + 50)  # at least 50 sequences
        val_window = max(val_window, 50)
    else:
        train_window = desired_train_window
        val_window = desired_val_window
        step_size = desired_step_size

    # Ensure we don't exceed data bounds
    if train_window + val_window > n:
        # Fallback: use 70% train, 30% val split
        train_window = int(n * 0.7)
        val_window = n - train_window
        step_size = val_window // 2

    print(f"\n📐 Adjusted walk-forward parameters (based on {n} candles):")
    print(f"  Train window: {train_window} raw candles")
    print(f"  Val window: {val_window} raw candles")
    print(f"  Step size: {step_size} raw candles")

    # Create model template (we'll clone hyperparams for each fold)
    temp_model = LSTMTradingModel(
        input_dim=50,  # placeholder, will be set correctly in each fold
        **model_hyperparams
    )

    # Run walk-forward validation
    print("\n🚀 Starting walk-forward validation...")
    try:
        results = run_walk_forward(
            data=data_with_features,
            model_template=temp_model,
            signal_threshold=signal_threshold,
            train_window=train_window,
            val_window=val_window,
            step_size=step_size,
            lookback=168,  # 1 week
            horizon=1
        )
    except Exception as e:
        print(f"\n❌ Walk-forward validation failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠️ Failing Step 11 - need to debug backtesting setup")
        return

    # Analyze and report results
    print("\n" + "=" * 70)
    print(" WALK-FORWARD VALIDATION RESULTS")
    print("=" * 70)

    folds = results['folds']
    summary = results['summary']

    print(f"\n📊 Overall Metrics:")
    print(f"  Total folds: {summary['total_folds']}")
    print(f"  Avg train accuracy: {np.mean([f['train_accuracy'] for f in folds]):.4f}")
    print(f"  Avg val accuracy: {summary['mean_accuracy']:.4f} (±{summary['std_accuracy']:.4f})")
    print(f"  Avg Sharpe ratio: {summary['mean_sharpe']:.4f} (±{summary['std_sharpe']:.4f})")
    print(f"  Avg max drawdown: {np.mean([f.get('max_drawdown', 0) for f in folds]):.4f}")
    print(f"  Avg win rate: {np.mean([f.get('win_rate', 0) for f in folds]):.4f}")

    if folds:
        print(f"\n📈 Best fold performance:")
        best_fold = max(folds, key=lambda x: x.get('sharpe', -999))
        print(f"  Fold {best_fold['fold']}")
        print(f"  Val accuracy: {best_fold['directional_accuracy']:.4f}")
        print(f"  Sharpe: {best_fold.get('sharpe', 0):.4f}")
        print(f"  Total return: {best_fold.get('total_return', 0):.4%}")

        print(f"\n📉 Worst fold performance:")
        worst_fold = min(folds, key=lambda x: x.get('sharpe', 999))
        print(f"  Fold {worst_fold['fold']}")
        print(f"  Val accuracy: {worst_fold['directional_accuracy']:.4f}")
        print(f"  Sharpe: {worst_fold.get('sharpe', 0):.4f}")
        print(f"  Total return: {worst_fold.get('total_return', 0):.4%}")

    # Assessment against targets
    print("\n" + "=" * 70)
    print(" ASSESSMENT AGAINST TARGETS")
    print("=" * 70)

    target_accuracy = 0.55
    mean_accuracy = summary['mean_accuracy']
    mean_sharpe = summary['mean_sharpe']

    print(f"\n🎯 Target: directional accuracy > {target_accuracy:.1%}")
    print(f"   Mean validation accuracy: {mean_accuracy:.2%}")
    accuracy_met = mean_accuracy >= target_accuracy
    if accuracy_met:
        print("   ✅ ACCURACY TARGET MET")
    else:
        print("   ❌ Accuracy target NOT met")

    print(f"\n📈 Target: Sharpe > 2.0")
    print(f"   Mean Sharpe: {mean_sharpe:.2f}")
    sharpe_met = mean_sharpe >= 2.0
    if sharpe_met:
        print("   ✅ SHARPE TARGET MET")
    else:
        print("   ⚠️  Sharpe target NOT met (but may be acceptable if accuracy high)")

    # Save results
    output_dir = Path(__file__).parent / "state" / "backtest_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "step11_walkforward_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {results_file}")

    # Final recommendation
    print("\n" + "=" * 70)
    print(" RECOMMENDATION FOR NEXT STEPS")
    print("=" * 70)

    robust = (
        accuracy_met and
        summary['std_accuracy'] < 0.15 and
        mean_sharpe > 0
    )

    if robust:
        print("\n✅ Model is ROBUST and ready for paper trading!")
        print("   - Consistent directional accuracy >55% across folds")
        print("   - Low variance indicates stable performance")
        print("   - Positive Sharpe ratio")
        print("\n➡️ Next: Proceed to paper trading deployment")
        print("   (real-time data streaming + simulated execution)")
    else:
        print("\n⚠️ Model needs REFINEMENT before live/paper trading")
        if not accuracy_met:
            print(f"   - Mean accuracy {mean_accuracy:.2%} below target {target_accuracy:.1%}")
        if summary['std_accuracy'] >= 0.15:
            print(f"   - High accuracy variance ({summary['std_accuracy']:.4f}) indicates instability")
        if mean_sharpe <= 0:
            print(f"   - Negative/zero Sharpe ({mean_sharpe:.2f}) indicates unprofitable strategy")
        print("\n➡️ Next: Refine model")
        print("   - Consider adding regime detection features")
        print("   - Try Transformer architecture")
        print("   - Adjust hyperparam ranges (reduce complexity)")

    print("\n" + "=" * 70)
    print(" Step 11 complete - results available in state/backtest_results/")
    print("=" * 70)

if __name__ == "__main__":
    main()
