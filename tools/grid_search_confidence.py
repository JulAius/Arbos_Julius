"""
Grid search over MIN_CONFIDENCE to find optimal threshold after leakage fix.
Runs the full walk-forward validation for each confidence value.
"""

import sys
import os
import random
import numpy as np
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fix seeds for reproducibility
np.random.seed(42)
random.seed(42)

from arbos_trading.main import TradingSystem
from arbos_trading import config as cfg

CONFIDENCE_VALUES = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65, 0.70, 0.75]

results = []

system = TradingSystem()

# Load and prepare data once
df_raw = system.load_or_fetch_data()

import pandas as pd
from datetime import timedelta
from arbos_trading.config import ModelConfig, ValidationConfig

if ModelConfig.ONLINE_DATA_DAYS is not None:
    cutoff = df_raw["open_time"].max() - timedelta(days=ModelConfig.ONLINE_DATA_DAYS)
    df_raw = df_raw[df_raw["open_time"] >= cutoff].copy()

print(f"Data: {len(df_raw)} rows", flush=True)
df = system.prepare_features(df_raw)
print(f"Features: {len(df)} rows, {len(system.feature_names)} features", flush=True)

# WFV data
if ValidationConfig.MAX_WALK_FORWARD_DAYS is not None:
    wf_cutoff = df["open_time"].max() - timedelta(days=ValidationConfig.MAX_WALK_FORWARD_DAYS)
    df_wf = df[df["open_time"] >= wf_cutoff].copy()
else:
    df_wf = df

print(f"WFV data: {len(df_wf)} rows", flush=True)

for conf in CONFIDENCE_VALUES:
    print(f"\n{'='*50}", flush=True)
    print(f"Testing MIN_CONFIDENCE = {conf}", flush=True)
    print(f"{'='*50}", flush=True)

    # Reset seeds for each config
    np.random.seed(42)
    random.seed(42)

    # Override consensus config
    cfg.ConsensusConfig.MIN_CONFIDENCE = conf
    system.consensus_gate.min_confidence = conf

    metrics = system.run_walk_forward(df_wf)

    row = {
        "min_confidence": conf,
        "accuracy": metrics.get("accuracy", 0.0),
        "bets_per_month": metrics.get("bets_per_month", 0.0),
        "sharpe": metrics.get("sharpe", 0.0),
        "total_return": metrics.get("total_return", 0.0),
        "n_trades": metrics.get("n_trades", 0),
        "n_folds": metrics.get("n_folds", 0),
    }
    results.append(row)
    print(f"-> acc={row['accuracy']:.4f}, bets/mo={row['bets_per_month']:.1f}, sharpe={row['sharpe']:.3f}, return={row['total_return']:.4f}", flush=True)

print("\n" + "="*60, flush=True)
print("GRID SEARCH RESULTS", flush=True)
print("="*60, flush=True)
print(f"{'Conf':>6} | {'Accuracy':>9} | {'Bets/mo':>8} | {'Sharpe':>8} | {'Return':>8}", flush=True)
print("-"*55, flush=True)
for r in results:
    print(f"{r['min_confidence']:>6.2f} | {r['accuracy']:>9.4f} | {r['bets_per_month']:>8.1f} | {r['sharpe']:>8.3f} | {r['total_return']:>8.4f}", flush=True)

# Find best by Sharpe (with accuracy >= 0.55 and bets >= 90)
valid = [r for r in results if r["accuracy"] >= 0.55 and r["bets_per_month"] >= 90]
if valid:
    best = max(valid, key=lambda x: x["sharpe"])
    print(f"\nBest (acc>=55%, bets>=90/mo): MIN_CONFIDENCE={best['min_confidence']:.2f}, Sharpe={best['sharpe']:.3f}, Acc={best['accuracy']:.4f}", flush=True)
else:
    print("\nNo config meets acc>=55% + bets>=90/mo", flush=True)

# Save results
out_path = "context/goals/1/runs/step85_grid_search.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}", flush=True)
