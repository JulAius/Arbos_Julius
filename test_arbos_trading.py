#!/usr/bin/env python3
"""
Test script for the BTC 15-minute directional trading system.

This script performs:
1. Data fetch from Binance
2. Feature engineering
3. Quick evolutionary run with a small population
4. Walk-forward validation
5. Report key metrics

Run this to verify the system works before deploying.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from arbos_trading.data.fetcher import fetch_and_save_historical
from arbos_trading.main import TradingSystem


def main():
    print("BTC 15-minute Trading System Test")
    print("=" * 60)

    # Step 1: Fetch recent data (last 30 days to test)
    print("\nStep 1: Fetching recent BTC data (last 30 days)...")
    end_date = datetime.utcnow()
    start_date = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")

    # Override for test: only fetch recent 30 days
    from arbos_trading.config import DataConfig
    original_start = DataConfig.START_DATE
    DataConfig.START_DATE = start_date

    try:
        df = fetch_and_save_historical(
            symbol=DataConfig.SYMBOL,
            interval=DataConfig.INTERVAL,
            start_date=start_date,
            end_date=None,
            filepath=None
        )
        print(f"Fetched {len(df)} rows")
        print(f"Date range: {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    if len(df) == 0:
        print("No data fetched. Exiting.")
        return

    # Step 2: Initialize system and run one evolution step
    print("\nStep 2: Running trading system step...")
    system = TradingSystem()
    system.df = df  # Use fetched data directly

    try:
        result = system.run_online_step()

        if result:
            print("\n" + "=" * 60)
            print("TEST RESULTS")
            print("=" * 60)
            print(json.dumps(result, indent=2))

            # Check against targets
            acc_ok = result["accuracy"] >= 0.65
            bets_ok = result["bets_per_month"] >= 90

            print("\nTARGET ASSESSMENT:")
            print(f"  Directional accuracy: {result['accuracy']:.2%} {'✓' if acc_ok else '✗'} (target: 65%)")
            print(f"  Bets per month: {result['bets_per_month']:.1f} {'✓' if bets_ok else '✗'} (target: >90)")

            if acc_ok and bets_ok:
                print("\n✓ SYSTEM MEETS TARGETS!")
            else:
                print("\n✗ System needs improvement.")
        else:
            print("No result returned from step.")

    except Exception as e:
        print(f"Error during system run: {e}")
        import traceback
        traceback.print_exc()

    # Restore config
    DataConfig.START_DATE = original_start


if __name__ == "__main__":
    from datetime import datetime, timedelta
    import json
    main()
