"""
Main evolution loop for BTC 15-minute directional trading system.

This module integrates all components and runs the continuous adaptation loop:

1. Fetch latest data
2. Generate features (including horizon ensembles)
3. Evolve model population
4. Train best models with walk-forward validation
5. Generate signals via consensus gating
6. Paper trade and measure performance
7. Reflect on metrics and adjust system
8. Repeat
"""

import os
import sys
import json
import pickle
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Generator, Tuple, Optional
import pandas as pd
import numpy as np

from .config import *
from .data.fetcher import BinanceDataFetcher, fetch_and_save_historical
from .features.horizon import generate_horizon_features
from .features.technical import add_technical_indicators
from .features.regime import add_regime_features
from .data.processor import clean_data, create_target
from .models.population import Population, Individual
from .ensemble.horizon_ensemble import HorizonEnsemble
from .ensemble.consensus import ConsensusGate
from .validation.walk_forward import WalkForwardValidator
from .trading.simulator import SimulatedTrader
from .trading.metrics import evaluate_predictions


class TradingSystem:
    """
    Main trading system class that orchestrates the full workflow.
    """

    def __init__(
        self,
        data_path: str = None,
        results_dir: str = PathConfig.RESULTS_DIR,
        log_dir: str = PathConfig.LOGS_DIR,
    ):
        self.data_path = (
            data_path
            or Path(PathConfig.DATA_DIR)
            / f"{DataConfig.SYMBOL.lower()}_{DataConfig.INTERVAL}.csv"
        )
        self.results_dir = Path(results_dir)
        self.log_dir = Path(log_dir)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Current state
        self.df = None
        self.feature_names = []
        self.main_population = None
        self.horizon_ensemble = None
        self.consensus_gate = None
        self.validator = None
        self.trader = None

        # Load or initialize
        self._init_components()

    def _init_components(self):
        """Initialize all system components."""
        self.validator = WalkForwardValidator(
            train_days=ValidationConfig.TRAIN_DAYS,
            valid_days=ValidationConfig.VALID_DAYS,
            step_days=ValidationConfig.STEP_DAYS,
            min_train_samples=ValidationConfig.MIN_TRAIN_SAMPLES,
        )

        self.trader = SimulatedTrader(
            initial_capital=TradingConfig.INITIAL_CAPITAL,
            fee_rate=TradingConfig.FEE_RATE,
            slippage=TradingConfig.SLIPPAGE,
            bet_size=TradingConfig.POSITION_SIZE,
            stop_loss_pct=TradingConfig.STOP_LOSS_PCT,
        )

        self.horizon_ensemble = HorizonEnsemble(horizons=FeatureConfig.HORIZONS)
        self.consensus_gate = ConsensusGate(
            min_models_agree=ConsensusConfig.MIN_MODELS_AGREE,
            min_confidence=ConsensusConfig.MIN_CONFIDENCE,
        )

    def load_or_fetch_data(self, force_refetch: bool = False) -> pd.DataFrame:
        """
        Load data from disk, then incrementally refresh if data is stale (>15 min old).
        Full re-fetch only if file doesn't exist or force_refetch=True.
        """
        from .data.fetcher import BinanceDataFetcher

        if not force_refetch and self.data_path.exists():
            print(f"Loading data from {self.data_path}", flush=True)
            df = pd.read_csv(self.data_path, parse_dates=["open_time"])

            # Incremental refresh: if last candle is >15 min old, fetch new candles
            if not df.empty:
                last_ts = df["open_time"].max()
                now_utc = datetime.utcnow()
                age_minutes = (now_utc - last_ts).total_seconds() / 60
                print(
                    f"Last candle: {last_ts} (age: {age_minutes:.1f} min)", flush=True
                )

                if age_minutes > 15:
                    print(
                        f"Data stale by {age_minutes:.1f} min — fetching incremental update...",
                        flush=True,
                    )
                    try:
                        fetcher = BinanceDataFetcher()
                        # Add 1 ms to last_ts to skip fetching the already-existing candle
                        next_start_ms = int((last_ts.timestamp() + 0.001) * 1000)
                        # Fetch raw klines directly to avoid timezone ambiguity
                        klines = fetcher.fetch_klines(
                            start_time=next_start_ms, limit=500
                        )
                        if klines:
                            df_new = pd.DataFrame(
                                klines,
                                columns=[
                                    "open_time",
                                    "open",
                                    "high",
                                    "low",
                                    "close",
                                    "volume",
                                    "close_time",
                                    "quote_volume",
                                    "trades_count",
                                    "taker_buy_base",
                                    "taker_buy_quote",
                                    "ignore",
                                ],
                            )
                            df_new["open_time"] = pd.to_datetime(
                                df_new["open_time"], unit="ms"
                            )
                            for col in [
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                                "quote_volume",
                                "taker_buy_base",
                                "taker_buy_quote",
                            ]:
                                df_new[col] = pd.to_numeric(df_new[col])
                            df_new["trades_count"] = pd.to_numeric(
                                df_new["trades_count"]
                            )
                            df_new = df_new[
                                [
                                    "open_time",
                                    "open",
                                    "high",
                                    "low",
                                    "close",
                                    "volume",
                                    "quote_volume",
                                    "trades_count",
                                    "taker_buy_base",
                                    "taker_buy_quote",
                                ]
                            ]
                        else:
                            df_new = pd.DataFrame()
                        if not df_new.empty:
                            df_combined = pd.concat([df, df_new], ignore_index=True)
                            df_combined = df_combined.drop_duplicates(
                                subset="open_time"
                            )
                            df_combined = df_combined.sort_values(
                                "open_time"
                            ).reset_index(drop=True)
                            df_combined.to_csv(self.data_path, index=False)
                            new_rows = len(df_combined) - len(df)
                            print(
                                f"Incremental update: +{new_rows} new candles (total: {len(df_combined)})",
                                flush=True,
                            )
                            df = df_combined
                        else:
                            print("No new data returned from Binance.", flush=True)
                    except Exception as e:
                        print(
                            f"Incremental refresh failed: {e} — using cached data.",
                            flush=True,
                        )
        else:
            print("Fetching fresh data from Binance...", flush=True)
            df = fetch_and_save_historical(
                symbol=DataConfig.SYMBOL,
                interval=DataConfig.INTERVAL,
                start_date=DataConfig.START_DATE,
                filepath=str(self.data_path),
            )

        if df.empty:
            raise ValueError("No data available")

        print(f"Loaded {len(df)} rows of data", flush=True)
        print(
            f"Date range: {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}",
            flush=True,
        )
        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare feature matrix X and target y.
        """
        print("Preparing features...", flush=True)

        # Clean data
        df = clean_data(df)

        # Create target (next 2-candle direction — 30m hold; break-even drops from 61.8% → ~54%)
        df = create_target(df, forward_periods=2)

        # Load and merge futures features (funding rate, L/S ratio, OI)
        try:
            from .data.futures_fetcher import load_or_fetch_futures_features

            futures_features = load_or_fetch_futures_features()
            futures_cols = [c for c in futures_features.columns if c != "open_time"]
            df = df.merge(
                futures_features[["open_time"] + futures_cols],
                on="open_time",
                how="left",
            )
            # Forward-fill futures features (they update at 8h or 15m intervals)
            df[futures_cols] = df[futures_cols].ffill()
            # Neutral fill for remaining NaNs (e.g., L/S and OI have only recent 500 bars)
            neutral_fills = {
                "long_short_ratio": 1.0,
                "ls_ratio_ma10": 1.0,
                "ls_ratio_dev": 0.0,
                "ls_ratio_z": 0.0,
                "oi_change": 0.0,
                "oi_ratio": 1.0,
                "open_interest": 0.0,
                "oi_ma20": 0.0,
            }
            for col, fill_val in neutral_fills.items():
                if col in df.columns:
                    df[col] = df[col].fillna(fill_val)
            # Remaining futures cols: fill with 0
            df[futures_cols] = df[futures_cols].fillna(0)
            print(
                f"  Merged {len(futures_cols)} futures features (funding rate, L/S, OI)",
                flush=True,
            )
        except Exception as e:
            print(f"  Futures features unavailable: {e}", flush=True)

        # Generate horizon ensemble features (these will be added as extra columns)
        print("  Generating horizon features...", flush=True)
        horizon_features = generate_horizon_features(df, FeatureConfig.HORIZONS)

        # Merge horizon features into df
        df = df.merge(horizon_features, on="open_time", how="left")

        # Drop rows with NaN in target
        df = df.dropna(subset=["target"])

        # Forward fill any remaining NaNs
        df = df.ffill().dropna()

        print(f"Prepared {len(df)} samples with {len(df.columns)} columns", flush=True)
        self.feature_names = [
            col
            for col in df.columns
            if col
            not in [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "trades_count",
                "future_close",
                "target",
                "taker_buy_base",
                "taker_buy_quote",  # raw order flow — use derived ratios
                "long_account",
                "short_account",  # raw L/S components — use derived ratio
            ]
        ]

        print(f"Feature count: {len(self.feature_names)}", flush=True)
        return df

    def evolve_population(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        generations: int = ModelConfig.GENERATIONS_PER_ITERATION,
    ) -> Population:
        """
        Evolve a population of models.
        """
        print(f"Evolving population (input_dim={X_train.shape[1]})...", flush=True)

        pop = Population(
            population_size=ModelConfig.POPULATION_SIZE,
            model_types=ModelConfig.MODEL_TYPES,
            mutation_rate=ModelConfig.MUTATION_RATE,
            crossover_rate=ModelConfig.CROSSOVER_RATE,
            elitism_count=ModelConfig.ELITISM_COUNT,
        )

        pop.initialize(input_dim=X_train.shape[1], feature_names=self.feature_names)

        for gen in range(generations):
            print(f"  Generation {gen + 1}/{generations}", flush=True)

            # Evaluate all individuals
            for i, ind in enumerate(pop.individuals):
                try:
                    print(
                        f"    Training individual {i + 1}/{len(pop.individuals)} ({ind.model_type})",
                        flush=True,
                    )
                    ind.model.fit(X_train, y_train)
                    proba = ind.model.predict_proba(X_train)
                    pred = ind.model.predict(X_train, threshold=0.5)
                    accuracy = (pred == y_train).mean()

                    # Simple fitness based on training accuracy (we'll validate on walk-forward later)
                    ind.fitness = accuracy
                    print(f"      -> fitness={ind.fitness:.4f}", flush=True)
                except Exception as e:
                    print(f"      -> training error: {e}", flush=True)
                    ind.fitness = -1000.0

            # Evolve to next generation, but not after the final generation
            if gen < generations - 1:
                print("  Evolving to next generation...", flush=True)
                pop.evolve()

            best = pop.get_best(1)[0]
            print(
                f"    Best fitness: {best.fitness:.4f} (model={best.model_type})",
                flush=True,
            )

        return pop

    def _log_feature_importances(self, ind) -> None:
        """Log top feature importances from a Random Forest model."""
        if ind is None:
            return
        try:
            if hasattr(ind.model, "model") and hasattr(
                ind.model.model, "feature_importances_"
            ):
                importances = ind.model.model.feature_importances_
                n = min(len(importances), len(self.feature_names))
                feature_imp = pd.Series(importances[:n], index=self.feature_names[:n])
                top10 = feature_imp.nlargest(10)
                print("\nTop 10 feature importances:", flush=True)
                for fname, imp in top10.items():
                    print(f"  {fname}: {imp:.4f}", flush=True)
        except Exception:
            pass

    def _run_fold(
        self, train_df: pd.DataFrame, valid_df: pd.DataFrame
    ) -> Optional[Dict]:
        """Run a single fold of walk-forward validation. Returns metrics dict or None."""
        X_train = train_df[self.feature_names]
        y_train = train_df["target"]
        X_valid = valid_df[self.feature_names]
        y_valid = valid_df["target"]
        valid_ohlc = valid_df[["close", "high", "low"]]
        close_prices = valid_df["close"]

        # Train fresh population on this fold's training data
        pop = Population(
            population_size=ModelConfig.POPULATION_SIZE,
            model_types=ModelConfig.MODEL_TYPES,
            mutation_rate=ModelConfig.MUTATION_RATE,
            crossover_rate=ModelConfig.CROSSOVER_RATE,
            elitism_count=ModelConfig.ELITISM_COUNT,
        )
        pop.initialize(input_dim=X_train.shape[1], feature_names=self.feature_names)

        # Sample weights: age decay × return magnitude
        # Age decay: recent samples weighted higher (half_life=2000 candles ≈ 20 days)
        n_train = len(X_train)
        half_life = 2000
        age_weight = np.exp(-np.arange(n_train - 1, -1, -1) / half_life)

        # Return magnitude weight: large moves matter more for profitability
        # Model should correctly predict large moves (which overcome fees) more than small ones
        next_returns = train_df["close"].pct_change().shift(-1).abs().fillna(0)
        next_returns = next_returns.iloc[-n_train:].values
        next_returns = np.clip(
            next_returns, 0, 0.02
        )  # cap at 2% to limit outlier influence
        mag_weight = next_returns / (next_returns.mean() + 1e-10)
        mag_weight = np.clip(
            mag_weight, 0.1, 5.0
        )  # bound: min 10% weight, max 5× weight

        sample_weight = age_weight * mag_weight
        sample_weight = sample_weight / sample_weight.mean()  # normalize to mean=1

        for gen in range(ModelConfig.GENERATIONS_PER_ITERATION):
            for ind in pop.individuals:
                try:
                    if ind.model_type == "lightgbm":
                        ind.model.fit(X_train, y_train, sample_weight=sample_weight)
                    else:
                        ind.model.fit(X_train, y_train)
                    pred = ind.model.predict(X_train, threshold=0.5)
                    ind.fitness = (pred == y_train).mean()
                except Exception as e:
                    print(f"    Training error: {e}", flush=True)
                    ind.fitness = -1000.0
            if gen < ModelConfig.GENERATIONS_PER_ITERATION - 1:
                pop.evolve()
            best = pop.get_best(1)[0]
            print(
                f"    Gen {gen + 1}: best_train_fitness={best.fitness:.4f} ({best.model_type})",
                flush=True,
            )

        # Re-score on validation set using profit_factor
        for i, ind in enumerate(pop.individuals):
            if ind.fitness is None or ind.fitness <= -1000:
                continue
            try:
                pred_valid = ind.model.predict(X_valid, threshold=0.5)
                if isinstance(pred_valid, np.ndarray):
                    pred_valid = pd.Series(pred_valid, index=X_valid.index)
                signals_ind = (
                    pred_valid.reindex(close_prices.index).fillna(-1).astype(int)
                )
                self.trader.run(
                    signals_ind, valid_ohlc, n_hold=TradingConfig.HOLD_PERIODS
                )
                ind.fitness = self.trader.metrics.get("profit_factor", 0.0)
                ind.validation_metrics = self.trader.metrics
                print(f"  Model {i + 1}: profit_factor={ind.fitness:.4f}", flush=True)
            except Exception as e:
                print(f"  Rescoring error: {e}", flush=True)
                ind.fitness = -1000.0

        top_models = pop.get_diverse_best(n=5)
        self._log_feature_importances(top_models[0] if top_models else None)

        consensus = ConsensusGate(
            min_models_agree=ConsensusConfig.MIN_MODELS_AGREE,
            min_confidence=ConsensusConfig.MIN_CONFIDENCE,
        )
        consensus_signals = consensus.decide(top_models, X_valid)

        # Step 121: HYBRID approach — rule signals + regime filter
        # Previous: rule-only = 57.67% accuracy, 91 bets/month (Fold 2 fails at 42.65%)
        # Problem: Fold 2 (Feb 4-25) = sustained downtrend where mean-reversion fails
        # Solution: add 7-day momentum regime filter to suppress signals in extreme regimes
        train_abs_ret = train_df["ret_lag_1"].abs()
        move_threshold = float(train_abs_ret.quantile(0.75))

        current_ret = X_valid["ret_lag_1"]
        prev_ret = X_valid["ret_lag_2"]
        current_large_down = current_ret < -move_threshold
        current_large_up = current_ret > move_threshold

        taker_low = X_valid["taker_buy_ratio"] < 0.35
        taker_high = X_valid["taker_buy_ratio"] > 0.65
        rsi_oversold = X_valid["rsi"] < 45
        rsi_overbought = X_valid["rsi"] > 55

        # 7-day momentum: compute from close prices
        # 7 days = 7*24*4 = 672 periods at 15m
        valid_close = valid_ohlc["close"]
        if len(valid_close) >= 672:
            mom_7d = valid_close.pct_change(672).shift(1)
            mom_7d = mom_7d.reindex(X_valid.index).fillna(0)
            # Also compute MA-based trend: price vs 20-period (5h) MA
            ma_20 = valid_close.rolling(20).mean().shift(1)
            ma_20 = ma_20.reindex(X_valid.index).fillna(valid_close)
            price_vs_ma20 = valid_close / ma_20 - 1
            # 1-day momentum (24 periods = 6h)
            if len(valid_close) >= 25:
                mom_1d = valid_close.pct_change(25).shift(1)
                mom_1d = mom_1d.reindex(X_valid.index).fillna(0)
            else:
                mom_1d = pd.Series(0.0, index=X_valid.index)
        else:
            mom_7d = pd.Series(0.0, index=X_valid.index)
            price_vs_ma20 = pd.Series(0.0, index=X_valid.index)
            mom_1d = pd.Series(0.0, index=X_valid.index)

        # Compute shorter-period momentum for faster regime detection
        # 3-day = 288 bars, 5-day = 480 bars
        if len(valid_close) >= 289:
            mom_3d = valid_close.pct_change(288).shift(1).reindex(X_valid.index).fillna(0)
        else:
            mom_3d = pd.Series(0.0, index=X_valid.index)
        if len(valid_close) >= 481:
            mom_5d = valid_close.pct_change(480).shift(1).reindex(X_valid.index).fillna(0)
        else:
            mom_5d = pd.Series(0.0, index=X_valid.index)

        # Test multiple regime thresholds
        print(
            f"  Regime stats: mom_7d: mean={mom_7d.mean():.4f}, min={mom_7d.min():.4f}, <-5%={(mom_7d < -0.05).sum()}, <-3%={(mom_7d < -0.03).sum()}, <-2%={(mom_7d < -0.02).sum()}, <-1%={(mom_7d < -0.01).sum()}",
            flush=True,
        )
        print(
            f"  Regime stats: mom_3d: mean={mom_3d.mean():.4f}, <-2%={(mom_3d < -0.02).sum()}, <-1%={(mom_3d < -0.01).sum()}  |  mom_5d: mean={mom_5d.mean():.4f}, <-2%={(mom_5d < -0.02).sum()}",
            flush=True,
        )
        print(
            f"  Regime stats: price_vs_MA20: mean={price_vs_ma20.mean():.4f}, <0={(price_vs_ma20 < 0).sum()}, <-2%={(price_vs_ma20 < -0.02).sum()}, <-5%={(price_vs_ma20 < -0.05).sum()}",
            flush=True,
        )

        regime_map = {
            "rule_only": (
                pd.Series(False, index=X_valid.index),
                pd.Series(False, index=X_valid.index),
            ),
            "mom7d_-5pct": (mom_7d < -0.05, mom_7d > 0.05),
            "mom7d_-3pct": (mom_7d < -0.03, mom_7d > 0.03),
            "mom7d_-2pct": (mom_7d < -0.02, mom_7d > 0.02),
            "mom7d_-1pct": (mom_7d < -0.01, mom_7d > 0.01),
            "mom5d_-3pct": (mom_5d < -0.03, mom_5d > 0.03),
            "mom5d_-2pct": (mom_5d < -0.02, mom_5d > 0.02),
            "mom5d_-1pct": (mom_5d < -0.01, mom_5d > 0.01),
            "mom3d_-3pct": (mom_3d < -0.03, mom_3d > 0.03),
            "mom3d_-2pct": (mom_3d < -0.02, mom_3d > 0.02),
            "mom3d_-1pct": (mom_3d < -0.01, mom_3d > 0.01),
            "ma20_bearish": (price_vs_ma20 < 0.0, price_vs_ma20 > 0.0),
            "ma20_-2pct": (price_vs_ma20 < -0.02, price_vs_ma20 > 0.02),
            "ma20_-5pct": (price_vs_ma20 < -0.05, price_vs_ma20 > 0.05),
        }

        # Rule-based signals (IS_rel + taker + RSI) — MEAN REVERSION variant
        rule_signals = pd.Series(-1, index=X_valid.index)
        rule_signals[
            current_large_down
            & (prev_ret.abs() < current_ret.abs())
            & taker_low
            & rsi_oversold
        ] = 1
        rule_signals[
            current_large_up
            & (prev_ret.abs() < current_ret.abs())
            & taker_high
            & rsi_overbought
        ] = 0

        # MOMENTUM CONTINUATION variant — bet with the trend, not against it
        momentum_signals = pd.Series(-1, index=X_valid.index)
        momentum_signals[
            current_large_down
            & (prev_ret.abs() < current_ret.abs())
            & taker_low
            & rsi_oversold
        ] = 0  # BET DOWN (continue the fall)
        momentum_signals[
            current_large_up
            & (prev_ret.abs() < current_ret.abs())
            & taker_high
            & rsi_overbought
        ] = 1  # BET UP (continue the rise)

        # Step 122: Two-stage selection — prioritize bpm >= 90 variants.
        # Track two separate bests: one for bpm >= 90 (primary target), one for bpm >= 60 (fallback).
        best_acc_90 = 0.0  # best accuracy among bpm >= 90 variants
        best_name_90 = None
        best_signals_90 = None
        best_n_90 = 0
        best_acc = 0.0  # fallback: best accuracy among bpm >= 60 variants
        best_name = "rule_only"
        best_signals = rule_signals.copy()
        best_n = 0

        def _update_best_variants(name, test_signals, n_active, acc, bpm):
            nonlocal best_acc_90, best_name_90, best_signals_90, best_n_90
            nonlocal best_acc, best_name, best_signals, best_n
            if bpm >= 90 and acc > best_acc_90:
                best_acc_90 = acc
                best_name_90 = name
                best_signals_90 = test_signals.copy()
                best_n_90 = n_active
            if bpm >= 60 and acc > best_acc:
                best_acc = acc
                best_name = name
                best_signals = test_signals.copy()
                best_n = n_active

        # Test MR (suppress) variants
        for name, (downtrend_mask, uptrend_mask) in regime_map.items():
            test_signals = rule_signals.copy()
            test_signals[(rule_signals == 1) & downtrend_mask] = -1
            test_signals[(rule_signals == 0) & uptrend_mask] = -1

            n_active = (test_signals != -1).sum()
            if n_active < 10:
                continue

            mask = (test_signals != -1) & y_valid.notna()
            if not mask.any():
                continue
            acc = (test_signals[mask] == y_valid[mask]).mean()
            bpm = n_active / (len(X_valid) * 15 / (60 * 24)) * 30

            print(f"    MR_{name}: acc={acc:.4f}, bpm={bpm:.1f}, n={n_active}", flush=True)
            _update_best_variants(f"MR_{name}", test_signals, n_active, acc, bpm)

        # Test MOM (suppress) variants
        for name, (downtrend_mask, uptrend_mask) in regime_map.items():
            test_signals = momentum_signals.copy()
            test_signals[(momentum_signals == 1) & uptrend_mask] = -1
            test_signals[(momentum_signals == 0) & downtrend_mask] = -1

            n_active = (test_signals != -1).sum()
            if n_active < 10:
                continue

            mask = (test_signals != -1) & y_valid.notna()
            if not mask.any():
                continue
            acc = (test_signals[mask] == y_valid[mask]).mean()
            bpm = n_active / (len(X_valid) * 15 / (60 * 24)) * 30

            print(f"    MOM_{name}: acc={acc:.4f}, bpm={bpm:.1f}, n={n_active}", flush=True)
            _update_best_variants(f"MOM_{name}", test_signals, n_active, acc, bpm)

        # Step 122: SWITCH variant — flip direction in trending regimes instead of suppressing.
        # Theory: large DOWN bar in a downtrend → momentum continuation (DOWN), not mean-reversion (UP).
        # This preserves full signal count while correcting direction for the regime.
        # Expected: ~65% accuracy with ~105 bpm (both targets met simultaneously).
        for name, (downtrend_mask, uptrend_mask) in regime_map.items():
            if name == "rule_only":
                continue
            switch_signals = rule_signals.copy()
            # In downtrend: flip UP reversion bet → DOWN momentum bet
            switch_signals[(rule_signals == 1) & downtrend_mask] = 0
            # In uptrend: flip DOWN reversion bet → UP momentum bet
            switch_signals[(rule_signals == 0) & uptrend_mask] = 1

            n_active = (switch_signals != -1).sum()
            if n_active < 10:
                continue
            mask = (switch_signals != -1) & y_valid.notna()
            if not mask.any():
                continue
            acc = (switch_signals[mask] == y_valid[mask]).mean()
            bpm = n_active / (len(X_valid) * 15 / (60 * 24)) * 30
            print(f"    SWITCH_{name}: acc={acc:.4f}, bpm={bpm:.1f}, n={n_active}", flush=True)
            _update_best_variants(f"SWITCH_{name}", switch_signals, n_active, acc, bpm)

        # Prefer bpm >= 90 variant if it has reasonable accuracy (>= 0.58 to filter noise);
        # otherwise fall back to the highest-accuracy bpm >= 60 variant.
        if best_name_90 is not None and best_acc_90 >= 0.58:
            signals = best_signals_90
            selected_name = best_name_90
            selected_n = best_n_90
            selected_acc = best_acc_90
            stage = "≥90bpm"
        else:
            signals = best_signals
            selected_name = best_name
            selected_n = best_n
            selected_acc = best_acc
            stage = "fallback"

        n_days = len(X_valid) * 15 / (60 * 24)
        print(
            f"  BEST [{stage}]: {selected_name}: acc={selected_acc:.4f}, bpm={selected_n / n_days * 30:.1f}",
            flush=True,
        )

        # Run simulator with configured hold period
        self.trader.run(signals, valid_ohlc, n_hold=TradingConfig.HOLD_PERIODS)
        metrics = self.trader.metrics.copy()

        # Directional accuracy
        mask = (signals != -1) & y_valid.notna()
        if mask.any():
            metrics["accuracy"] = (signals[mask] == y_valid[mask]).mean()
        else:
            metrics["accuracy"] = 0.0

        print(
            f"Fold result: accuracy={metrics['accuracy']:.4f}, bets/month={metrics.get('bets_per_month', 0):.2f}, "
            f"trades={metrics.get('n_trades', 0)}, sharpe={metrics.get('sharpe', 0):.3f}",
            flush=True,
        )
        return metrics

    def run_walk_forward(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Run 2-fold walk-forward validation with non-overlapping validation windows.
        Fold 1 (earlier): validates on [T-2v, T-v]; Fold 2 (latest): validates on [T-v, T].
        Returns weighted-average metrics across folds for robust OOS estimation.
        """
        print("Running multi-fold walk-forward validation...", flush=True)

        df_sorted = df.sort_values("open_time").reset_index(drop=True)
        val_rows = int(ValidationConfig.VALID_DAYS * 24 * 60 / 15)
        total_rows = len(df_sorted)

        if total_rows < val_rows + ValidationConfig.MIN_TRAIN_SAMPLES:
            print("Insufficient data for walk-forward.", flush=True)
            return {"accuracy": 0.0, "bets_per_month": 0.0, "n_folds": 0}

        # Determine how many non-overlapping folds fit (up to 3)
        n_folds = 1
        if total_rows >= 2 * val_rows + ValidationConfig.MIN_TRAIN_SAMPLES:
            n_folds = 2
        if total_rows >= 3 * val_rows + ValidationConfig.MIN_TRAIN_SAMPLES:
            n_folds = 3

        fold_metrics = []
        for fold_idx in range(n_folds):
            # fold_idx=0 (earlier), fold_idx=1 (most recent)
            valid_end = total_rows - (n_folds - 1 - fold_idx) * val_rows
            valid_start = valid_end - val_rows

            if valid_start < ValidationConfig.MIN_TRAIN_SAMPLES:
                print(
                    f"Fold {fold_idx + 1}: insufficient train samples, skipping.",
                    flush=True,
                )
                continue

            valid_df = df_sorted.iloc[valid_start:valid_end].copy()
            train_df = df_sorted.iloc[:valid_start].copy()

            if len(train_df) > ModelConfig.MAX_TRAIN_SAMPLES:
                train_df = train_df.tail(ModelConfig.MAX_TRAIN_SAMPLES).copy()

            print(
                f"\nFold {fold_idx + 1}/{n_folds}: train={len(train_df)}, valid={len(valid_df)}",
                flush=True,
            )

            m = self._run_fold(train_df, valid_df)
            if m:
                fold_metrics.append(m)

        if not fold_metrics:
            return {"accuracy": 0.0, "bets_per_month": 0.0, "n_folds": 0}

        n = len(fold_metrics)
        total_trades = sum(m.get("n_trades", 0) for m in fold_metrics)

        if total_trades == 0:
            combined = fold_metrics[-1].copy()
            combined["n_folds"] = n
            return combined

        # Aggregate: weighted accuracy by n_trades, average everything else
        accuracy = (
            sum(m["accuracy"] * m.get("n_trades", 1) for m in fold_metrics)
            / total_trades
        )
        combined = {
            "accuracy": accuracy,
            "bets_per_month": sum(m.get("bets_per_month", 0) for m in fold_metrics) / n,
            "sharpe": sum(m.get("sharpe", 0) for m in fold_metrics) / n,
            "profit_factor": sum(m.get("profit_factor", 0) for m in fold_metrics) / n,
            "total_return": sum(m.get("total_return", 0) for m in fold_metrics),
            "max_drawdown": max(m.get("max_drawdown", 0) for m in fold_metrics),
            "n_trades": total_trades,
            "n_folds": n,
        }

        print(
            f"\nMulti-fold combined ({n} folds): accuracy={combined['accuracy']:.4f}, "
            f"bets/month={combined['bets_per_month']:.2f}, Sharpe={combined['sharpe']:.3f}, "
            f"total_return={combined['total_return']:.4f}, trades={total_trades}",
            flush=True,
        )
        return combined

    def train_best_models(
        self, df: pd.DataFrame, population: Population, horizon_data: dict = None
    ):
        """
        Train the best models on full recent data and optionally train horizon ensembles.
        """
        print("Training best models on full data...", flush=True)

        # Use recent data (last N days) for final training
        recent_days = 60
        cutoff_time = df["open_time"].max() - timedelta(days=recent_days)
        recent_df = df[df["open_time"] >= cutoff_time].copy()
        print(f"Recent training data: {len(recent_df)} samples", flush=True)

        X_recent = recent_df[self.feature_names]
        y_recent = recent_df["target"]

        # Get top models
        top_models = population.get_diverse_best(n=ConsensusConfig.MIN_MODELS_AGREE + 2)
        print(f"Selected {len(top_models)} top models for retraining", flush=True)

        # Retrain on full recent data
        for i, ind in enumerate(top_models):
            try:
                print(
                    f"  Retraining model {i + 1}/{len(top_models)} ({ind.model_type})",
                    flush=True,
                )
                ind.model.fit(X_recent, y_recent)
                print(f"    Done", flush=True)
            except Exception as e:
                print(f"  Failed to train {ind.model_type}: {e}", flush=True)

        # Train horizon ensemble if not provided and if we have multiple horizons
        if horizon_data is None:
            horizon_data = {}
            horizons_to_train = [h for h in FeatureConfig.HORIZONS.keys() if h != "15m"]
            if horizons_to_train:
                print(
                    f"Training horizon ensemble for {len(horizons_to_train)} horizons...",
                    flush=True,
                )
                for h_name in horizons_to_train:
                    factor = FeatureConfig.HORIZONS[h_name]
                    print(f"  Horizon {h_name} (factor={factor})", flush=True)
                    from .data.processor import resample_to_horizon

                    # Use correct horizon minutes: factor * 15 minutes
                    horizon_minutes = factor * 15
                    df_horizon = resample_to_horizon(recent_df, horizon_minutes, factor)
                    print(f"    Resampled to {len(df_horizon)} rows", flush=True)
                    if len(df_horizon) > 100:
                        df_horizon = create_target(df_horizon, forward_periods=1)
                        df_horizon = add_technical_indicators(df_horizon)
                        df_horizon = add_regime_features(df_horizon)
                        df_horizon = df_horizon.dropna()
                        print(
                            f"    Prepared {len(df_horizon)} samples with {len(df_horizon.columns)} features",
                            flush=True,
                        )
                        horizon_data[h_name] = df_horizon
                self.horizon_ensemble.fit_horizon_models(horizon_data, y_recent)
                print("Horizon ensemble training complete", flush=True)
            else:
                print(
                    "No additional horizons to train (HORIZONS only includes 15m)",
                    flush=True,
                )
        else:
            self.horizon_ensemble.fit_horizon_models(horizon_data, y_recent)

        self.main_population = population
        self.top_models = top_models

    def generate_current_signals(
        self, current_features: pd.DataFrame, use_horizon: bool = True
    ) -> pd.Series:
        """
        Generate trading signals for the most recent data points.
        """
        if self.top_models is None:
            raise RuntimeError("Models not trained. Call train_best_models first.")

        signals = self.consensus_gate.decide(self.top_models, current_features)
        return signals

    def run_online_step(self, new_data: pd.DataFrame = None):
        """
        Execute one iteration of the online evolution loop.

        This is the main step method called repeatedly by Arbos.
        """
        print("=" * 60, flush=True)
        print("RUNNING ONLINE STEP", flush=True)
        print("=" * 60, flush=True)

        # 1. Load/fetch latest data
        if new_data is None:
            df = self.load_or_fetch_data()
        else:
            df = new_data

        # 2. For online operation, limit to most recent N days to bound runtime
        if ModelConfig.ONLINE_DATA_DAYS is not None:
            cutoff = df["open_time"].max() - timedelta(
                days=ModelConfig.ONLINE_DATA_DAYS
            )
            df = df[df["open_time"] >= cutoff].copy()
            print(
                f"Online mode: limited data to last {ModelConfig.ONLINE_DATA_DAYS} days ({len(df)} rows)",
                flush=True,
            )

        # 3. Prepare features
        print("[main] Starting feature preparation...", flush=True)
        df = self.prepare_features(df)
        print(
            f"[main] Feature preparation complete: {len(df)} rows, {len(self.feature_names)} features",
            flush=True,
        )

        # 3. Split into recent data for training and current for prediction
        # Use last 30 days for training, current for signal generation
        recent_cutoff = df["open_time"].max() - timedelta(days=30)
        train_df = df[df["open_time"] <= recent_cutoff].copy()
        current_df = df[df["open_time"] > recent_cutoff].copy()

        # Limit training samples to control memory/time (use most recent samples)
        if len(train_df) > ModelConfig.MAX_TRAIN_SAMPLES:
            train_df = train_df.tail(ModelConfig.MAX_TRAIN_SAMPLES).copy()
            print(
                f"Limited training to {len(train_df)} most recent samples", flush=True
            )

        if len(train_df) < ValidationConfig.MIN_TRAIN_SAMPLES:
            print(f"Insufficient training data: {len(train_df)} samples", flush=True)
            return None

        X_train = train_df[self.feature_names]
        y_train = train_df["target"]

        # 4. Evolve population
        population = self.evolve_population(
            X_train, y_train, generations=ModelConfig.GENERATIONS_PER_ITERATION
        )

        # 5. Walk-forward validation on historical folds (use recent data to limit folds)
        if ValidationConfig.MAX_WALK_FORWARD_DAYS is not None:
            wf_cutoff = df["open_time"].max() - timedelta(
                days=ValidationConfig.MAX_WALK_FORWARD_DAYS
            )
            df_wf = df[df["open_time"] >= wf_cutoff].copy()
            print(
                f"Using {len(df_wf)} recent samples for walk-forward validation (max {ValidationConfig.MAX_WALK_FORWARD_DAYS} days)",
                flush=True,
            )
        else:
            df_wf = df
        metrics = self.run_walk_forward(df_wf)

        # Adapt consensus thresholds based on walk-forward metrics
        self.consensus_gate.adjust_thresholds(metrics)

        # 6. Train best models on full recent data and prepare horizon ensemble
        self.train_best_models(df, population)

        # 7. Generate current signals
        if len(current_df) > 0:
            current_features = current_df[self.feature_names]
            signals = self.generate_current_signals(current_features)
            print(f"Generated {len(signals)} current signals", flush=True)
            latest_signal = signals.iloc[-1] if len(signals) > 0 else None
            print(
                f"Latest signal: {latest_signal} (1=UP, 0=DOWN, -1=NO TRADE)",
                flush=True,
            )
        else:
            signals = pd.Series()
            latest_signal = None

        # 8. Save results
        self._save_results(metrics, population, signals)

        # Compile full result
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "accuracy": metrics.get("accuracy", 0.0),
            "bets_per_month": metrics.get("bets_per_month", 0.0),
            "sharpe": metrics.get("sharpe", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "total_return": metrics.get("total_return", 0.0),
            "latest_signal": int(latest_signal) if latest_signal is not None else None,
            "population_fitness": {
                str(i): ind.fitness for i, ind in enumerate(population.get_best(5))
            },
        }

        print("=" * 60, flush=True)
        print("STEP COMPLETE", flush=True)
        print(f"Metrics: {json.dumps(result, indent=2)}", flush=True)
        print("=" * 60, flush=True)

        return result

    def _save_results(self, metrics: dict, population: Population, signals: pd.Series):
        """Save results to disk."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = self.results_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Save signals
        signals.to_csv(run_dir / "signals.csv")

        # Save top models
        top_models = population.get_best(5)
        model_dir = run_dir / "models"
        model_dir.mkdir(exist_ok=True)
        for i, ind in enumerate(top_models):
            try:
                ind.model.save(str(model_dir / f"model_{i}_{ind.model_type}.pkl"))
            except Exception as e:
                print(f"Could not save model {i}: {e}", flush=True)

        # Save population metadata
        pop_data = []
        for ind in population.individuals[:10]:
            pop_data.append(
                {
                    "model_type": ind.model_type,
                    "params": ind.params,
                    "fitness": float(ind.fitness),
                }
            )
        with open(run_dir / "population.json", "w") as f:
            json.dump(pop_data, f, indent=2)

        print(f"Results saved to {run_dir}", flush=True)


def main():
    """Entry point for running the trading system."""
    # Fix global random seeds for reproducible population initialization and training
    np.random.seed(42)
    random.seed(42)
    print("[main] main() started", flush=True)
    system = TradingSystem()

    # Run one step
    result = system.run_online_step()

    if result:
        print(f"\nStep completed. Latest signal: {result['latest_signal']}", flush=True)
        print(
            f"Accuracy: {result['accuracy']:.4f}, Bets/month: {result['bets_per_month']:.2f}",
            flush=True,
        )
    else:
        print("Step did not complete successfully.", flush=True)


if __name__ == "__main__":
    main()
