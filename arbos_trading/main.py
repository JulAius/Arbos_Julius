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
        log_dir: str = PathConfig.LOGS_DIR
    ):
        self.data_path = data_path or Path(PathConfig.DATA_DIR) / f"{DataConfig.SYMBOL.lower()}_{DataConfig.INTERVAL}.csv"
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
            min_train_samples=ValidationConfig.MIN_TRAIN_SAMPLES
        )

        self.trader = SimulatedTrader(
            initial_capital=TradingConfig.INITIAL_CAPITAL,
            fee_rate=TradingConfig.FEE_RATE,
            slippage=TradingConfig.SLIPPAGE,
            bet_size=TradingConfig.POSITION_SIZE,
            stop_loss_pct=TradingConfig.STOP_LOSS_PCT
        )

        self.horizon_ensemble = HorizonEnsemble(horizons=FeatureConfig.HORIZONS)
        self.consensus_gate = ConsensusGate(
            min_models_agree=ConsensusConfig.MIN_MODELS_AGREE,
            min_confidence=ConsensusConfig.MIN_CONFIDENCE
        )

    def load_or_fetch_data(self, force_refetch: bool = False) -> pd.DataFrame:
        """
        Load data from disk or fetch from Binance if not available/forced.
        """
        if not force_refetch and self.data_path.exists():
            print(f"Loading data from {self.data_path}", flush=True)
            df = pd.read_csv(self.data_path, parse_dates=["open_time"])
        else:
            print("Fetching fresh data from Binance...", flush=True)
            df = fetch_and_save_historical(
                symbol=DataConfig.SYMBOL,
                interval=DataConfig.INTERVAL,
                start_date=DataConfig.START_DATE,
                filepath=str(self.data_path)
            )

        if df.empty:
            raise ValueError("No data available")

        print(f"Loaded {len(df)} rows of data", flush=True)
        print(f"Date range: {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}", flush=True)
        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare feature matrix X and target y.
        """
        print("Preparing features...", flush=True)

        # Clean data
        df = clean_data(df)

        # Create target (next 15-minute direction)
        df = create_target(df, forward_periods=1)

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
            col for col in df.columns
            if col not in [
                "open_time", "open", "high", "low", "close", "volume",
                "quote_volume", "trades_count", "future_close", "target"
            ]
        ]

        print(f"Feature count: {len(self.feature_names)}", flush=True)
        return df

    def evolve_population(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        generations: int = ModelConfig.GENERATIONS_PER_ITERATION
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
            elitism_count=ModelConfig.ELITISM_COUNT
        )

        pop.initialize(input_dim=X_train.shape[1], feature_names=self.feature_names)

        for gen in range(generations):
            print(f"  Generation {gen + 1}/{generations}", flush=True)

            # Evaluate all individuals
            for i, ind in enumerate(pop.individuals):
                try:
                    print(f"    Training individual {i+1}/{len(pop.individuals)} ({ind.model_type})", flush=True)
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
            print(f"    Best fitness: {best.fitness:.4f} (model={best.model_type})", flush=True)

        return pop

    def _log_feature_importances(self, ind) -> None:
        """Log top feature importances from a Random Forest model."""
        if ind is None:
            return
        try:
            if hasattr(ind.model, 'model') and hasattr(ind.model.model, 'feature_importances_'):
                importances = ind.model.model.feature_importances_
                n = min(len(importances), len(self.feature_names))
                feature_imp = pd.Series(importances[:n], index=self.feature_names[:n])
                top10 = feature_imp.nlargest(10)
                print("\nTop 10 feature importances:", flush=True)
                for fname, imp in top10.items():
                    print(f"  {fname}: {imp:.4f}", flush=True)
        except Exception:
            pass

    def _run_fold(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> Optional[Dict]:
        """Run a single fold of walk-forward validation. Returns metrics dict or None."""
        X_train = train_df[self.feature_names]
        y_train = train_df["target"]
        X_valid = valid_df[self.feature_names]
        y_valid = valid_df["target"]
        valid_ohlc = valid_df[['close', 'high', 'low']]
        close_prices = valid_df["close"]

        # Train fresh population on this fold's training data
        pop = Population(
            population_size=ModelConfig.POPULATION_SIZE,
            model_types=ModelConfig.MODEL_TYPES,
            mutation_rate=ModelConfig.MUTATION_RATE,
            crossover_rate=ModelConfig.CROSSOVER_RATE,
            elitism_count=ModelConfig.ELITISM_COUNT
        )
        pop.initialize(input_dim=X_train.shape[1], feature_names=self.feature_names)

        for gen in range(ModelConfig.GENERATIONS_PER_ITERATION):
            for ind in pop.individuals:
                try:
                    ind.model.fit(X_train, y_train)
                    pred = ind.model.predict(X_train, threshold=0.5)
                    ind.fitness = (pred == y_train).mean()
                except Exception as e:
                    print(f"    Training error: {e}", flush=True)
                    ind.fitness = -1000.0
            if gen < ModelConfig.GENERATIONS_PER_ITERATION - 1:
                pop.evolve()
            best = pop.get_best(1)[0]
            print(f"    Gen {gen+1}: best_train_fitness={best.fitness:.4f} ({best.model_type})", flush=True)

        # Re-score on validation set using profit_factor
        for i, ind in enumerate(pop.individuals):
            if ind.fitness is None or ind.fitness <= -1000:
                continue
            try:
                pred_valid = ind.model.predict(X_valid, threshold=0.5)
                if isinstance(pred_valid, np.ndarray):
                    pred_valid = pd.Series(pred_valid, index=X_valid.index)
                signals_ind = pred_valid.reindex(close_prices.index).fillna(-1).astype(int)
                self.trader.run(signals_ind, valid_ohlc)
                ind.fitness = self.trader.metrics.get('profit_factor', 0.0)
                ind.validation_metrics = self.trader.metrics
                print(f"  Model {i+1}: profit_factor={ind.fitness:.4f}", flush=True)
            except Exception as e:
                print(f"  Rescoring error: {e}", flush=True)
                ind.fitness = -1000.0

        top_models = pop.get_diverse_best(n=5)
        self._log_feature_importances(top_models[0] if top_models else None)

        # Generate consensus signals
        consensus = ConsensusGate(
            min_models_agree=ConsensusConfig.MIN_MODELS_AGREE,
            min_confidence=ConsensusConfig.MIN_CONFIDENCE
        )
        signals = consensus.decide(top_models, X_valid)

        # Run simulator
        self.trader.run(signals, valid_ohlc)
        metrics = self.trader.metrics.copy()

        # Directional accuracy
        mask = (signals != -1) & y_valid.notna()
        if mask.any():
            metrics["accuracy"] = (signals[mask] == y_valid[mask]).mean()
        else:
            metrics["accuracy"] = 0.0

        print(f"Fold result: accuracy={metrics['accuracy']:.4f}, bets/month={metrics.get('bets_per_month', 0):.2f}, "
              f"trades={metrics.get('n_trades', 0)}, sharpe={metrics.get('sharpe', 0):.3f}", flush=True)
        return metrics

    def run_walk_forward(
        self,
        df: pd.DataFrame
    ) -> Dict[str, float]:
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
                print(f"Fold {fold_idx+1}: insufficient train samples, skipping.", flush=True)
                continue

            valid_df = df_sorted.iloc[valid_start:valid_end].copy()
            train_df = df_sorted.iloc[:valid_start].copy()

            if len(train_df) > ModelConfig.MAX_TRAIN_SAMPLES:
                train_df = train_df.tail(ModelConfig.MAX_TRAIN_SAMPLES).copy()

            print(f"\nFold {fold_idx+1}/{n_folds}: train={len(train_df)}, valid={len(valid_df)}", flush=True)

            m = self._run_fold(train_df, valid_df)
            if m:
                fold_metrics.append(m)

        if not fold_metrics:
            return {"accuracy": 0.0, "bets_per_month": 0.0, "n_folds": 0}

        n = len(fold_metrics)
        total_trades = sum(m.get('n_trades', 0) for m in fold_metrics)

        if total_trades == 0:
            combined = fold_metrics[-1].copy()
            combined["n_folds"] = n
            return combined

        # Aggregate: weighted accuracy by n_trades, average everything else
        accuracy = sum(m['accuracy'] * m.get('n_trades', 1) for m in fold_metrics) / total_trades
        combined = {
            "accuracy": accuracy,
            "bets_per_month": sum(m.get('bets_per_month', 0) for m in fold_metrics) / n,
            "sharpe": sum(m.get('sharpe', 0) for m in fold_metrics) / n,
            "profit_factor": sum(m.get('profit_factor', 0) for m in fold_metrics) / n,
            "total_return": sum(m.get('total_return', 0) for m in fold_metrics),
            "max_drawdown": max(m.get('max_drawdown', 0) for m in fold_metrics),
            "n_trades": total_trades,
            "n_folds": n,
        }

        print(f"\nMulti-fold combined ({n} folds): accuracy={combined['accuracy']:.4f}, "
              f"bets/month={combined['bets_per_month']:.2f}, Sharpe={combined['sharpe']:.3f}, "
              f"total_return={combined['total_return']:.4f}, trades={total_trades}", flush=True)
        return combined

    def train_best_models(
        self,
        df: pd.DataFrame,
        population: Population,
        horizon_data: dict = None
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
                print(f"  Retraining model {i+1}/{len(top_models)} ({ind.model_type})", flush=True)
                ind.model.fit(X_recent, y_recent)
                print(f"    Done", flush=True)
            except Exception as e:
                print(f"  Failed to train {ind.model_type}: {e}", flush=True)

        # Train horizon ensemble if not provided and if we have multiple horizons
        if horizon_data is None:
            horizon_data = {}
            horizons_to_train = [h for h in FeatureConfig.HORIZONS.keys() if h != "15m"]
            if horizons_to_train:
                print(f"Training horizon ensemble for {len(horizons_to_train)} horizons...", flush=True)
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
                        print(f"    Prepared {len(df_horizon)} samples with {len(df_horizon.columns)} features", flush=True)
                        horizon_data[h_name] = df_horizon
                self.horizon_ensemble.fit_horizon_models(horizon_data, y_recent)
                print("Horizon ensemble training complete", flush=True)
            else:
                print("No additional horizons to train (HORIZONS only includes 15m)", flush=True)
        else:
            self.horizon_ensemble.fit_horizon_models(horizon_data, y_recent)

        self.main_population = population
        self.top_models = top_models

    def generate_current_signals(
        self,
        current_features: pd.DataFrame,
        use_horizon: bool = True
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
            cutoff = df["open_time"].max() - timedelta(days=ModelConfig.ONLINE_DATA_DAYS)
            df = df[df["open_time"] >= cutoff].copy()
            print(f"Online mode: limited data to last {ModelConfig.ONLINE_DATA_DAYS} days ({len(df)} rows)", flush=True)

        # 3. Prepare features
        print("[main] Starting feature preparation...", flush=True)
        df = self.prepare_features(df)
        print(f"[main] Feature preparation complete: {len(df)} rows, {len(self.feature_names)} features", flush=True)

        # 3. Split into recent data for training and current for prediction
        # Use last 30 days for training, current for signal generation
        recent_cutoff = df["open_time"].max() - timedelta(days=30)
        train_df = df[df["open_time"] <= recent_cutoff].copy()
        current_df = df[df["open_time"] > recent_cutoff].copy()

        # Limit training samples to control memory/time (use most recent samples)
        if len(train_df) > ModelConfig.MAX_TRAIN_SAMPLES:
            train_df = train_df.tail(ModelConfig.MAX_TRAIN_SAMPLES).copy()
            print(f"Limited training to {len(train_df)} most recent samples", flush=True)

        if len(train_df) < ValidationConfig.MIN_TRAIN_SAMPLES:
            print(f"Insufficient training data: {len(train_df)} samples", flush=True)
            return None

        X_train = train_df[self.feature_names]
        y_train = train_df["target"]

        # 4. Evolve population
        population = self.evolve_population(X_train, y_train, generations=ModelConfig.GENERATIONS_PER_ITERATION)

        # 5. Walk-forward validation on historical folds (use recent data to limit folds)
        if ValidationConfig.MAX_WALK_FORWARD_DAYS is not None:
            wf_cutoff = df["open_time"].max() - timedelta(days=ValidationConfig.MAX_WALK_FORWARD_DAYS)
            df_wf = df[df["open_time"] >= wf_cutoff].copy()
            print(f"Using {len(df_wf)} recent samples for walk-forward validation (max {ValidationConfig.MAX_WALK_FORWARD_DAYS} days)", flush=True)
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
            print(f"Latest signal: {latest_signal} (1=UP, 0=DOWN, -1=NO TRADE)", flush=True)
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
            "population_fitness": {str(i): ind.fitness for i, ind in enumerate(population.get_best(5))}
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
            pop_data.append({
                "model_type": ind.model_type,
                "params": ind.params,
                "fitness": float(ind.fitness)
            })
        with open(run_dir / "population.json", "w") as f:
            json.dump(pop_data, f, indent=2)

        print(f"Results saved to {run_dir}", flush=True)


def main():
    """Entry point for running the trading system."""
    print("[main] main() started", flush=True)
    system = TradingSystem()

    # Run one step
    result = system.run_online_step()

    if result:
        print(f"\nStep completed. Latest signal: {result['latest_signal']}", flush=True)
        print(f"Accuracy: {result['accuracy']:.4f}, Bets/month: {result['bets_per_month']:.2f}", flush=True)
    else:
        print("Step did not complete successfully.", flush=True)


if __name__ == "__main__":
    main()
