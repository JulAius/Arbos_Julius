"""
Orchestrator - Main coordination loop for the trading system.

Manages the full pipeline: data → features → train → validate → trade →
reflect & improve. Designed to run continuously, evolving the system over time.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass, field
import json
import os

from data.connector import DataConnector
from data.features import compute_all_features, prepare_xy
from models.base import TradingModel
from validation.walk_forward import WalkForwardValidator
from execution.backtest import BacktestEngine, BacktestResult
from utils.logger import setup_logger
from evolution import EvolutionEngine, Individual


@dataclass
class SystemState:
    """Current state of the trading system."""
    model: Optional[TradingModel] = None
    last_training_data: Optional[pd.DataFrame] = None
    last_validation_result: Optional[Any] = None
    last_backtest_result: Optional[BacktestResult] = None
    current_equity: float = 10000.0
    performance_history: list = field(default_factory=list)
    iteration: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    def save(self, path: str) -> None:
        """Save state to disk."""
        os.makedirs(path, exist_ok=True)

        state_dict = {
            'iteration': self.iteration,
            'current_equity': self.current_equity,
            'performance_history': self.performance_history,
            'created_at': self.created_at.isoformat(),
            'has_model': self.model is not None,
            'has_validation': self.last_validation_result is not None,
            'has_backtest': self.last_backtest_result is not None,
        }

        with open(os.path.join(path, 'state.json'), 'w') as f:
            json.dump(state_dict, f, indent=2)

        # Save model if exists
        if self.model:
            model_path = os.path.join(path, 'model')
            self.model.save(model_path)

    @classmethod
    def load(cls, path: str) -> 'SystemState':
        """Load state from disk."""
        state_file = os.path.join(path, 'state.json')
        if not os.path.exists(state_file):
            return cls()

        with open(state_file, 'r') as f:
            state_dict = json.load(f)

        state = cls(
            iteration=state_dict['iteration'],
            current_equity=state_dict['current_equity'],
            performance_history=state_dict['performance_history'],
            created_at=datetime.fromisoformat(state_dict['created_at'])
        )

        # Load model if exists
        model_path = os.path.join(path, 'model')
        if os.path.exists(model_path):
            # Dynamically load the model class based on saved type
            from models.base import BaselineMomentumModel
            model = BaselineMomentumModel()
            model.load(model_path)
            state.model = model

        return state


class Orchestrator:
    """
    Main orchestrator for the trading system.

    Implements the design loop:
        S_t = design_or_modify(S_{t-1})
        O_t = run(S_t)
        P_t = measure(O_t)
        Δ_t = reflect(S_t, P_t)
        S_{t+1} = improve(S_t, Δ_t)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        state_dir: str = "./trading_system/state"
    ):
        """
        Initialize orchestrator.

        Args:
            config: Configuration dictionary
            state_dir: Directory for persisting state
        """
        self.config = config or self._default_config()
        self.state_dir = state_dir
        self.logger = setup_logger("orchestrator")
        self.state = SystemState.load(state_dir)

        # Components
        self.validator = WalkForwardValidator(
            train_window=self.config['validation']['train_window'],
            validation_window=self.config['validation']['validation_window'],
            step_size=self.config['validation']['step_size'],
            expanding=True
        )
        self.backtest_engine = BacktestEngine(
            initial_capital=self.config['trading']['initial_capital'],
            commission_rate=self.config['trading']['commission'],
            max_position_size=self.config['trading']['max_position_size']
        )

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'data': {
                'exchange': 'binance',
                'symbol': 'BTC/USDT',
                'timeframe': '1h',
                'lookback_days': 365
            },
            'model': {
                'type': 'momentum_baseline',
                'horizon': 1,
                'params': {}
            },
            'validation': {
                'train_window': 1000,
                'validation_window': 168,
                'step_size': 24
            },
            'trading': {
                'initial_capital': 10000.0,
                'commission': 0.0006,
                'max_position_size': 1.0
            },
            'evolution': {
                'enabled': False,
                'population_size': 10
            }
        }

    async def fetch_data(self) -> pd.DataFrame:
        """Fetch latest market data."""
        self.logger.info("Fetching market data...")

        symbol = self.config['data']['symbol']
        timeframe = self.config['data']['timeframe']
        lookback_days = self.config['data']['lookback_days']

        # Calculate how many candles we need
        candles_per_day = 24 if timeframe == '1h' else 1
        limit = lookback_days * candles_per_day

        async with DataConnector(self.config['data']['exchange']) as dc:
            df = await dc.fetch_ohlcv(symbol, timeframe, limit=limit)

        self.logger.info(f"Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        return df

    def process_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute all features."""
        self.logger.info("Computing features...")
        features = compute_all_features(data)
        self.logger.info(f"Computed {len(features.columns)} feature columns")
        return features

    def train_model(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Train the current model."""
        self.logger.info("Training model...")

        # Prepare training data
        X, y, _ = prepare_xy(
            features,
            target_column='close_return',
            lookback=168,  # 1 week
            horizon=1
        )

        if len(X) == 0:
            raise ValueError("No training data after preprocessing")

        # Get or create model
        model = self.state.model
        if model is None:
            from models.base import BaselineMomentumModel
            model = BaselineMomentumModel(
                name=f"momentum_{self.state.iteration}",
                horizon=self.config['model']['horizon']
            )
            model.feature_columns = features.columns.tolist()

        # Train
        metrics = model.train(X, y)
        self.state.model = model

        self.logger.info(f"Training complete: accuracy={metrics.get('accuracy', 0):.2%}, mse={metrics.get('mse', 0):.6f}")
        return metrics

    def validate_model(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Run walk-forward validation."""
        self.logger.info("Running walk-forward validation...")

        feature_cols = [c for c in features.columns if c != 'close_return']

        result = self.validator.evaluate_model(
            data=features,
            model=self.state.model,
            train_func=lambda m, X, y: m.train(X, y),
            predict_func=lambda m, X: m.predict(X),
            feature_columns=feature_cols
        )

        summary = result.summary()
        self.state.last_validation_result = result

        self.logger.info(f"Validation complete: val_sharpe_ratio_mean={summary.get('val_sharpe_ratio_mean', 0):.2f}")
        return summary

    def run_backtest(self, data: pd.DataFrame, signals: pd.Series) -> BacktestResult:
        """Run backtest on recent data."""
        self.logger.info("Running backtest...")

        result = self.backtest_engine.run(
            data=data,
            signals=signals,
            prices=data['close'],
            signal_threshold=0.3
        )

        self.logger.info(f"Backtest: {len(result.trades)} trades, total return={result.total_return:.2%}")
        self.state.last_backtest_result = result

        return result

    async def design_phase(self) -> List[str]:
        """
        Design phase: decide what changes to make to the system.

        This is a placeholder for future auto-design. Currently returns
        a simple plan.
        """
        self.logger.info("Design phase: reviewing current system state")

        # For now, just return a basic plan
        actions = [
            "Fetch latest market data",
            "Compute features",
            "Train model",
            "Validate via walk-forward",
            "Run backtest",
        ]

        return actions

    async def run_phase(self, plan: List[str]) -> Optional[BacktestResult]:
        """Run the planned actions."""
        self.logger.info("Run phase: executing plan")

        # 1. Fetch data
        data = await self.fetch_data()
        features = self.process_features(data)

        # 2. Train model
        train_metrics = self.train_model(features)

        # 3. Validate
        val_metrics = self.validate_model(features)

        # 4. Generate signals on recent data and backtest
        feature_cols = [c for c in features.columns if c != 'close_return']
        X_test = features[feature_cols].values[-500:]  # Last 500 hours
        signals = self.state.model.predict(X_test)

        backtest = self.run_backtest(features.iloc[-500:], pd.Series(signals, index=features.index[-500:]))

        # Update state
        self.state.current_equity = backtest.final_equity
        self.state.performance_history.append({
            'iteration': self.state.iteration,
            'total_return': backtest.total_return,
            'sharpe_ratio': backtest.metrics.get('sharpe_ratio', 0),
            'max_drawdown': backtest.metrics.get('max_drawdown', 0),
            'timestamp': datetime.now().isoformat()
        })

        return backtest

    def reflect_phase(self, result: BacktestResult) -> Dict[str, Any]:
        """
        Reflection phase: analyze what worked and what didn't.

        Identifies weaknesses in the current design to inform improvements.
        """
        self.logger.info("Reflection phase: analyzing performance")

        reflections = {
            'sharpe_ratio': result.metrics.get('sharpe_ratio', 0),
            'max_drawdown': result.metrics.get('max_drawdown', 0),
            'win_rate': result.metrics.get('win_rate', 0),
            'profit_factor': result.metrics.get('profit_factor', 0),
            'n_trades': len(result.trades),
            'issues': [],
            'suggestions': []
        }

        # Identify issues
        if reflections['sharpe_ratio'] < 1.0:
            reflections['issues'].append("Sharpe ratio below threshold (1.0)")
        if reflections['max_drawdown'] > 0.2:
            reflections['issues'].append("Max drawdown exceeds 20%")
        if reflections['win_rate'] < 0.45:
            reflections['issues'].append("Win rate below 45%")
        if reflections['profit_factor'] < 1.2:
            reflections['issues'].append("Profit factor below 1.2")

        # Generate improvement suggestions (basic)
        if reflections['max_drawdown'] > 0.15:
            reflections['suggestions'].append("Add tighter risk management / stop losses")
        if reflections['sharpe_ratio'] < 0.5:
            reflections['suggestions'].append("Add more features or try different model architecture")

        self.logger.info(f"Reflection: {len(reflections['issues'])} issues, {len(reflections['suggestions'])} suggestions")
        return reflections

    def improve_phase(self, reflections: Dict[str, Any]) -> List[str]:
        """
        Improvement phase: decide on changes for next iteration.

        Currently simple - would be replaced with automated evolution in
        advanced versions.
        """
        self.logger.info("Improvement phase: planning next iteration")

        actions = ["Iterate with current design (no changes)"]

        if 'use different model architecture' in str(reflections.get('suggestions', [])).lower():
            actions = ["Explore adding more technical indicators", "Try adding LSTM/Transformer model"]

        self.logger.info(f"Next iteration plan: {actions}")
        return actions

    async def run_one_iteration(self) -> None:
        """Run a single iteration of the design→run→measure→reflect→improve loop."""
        self.logger.info(f"=== Starting iteration {self.state.iteration} ===")

        try:
            # Design
            plan = await self.design_phase()
            self.logger.info(f"Plan: {plan}")

            # Run
            result = await self.run_phase(plan)

            # Measure
            reflections = self.reflect_phase(result) if result else {}

            # Improve
            next_actions = self.improve_phase(reflections)
            self.logger.info(f"Next actions: {next_actions}")

            # Increment and save state
            self.state.iteration += 1
            self.state.save(self.state_dir)

            self.logger.info(f"Iteration {self.state.iteration - 1} complete. Equity: ${self.state.current_equity:.2f}")

        except Exception as e:
            self.logger.error(f"Iteration failed: {e}", exc_info=True)
            raise

    async def run_evolution_cycle(self) -> Individual:
        """
        Run an evolutionary optimization cycle.

        Returns:
            Best evolved individual
        """
        self.logger.info("Starting evolution cycle")

        # Get evolution config
        evo_config = self.config.get('evolution', {})
        population_size = evo_config.get('population_size', 50)
        generations = evo_config.get('generations', 20)
        elite_count = evo_config.get('elite_count', 5)
        fitness_metric = evo_config.get('fitness_metric', 'sharpe_ratio')
        hyperparam_ranges = evo_config.get('hyperparam_ranges', {})

        # Determine model class from config
        model_type = self.config['model']['type']
        if model_type == 'momentum_baseline':
            from models.base import BaselineMomentumModel
            model_class = BaselineMomentumModel
        elif model_type == 'lstm':
            from models.lstm import LSTMTradingModel
            model_class = LSTMTradingModel
        elif model_type == 'transformer':
            from models.transformer import TransformerTradingModel
            model_class = TransformerTradingModel
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Create evolution engine
        engine = EvolutionEngine(
            model_class=model_class,
            hyperparam_ranges=hyperparam_ranges,
            population_size=population_size,
            generations=generations,
            fitness_metric=fitness_metric,
            elite_count=elite_count,
            selection_method='tournament',
            tournament_size=3,
            data_lookback_days=self.config['data']['lookback_days'],
            exchange=self.config['data']['exchange'],
            symbol=self.config['data']['symbol'],
            timeframe=self.config['data']['timeframe'],
            random_state=42,
            logger=self.logger
        )

        # Run evolution
        best = await engine.run_evolution()

        # Save results
        from pathlib import Path
        results_dir = Path(self.state_dir) / 'evolution_results'
        engine.save_results(str(results_dir))

        return best

    async def run_forever(self) -> None:
        """Run the orchestrator in an infinite loop."""
        self.logger.info("Starting orchestrator infinite loop")

        # Check if evolution mode is enabled
        evolution_mode = self.config.get('evolution', {}).get('enabled', False)

        while True:
            try:
                if evolution_mode:
                    # Run evolution cycle instead of normal iteration
                    await self.run_evolution_cycle()
                    # After evolution, can turn off or continue? For now, just continue loop
                else:
                    # Normal iteration
                    await self.run_one_iteration()

                # Wait before next iteration (configurable)
                wait_hours = self.config.get('loop', {}).get('wait_hours', 1)
                self.logger.info(f"Waiting {wait_hours} hour(s) before next iteration...")
                await asyncio.sleep(wait_hours * 3600)

            except KeyboardInterrupt:
                self.logger.info("Shutting down orchestrator")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}, retrying in 1 hour...")
                await asyncio.sleep(3600)
