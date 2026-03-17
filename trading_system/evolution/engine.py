"""
EvolutionEngine - Main orchestrator for evolutionary model optimization.

Integrates with:
- Data pipeline (connector, features)
- Model zoo (TradingModel subclasses)
- Validation framework (walk-forward)

Runs evolutionary cycles to optimize model hyperparameters
and weights over multiple generations.
"""

from typing import Dict, Any, Optional, Type, List
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
import json

from data.connector import DataConnector
from data.features import compute_all_features, prepare_xy
from models.base import TradingModel
from validation.walk_forward import WalkForwardValidator
from .population import Population
from .individual import Individual


class EvolutionEngine:
    """
    Evolutionary optimization engine for trading models.

    Workflow:
    1. Fetch data and compute features
    2. Split into train/validation via walk-forward
    3. Initialize population of model individuals
    4. For each generation:
        - Evaluate fitness on validation folds
        - Select parents (tournament/roulette)
        - Create offspring (crossover + mutation)
        - Apply elitism (preserve best)
    5. Return best individual and population stats

    Configuration via hyperparam_ranges to define search space.
    """

    def __init__(
        self,
        model_class: Type[TradingModel],
        hyperparam_ranges: Dict[str, Any],
        population_size: int = 50,
        generations: int = 20,
        fitness_metric: str = 'sharpe_ratio',
        elite_count: int = 5,
        selection_method: str = 'tournament',
        tournament_size: int = 3,
        crossover_rate: float = 0.5,
        mutation_rate: float = 0.2,
        mutation_scale: float = 0.2,
        validation_folds: int = 5,
        data_lookback_days: int = 365,
        exchange: str = 'binance',
        symbol: str = 'BTC/USDT',
        timeframe: str = '1h',
        random_state: int = 42,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the evolution engine.

        Args:
            model_class: TradingModel class to evolve
            hyperparam_ranges: Dict defining search space
                Example: {
                    'momentum_periods': (6, 168),  # int range
                    'volume_period': (12, 48),     # int range
                    'threshold': (0.1, 0.5),      # float range
                }
            population_size: Number of individuals per generation
            generations: Number of generations to evolve
            fitness_metric: 'sharpe_ratio', 'total_return', 'calmar_ratio'
            elite_count: Number of top individuals to preserve each generation
            selection_method: 'tournament' or 'roulette'
            tournament_size: Tournament size (if tournament selection)
            crossover_rate: Uniform crossover probability (0-1)
            mutation_rate: Per-parameter mutation probability
            mutation_scale: Perturbation magnitude (relative)
            validation_folds: Number of walk-forward folds for fitness evaluation
            data_lookback_days: How many days of data to fetch
            exchange, symbol, timeframe: Data source configuration
            random_state: Random seed
            logger: Optional logger instance
        """
        self.model_class = model_class
        self.hyperparam_ranges = hyperparam_ranges
        self.population_size = population_size
        self.generations = generations
        self.fitness_metric = fitness_metric
        self.elite_count = elite_count
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.validation_folds = validation_folds
        self.data_lookback_days = data_lookback_days
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.random_state = random_state

        self.logger = logger or logging.getLogger(__name__)

        # State
        self.data_connector: Optional[DataConnector] = None
        self.raw_data: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.validator: Optional[WalkForwardValidator] = None
        self.population: Optional[Population] = None

        # Results
        self.best_individual: Optional[Individual] = None
        self.evolution_stats: List[Dict[str, float]] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    async def run_evolution(self) -> Individual:
        """
        Run the full evolutionary optimization.

        Returns:
            Best individual found
        """
        self.start_time = datetime.now()
        self.logger.info(f"Starting evolution: {self.population_size} pop × {self.generations} gen")

        # 1. Fetch and prepare data
        await self._prepare_data()

        # 2. Create population
        self._initialize_population()

        # 3. Run evolution loop
        for gen in range(self.generations):
            self.logger.info(f"Generation {gen+1}/{self.generations}")

            # Evaluate population
            self._evaluate_population()

            # Evolve to next generation (except last)
            if gen < self.generations - 1:
                self.population.evolve(
                    X_train=self._get_train_X(),
                    y_train=self._get_train_y(),
                    X_val=self._get_val_X(),
                    y_val=self._get_val_y(),
                    selection_method=self.selection_method,
                    tournament_size=self.tournament_size,
                    mutation_rate=self.mutation_rate,
                    mutation_scale=self.mutation_scale,
                    crossover_rate=self.crossover_rate,
                    preserve_elites=self.elite_count
                )

            # Track stats
            stats = self.population.get_population_stats()
            self.evolution_stats.append(stats)
            self.logger.info(f"  Best fitness: {stats.get('best', 'N/A'):.4f}")

        # 4. Select best individual
        self.best_individual = self.population.get_best_individual()

        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        self.logger.info(f"Evolution complete in {duration:.1f}s")
        self.logger.info(f"Best individual: {self.best_individual}")

        return self.best_individual

    async def _prepare_data(self):
        """Fetch data and compute features."""
        self.logger.info("Fetching market data...")

        # Convert lookback_days to limit (candles per day depends on timeframe)
        if self.timeframe == '1h':
            candles_per_day = 24
        elif self.timeframe == '1d':
            candles_per_day = 1
        elif self.timeframe == '5m':
            candles_per_day = 24 * 12
        elif self.timeframe == '15m':
            candles_per_day = 24 * 4
        else:
            candles_per_day = 24  # default to hourly

        limit = self.data_lookback_days * candles_per_day

        async with DataConnector(self.exchange) as dc:
            self.raw_data = await dc.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=limit
            )
        self.logger.info(f"Fetched {len(self.raw_data)} candles")

        self.logger.info("Computing features...")
        self.features = compute_all_features(self.raw_data)
        self.logger.info(f"Computed {len(self.features.columns)} feature columns")

        # Initialize validator
        self.validator = WalkForwardValidator(
            train_window=1000,
            validation_window=168,
            step_size=24
        )

    def _initialize_population(self):
        """Create initial population."""
        self.population = Population(
            model_class=self.model_class,
            population_size=self.population_size,
            hyperparam_ranges=self.hyperparam_ranges,
            fitness_metric=self.fitness_metric,
            elite_count=self.elite_count,
            random_state=self.random_state
        )
        self.population.initialize(method='random')
        # Assign feature columns to each individual for proper model feature identification
        feature_cols = self.features.columns.tolist()
        for ind in self.population.individuals:
            ind.feature_columns = feature_cols
        self.logger.info(f"Initialized population with {len(self.population)} individuals")

    def _get_datasets(self):
        """Prepare training and validation datasets."""
        # Use most recent data for validation (last 500), earlier for training
        total_len = len(self.features)
        val_end = total_len
        val_start = max(0, total_len - 500)
        train_end = val_start
        train_start = max(0, train_end - 2000)

        train_df = self.features.iloc[train_start:train_end]
        val_df = self.features.iloc[val_start:val_end]

        # Use close_return as target (like orchestrator)
        lookback = 168
        horizon = 1

        X_train, y_train, _ = prepare_xy(
            train_df,
            target_column='close_return',
            lookback=lookback,
            horizon=horizon
        )
        X_val, y_val, _ = prepare_xy(
            val_df,
            target_column='close_return',
            lookback=lookback,
            horizon=horizon
        )

        return X_train, y_train, X_val, y_val

    def _get_train_X(self) -> np.ndarray:
        X_train, _, _, _ = self._get_datasets()
        return X_train

    def _get_train_y(self) -> np.ndarray:
        _, y_train, _, _ = self._get_datasets()
        return y_train

    def _get_val_X(self) -> np.ndarray:
        _, _, X_val, _ = self._get_datasets()
        return X_val

    def _get_val_y(self) -> np.ndarray:
        _, _, _, y_val = self._get_datasets()
        return y_val

    def _evaluate_population(self):
        """Evaluate all individuals in population."""
        self.population.evaluate_all(
            X_train=self._get_train_X(),
            y_train=self._get_train_y(),
            X_val=self._get_val_X(),
            y_val=self._get_val_y()
        )

    def save_results(self, output_dir: str):
        """
        Save evolution results to disk.

        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save best individual hyperparams
        if self.best_individual:
            best_params = self.best_individual.to_dict()
            with open(output_path / 'best_individual.json', 'w') as f:
                json.dump(best_params, f, indent=2, default=str)

        # Save evolution statistics
        stats_file = output_path / 'evolution_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(self.evolution_stats, f, indent=2)

        # Save summary report
        summary = self._generate_summary()
        with open(output_path / 'summary.txt', 'w') as f:
            f.write(summary)

        self.logger.info(f"Evolution results saved to {output_dir}")

    def _generate_summary(self) -> str:
        """Generate human-readable summary."""
        if not self.evolution_stats:
            return "No evolution data available."

        lines = []
        lines.append("Evolution Summary")
        lines.append("=" * 50)
        lines.append(f"Model: {self.model_class.__name__}")
        lines.append(f"Generations: {self.generations}")
        lines.append(f"Population size: {self.population_size}")
        lines.append(f"Fitness metric: {self.fitness_metric}")
        lines.append("")

        lines.append("Best fitness by generation:")
        for i, stats in enumerate(self.evolution_stats):
            lines.append(f"  Gen {i+1}: {stats.get('best', 'N/A'):.4f}")

        lines.append("")
        if self.best_individual:
            lines.append("Best Individual Configuration:")
            for k, v in self.best_individual.hyperparams.items():
                lines.append(f"  {k}: {v}")
            lines.append(f"Fitness: {self.best_individual.fitness:.4f}")

        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        lines.append(f"\nTotal time: {duration:.1f}s")

        return "\n".join(lines)
