#!/usr/bin/env python3
"""
Demo script for evolutionary model population.

Runs a short evolution to demonstrate the system.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.base import BaselineMomentumModel
from evolution import Population, Individual, EvolutionEngine
from validation.metrics import compute_all_metrics
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo():
    """Run demo evolution."""
    # Define hyperparameter search space
    hyperparam_ranges = {
        'momentum_periods': (6, 12, 24, 168),  # categorical choices
        'volume_period': (12, 24, 48),
        'signal_threshold': (0.2, 0.3, 0.4, 0.5),
    }

    # Create evolution engine
    engine = EvolutionEngine(
        model_class=BaselineMomentumModel,
        hyperparam_ranges=hyperparam_ranges,
        population_size=10,  # Small for demo
        generations=3,      # Short demo
        fitness_metric='sharpe_ratio',
        elite_count=2,
        selection_method='tournament',
        tournament_size=3,
        crossover_rate=0.5,
        mutation_rate=0.3,
        mutation_scale=0.2,
        data_lookback_days=60,
        random_state=42
    )

    # Run evolution
    try:
        best = await engine.run_evolution()
        logger.info(f"Demo complete. Best individual: {best}")
        logger.info(f"Best hyperparams: {best.hyperparams}")
        logger.info(f"Best fitness: {best.fitness}")

        # Save results
        engine.save_results('./demo_evolution_results')
        logger.info("Results saved to ./demo_evolution_results")

        # Print summary
        print("\n" + engine._generate_summary())

    except Exception as e:
        logger.error(f"Evolution failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    asyncio.run(demo())
