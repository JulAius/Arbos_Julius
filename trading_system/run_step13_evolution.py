#!/usr/bin/env python3
"""
Step 13: Full-scale Evolution with Refined Configuration

Run evolution using LSTM model with tightened hyperparameter ranges to improve generalization.
Population: 50, Generations: 20, with ADX regime features included.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import Orchestrator
from utils.config import load_config
from utils.logger import setup_logger
import logging

async def run_step13():
    """Run the full-scale evolution cycle."""

    # Load configuration
    config = load_config('trading_system/config.yaml')

    # Ensure evolution is enabled and use LSTM
    config.setdefault('evolution', {})['enabled'] = True
    config['model']['type'] = 'lstm'  # Use LSTM as primary

    # Setup logger
    logger = setup_logger(
        name="step13_evolution",
        log_dir="./logs",
        level=getattr(logging, config.get('logging', {}).get('level', 'INFO')),
        console=config.get('logging', {}).get('console', True),
        file=config.get('logging', {}).get('file', True)
    )

    logger.info("=" * 60)
    logger.info("STEP 13: Full-scale Evolution with Refined Configuration")
    logger.info("=" * 60)
    logger.info(f"Population size: {config['evolution']['population_size']}")
    logger.info(f"Generations: {config['evolution']['generations']}")
    logger.info(f"Model type: {config['model']['type']}")
    logger.info(f"Hyperparam ranges: {config['evolution']['hyperparam_ranges']}")
    logger.info(f"Data lookback: {config['data']['lookback_days']} days")
    logger.info("=" * 60)

    # Create orchestrator
    orchestrator = Orchestrator(
        config=config,
        state_dir="./trading_system/state"
    )

    # Run a single evolution cycle
    try:
        best_individual = await orchestrator.run_evolution_cycle()

        logger.info("=" * 60)
        logger.info("STEP 13 COMPLETE - Evolution Results")
        logger.info("=" * 60)
        logger.info(f"Best individual ID: {best_individual.id}")
        logger.info(f"Best fitness ({config['evolution']['fitness_metric']}): {best_individual.fitness:.6f}")
        logger.info(f"Best hyperparameters: {best_individual.hyperparams}")
        logger.info(f"Results saved to: trading_system/state/evolution_results/")
        logger.info("=" * 60)

        # Print summary to stdout for arbos capture
        print("\n" + "=" * 60)
        print("STEP 13 EVOLUTION COMPLETE")
        print("=" * 60)
        print(f"Best fitness: {best_individual.fitness:.6f}")
        print(f"Model: {best_individual.model_config.get('model_type')}")
        print(f"Hyperparams: {best_individual.hyperparams}")
        print("=" * 60)

        return best_individual

    except Exception as e:
        logger.error(f"Step 13 evolution failed: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    result = asyncio.run(run_step13())
    print(f"\nFinal best fitness: {result.fitness if result else 'N/A'}")
