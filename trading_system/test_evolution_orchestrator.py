#!/usr/bin/env python3
"""
Test script for evolution integration in orchestrator.

Runs one evolution cycle via orchestrator.run_evolution_cycle()
and verifies it completes successfully.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import Orchestrator
from utils.config import load_config
from utils.logger import setup_logger
import logging

async def test_evolution_cycle():
    """Test the evolution cycle through orchestrator."""
    print("=" * 70)
    print("EVOLUTION INTEGRATION TEST")
    print("=" * 70)

    # Load config relative to this script's directory
    script_dir = Path(__file__).parent
    config = load_config(str(script_dir / 'config.yaml'))
    print(f"Config loaded: timeframe={config['data']['timeframe']}")
    print(f"  evolution.enabled = {config.get('evolution', {}).get('enabled', False)}")
    print(f"  population_size = {config['evolution']['population_size']}")
    print(f"  generations = {config['evolution']['generations']}")

    # Setup logger
    logger = setup_logger(
        name="test_evolution",
        log_dir="./logs",
        level=getattr(logging, config.get('logging', {}).get('level', 'INFO')),
        console=True,
        file=False  # Don't log to file for test
    )

    # Create orchestrator
    script_dir = Path(__file__).parent
    orchestrator = Orchestrator(
        config=config,
        state_dir=str(script_dir / "state")
    )

    # Run one evolution cycle
    print("\nStarting evolution cycle...")
    try:
        best_individual = await orchestrator.run_evolution_cycle()

        print("\n" + "=" * 70)
        print("✅ EVOLUTION CYCLE COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"\nBest individual found:")
        print(f"  Hyperparams: {best_individual.hyperparams}")
        print(f"  Fitness ({config['evolution']['fitness_metric']}): {best_individual.fitness:.4f}")

        # Check results were saved
        results_dir = Path('./trading_system/state/evolution_results')
        if results_dir.exists():
            print(f"\nResults saved to: {results_dir}")
            summary_file = results_dir / 'summary.txt'
            if summary_file.exists():
                print("\n--- Summary ---")
                print(summary_file.read_text())

        return True

    except Exception as e:
        print(f"\n❌ Evolution cycle failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = asyncio.run(test_evolution_cycle())
    sys.exit(0 if success else 1)
