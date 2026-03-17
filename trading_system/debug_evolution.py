#!/usr/bin/env python3
"""
Debug evolution: print all individual hyperparams and fitness to see diversity.
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import Orchestrator
from utils.config import load_config
from utils.logger import setup_logger
import logging

async def debug_evolution():
    # Load config
    config = load_config('./config.yaml')

    # Override for quick debug: smaller population to see all
    config['evolution']['population_size'] = 10
    config['evolution']['generations'] = 2

    logger = setup_logger(name="debug", log_dir="./logs", level=logging.INFO, console=True, file=False)
    orchestrator = Orchestrator(config=config, state_dir="./trading_system/state")

    # Run evolution
    best = await orchestrator.run_evolution_cycle()

    # After run, access the population from the engine? Not directly stored.
    # But we can look at saved results
    results_dir = Path('./trading_system/state/evolution_results')
    summary_file = results_dir / 'summary.txt'
    if summary_file.exists():
        print("\n--- Summary file ---")
        print(summary_file.read_text())

    # Also, let's manually check the hyperparam ranges in config
    print("\n--- Config hyperparam_ranges ---")
    print(config['evolution']['hyperparam_ranges'])

    return best

if __name__ == '__main__':
    asyncio.run(debug_evolution())
