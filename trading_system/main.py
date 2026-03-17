#!/usr/bin/env python3
"""
Main entry point for the SOTA BTC Trading System.

Usage:
    python main.py [--config path/to/config.yaml] [--iterate] [--once]

Options:
    --config    Path to configuration YAML file (default: ./config.yaml)
    --iterate   Run in continuous loop (default: run once then exit)
    --once      Alias for --iterate=false (run single iteration)
"""

import asyncio
import argparse
import sys
from pathlib import Path

from orchestrator import Orchestrator
from utils.config import load_config
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="SOTA BTC Trading System")
    parser.add_argument(
        '--config',
        type=str,
        default='./config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--iterate',
        action='store_true',
        default=True,
        help='Run continuously (default: True)'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run single iteration and exit'
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logger
    logger = setup_logger(
        name="main",
        log_dir="./logs",
        level=getattr(logging, config.get('logging', {}).get('level', 'INFO')),
        console=config.get('logging', {}).get('console', True),
        file=config.get('logging', {}).get('file', True)
    )

    # Create orchestrator
    logger.info(f"Loading config from {args.config}")
    orchestrator = Orchestrator(
        config=config,
        state_dir="./trading_system/state"
    )

    # Run
    if args.once or not args.iterate:
        logger.info("Running single iteration")
        await orchestrator.run_one_iteration()
        logger.info("Single iteration complete")
    else:
        logger.info("Starting continuous run loop")
        await orchestrator.run_forever()


if __name__ == "__main__":
    import logging
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
