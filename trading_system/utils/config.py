"""
Configuration loader - Load config from YAML or use defaults.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "./config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file or use defaults.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)

    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Merge with sensible defaults
    defaults = {
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
        'loop': {
            'wait_hours': 1
        }
    }

    # Deep merge
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = deep_update(d[k], v)
            else:
                d[k] = v
        return d

    merged = defaults.copy()
    if config:
        merged = deep_update(merged, config)

    return merged
