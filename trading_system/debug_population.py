#!/usr/bin/env python3
"""
Debug: Check what hyperparams are sampled in initial population.
"""

import asyncio
import numpy as np
from evolution import Population, Individual
from models.base import BaselineMomentumModel

def test_sampling():
    hyperparam_ranges = {
        'momentum_periods': [12, 24, 48, 96],
        'volume_period': [12, 24, 48],
        'signal_threshold': [0.2, 0.3, 0.4],
    }

    pop = Population(
        model_class=BaselineMomentumModel,
        population_size=20,
        hyperparam_ranges=hyperparam_ranges,
        fitness_metric='directional_accuracy',
        elite_count=5,
        random_state=42
    )
    pop.initialize(method='random')

    print("Initial population hyperparams:")
    for i, ind in enumerate(pop.individuals):
        print(f"  {i}: {ind.hyperparams}")

    # Check count of momentum=24
    mom_24_count = sum(1 for ind in pop.individuals if ind.hyperparams.get('momentum_periods') == 24)
    print(f"\nCount with momentum=24: {mom_24_count} / {len(pop.individuals)}")

if __name__ == '__main__':
    test_sampling()
