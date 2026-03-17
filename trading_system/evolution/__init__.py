"""
Evolutionary algorithms for model search and optimization.

Components:
- Individual: A model instance with hyperparameters and fitness
- Population: Manages a population of individuals
- GeneticOperators: Mutation, crossover, selection
- Evolution: Main evolution loop
"""

from .individual import Individual
from .population import Population
from .operators import (
    tournament_selection,
    roulette_wheel_selection,
    uniform_crossover,
    mutate_hyperparameters,
)
from .engine import EvolutionEngine

__all__ = [
    'Individual',
    'Population',
    'tournament_selection',
    'roulette_wheel_selection',
    'uniform_crossover',
    'mutate_hyperparameters',
    'EvolutionEngine',
]
