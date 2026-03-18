"""
Evolutionary population management, mutation, and crossover.
"""

import random
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .base import BaseModel, create_model
from ..config import ModelConfig


@dataclass
class Individual:
    """An individual in the population."""
    model: BaseModel
    model_type: str
    params: dict
    fitness: float = 0.0
    metrics: Dict[str, float] = None  # accuracy, bets, sharpe, pnl, etc.

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class Population:
    """Manages a population of models with evolutionary operations."""

    def __init__(
        self,
        population_size: int = ModelConfig.POPULATION_SIZE,
        model_types: List[str] = None,
        mutation_rate: float = ModelConfig.MUTATION_RATE,
        crossover_rate: float = ModelConfig.CROSSOVER_RATE,
        elitism_count: int = ModelConfig.ELITISM_COUNT
    ):
        self.population_size = population_size
        self.model_types = model_types or ModelConfig.MODEL_TYPES
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = min(elitism_count, population_size // 4)

        self.individuals: List[Individual] = []
        self.generation = 0
        self.input_dim = None

    def initialize(self, input_dim: int, feature_names: List[str]):
        """Initialize random population."""
        self.input_dim = input_dim

        for _ in range(self.population_size):
            model_type = random.choice(self.model_types)
            params = self._random_params(model_type)
            model = create_model(model_type, input_dim=input_dim, **params)
            individual = Individual(
                model=model,
                model_type=model_type,
                params=params
            )
            self.individuals.append(individual)

        self.generation = 0

    def _random_params(self, model_type: str) -> dict:
        """Generate random hyperparameters for a model type."""
        if model_type == "lightgbm":
            n_estimators = int(np.random.choice(getattr(ModelConfig, 'LIGHTGBM_N_ESTIMATORS', [200, 300, 400])))
            learning_rate = float(np.random.choice(getattr(ModelConfig, 'LIGHTGBM_LEARNING_RATE', [0.02, 0.05, 0.1])))
            num_leaves = int(np.random.choice(getattr(ModelConfig, 'LIGHTGBM_NUM_LEAVES', [31, 63, 127])))
            min_child_samples = int(np.random.choice(getattr(ModelConfig, 'LIGHTGBM_MIN_CHILD_SAMPLES', [10, 20, 50])))
            return {"n_estimators": n_estimators, "learning_rate": learning_rate, "num_leaves": num_leaves, "min_child_samples": min_child_samples}
        elif model_type == "extra_trees":
            n_estimators = int(np.random.choice(getattr(ModelConfig, 'EXTRA_TREES_N_ESTIMATORS', [50, 100])))
            max_depth = random.choice(getattr(ModelConfig, 'EXTRA_TREES_MAX_DEPTH', [8, 10, 12]))
            min_samples_leaf = random.choice(getattr(ModelConfig, 'EXTRA_TREES_MIN_SAMPLES_LEAF', [5, 10]))
            return {"n_estimators": n_estimators, "max_depth": max_depth, "min_samples_leaf": min_samples_leaf}
        elif model_type == "random_forest":
            n_estimators = int(np.random.choice(ModelConfig.RANDOM_FOREST_N_ESTIMATORS))
            max_depth = random.choice(ModelConfig.RANDOM_FOREST_MAX_DEPTH)
            min_samples_leaf = random.choice(ModelConfig.RANDOM_FOREST_MIN_SAMPLES_LEAF) if hasattr(ModelConfig, 'RANDOM_FOREST_MIN_SAMPLES_LEAF') else 1
            return {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "n_jobs": getattr(ModelConfig, 'RANDOM_FOREST_N_JOBS', 1)
            }
        elif model_type == "gradient_boosting":
            lr = float(np.random.choice(ModelConfig.GRADIENT_BOOSTING_LEARNING_RATE))
            n_est = int(np.random.choice(getattr(ModelConfig, 'GRADIENT_BOOSTING_N_ESTIMATORS', [50, 100])))
            max_depth = int(np.random.choice(getattr(ModelConfig, 'GRADIENT_BOOSTING_MAX_DEPTH', [3, 5])))
            return {"n_estimators": n_est, "learning_rate": lr, "max_depth": max_depth}
        elif model_type == "simple_nn":
            hidden_sizes = random.choice(ModelConfig.SIMPLE_NN_HIDDEN_SIZE)
            layers = random.choice(ModelConfig.SIMPLE_NN_LAYERS)
            # Build a multi-layer structure
            sizes = [int(hidden_sizes)] * layers
            return {
                "hidden_sizes": sizes,
                "dropout": np.random.uniform(0.1, 0.5),
                "learning_rate": np.random.uniform(0.0001, 0.001),
                "epochs": 30
            }
        return {}

    def evaluate_fitness(self, individual: Individual, metrics: Dict[str, float]):
        """
        Compute composite fitness score.

        Fitness = w1*accuracy + w2*bet_ratio + w3*sharpe - w4*drawdown
        where bet_ratio = (bets_per_month / target_bets) capped at 1
        """
        accuracy = metrics.get("accuracy", 0.0)
        bets_per_month = metrics.get("bets_per_month", 0.0)
        sharpe = metrics.get("sharpe", 0.0)
        drawdown = metrics.get("max_drawdown", 0.0)

        # Normalize bet count (encourage reaching 90)
        target_bets = 90.0
        bet_ratio = min(bets_per_month / target_bets, 1.0)

        # Dual penalty for missing targets
        accuracy_penalty = 0.0 if accuracy >= 0.65 else (0.65 - accuracy) * 10
        bet_penalty = 0.0 if bets_per_month >= 90 else (90 - bets_per_month) / 90.0 * 5

        # Weighted sum (tunable)
        fitness = (
            2.0 * accuracy +
            1.0 * bet_ratio +
            0.5 * max(sharpe, 0) -
            0.5 * drawdown
        ) - accuracy_penalty - bet_penalty

        individual.fitness = fitness
        individual.metrics = metrics.copy()

    def _select_parents(self) -> Tuple[Individual, Individual]:
        """Tournament selection."""
        tournament_size = 3

        def tournament():
            contestants = random.sample(self.individuals, tournament_size)
            return max(contestants, key=lambda ind: ind.fitness)

        return tournament(), tournament()

    def _mutate(self, individual: Individual) -> Individual:
        """Apply random mutations to hyperparameters."""
        if random.random() > self.mutation_rate:
            return individual

        new_params = individual.params.copy()
        model_type = individual.model_type

        if model_type == "lightgbm":
            if random.random() < 0.5:
                new_params["n_estimators"] = int(np.random.choice(getattr(ModelConfig, 'LIGHTGBM_N_ESTIMATORS', [200, 300, 400])))
            if random.random() < 0.5:
                new_params["learning_rate"] = float(np.random.choice(getattr(ModelConfig, 'LIGHTGBM_LEARNING_RATE', [0.02, 0.05, 0.1])))
            if random.random() < 0.3:
                new_params["num_leaves"] = int(np.random.choice(getattr(ModelConfig, 'LIGHTGBM_NUM_LEAVES', [31, 63, 127])))
            if random.random() < 0.3:
                new_params["min_child_samples"] = int(np.random.choice(getattr(ModelConfig, 'LIGHTGBM_MIN_CHILD_SAMPLES', [10, 20, 50])))
        elif model_type == "extra_trees":
            if random.random() < 0.5:
                new_params["n_estimators"] = int(np.random.choice(getattr(ModelConfig, 'EXTRA_TREES_N_ESTIMATORS', [50, 100])))
            if random.random() < 0.5:
                new_params["max_depth"] = random.choice(getattr(ModelConfig, 'EXTRA_TREES_MAX_DEPTH', [8, 10, 12]))
            if random.random() < 0.3:
                new_params["min_samples_leaf"] = random.choice(getattr(ModelConfig, 'EXTRA_TREES_MIN_SAMPLES_LEAF', [5, 10]))
        elif model_type == "random_forest":
            if random.random() < 0.5:
                new_params["n_estimators"] = int(np.random.choice(ModelConfig.RANDOM_FOREST_N_ESTIMATORS))
            if random.random() < 0.5:
                new_params["max_depth"] = random.choice(ModelConfig.RANDOM_FOREST_MAX_DEPTH)
            if hasattr(ModelConfig, 'RANDOM_FOREST_MIN_SAMPLES_LEAF'):
                if random.random() < 0.3:
                    new_params["min_samples_leaf"] = random.choice(ModelConfig.RANDOM_FOREST_MIN_SAMPLES_LEAF)
        elif model_type == "gradient_boosting":
            if random.random() < 0.5:
                new_params["learning_rate"] = float(np.random.choice(ModelConfig.GRADIENT_BOOSTING_LEARNING_RATE))
            if random.random() < 0.5:
                new_params["n_estimators"] = int(np.random.choice(getattr(ModelConfig, 'GRADIENT_BOOSTING_N_ESTIMATORS', [50, 100])))
            if random.random() < 0.3:
                new_params["max_depth"] = int(np.random.choice(getattr(ModelConfig, 'GRADIENT_BOOSTING_MAX_DEPTH', [3, 5])))
        elif model_type == "simple_nn":
            if random.random() < 0.3:
                # Change number of layers
                layers = random.choice(ModelConfig.SIMPLE_NN_LAYERS)
                hidden = random.choice(ModelConfig.SIMPLE_NN_HIDDEN_SIZE)
                new_params["hidden_sizes"] = [int(hidden)] * layers
            if random.random() < 0.3:
                new_params["dropout"] = np.random.uniform(0.1, 0.5)

        # Create new individual with mutated params
        new_model = create_model(model_type, input_dim=self.input_dim, **new_params)
        return Individual(
            model=new_model,
            model_type=model_type,
            params=new_params
        )

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Crossover between two parents (blend or swap)."""
        if random.random() > self.crossover_rate:
            return parent1

        # Simple parameter averaging or swap for same model type
        if parent1.model_type != parent2.model_type:
            return parent1

        child_params = parent1.params.copy()

        # For each parameter, randomly take from either parent or blend
        for key in child_params:
            if key in parent2.params:
                if random.random() < 0.5:
                    child_params[key] = parent2.params[key]
                elif isinstance(parent1.params[key], (int, float)) and isinstance(parent2.params[key], (int, float)):
                    # Blend numeric parameters
                    blended = (parent1.params[key] + parent2.params[key]) / 2
                    # If both parents had integer values, round to int
                    if isinstance(parent1.params[key], int) and isinstance(parent2.params[key], int):
                        blended = int(round(blended))
                    child_params[key] = blended

        child_model = create_model(parent1.model_type, input_dim=self.input_dim, **child_params)
        return Individual(
            model=child_model,
            model_type=parent1.model_type,
            params=child_params
        )

    def evolve(self):
        """Create next generation using selection, crossover, mutation."""
        self.generation += 1

        # Sort by fitness (descending)
        self.individuals.sort(key=lambda ind: ind.fitness, reverse=True)

        # Elitism: keep best individuals
        elites = self.individuals[:self.elitism_count]

        # Generate offspring
        offspring = []
        while len(offspring) < self.population_size - self.elitism_count:
            parent1, parent2 = self._select_parents()
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            offspring.append(child)

        self.individuals = elites + offspring

        # Ensure we have exactly population_size
        self.individuals = self.individuals[:self.population_size]

    def get_best(self, n: int = 1) -> List[Individual]:
        """Get top n individuals."""
        sorted_inds = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
        return sorted_inds[:n]

    def get_diverse_best(self, n: int = 5) -> List[Individual]:
        """Get best individuals ensuring model type diversity."""
        sorted_inds = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
        selected = []
        types_seen = set()

        for ind in sorted_inds:
            if len(selected) >= n:
                break
            if ind.model_type not in types_seen or len(selected) < 2:
                selected.append(ind)
                types_seen.add(ind.model_type)

        return selected
