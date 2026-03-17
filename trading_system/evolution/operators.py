"""
Genetic Operators - Functions for evolutionary algorithms.

These are standalone utility functions that can be used
independently or by the Population class.
"""

from typing import List, Dict, Any, Optional, Type
import numpy as np
from .individual import Individual


def tournament_selection(
    individuals: List[Individual],
    tournament_size: int = 3,
    n_select: int = 1
) -> List[Individual]:
    """
    Select individuals via tournament selection.

    Args:
        individuals: Pool of individuals to select from (must be evaluated)
        tournament_size: Number of individuals per tournament
        n_select: Number of individuals to select

    Returns:
        List of selected individuals
    """
    selected = []
    for _ in range(n_select):
        contestants = np.random.choice(individuals, size=tournament_size, replace=True)
        best = max(contestants, key=lambda ind: ind.fitness if ind.fitness is not None else -np.inf)
        selected.append(best)
    return selected if n_select > 1 else [selected[0]]


def roulette_wheel_selection(
    individuals: List[Individual],
    n_select: int = 1,
    fit_transform: str = 'raw'
) -> List[Individual]:
    """
    Select individuals via roulette wheel (fitness proportionate).

    Args:
        individuals: Pool of individuals to select from (must be evaluated)
        n_select: Number of individuals to select
        fit_transform: How to handle negative/zero fitness
            - 'raw': use fitness as-is (may produce negatives)
            - 'shift': shift to make all positive
            - 'rank': use rank-based probabilities

    Returns:
        List of selected individuals
    """
    fitnesses = np.array([ind.fitness for ind in individuals])

    if fit_transform == 'raw':
        # Shift to positive if any negative
        if fitnesses.min() < 0:
            fitnesses = fitnesses - fitnesses.min() + 1e-6
        probs = fitnesses / fitnesses.sum()
    elif fit_transform == 'shift':
        # Minimum becomes zero
        fitnesses = fitnesses - fitnesses.min()
        if fitnesses.sum() == 0:
            probs = np.ones(len(individuals)) / len(individuals)
        else:
            probs = fitnesses / fitnesses.sum()
    elif fit_transform == 'rank':
        # Rank-based: higher rank = higher probability
        ranks = np.argsort(fitnesses)[::-1] + 1  # rank 1 = best
        probs = ranks / ranks.sum()
    else:
        raise ValueError(f"Unknown fit_transform: {fit_transform}")

    selected = np.random.choice(
        individuals,
        size=n_select,
        replace=True,
        p=probs
    )
    return list(selected)


def uniform_crossover(
    parent1: Individual,
    parent2: Individual,
    crossover_rate: float = 0.5
) -> Individual:
    """
    Perform uniform crossover on hyperparameters.

    For each hyperparameter, randomly choose from parent1 or parent2.

    Args:
        parent1, parent2: Parents to crossover
        crossover_rate: Probability of taking param from parent2 (else parent1)

    Returns:
        New child Individual
    """
    if parent1.model_class != parent2.model_class:
        raise ValueError("Parents must be same model class")

    child_hyperparams = {}
    all_keys = set(parent1.hyperparams.keys()) | set(parent2.hyperparams.keys())

    for key in all_keys:
        if key in parent1.hyperparams and key in parent2.hyperparams:
            # Both have this param - choose based on crossover_rate
            if np.random.random() < crossover_rate:
                child_hyperparams[key] = parent2.hyperparams[key]
            else:
                child_hyperparams[key] = parent1.hyperparams[key]
        elif key in parent1.hyperparams:
            child_hyperparams[key] = parent1.hyperparams[key]
        else:
            child_hyperparams[key] = parent2.hyperparams[key]

    return Individual(
        model_class=parent1.model_class,
        hyperparams=child_hyperparams,
        name=f"crossover_{parent1.name}_{parent2.name}"
    )


def mutate_hyperparameters(
    individual: Individual,
    mutation_rate: float = 0.1,
    mutation_scale: float = 0.2,
    param_ranges: Optional[Dict[str, Any]] = None,
    inplace: bool = False
) -> Individual:
    """
    Mutate hyperparameters of an individual.

    Args:
        individual: Individual to mutate
        mutation_rate: Probability of mutating each hyperparameter
        mutation_scale: Perturbation scale for numeric params
        param_ranges: Optional dict of {param: (min, max)} to clip mutations
        inplace: If True, modify individual in place; else return copy

    Returns:
        Mutated Individual (same object if inplace=True)
    """
    if not inplace:
        new_individual = Individual(
            model_class=individual.model_class,
            hyperparams=individual.hyperparams.copy(),
            name=f"{individual.name}_mutated"
        )
        target = new_individual
    else:
        target = individual

    for key, value in target.hyperparams.items():
        if np.random.random() >= mutation_rate:
            continue  # No mutation for this param

        if isinstance(value, (int, float)):
            # Numeric mutation: add Gaussian noise
            if isinstance(value, int):
                # Perturb integer
                perturbation = int(np.random.normal(0, mutation_scale * max(1, abs(value))))
                new_value = value + perturbation
                if param_ranges and key in param_ranges:
                    min_val, max_val = param_ranges[key]
                    new_value = max(min_val, min(max_val, new_value))
                target.hyperparams[key] = max(1, new_value)  # Ensure positive int
            else:
                # Float perturbation
                perturbation = np.random.normal(0, mutation_scale * abs(value))
                new_value = value + perturbation
                if param_ranges and key in param_ranges:
                    min_val, max_val = param_ranges[key]
                    new_value = max(min_val, min(max_val, new_value))
                target.hyperparams[key] = float(new_value)

        elif isinstance(value, (list, tuple)):
            # For sequences, perturb each element with some probability
            original_type = type(value)
            new_sequence = list(value)
            for i in range(len(new_sequence)):
                if isinstance(new_sequence[i], (int, float)) and np.random.random() < mutation_rate:
                    elem = new_sequence[i]
                    if isinstance(elem, int):
                        perturbation = int(np.random.normal(0, mutation_scale * max(1, abs(elem))))
                        new_sequence[i] = max(1, elem + perturbation)
                    else:
                        perturbation = np.random.normal(0, mutation_scale * abs(elem))
                        new_sequence[i] = float(elem + perturbation)
            target.hyperparams[key] = original_type(new_sequence)

        # For strings/categoricals, could add swap/replace if vocabulary provides

    if not inplace:
        return new_individual
    return individual


def gaussian_mutation(
    individual: Individual,
    mutation_scale: float = 0.1,
    params_to_mutate: Optional[List[str]] = None
) -> Individual:
    """
    Specific mutation: Gaussian perturbation on selected numeric parameters.

    Args:
        individual: Individual to mutate
        mutation_scale: Standard deviation as fraction of value
        params_to_mutate: Specific parameters to mutate (None = all numeric)

    Returns:
        New mutated Individual (copy)
    """
    new_hyperparams = individual.hyperparams.copy()

    for key in (params_to_mutate or individual.hyperparams.keys()):
        value = individual.hyperparams.get(key)
        if isinstance(value, (int, float)):
            if isinstance(value, int):
                sigma = max(1, int(abs(value) * mutation_scale))
                new_value = value + np.random.randint(-sigma, sigma + 1)
                new_hyperparams[key] = max(1, new_value)
            else:
                sigma = abs(value) * mutation_scale
                new_value = value + np.random.normal(0, sigma)
                new_hyperparams[key] = float(new_value)

    return Individual(
        model_class=individual.model_class,
        hyperparams=new_hyperparams,
        name=f"{individual.name}_gaussian"
    )


def blend_crossover(
    parent1: Individual,
    parent2: Individual,
    alpha: float = 0.5
) -> Individual:
    """
    Blend crossover (BLX-alpha) for numeric hyperparameters.

    Creates child by taking values from interval around parents' values.

    Args:
        parent1, parent2: Two parents
        alpha: Blend factor (0 = no blending, 1 = uniform between extremes)

    Returns:
        New child Individual
    """
    if parent1.model_class != parent2.model_class:
        raise ValueError("Parents must be same model class")

    child_hyperparams = {}
    all_keys = set(parent1.hyperparams.keys()) | set(parent2.hyperparams.keys())

    for key in all_keys:
        v1 = parent1.hyperparams.get(key)
        v2 = parent2.hyperparams.get(key)

        if v1 is not None and v2 is not None and isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            # Numeric: blend
            if isinstance(v1, int):
                # For ints, blend and round
                diff = abs(v2 - v1)
                blend_range = alpha * diff
                lower = min(v1, v2) - blend_range
                upper = max(v1, v2) + blend_range
                child_hyperparams[key] = int(np.random.uniform(lower, upper))
            else:
                # Floats
                diff = abs(v2 - v1)
                blend_range = alpha * diff
                lower = min(v1, v2) - blend_range
                upper = max(v1, v2) + blend_range
                child_hyperparams[key] = float(np.random.uniform(lower, upper))
        else:
            # Non-numeric or missing from one parent: default to parent1 or random choice
            if v1 is not None:
                child_hyperparams[key] = v1
            elif v2 is not None:
                child_hyperparams[key] = v2
            # If both None, skip (shouldn't happen)

    return Individual(
        model_class=parent1.model_class,
        hyperparams=child_hyperparams,
        name=f"blend_{parent1.name}_{parent2.name}"
    )
