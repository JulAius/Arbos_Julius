"""
Population - Manages a collection of individuals.

Provides evolutionary operations: selection, breeding, mutation,
and tracking of best individuals across generations.
"""

from typing import List, Dict, Any, Optional, Type, Callable
import numpy as np
from .individual import Individual
from models.base import TradingModel


class Population:
    """
    A population of model individuals undergoing evolution.

    Manages:
    - Initialization (random or seeded)
    - Fitness evaluation tracking
    - Selection for breeding
    - Crossover and mutation
    - Elitism (preserving best individuals)
    """

    def __init__(
        self,
        model_class: Type[TradingModel],
        population_size: int = 50,
        hyperparam_ranges: Optional[Dict[str, Any]] = None,
        fitness_metric: str = 'sharpe_ratio',
        elite_count: int = 5,
        random_state: int = 42
    ):
        """
        Initialize population.

        Args:
            model_class: TradingModel class to instantiate for all individuals
            population_size: Number of individuals in population
            hyperparam_ranges: Dict of {param_name: (min, max) or list of values} for random sampling
            fitness_metric: Metric to optimize ('sharpe_ratio', 'total_return', 'calmar_ratio')
            elite_count: Number of top individuals to preserve unchanged (elitism)
            random_state: Random seed for reproducibility
        """
        self.model_class = model_class
        self.population_size = population_size
        self.hyperparam_ranges = hyperparam_ranges or {}
        self.fitness_metric = fitness_metric
        self.elite_count = min(elite_count, population_size)
        self.random_state = random_state

        np.random.seed(random_state)

        # Population state
        self.individuals: List[Individual] = []
        self.generation: int = 0
        self.best_individual_history: List[Individual] = []
        self.fitness_history: List[float] = []
        self.mean_fitness_history: List[float] = []

    def initialize(self, method: str = 'random', seed_hyperparams: Optional[List[Dict]] = None):
        """
        Initialize the population.

        Args:
            method: 'random' or 'seeded'
            seed_hyperparams: List of hyperparameter dicts to seed population with (for 'seeded' method)
        """
        if method == 'random':
            self._initialize_random()
        elif method == 'seeded' and seed_hyperparams:
            self._initialize_seeded(seed_hyperparams)
        else:
            raise ValueError(f"Invalid initialization method: {method}")

    def _initialize_random(self):
        """Create random hyperparameters within specified ranges."""
        self.individuals = []
        for i in range(self.population_size):
            hyperparams = self._sample_random_hyperparams()
            individual = Individual(
                model_class=self.model_class,
                hyperparams=hyperparams,
                name=f"gen{self.generation}_ind{i}"
            )
            self.individuals.append(individual)

    def _sample_random_hyperparams(self) -> Dict[str, Any]:
        """Sample a random hyperparameter set from ranges."""
        hyperparams = {'horizon': 1}  # Default horizon
        for key, value_range in self.hyperparam_ranges.items():
            if isinstance(value_range, (list, tuple)):
                if len(value_range) == 2 and all(isinstance(x, (int, float)) for x in value_range):
                    # Numeric range (min, max)
                    min_val, max_val = value_range
                    if isinstance(min_val, int):
                        hyperparams[key] = np.random.randint(min_val, max_val + 1)
                    else:
                        hyperparams[key] = np.random.uniform(min_val, max_val)
                else:
                    # Categorical - pick one
                    hyperparams[key] = np.random.choice(value_range)
            else:
                # Single value? just use it
                hyperparams[key] = value_range
        return hyperparams

    def _initialize_seeded(self, seed_hyperparams: List[Dict]):
        """Initialize with seeded individuals and fill rest with random."""
        # Add seeded individuals
        for i, hyperparams in enumerate(seed_hyperparams[:self.population_size]):
            individual = Individual(
                model_class=self.model_class,
                hyperparams=hyperparams,
                name=f"gen{self.generation}_seed{i}"
            )
            self.individuals.append(individual)

        # Fill remaining slots with random
        remaining = self.population_size - len(self.individuals)
        for i in range(remaining):
            hyperparams = self._sample_random_hyperparams()
            individual = Individual(
                model_class=self.model_class,
                hyperparams=hyperparams,
                name=f"gen{self.generation}_rand{i}"
            )
            self.individuals.append(individual)

    def evaluate_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> None:
        """
        Evaluate fitness of all unevaluated individuals.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
        """
        for individual in self.individuals:
            if not individual.is_evaluated:
                individual.evaluate(X_train, y_train, X_val, y_val, metric=self.fitness_metric)

    def get_fitness_scores(self) -> np.ndarray:
        """Return fitness scores of all evaluated individuals."""
        scores = []
        for ind in self.individuals:
            if ind.is_evaluated:
                scores.append(ind.fitness if ind.fitness is not None else -np.inf)
        return np.array(scores)

    def select_parents(self, method: str = 'tournament', tournament_size: int = 3) -> List[Individual]:
        """
        Select individuals for breeding.

        Args:
            method: 'tournament' or 'roulette'
            tournament_size: For tournament selection

        Returns:
            List of selected individuals (with replacement)
        """
        # Filter to evaluated individuals
        evaluated = [ind for ind in self.individuals if ind.is_evaluated]
        if not evaluated:
            raise RuntimeError("Cannot select parents - population not evaluated")

        if method == 'tournament':
            return self._tournament_selection(evaluated, tournament_size)
        elif method == 'roulette':
            return self._roulette_wheel_selection(evaluated)
        else:
            raise ValueError(f"Unknown selection method: {method}")

    def _tournament_selection(self, evaluated: List[Individual], tournament_size: int) -> List[Individual]:
        """Tournament selection: pick k random, take best."""
        selected = []
        for _ in range(len(evaluated)):  # Select same count as evaluated pool
            contestants = np.random.choice(evaluated, size=tournament_size, replace=True)
            best = max(contestants, key=lambda ind: ind.fitness if ind.fitness is not None else -np.inf)
            selected.append(best)
        return selected

    def _roulette_wheel_selection(self, evaluated: List[Individual]) -> List[Individual]:
        """Roulette wheel selection based on fitness."""
        # Get positive fitness only (shift if negative)
        fitnesses = np.array([max(ind.fitness or -np.inf, 0) for ind in evaluated])
        if fitnesses.sum() == 0:
            # All zero/negative, uniform selection
            probs = np.ones(len(evaluated)) / len(evaluated)
        else:
            probs = fitnesses / fitnesses.sum()

        selected = np.random.choice(
            evaluated,
            size=len(evaluated),
            replace=True,
            p=probs
        )
        return list(selected)

    def evolve(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        selection_method: str = 'tournament',
        tournament_size: int = 3,
        mutation_rate: float = 0.2,
        mutation_scale: float = 0.2,
        crossover_rate: float = 0.5,
        preserve_elites: bool = True
    ) -> None:
        """
        Perform one generation of evolution.

        Steps:
        1. Evaluate new individuals (if any unevaluated)
        2. Select parents
        3. Create offspring via crossover + mutation
        4. Apply elitism (preserve best individuals)

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data for fitness
            selection_method: 'tournament' or 'roulette'
            tournament_size: Tournament size (if tournament selection)
            mutation_rate: Probability of mutating each hyperparameter
            mutation_scale: Perturbation scale for numeric params
            crossover_rate: Probability of taking param from parent2
            preserve_elites: Number of top individuals to preserve
        """
        # Evaluate all individuals (only those not already evaluated)
        self.evaluate_all(X_train, y_train, X_val, y_val)

        # Track generation stats
        fitness_scores = self.get_fitness_scores()
        if len(fitness_scores) > 0:
            mean_fitness = float(np.nanmean(fitness_scores))
            best_fitness = float(np.nanmax(fitness_scores))
            best_idx = np.nanargmax(fitness_scores)
            best_individual = self.individuals[best_idx]

            self.fitness_history.append(best_fitness)
            self.mean_fitness_history.append(mean_fitness)
            self.best_individual_history.append(best_individual)

            print(f"Generation {self.generation}: best={best_fitness:.4f}, mean={mean_fitness:.4f}")

        # Elitism: preserve top individuals
        if preserve_elites and self.elite_count > 0:
            evaluated = [ind for ind in self.individuals if ind.is_evaluated]
            evaluated.sort(key=lambda ind: ind.fitness if ind.fitness is not None else -np.inf, reverse=True)
            elites = evaluated[:self.elite_count]
        else:
            elites = []

        # Select parents from evaluated pool
        parents = self.select_parents(method=selection_method, tournament_size=tournament_size)

        # Create offspring to fill population
        offspring = []
        while len(offspring) + len(elites) < self.population_size:
            # Select two random parents
            if len(parents) >= 2:
                p1, p2 = np.random.choice(parents, size=2, replace=False)
                # Crossover
                child = Individual.crossover(p1, p2, crossover_rate=crossover_rate)
                # Mutate
                child = child.mutate(mutation_rate=mutation_rate, mutation_scale=mutation_scale)
                offspring.append(child)
            else:
                # Fallback: clone random parent
                parent = np.random.choice(parents)
                child = Individual(
                    model_class=parent.model_class,
                    hyperparams=parent.hyperparams.copy(),
                    name=f"gen{self.generation+1}_clone_{len(offspring)}"
                )
                child.feature_columns = parent.feature_columns
                child = child.mutate(mutation_rate=mutation_rate, mutation_scale=mutation_scale)
                offspring.append(child)

        # Replace population: elites + offspring
        self.individuals = elites + offspring[:self.population_size - len(elites)]
        self.generation += 1

    def get_best_individual(self) -> Optional[Individual]:
        """Return the best individual across all generations."""
        if not self.best_individual_history:
            return None
        return max(self.best_individual_history, key=lambda ind: ind.fitness if ind.fitness is not None else -np.inf)

    def get_population_stats(self) -> Dict[str, float]:
        """Get current population statistics."""
        evaluated = [ind for ind in self.individuals if ind.is_evaluated]
        if not evaluated:
            return {'generation': self.generation, 'evaluated': 0}

        fitnesses = [ind.fitness for ind in evaluated if ind.fitness is not None]
        if not fitnesses:
            return {'generation': self.generation, 'evaluated': len(evaluated)}

        return {
            'generation': self.generation,
            'evaluated': len(evaluated),
            'best': float(max(fitnesses)),
            'worst': float(min(fitnesses)),
            'mean': float(np.mean(fitnesses)),
            'std': float(np.std(fitnesses)),
        }

    def __len__(self) -> int:
        return len(self.individuals)

    def __repr__(self) -> str:
        stats = self.get_population_stats()
        return f"Population(size={self.population_size}, generation={self.generation}, evaluated={stats.get('evaluated', 0)}, best={stats.get('best', 'N/A')})"
