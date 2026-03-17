"""
Individual - A single model instance in the population.

Each individual represents a specific model configuration (hyperparameters)
with associated fitness score from evaluation.
"""

from typing import Dict, Any, Optional, Type
import numpy as np
import pandas as pd
from models.base import TradingModel


class Individual:
    """
    An individual in the evolutionary population.

    Wraps a TradingModel with hyperparameters and fitness metrics.
    Supports genetic operations: mutation, crossover (on hyperparameters).
    """

    def __init__(
        self,
        model_class: Type[TradingModel],
        hyperparams: Dict[str, Any],
        name: Optional[str] = None,
        feature_columns: Optional[list] = None,
    ):
        """
        Initialize an individual.

        Args:
            model_class: The TradingModel class to instantiate
            hyperparams: Dictionary of hyperparameters specific to the model
            name: Unique identifier (generated if None)
            feature_columns: List of feature column names (for model to identify features)
        """
        self.model_class = model_class
        self.hyperparams = hyperparams.copy()
        self.name = name or f"{model_class.__name__}_{id(self) & 0xffffffff:x}"
        self.feature_columns = feature_columns

        # State (set during evolution)
        self.model: Optional[TradingModel] = None
        self.fitness: Optional[float] = None
        self.fitness_details: Optional[Dict[str, float]] = None
        self.is_evaluated: bool = False

    def create_model(self) -> TradingModel:
        """Instantiate the model with current hyperparameters."""
        self.model = self.model_class(
            name=self.name,
            horizon=self.hyperparams.get('horizon', 1)
        )
        # Apply hyperparameters to model instance by overriding existing attributes
        for key, value in self.hyperparams.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
        # Store hyperparams on model for later reference
        self.model.hyperparams = self.hyperparams
        # Set feature columns if provided
        if self.feature_columns is not None:
            self.model.feature_columns = self.feature_columns
        return self.model

    def evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = 'sharpe_ratio'
    ) -> float:
        """
        Train the model and compute fitness on validation set.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data (for fitness calculation)
            metric: Fitness metric ('sharpe_ratio', 'total_return', 'calmar_ratio', 'directional_accuracy')

        Returns:
            Fitness score (higher is better)
        """
        if self.model is None:
            self.create_model()

        # Train
        train_metrics = self.model.train(X_train, y_train, **self.hyperparams.get('train_params', {}))

        # Get predictions on validation
        y_pred = self.model.predict(X_val)

        # Calculate fitness metric
        if metric == 'sharpe_ratio':
            # Convert predictions to trading signals, compute returns
            fitness = self._compute_sharpe_fitness(y_val, y_pred)
        elif metric == 'total_return':
            fitness = self._compute_total_return_fitness(y_val, y_pred)
        elif metric == 'calmar_ratio':
            fitness = self._compute_calmar_fitness(y_val, y_pred)
        elif metric == 'directional_accuracy':
            fitness = self._compute_directional_accuracy_fitness(y_val, y_pred)
        else:
            raise ValueError(f"Unknown fitness metric: {metric}")

        self.fitness = fitness
        self.fitness_details = {
            'train_accuracy': train_metrics.get('accuracy', 0.0),
            'train_mse': train_metrics.get('mse', 0.0),
            'validation_metric': fitness,
        }
        self.is_evaluated = True

        return fitness

    def _compute_sharpe_fitness(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Sharpe ratio of simple strategy based on predictions with threshold."""
        threshold = self.hyperparams.get('signal_threshold', 0.0)
        # Determine position: long (1) if pred >= +threshold, short (-1) if pred <= -threshold, else neutral (0)
        positions = np.where(np.abs(y_pred) >= threshold, np.sign(y_pred), 0)
        returns = positions * y_true
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        sharpe = returns.mean() / returns.std()
        # Annualize (assuming hourly data: 365*24) - but actually depends on timeframe
        # We'll keep the same annualization as before (for 1h data) but might need adjustment
        sharpe_annual = sharpe * np.sqrt(365 * 24)
        return float(sharpe_annual)

    def _compute_total_return_fitness(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute total return with signal threshold."""
        threshold = self.hyperparams.get('signal_threshold', 0.0)
        positions = np.where(np.abs(y_pred) >= threshold, np.sign(y_pred), 0)
        returns = positions * y_true
        total_return = returns.sum()
        return float(total_return)

    def _compute_calmar_fitness(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Calmar ratio (return / max drawdown) with threshold."""
        threshold = self.hyperparams.get('signal_threshold', 0.0)
        positions = np.where(np.abs(y_pred) >= threshold, np.sign(y_pred), 0)
        returns = positions * y_true

        # Compute equity curve
        equity = np.cumprod(1 + returns)
        if len(equity) == 0 or (equity == 0).any():
            return 0.0

        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_dd = abs(drawdown.min())

        if max_dd == 0:
            return np.inf if returns.sum() > 0 else 0.0

        calmar = returns.mean() / max_dd
        return float(calmar)

    def _compute_directional_accuracy_fitness(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute directional accuracy as fitness.

        Only considers predictions that exceed a confidence threshold (signal_threshold).
        This ensures that only actionable, high-conviction trades are counted.
        Weak predictions are ignored (not penalized), making the metric more meaningful.

        Returns:
            Accuracy in range [0, 1]. Higher is better.
        """
        threshold = self.hyperparams.get('signal_threshold', 0.0)

        # Mask: only consider predictions with absolute value >= threshold
        mask = np.abs(y_pred) >= threshold
        if not np.any(mask):
            # No predictions meet threshold - worst score (could also be 0.0)
            return 0.0

        y_true_selected = y_true[mask]
        y_pred_selected = y_pred[mask]

        y_true_dir = np.sign(y_true_selected)
        y_pred_dir = np.sign(y_pred_selected)

        correct = (y_true_dir == y_pred_dir)
        accuracy = correct.mean()
        return float(accuracy)

    def mutate(self, mutation_rate: float = 0.1, mutation_scale: float = 0.2) -> 'Individual':
        """
        Create a mutated copy of this individual.

        Args:
            mutation_rate: Probability of mutating each hyperparameter
            mutation_scale: Magnitude of perturbation for numeric params

        Returns:
            New Individual with mutated hyperparameters
        """
        new_hyperparams = self.hyperparams.copy()

        for key, value in new_hyperparams.items():
            if isinstance(value, (int, float)) and np.random.random() < mutation_rate:
                # Numeric mutation: add Gaussian noise
                if isinstance(value, int):
                    # Perturb integer by small amount and round
                    perturbation = int(np.random.normal(0, mutation_scale * max(1, abs(value))))
                    new_hyperparams[key] = max(1, int(value + perturbation))
                else:
                    # Float perturbation
                    perturbation = np.random.normal(0, mutation_scale * abs(value))
                    new_hyperparams[key] = float(value + perturbation)
            elif isinstance(value, (list, tuple)) and np.random.random() < mutation_rate:
                # List mutation: randomly add/remove/replace
                new_list = list(value)
                if len(new_list) > 0 and np.random.random() < 0.5:
                    # Replace one element
                    idx = np.random.randint(len(new_list))
                    if all(isinstance(x, (int, float)) for x in new_list):
                        # Numeric list - perturb one element
                        old_val = new_list[idx]
                        new_list[idx] = old_val * (1 + np.random.normal(0, mutation_scale))
                    else:
                        # Random choice from original pool? For now, skip
                        pass
                new_hyperparams[key] = tuple(new_list) if isinstance(value, tuple) else new_list
            # String/categorical: skip (would need predefined values)

        child = Individual(
            model_class=self.model_class,
            hyperparams=new_hyperparams,
            name=f"{self.model_class.__name__}_mut_{id(self) & 0xffffffff:x}"
        )
        # Propagate feature_columns
        child.feature_columns = self.feature_columns
        return child

    @classmethod
    def crossover(
        cls,
        parent1: 'Individual',
        parent2: 'Individual',
        crossover_rate: float = 0.5
    ) -> 'Individual':
        """
        Create child by crossing over hyperparameters from two parents.

        Args:
            parent1, parent2: Two parent individuals
            crossover_rate: Probability of taking each param from parent2

        Returns:
            New child Individual
        """
        if parent1.model_class != parent2.model_class:
            raise ValueError("Parents must be same model class for crossover")

        child_hyperparams = {}
        all_keys = set(parent1.hyperparams.keys()) | set(parent2.hyperparams.keys())

        for key in all_keys:
            if key in parent1.hyperparams and key in parent2.hyperparams:
                # Both have the key - uniform crossover
                if np.random.random() < crossover_rate:
                    child_hyperparams[key] = parent2.hyperparams[key].copy() if hasattr(parent2.hyperparams[key], 'copy') else parent2.hyperparams[key]
                else:
                    child_hyperparams[key] = parent1.hyperparams[key].copy() if hasattr(parent1.hyperparams[key], 'copy') else parent1.hyperparams[key]
            elif key in parent1.hyperparams:
                # Only parent1 has it
                child_hyperparams[key] = parent1.hyperparams[key].copy() if hasattr(parent1.hyperparams[key], 'copy') else parent1.hyperparams[key]
            else:
                # Only parent2 has it
                child_hyperparams[key] = parent2.hyperparams[key].copy() if hasattr(parent2.hyperparams[key], 'copy') else parent2.hyperparams[key]

        child = Individual(
            model_class=parent1.model_class,
            hyperparams=child_hyperparams,
            name=f"{parent1.model_class.__name__}_crossover_{id(parent1) & 0xffffffff:x}_{id(parent2) & 0xffffffff:x}"
        )
        # Propagate feature_columns (both parents should have same)
        child.feature_columns = parent1.feature_columns or parent2.feature_columns
        return child

    def to_dict(self) -> Dict[str, Any]:
        """Serialize individual to dictionary."""
        return {
            'name': self.name,
            'model_class': self.model_class.__name__,
            'hyperparams': self.hyperparams,
            'fitness': self.fitness,
            'fitness_details': self.fitness_details,
            'is_evaluated': self.is_evaluated,
        }

    def __repr__(self) -> str:
        fitness_str = f"{self.fitness:.4f}" if self.fitness is not None else "None"
        return f"Individual({self.name}, fitness={fitness_str}, hyperparams={self.hyperparams})"
