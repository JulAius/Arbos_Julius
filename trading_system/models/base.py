"""
Base Model - Abstract interface for all trading models.

Defines the contract that all models must follow: train, predict, save, load.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import os


class TradingModel(ABC):
    """
    Abstract base class for trading models.

    All trading models should inherit from this class and implement
    the required methods.
    """

    def __init__(self, name: str, horizon: int = 1):
        """
        Initialize the model.

        Args:
            name: Unique name for this model instance
            horizon: Prediction horizon in steps (1h, 4h, 8h, 12h, 24h, etc.)
        """
        self.name = name
        self.horizon = horizon
        self.is_trained = False

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the model on historical data.

        Args:
            X: Input features, shape (n_samples, lookback, n_features) or (n_samples, n_features)
            y: Target values, shape (n_samples,)
            **kwargs: Additional training parameters

        Returns:
            Dictionary with training metrics (loss, accuracy, etc.)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Input features, shape (n_samples, lookback, n_features) or (n_samples, n_features)

        Returns:
            Predictions, shape (n_samples,)
        """
        pass

    @abstractmethod
    def get_signal(self, recent_data: pd.DataFrame) -> float:
        """
        Generate a trading signal for the next step.

        Args:
            recent_data: Recent OHLCV + features data, at least `lookback` rows

        Returns:
            Signal value: positive for long, negative for short, 0 for neutral
            Typically in range [-1, 1] representing position sizing suggestion
        """
        pass

    def save(self, path: str) -> None:
        """
        Save model state to disk.

        Args:
            path: Directory path to save model artifacts
        """
        raise NotImplementedError("Save not implemented for this model")

    def load(self, path: str) -> None:
        """
        Load model state from disk.

        Args:
            path: Directory path to load model artifacts from
        """
        raise NotImplementedError("Load not implemented for this model")

    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            'name': self.name,
            'horizon': self.horizon,
            'is_trained': self.is_trained
        }


class BaselineMomentumModel(TradingModel):
    """
    Simple momentum-based trading strategy.

    This is a baseline model that uses price momentum over multiple
    timeframes to generate trading signals. It's designed to be fast
    to train and serve as a benchmark.
    """

    def __init__(
        self,
        name: str = "momentum_baseline",
        horizon: int = 1,
        momentum_periods: list = [6, 24, 168],
        volume_period: int = 24
    ):
        """
        Initialize momentum model.

        Args:
            name: Model name
            horizon: Prediction horizon
            momentum_periods: List of lookback periods for momentum calculation
            volume_period: Period for volume spike detection
        """
        super().__init__(name, horizon)
        self.momentum_periods = momentum_periods
        self.volume_period = volume_period
        self.feature_columns = None
        self.mom_weights = None

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the momentum model by learning optimal weights for each momentum period.

        The model uses simple linear regression or correlation to weight signals.
        """
        # Extract momentum features from X
        # X shape: (n_samples, lookback, n_features) or (n_samples, n_features)
        if X.ndim == 3:
            n_samples, lookback, n_features = X.shape
            # Flatten temporal dimension by taking last values of each feature
            X_flat = X[:, -1, :]  # Use most recent timestep
        else:
            X_flat = X

        # Find momentum feature columns (heuristic based on naming)
        momentum_cols = []
        for i, col in enumerate(self.feature_columns or []):
            if any(f'momentum_{p}h' in col for p in self.momentum_periods):
                momentum_cols.append(i)

        if not momentum_cols:
            # No precomputed momentum features; use raw price features
            # Assume first few columns are price-related
            momentum_cols = list(range(min(3, X_flat.shape[1])))

        X_mom = X_flat[:, momentum_cols]

        # Learn weights by minimizing MSE between weighted sum and target
        # Use simple linear regression (no intercept)
        try:
            # Normalize features
            X_norm = (X_mom - X_mom.mean(axis=0)) / (X_mom.std(axis=0) + 1e-8)

            # Compute correlation of each feature with target
            correlations = np.array([
                np.corrcoef(X_norm[:, i], y)[0, 1]
                for i in range(X_norm.shape[1])
            ])
            # Handle NaN correlations
            correlations = np.nan_to_num(correlations, nan=0.0)

            # Use absolute correlation as weight, sign determines direction
            weights = correlations / (np.abs(correlations).sum() + 1e-8)
            self.mom_weights = weights
        except Exception as e:
            print(f"Warning: training failed, using equal weights: {e}")
            self.mom_weights = np.ones(len(momentum_cols)) / len(momentum_cols) if momentum_cols else np.array([1.0])

        self.is_trained = True

        # Compute training metrics
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        accuracy = np.mean(np.sign(predictions) == np.sign(y))

        return {
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'accuracy': float(accuracy),
            'n_samples': len(y)
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict returns using weighted momentum signals.
        """
        if X.ndim == 3:
            X_flat = X[:, -1, :]  # Most recent timestep
        else:
            X_flat = X

        if self.mom_weights is None:
            self.mom_weights = np.array([1.0])

        # Extract momentum features
        momentum_cols = []
        for i, col in enumerate(self.feature_columns or []):
            if any(f'momentum_{p}h' in col for p in self.momentum_periods):
                momentum_cols.append(i)

        if not momentum_cols:
            momentum_cols = list(range(min(3, X_flat.shape[1])))

        X_mom = X_flat[:, momentum_cols]

        # Normalize each feature by its rolling average (simulate z-score)
        X_normalized = X_mom / (np.abs(X_mom.mean(axis=0, keepdims=True)) + 1e-8)
        X_normalized = np.clip(X_normalized, -3, 3)  # Clip extremes

        predictions = X_normalized @ self.mom_weights

        return predictions

    def get_signal(self, recent_data: pd.DataFrame) -> float:
        """
        Generate trading signal based on recent momentum.
        """
        if len(recent_data) < max(self.momentum_periods):
            return 0.0

        signals = []
        for period in self.momentum_periods:
            mom = recent_data['close'].pct_change(periods=period).iloc[-1]
            signals.append(mom)

        if self.mom_weights is not None and len(signals) == len(self.mom_weights):
            weighted_signal = sum(s * w for s, w in zip(signals, self.mom_weights))
        else:
            weighted_signal = np.mean(signals) if signals else 0.0

        # Scale signal to reasonable bounds and apply nonlinearity
        signal = np.tanh(weighted_signal * 5)

        return float(signal)

    def save(self, path: str) -> None:
        """Save model weights and parameters."""
        import json
        import os
        os.makedirs(path, exist_ok=True)

        params = {
            'name': self.name,
            'horizon': self.horizon,
            'momentum_periods': self.momentum_periods,
            'volume_period': self.volume_period,
            'is_trained': self.is_trained,
            'mom_weights': self.mom_weights.tolist() if self.mom_weights is not None else None
        }

        with open(os.path.join(path, 'model.json'), 'w') as f:
            json.dump(params, f, indent=2)

    def load(self, path: str) -> None:
        """Load model weights and parameters."""
        import json

        with open(os.path.join(path, 'model.json'), 'r') as f:
            params = json.load(f)

        self.name = params['name']
        self.horizon = params['horizon']
        self.momentum_periods = params['momentum_periods']
        self.volume_period = params['volume_period']
        self.is_trained = params['is_trained']
        self.mom_weights = np.array(params['mom_weights']) if params['mom_weights'] else None
