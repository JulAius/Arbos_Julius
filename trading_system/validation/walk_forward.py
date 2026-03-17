"""
Walk-Forward Validation - Strict time-series cross validation.

Implements walk-forward analysis (WFA) with expanding or rolling window
to prevent lookahead bias and evaluate model performance on out-of-sample data.
"""

import pandas as pd
import numpy as np
from typing import Callable, Dict, Any, Optional, Tuple
from datetime import timedelta
from dataclasses import dataclass, field


@dataclass
class WalkForwardResult:
    """Results from walk-forward validation."""
    fold_metrics: list = field(default_factory=list)
    predictions: pd.Series = field(default=None)
    actuals: pd.Series = field(default=None)
    fold_boundaries: list = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        """Aggregate metrics across all folds."""
        if not self.fold_metrics:
            return {}

        # Average metrics across folds
        avg_metrics = {}
        keys = self.fold_metrics[0].keys() if self.fold_metrics else []

        for key in keys:
            values = [m[key] for m in self.fold_metrics if key in m]
            if values:
                avg_metrics[f'{key}_mean'] = np.mean(values)
                avg_metrics[f'{key}_std'] = np.std(values)
                avg_metrics[f'{key}_min'] = np.min(values)
                avg_metrics[f'{key}_max'] = np.max(values)

        return avg_metrics


class WalkForwardValidator:
    """
    Walk-forward validation for time-series models.

    Splits data chronologically into training and validation sets,
    rolling forward through time to simulate live trading.
    """

    def __init__(
        self,
        train_window: int = 1000,       # Initial training size (hours)
        validation_window: int = 168,   # Validation size (1 week)
        step_size: int = 24,            # How much to advance each fold (1 day)
        expanding: bool = True          # If True, training window grows; if False, rolling
    ):
        """
        Initialize walk-forward validator.

        Args:
            train_window: Initial number of periods for training
            validation_window: Number of periods for validation in each fold
            step_size: Number of periods to advance for each fold
            expanding: If True, training window expands; if False, it rolls forward
        """
        self.train_window = train_window
        self.validation_window = validation_window
        self.step_size = step_size
        self.expanding = expanding

    def split(
        self,
        data: pd.DataFrame,
        start_idx: int = 0
    ) -> list[Tuple[pd.Index, pd.Index]]:
        """
        Generate train/validation splits for walk-forward validation.

        Args:
            data: DataFrame with datetime index
            start_idx: Starting index (for skipping initial warm-up period)

        Returns:
            List of (train_idx, val_idx) tuples
        """
        n = len(data)
        splits = []

        # Initial train window
        train_start = start_idx
        train_end = start_idx + self.train_window

        while train_end + self.validation_window <= n:
            train_idx = slice(train_start, train_end)

            if self.expanding:
                # Expanding window: train from start to current
                val_start = train_end
            else:
                # Rolling window: train window rolls forward
                train_start = train_end - self.train_window
                train_idx = slice(train_start, train_end)
                val_start = train_end

            val_end = val_start + self.validation_window
            val_idx = slice(val_start, val_end)

            splits.append((data.index[train_idx], data.index[val_idx]))

            # Advance
            train_end += self.step_size

        return splits

    def evaluate_model(
        self,
        data: pd.DataFrame,
        model: Any,
        train_func: Callable,
        predict_func: Callable,
        feature_columns: list,
        target_column: str = 'close_return',
        metric_func: Callable = None,
        **train_kwargs
    ) -> WalkForwardResult:
        """
        Run walk-forward validation on a model.

        Args:
            data: Full dataset with features and target
            model: Model instance (must have fit/predict or train_func/predict_func)
            train_func: Function to train model (X_train, y_train) -> metrics
            predict_func: Function to generate predictions (X_val) -> predictions
            feature_columns: List of feature column names
            target_column: Name of target column
            metric_func: Function to compute metrics (actuals, predictions) -> dict
            **train_kwargs: Additional arguments passed to train_func

        Returns:
            WalkForwardResult with fold results
        """
        if metric_func is None:
            from .metrics import compute_all_metrics
            metric_func = compute_all_metrics

        splits = self.split(data)
        result = WalkForwardResult()
        all_predictions = []
        all_actuals = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_data = data.loc[train_idx]
            val_data = data.loc[val_idx]

            # Prepare features and target
            X_train = train_data[feature_columns].values
            y_train = train_data[target_column].values

            X_val = val_data[feature_columns].values
            y_val = val_data[target_column].values

            # Reshape for sequence models if needed (add time dimension)
            if hasattr(model, 'predict') and getattr(model, 'needs_sequences', False):
                # For now, skip sequence reshaping - models will handle internally
                pass

            # Train
            train_metrics = train_func(model, X_train, y_train, **train_kwargs)

            # Predict
            predictions = predict_func(model, X_val)

            # Compute validation metrics
            val_metrics = metric_func(
                returns=pd.Series(y_val, index=val_idx),
                equity_curve=pd.Series((1 + y_val).cumprod(), index=val_idx)
            )

            # Combine metrics
            fold_metrics = {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
            result.fold_metrics.append(fold_metrics)
            result.fold_boundaries.append((train_idx, val_idx))

            # Collect predictions
            all_predictions.extend(predictions)
            all_actuals.extend(y_val)

        result.predictions = pd.Series(all_predictions)
        result.actuals = pd.Series(all_actuals)

        return result


def walk_forward_validation(
    data: pd.DataFrame,
    model: Any,
    feature_columns: list,
    target_column: str = 'close_return',
    train_window: int = 1000,
    validation_window: int = 168,
    step_size: int = 24,
    expanding: bool = True,
    min_train_size: int = 100
) -> WalkForwardResult:
    """
    Convenience function for walk-forward validation.

    Example:
        validator = WalkForwardValidator(train_window=1000, validation_window=168)
        result = validator.evaluate_model(
            data=df,
            model=my_model,
            train_func=lambda m, X, y: m.train(X, y),
            predict_func=lambda m, X: m.predict(X),
            feature_columns=feature_cols
        )
    """
    # Adjust train window to be at least min_train_size
    if train_window < min_train_size and len(data) >= min_train_size:
        train_window = min_train_size

    validator = WalkForwardValidator(
        train_window=train_window,
        validation_window=validation_window,
        step_size=step_size,
        expanding=expanding
    )

    # Default train/predict functions using model methods
    def default_train(m, X, y, **kwargs):
        return m.train(X, y, **kwargs)

    def default_predict(m, X):
        return m.predict(X)

    result = validator.evaluate_model(
        data=data,
        model=model,
        train_func=default_train,
        predict_func=default_predict,
        feature_columns=feature_columns,
        target_column=target_column,
        **{}
    )

    return result
