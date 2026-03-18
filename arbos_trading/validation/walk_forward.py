"""
Walk-forward validation with expanding window.
"""

import pandas as pd
import numpy as np
from typing import Generator, Tuple, Dict
from datetime import timedelta


class WalkForwardValidator:
    """
    Strict temporal walk-forward validation.

    Splits data into expanding training window and fixed validation window.
    Each fold tests on a future period unseen during training.
    """

    def __init__(
        self,
        train_days: int = 30,
        valid_days: int = 7,
        step_days: int = 7,
        min_train_samples: int = 1000
    ):
        self.train_days = train_days
        self.valid_days = valid_days
        self.step_days = step_days
        self.min_train_samples = min_train_samples

    def generate_folds(
        self,
        df: pd.DataFrame,
        time_col: str = "open_time"
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, pd.Index], None, None]:
        """
        Generate train/validation splits for walk-forward validation.

        Yields:
            train_df: training data (expanding window)
            valid_df: validation data (fixed window)
            test_index: index of test samples (for predictions)
        """
        df = df.sort_values(time_col).reset_index(drop=True)
        times = df[time_col]

        start_idx = 0
        end_idx = 0
        total_rows = len(df)

        while True:
            # Define train window: [start_idx, end_idx)
            train_end = end_idx
            train_start = start_idx

            # Define valid window: [end_idx, end_idx + valid_rows)
            valid_start = end_idx
            valid_end = end_idx + int(self.valid_days * 24 * 60 / 15)  # 15-min candles per day

            if valid_end >= total_rows:
                break

            # Check minimum training samples
            if train_end - train_start >= self.min_train_samples:
                train_df = df.iloc[train_start:train_end].copy()
                valid_df = df.iloc[valid_start:valid_end].copy()

                if len(valid_df) > 0:
                    yield train_df, valid_df, valid_df.index

            # Step forward
            step_rows = int(self.step_days * 24 * 60 / 15)
            end_idx += step_rows

            # Optionally also move start forward for sliding window
            # For expanding, start_idx stays at 0

    def evaluate_folds(
        self,
        df: pd.DataFrame,
        model_builder,
        feature_fn,
        target_col: str = "target",
        time_col: str = "open_time"
    ) -> Dict[str, float]:
        """
        Run walk-forward evaluation.

        Args:
            df: full dataset
            model_builder: function that returns a trained model given X_train, y_train
            feature_fn: function to extract features from df
            target_col: name of target column
            time_col: name of time column

        Returns:
            dict of aggregated metrics across all folds
        """
        all_metrics = []

        for train_df, valid_df, test_idx in self.generate_folds(df):
            X_train = feature_fn(train_df)
            y_train = train_df[target_col]

            X_valid = feature_fn(valid_df)
            y_valid = valid_df[target_col]

            if len(X_valid) == 0:
                continue

            # Train model
            model = model_builder()
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                continue

            # Predict
            proba = model.predict_proba(X_valid)
            pred = model.predict(X_valid, threshold=0.5)

            # Metrics
            accuracy = (pred == y_valid).mean()
            all_metrics.append({
                "accuracy": accuracy,
                "n_samples": len(X_valid),
                "fold": len(all_metrics)
            })

        if not all_metrics:
            return {"accuracy": 0.0, "n_folds": 0}

        # Weighted average by fold size
        total_samples = sum(m["n_samples"] for m in all_metrics)
        weighted_accuracy = sum(m["accuracy"] * m["n_samples"] for m in all_metrics) / total_samples

        return {
            "accuracy": weighted_accuracy,
            "n_folds": len(all_metrics),
            "total_valid_samples": total_samples
        }
