"""
Horizon ensemble: Integrate multi-horizon contextual signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from ..models.population import Individual, Population


class HorizonEnsemble:
    """
    Ensemble that uses multiple horizon-specific models as contextual features.

    For each horizon (1h, 4h, 8h, 12h, 24h), we train a model on resampled data
    and use its prediction as a feature for the main 15m model.
    """

    def __init__(self, horizons: List[str] = None, population: Population = None):
        self.horizons = horizons or ["1h", "4h", "8h", "12h", "24h"]
        self.population = population
        self.horizon_models: Dict[str, Population] = {}  # horizon -> population
        self.main_features = []  # list of feature names produced by this ensemble
        self.is_fitted = False

    def fit_horizon_models(
        self,
        horizon_data: Dict[str, pd.DataFrame],
        target_series: pd.Series = None
    ):
        """
        Train a separate model population for each horizon.

        Args:
            horizon_data: dict of {horizon: DataFrame} with features and 'target' column for that horizon
            target_series: ignored (for backward compatibility)
        """
        self.horizon_models = {}

        horizons = list(horizon_data.items())
        print(f"Fitting horizon ensembles for {len(horizons)} horizons", flush=True)
        for idx, (horizon, df) in enumerate(horizons):
            if df.empty:
                print(f"  Horizon {horizon}: empty, skipping", flush=True)
                continue

            print(f"  Horizon {horizon} ({idx+1}/{len(horizons)}): processing {len(df)} rows", flush=True)

            # Set index for alignment
            df_indexed = df.set_index("open_time")

            # Use the horizon-specific target from df['target']
            if 'target' not in df_indexed.columns:
                print(f"    WARNING: no 'target' column in horizon data; skipping", flush=True)
                continue
            y = df_indexed['target']

            # Prepare features (exclude time and price/volume and target)
            exclude_cols = ["open_time", "open", "high", "low", "close", "volume", "quote_volume", "target", "future_close"]
            feature_cols = [c for c in df_indexed.columns if c not in exclude_cols]
            X = df_indexed[feature_cols]

            # Drop rows where target is NaN
            valid_idx = y.notna()
            if not valid_idx.any():
                print(f"    WARNING: all targets NaN; skipping", flush=True)
                continue
            X = X[valid_idx]
            y = y[valid_idx]

            # Double-check: ensure no NaNs remain in y (some models are sensitive)
            if y.isna().any():
                # Drop any remaining NaNs just in case
                nan_count = y.isna().sum()
                print(f"    Warning: {nan_count} NaNs in y after filtering; dropping", flush=True)
                X = X[~y.isna()]
                y = y[~y.isna()]

            print(f"    Feature count: {len(feature_cols)}, training samples: {len(X)}", flush=True)

            # Create population (use only random_forest for speed and stability)
            pop = Population(
                population_size=5,  # small for horizon models
                model_types=["random_forest"],
                elitism_count=1
            )
            pop.initialize(input_dim=len(feature_cols), feature_names=feature_cols)

            # Quick training: fit all models
            print(f"    Training {len(pop.individuals)} models...", flush=True)
            self._train_population(pop, X, y)
            self.horizon_models[horizon] = pop
            print(f"    Completed horizon {horizon}", flush=True)

        self.is_fitted = True
        print("All horizon ensembles fitted", flush=True)

    def _train_population(self, population: Population, X: pd.DataFrame, y: pd.Series):
        """Fit all models in a population."""
        total = len(population.individuals)
        for i, ind in enumerate(population.individuals):
            try:
                print(f"[horizon]   Training model {i+1}/{total} ({ind.model_type})...", flush=True)
                ind.model.fit(X, y)
                print(f"[horizon]   Model {i+1} trained", flush=True)
            except Exception as e:
                print(f"[horizon]   Model {i+1} failed: {e}", flush=True)
                # Set very low fitness if model fails
                ind.fitness = -1000.0

    def generate_features(
        self,
        horizon_data: Dict[str, pd.DataFrame],
        base_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Generate horizon ensemble features: predictions from each horizon's best model.

        Returns:
            DataFrame with columns like: h_1h_pred, h_1h_proba, h_4h_pred, ...
        """
        if not self.is_fitted:
            raise RuntimeError("HorizonEnsemble must be fitted first")

        features = pd.DataFrame(index=base_index)

        for horizon, pop in self.horizon_models.items():
            best = pop.get_best(1)[0]

            # Get horizon data aligned to base index
            df_h = horizon_data[horizon].set_index("open_time")
            X_h = df_h.reindex(base_index)

            # Drop non-feature columns
            feature_cols = [c for c in X_h.columns if c not in ["open", "high", "low", "close", "volume", "quote_volume"]]
            X_h_features = X_h[feature_cols]

            # Forward fill any missing features (from resampling gaps)
            X_h_features = X_h_features.ffill()

            # Generate predictions
            try:
                proba = best.model.predict_proba(X_h_features)
                pred = best.model.predict(X_h_features, threshold=0.5)
                features[f"h_{horizon}_proba"] = proba
                features[f"h_{horizon}_pred"] = pred
            except Exception as e:
                # If prediction fails, fill with 0.5 (neutral)
                features[f"h_{horizon}_proba"] = 0.5
                features[f"h_{horizon}_pred"] = 0

        self.main_features = list(features.columns)
        return features

    def get_feature_names(self) -> List[str]:
        return self.main_features
