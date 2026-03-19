"""
Base model interface and concrete implementations.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import joblib
from pathlib import Path


class BaseModel(ABC):
    """Abstract base class for all models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Train the model."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of UP (class 1)."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary prediction."""
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """Get model hyperparameters."""
        pass

    @abstractmethod
    def set_params(self, params: dict):
        """Set model hyperparameters."""
        pass

    def save(self, path: str):
        """Save model to disk."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """Load model from disk."""
        return joblib.load(path)


class LogisticModel(BaseModel):
    """Logistic regression using sklearn."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42, input_dim: int = None, **kwargs):
        from sklearn.linear_model import LogisticRegression
        from ..config import ModelConfig
        solver = getattr(ModelConfig, 'LOGISTIC_SOLVER', 'saga')
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state, solver=solver)
        self._params = {"C": C, "max_iter": max_iter, "random_state": random_state}
        self.is_fitted = False
        self.scaler = None  # Will hold StandardScaler

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticModel":
        from sklearn.preprocessing import StandardScaler
        # Scale features for faster convergence
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_params(self) -> dict:
        return self.model.get_params()

    def set_params(self, params: dict):
        self.model.set_params(**params)
        self._params.update(params)
        self.is_fitted = False  # need to retrain


class RandomForestModel(BaseModel):
    """Random Forest classifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        n_jobs: int = -1,
        input_dim: int = None,
        **kwargs
    ):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self._params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
            "n_jobs": n_jobs
        }
        self.is_fitted = False
        self._calibrated = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        from ..config import ModelConfig
        cal_split = getattr(ModelConfig, 'CALIBRATION_SPLIT', 0.2)
        n_cal = int(len(X) * cal_split)
        use_cal = getattr(ModelConfig, 'USE_CALIBRATION', False) and n_cal >= 60
        if use_cal:
            from sklearn.calibration import CalibratedClassifierCV
            method = getattr(ModelConfig, 'CALIBRATION_METHOD', 'sigmoid')
            X_base, y_base = X.iloc[:-n_cal], y.iloc[:-n_cal]
            X_cal, y_cal = X.iloc[-n_cal:], y.iloc[-n_cal:]
            self.model.fit(X_base, y_base)
            from sklearn.frozen import FrozenEstimator
            cal = CalibratedClassifierCV(FrozenEstimator(self.model), method=method)
            cal.fit(X_cal, y_cal)
            self._calibrated = cal
        else:
            self.model.fit(X, y)
            self._calibrated = None
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        if self._calibrated is not None:
            return self._calibrated.predict_proba(X)[:, 1]
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_params(self) -> dict:
        return self.model.get_params()

    def set_params(self, params: dict):
        self.model.set_params(**params)
        self._params.update(params)
        self.is_fitted = False
        self._calibrated = None


class GradientBoostingModel(BaseModel):
    """Gradient Boosting classifier using sklearn's HistGradientBoostingClassifier (fast)."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int = 42,
        input_dim: int = None,
        **kwargs
    ):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.model = HistGradientBoostingClassifier(
            max_iter=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            verbose=0
        )
        self._params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "random_state": random_state
        }
        self.is_fitted = False
        self._calibrated = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GradientBoostingModel":
        from ..config import ModelConfig
        cal_split = getattr(ModelConfig, 'CALIBRATION_SPLIT', 0.2)
        n_cal = int(len(X) * cal_split)
        use_cal = getattr(ModelConfig, 'USE_CALIBRATION', False) and n_cal >= 60
        if use_cal:
            from sklearn.calibration import CalibratedClassifierCV
            method = getattr(ModelConfig, 'CALIBRATION_METHOD', 'sigmoid')
            X_base, y_base = X.iloc[:-n_cal], y.iloc[:-n_cal]
            X_cal, y_cal = X.iloc[-n_cal:], y.iloc[-n_cal:]
            self.model.fit(X_base, y_base)
            from sklearn.frozen import FrozenEstimator
            cal = CalibratedClassifierCV(FrozenEstimator(self.model), method=method)
            cal.fit(X_cal, y_cal)
            self._calibrated = cal
        else:
            self.model.fit(X, y)
            self._calibrated = None
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        if self._calibrated is not None:
            return self._calibrated.predict_proba(X)[:, 1]
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_params(self) -> dict:
        return self._params.copy()

    def set_params(self, params: dict):
        self._params.update(params)
        # Map n_estimators -> max_iter for HistGBM
        hist_params = {}
        for k, v in params.items():
            if k == "n_estimators":
                hist_params["max_iter"] = v
            elif k != "random_state":  # random_state not settable post-init in HistGBM
                hist_params[k] = v
        if hist_params:
            self.model.set_params(**hist_params)
        self.is_fitted = False
        self._calibrated = None


class SimpleNNModel(BaseModel):
    """Simple feedforward neural network using PyTorch or sklearn MLP."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list = [64, 32],
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        random_state: int = 42
    ):
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self._params = {
            "input_dim": input_dim,
            "hidden_sizes": hidden_sizes,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "random_state": random_state
        }

        self.model = None
        self.is_fitted = False

    def _build_model(self):
        """Build the neural network."""
        try:
            import torch
            import torch.nn as nn

            class Net(nn.Module):
                def __init__(self, input_dim, hidden_sizes, dropout):
                    super().__init__()
                    layers = []
                    prev_dim = input_dim
                    for h in hidden_sizes:
                        layers.append(nn.Linear(prev_dim, h))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(dropout))
                        prev_dim = h
                    layers.append(nn.Linear(prev_dim, 1))
                    self.net = nn.Sequential(*layers)

                def forward(self, x):
                    return torch.sigmoid(self.net(x))

            self.model = Net(self.input_dim, self.hidden_sizes, self.dropout)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.criterion = nn.BCELoss()
            self.use_torch = True
        except ImportError:
            from sklearn.neural_network import MLPClassifier
            self.model = MLPClassifier(
                hidden_layer_sizes=self.hidden_sizes,
                activation="relu",
                solver="adam",
                alpha=self.dropout,  # L2 penalty as proxy for dropout
                learning_rate_init=self.learning_rate,
                max_iter=self.epochs,
                batch_size=self.batch_size,
                random_state=self.random_state,
                verbose=False
            )
            self.use_torch = False
            self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SimpleNNModel":
        if self.model is None:
            self._build_model()

        X_arr = X.values.astype(np.float32)
        y_arr = y.values.astype(np.float32)

        if self.use_torch:
            import torch
            from torch.utils.data import TensorDataset, DataLoader

            X_tensor = torch.FloatTensor(X_arr)
            y_tensor = torch.FloatTensor(y_arr).unsqueeze(1)
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            self.model.train()
            for epoch in range(self.epochs):
                for X_batch, y_batch in loader:
                    self.optimizer.zero_grad()
                    preds = self.model(X_batch)
                    loss = self.criterion(preds, y_batch)
                    loss.backward()
                    self.optimizer.step()
        else:
            self.model.fit(X, y)

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        X_arr = X.values.astype(np.float32)

        if self.use_torch:
            import torch
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_arr)
                probs = self.model(X_tensor).numpy().flatten()
            return probs
        else:
            return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_params(self) -> dict:
        return self._params.copy()

    def set_params(self, params: dict):
        self._params.update(params)
        # Rebuild model if architecture params change
        hidden_changed = "hidden_sizes" in params and params["hidden_sizes"] != self.hidden_sizes
        input_changed = "input_dim" in params and params["input_dim"] != self.input_dim
        if hidden_changed or input_changed:
            self.model = None
            self.is_fitted = False
        for k, v in params.items():
            setattr(self, k, v)
        # For sklearn, set_params also updates the sklearn model
        if not self.use_torch and self.model is not None:
            self.model.set_params(**params)


class ExtraTreesModel(BaseModel):
    """Extra Trees classifier — high variance, fast, good diversity complement to RF."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_leaf: int = 5,
        random_state: int = 42,
        n_jobs: int = 1,
        input_dim: int = None,
        **kwargs
    ):
        from sklearn.ensemble import ExtraTreesClassifier
        self.model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self._params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
            "n_jobs": n_jobs
        }
        self.is_fitted = False
        self._calibrated = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ExtraTreesModel":
        from ..config import ModelConfig
        cal_split = getattr(ModelConfig, 'CALIBRATION_SPLIT', 0.2)
        n_cal = int(len(X) * cal_split)
        use_cal = getattr(ModelConfig, 'USE_CALIBRATION', False) and n_cal >= 60
        if use_cal:
            from sklearn.calibration import CalibratedClassifierCV
            method = getattr(ModelConfig, 'CALIBRATION_METHOD', 'sigmoid')
            X_base, y_base = X.iloc[:-n_cal], y.iloc[:-n_cal]
            X_cal, y_cal = X.iloc[-n_cal:], y.iloc[-n_cal:]
            self.model.fit(X_base, y_base)
            from sklearn.frozen import FrozenEstimator
            cal = CalibratedClassifierCV(FrozenEstimator(self.model), method=method)
            cal.fit(X_cal, y_cal)
            self._calibrated = cal
        else:
            self.model.fit(X, y)
            self._calibrated = None
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        if self._calibrated is not None:
            return self._calibrated.predict_proba(X)[:, 1]
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_params(self) -> dict:
        return self.model.get_params()

    def set_params(self, params: dict):
        self.model.set_params(**params)
        self._params.update(params)
        self.is_fitted = False
        self._calibrated = None


class LightGBMModel(BaseModel):
    """LightGBM classifier — fast gradient boosting with leaf-wise growth."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        min_child_samples: int = 20,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        random_state: int = 42,
        input_dim: int = None,
        **kwargs
    ):
        import lightgbm as lgb
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            verbose=-1,
            n_jobs=1,
            force_row_wise=True  # deterministic: disable parallel histogram building
        )
        self._params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
        }
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> "LightGBMModel":
        self.model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_params(self) -> dict:
        return self._params.copy()

    def set_params(self, params: dict):
        self._params.update(params)
        self.model.set_params(**{k: v for k, v in params.items() if k != "random_state"})
        self.is_fitted = False


# Factory function
MODEL_REGISTRY = {
    "logistic": LogisticModel,
    "random_forest": RandomForestModel,
    "gradient_boosting": GradientBoostingModel,
    "extra_trees": ExtraTreesModel,
    "simple_nn": SimpleNNModel,
    "lightgbm": LightGBMModel,
}

def create_model(model_type: str, **kwargs) -> BaseModel:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_REGISTRY[model_type](**kwargs)
