"""
LSTM Trading Model - Deep learning model for time series prediction.

Uses LSTM layers to capture sequential dependencies in market data.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import os

from .base import TradingModel


class LSTMModel(nn.Module):
    """Internal PyTorch LSTM model."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        lstm_out, (hn, cn) = self.lstm(x)
        # Use output from last time step
        out = self.fc(lstm_out[:, -1, :])
        return out


class LSTMTradingModel(TradingModel):
    """
    LSTM-based trading model with hyperparameter optimization support.

    Hyperparameters (to be tuned by evolution):
      - hidden_dim: size of LSTM hidden state (default 64)
      - num_layers: number of LSTM layers (default 2)
      - dropout: dropout rate (default 0.2)
      - lr: learning rate (default 1e-3)
      - epochs: training epochs (default 50)
    """

    def __init__(
        self,
        name: str = "lstm",
        horizon: int = 1,
        input_dim: int = None,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 50  # Increased for full training
    ):
        super().__init__(name, horizon)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs

        self.model: Optional[nn.Module] = None
        self.scaler_mean: Optional[torch.Tensor] = None
        self.scaler_std: Optional[torch.Tensor] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_columns = None  # To remember feature names if needed

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the LSTM model.

        Args:
            X: Input features, shape (n_samples, lookback, n_features) or (n_samples, n_features)
            y: Target values, shape (n_samples,)

        Returns:
            Dictionary with training metrics (loss, accuracy, etc.)
        """
        # Handle input shape
        if X.ndim == 2:
            # Assume missing lookback dimension, add it as 1
            X = X[:, np.newaxis, :]
        n_samples, seq_len, n_features = X.shape
        if self.input_dim is None:
            self.input_dim = n_features
        # Convert to torch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        # Normalize features (per-feature normalization)
        if self.scaler_mean is None or self.scaler_std is None:
            # Compute mean/std over the entire flattened dataset (across all timesteps)
            self.scaler_mean = X_tensor.mean(dim=(0, 1), keepdim=True)
            self.scaler_std = X_tensor.std(dim=(0, 1), keepdim=True)
            self.scaler_std[self.scaler_std == 0] = 1.0
        X_tensor = (X_tensor - self.scaler_mean) / self.scaler_std
        # Build model if needed
        if self.model is None:
            self.model = LSTMModel(
                self.input_dim,
                self.hidden_dim,
                self.num_layers,
                1,
                self.dropout
            ).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor).squeeze()
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        self.is_trained = True
        # Compute metrics
        with torch.no_grad():
            preds = self.model(X_tensor).squeeze().cpu().numpy()
        mse = np.mean((preds - y) ** 2)
        accuracy = np.mean(np.sign(preds) == np.sign(y))
        return {
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'accuracy': float(accuracy),
            'n_samples': n_samples
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for X."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        X_tensor = torch.FloatTensor(X).to(self.device)
        if self.scaler_mean is not None and self.scaler_std is not None:
            X_tensor = (X_tensor - self.scaler_mean) / self.scaler_std
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).squeeze()
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            preds = preds.cpu().numpy()
        return preds

    def get_signal(self, recent_data: pd.DataFrame) -> float:
        """
        Generate a trading signal from recent data.

        For LSTM, we need a lookback window equal to the seq_len expected.
        Since we don't know the fixed lookback, we'll use the data as is,
        assuming recent_data has shape (lookback, n_features).
        """
        if len(recent_data) < 1:
            return 0.0
        # We'll treat the entire recent_data as one sequence.
        # Ensure we have feature_columns set; if not, we assume recent_data contains all needed features.
        features = recent_data.values.astype(np.float32)
        # Shape: (lookback, n_features) -> (1, lookback, n_features)
        X = features[np.newaxis, :, :]
        preds = self.predict(X)
        return float(preds[0]) if preds.ndim > 0 else float(preds)

    def save(self, path: str) -> None:
        """Save model state and hyperparameters."""
        os.makedirs(path, exist_ok=True)
        # Save model state dict
        torch.save(self.model.state_dict(), Path(path) / 'model.pt')
        # Save hyperparameters and scaler
        params = {
            'name': self.name,
            'horizon': self.horizon,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'lr': self.lr,
            'epochs': self.epochs,
        }
        if self.scaler_mean is not None:
            params['scaler_mean'] = self.scaler_mean.cpu().numpy().tolist()
        if self.scaler_std is not None:
            params['scaler_std'] = self.scaler_std.cpu().numpy().tolist()
        with open(Path(path) / 'params.json', 'w') as f:
            json.dump(params, f, indent=2)

    def load(self, path: str) -> None:
        """Load model state and hyperparameters."""
        # Load params
        with open(Path(path) / 'params.json', 'r') as f:
            params = json.load(f)
        self.name = params['name']
        self.horizon = params['horizon']
        self.input_dim = params['input_dim']
        self.hidden_dim = params['hidden_dim']
        self.num_layers = params['num_layers']
        self.dropout = params['dropout']
        self.lr = params['lr']
        self.epochs = params['epochs']
        if 'scaler_mean' in params:
            self.scaler_mean = torch.FloatTensor(params['scaler_mean']).to(self.device)
        if 'scaler_std' in params:
            self.scaler_std = torch.FloatTensor(params['scaler_std']).to(self.device)
        # Build model and load state
        self.model = LSTMModel(
            self.input_dim,
            self.hidden_dim,
            self.num_layers,
            1,
            self.dropout
        ).to(self.device)
        self.model.load_state_dict(torch.load(Path(path) / 'model.pt', map_location=self.device))
        self.is_trained = True
