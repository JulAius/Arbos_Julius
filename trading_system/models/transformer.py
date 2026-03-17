"""
Transformer Trading Model - Attention-based model for time series prediction.

Uses Transformer encoder layers to capture long-range dependencies.
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


class PositionalEncoding(nn.Module):
    """Positional encoding for sequences."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerModel(nn.Module):
    """Internal PyTorch Transformer model."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        output_dim: int = 1
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Take output from last time step
        out = self.fc_out(x[:, -1, :])
        return out


class TransformerTradingModel(TradingModel):
    """
    Transformer-based trading model with attention mechanism.

    Hyperparameters (to be tuned by evolution):
      - d_model: embedding/hidden dimension (default 64)
      - nhead: number of attention heads (default 4)
      - num_encoder_layers: number of transformer layers (default 3)
      - dim_feedforward: size of feedforward network (default 256)
      - dropout: dropout rate (default 0.2)
      - lr: learning rate (default 1e-3)
      - epochs: training epochs (default 50)
    """

    def __init__(
        self,
        name: str = "transformer",
        horizon: int = 1,
        input_dim: int = None,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 50  # Increased for full training
    ):
        super().__init__(name, horizon)
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs

        self.model: Optional[nn.Module] = None
        self.scaler_mean: Optional[torch.Tensor] = None
        self.scaler_std: Optional[torch.Tensor] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_columns = None

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train the Transformer model."""
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        n_samples, seq_len, n_features = X.shape
        if self.input_dim is None:
            self.input_dim = n_features
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        # Normalize features
        if self.scaler_mean is None or self.scaler_std is None:
            self.scaler_mean = X_tensor.mean(dim=(0, 1), keepdim=True)
            self.scaler_std = X_tensor.std(dim=(0, 1), keepdim=True)
            self.scaler_std[self.scaler_std == 0] = 1.0
        X_tensor = (X_tensor - self.scaler_mean) / self.scaler_std
        # Build model
        if self.model is None:
            self.model = TransformerModel(
                input_dim=self.input_dim,
                d_model=self.d_model,
                nhead=self.nhead,
                num_encoder_layers=self.num_encoder_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                output_dim=1
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
        """Generate predictions."""
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
        """Generate trading signal."""
        if len(recent_data) < 1:
            return 0.0
        features = recent_data.values.astype(np.float32)
        X = features[np.newaxis, :, :]
        preds = self.predict(X)
        return float(preds[0]) if preds.ndim > 0 else float(preds)

    def save(self, path: str) -> None:
        """Save model."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), Path(path) / 'model.pt')
        params = {
            'name': self.name,
            'horizon': self.horizon,
            'input_dim': self.input_dim,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_encoder_layers': self.num_encoder_layers,
            'dim_feedforward': self.dim_feedforward,
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
        """Load model."""
        with open(Path(path) / 'params.json', 'r') as f:
            params = json.load(f)
        self.name = params['name']
        self.horizon = params['horizon']
        self.input_dim = params['input_dim']
        self.d_model = params['d_model']
        self.nhead = params['nhead']
        self.num_encoder_layers = params['num_encoder_layers']
        self.dim_feedforward = params['dim_feedforward']
        self.dropout = params['dropout']
        self.lr = params['lr']
        self.epochs = params['epochs']
        if 'scaler_mean' in params:
            self.scaler_mean = torch.FloatTensor(params['scaler_mean']).to(self.device)
        if 'scaler_std' in params:
            self.scaler_std = torch.FloatTensor(params['scaler_std']).to(self.device)
        self.model = TransformerModel(
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            output_dim=1
        ).to(self.device)
        self.model.load_state_dict(torch.load(Path(path) / 'model.pt', map_location=self.device))
        self.is_trained = True
