"""
Performance Metrics - Calculate trading performance metrics.

Common metrics: Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio,
profit factor, win rate, etc.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


def calculate_returns_series(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate log returns from price series."""
    return np.log(prices / prices.shift(periods))


def calculate_sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = 365 * 24,  # Hourly data by default
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Series of per-period returns
        periods_per_year: Number of periods in a year (for annualization)
        risk_free_rate: Annual risk-free rate

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()

    sharpe = mean_excess / std_excess
    sharpe_annualized = sharpe * np.sqrt(periods_per_year)

    return float(sharpe_annualized)


def calculate_sortino_ratio(
    returns: pd.Series,
    periods_per_year: int = 365 * 24,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0
) -> float:
    """
    Calculate Sortino ratio (uses downside deviation instead of total std).

    Args:
        returns: Series of per-period returns
        periods_per_year: Number of periods in a year
        risk_free_rate: Annual risk-free rate
        target_return: Minimum acceptable return (MAR)

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - target_return
    downside = excess_returns[excess_returns < 0]
    downside_dev = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 0.0

    if downside_dev == 0:
        return np.inf if excess_returns.mean() > 0 else 0.0

    sortino = (excess_returns.mean() / downside_dev) * np.sqrt(periods_per_year)
    return float(sortino)


def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown and its duration.

    Args:
        equity_curve: Series of portfolio values (cumulative returns or equity)

    Returns:
        (max_drawdown_pct, peak_idx, trough_idx) where peak_idx and trough_idx are integer positions
    """
    if len(equity_curve) == 0:
        return 0.0, 0, 0

    # Calculate running maximum
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max

    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()

    # Find the peak before this trough
    peak_idx = running_max.loc[:max_dd_idx].idxmax()

    # Convert index labels to integer positions
    try:
        peak_pos = equity_curve.index.get_loc(peak_idx)
        trough_pos = equity_curve.index.get_loc(max_dd_idx)
    except (KeyError, TypeError):
        # Fallback: use positional if index is not unique or not found
        peak_pos = 0
        trough_pos = 0

    return float(abs(max_dd)), int(peak_pos), int(trough_pos)


def calculate_calmar_ratio(
    returns: pd.Series,
    equity_curve: pd.Series,
    periods_per_year: int = 365 * 24
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        returns: Series of per-period returns
        equity_curve: Series of portfolio values
        periods_per_year: Number of periods for annualization

    Returns:
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0

    annual_return = returns.mean() * periods_per_year
    max_dd, _, _ = calculate_max_drawdown(equity_curve)

    if max_dd == 0:
        return np.inf if annual_return > 0 else 0.0

    return float(annual_return / max_dd)


def calculate_profit_factor(returns: pd.Series) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        returns: Series of period returns

    Returns:
        Profit factor (> 1 is profitable)
    """
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0

    return float(gross_profit / gross_loss)


def calculate_win_rate(returns: pd.Series) -> float:
    """
    Calculate win rate (percentage of winning periods).

    Args:
        returns: Series of period returns

    Returns:
        Win rate as a fraction
    """
    if len(returns) == 0:
        return 0.0

    wins = (returns > 0).sum()
    total = len(returns)

    return float(wins / total)


def compute_all_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    periods_per_year: int = 365 * 24,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Compute all performance metrics in one call.

    Args:
        returns: Period returns
        equity_curve: Cumulative equity/portfolio values
        periods_per_year: Annualization factor
        risk_free_rate: Risk-free rate

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'total_return': float((1 + returns).prod() - 1) if len(returns) > 0 else 0.0,
        'annualized_return': float(returns.mean() * periods_per_year) if len(returns) > 0 else 0.0,
        'sharpe_ratio': calculate_sharpe_ratio(returns, periods_per_year, risk_free_rate),
        'sortino_ratio': calculate_sortino_ratio(returns, periods_per_year, risk_free_rate),
        'max_drawdown': calculate_max_drawdown(equity_curve)[0],
        'calmar_ratio': calculate_calmar_ratio(returns, equity_curve, periods_per_year),
        'profit_factor': calculate_profit_factor(returns),
        'win_rate': calculate_win_rate(returns),
        'volatility': float(returns.std() * np.sqrt(periods_per_year)),
        'skewness': float(returns.skew()) if len(returns) > 2 else 0.0,
        'kurtosis': float(returns.kurtosis()) if len(returns) > 3 else 0.0,
        'n_periods': len(returns)
    }

    return metrics


def evaluate_single_metric(returns: pd.Series, equity_curve: pd.Series, metric: str = 'sharpe_ratio') -> float:
    """
    Evaluate a single metric for use in optimization/selection.

    Args:
        returns: Period returns
        equity_curve: Cumulative equity
        metric: Name of metric to compute

    Returns:
        Metric value
    """
    all_metrics = compute_all_metrics(returns, equity_curve)
    if metric not in all_metrics:
        raise ValueError(f"Unknown metric: {metric}. Available: {list(all_metrics.keys())}")
    return all_metrics[metric]
