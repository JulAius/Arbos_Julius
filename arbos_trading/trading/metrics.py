"""
Performance metrics calculation for trading system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List


def calculate_metrics(
    signals: pd.Series,
    prices: pd.Series,
    initial_capital: float = 10000.0,
    fee_rate: float = 0.001,
    slippage: float = 0.0005,
    bet_size: float = 1.0
) -> Dict[str, float]:
    """
    Calculate comprehensive trading metrics from signals and prices.

    Args:
        signals: Series of signals (1=UP/buy, 0=DOWN/sell, -1=no trade)
        prices: Series of closing prices (aligned with signals)
        initial_capital: starting capital
        fee_rate: trading fee per trade (e.g., 0.001 = 0.1%)
        slippage: assumed slippage per trade
        bet_size: fraction of capital per bet

    Returns:
        dict of metrics
    """
    # Align
    signals = signals.reindex(prices.index)
    signals = signals.fillna(-1)

    # Track equity curve
    equity = [initial_capital]
    positions = []  # track position at each step
    capital = initial_capital
    position = 0  # 1=long, 0=flat

    trades = []  # list of trade returns

    for i in range(len(signals)):
        sig = signals.iloc[i]
        price = prices.iloc[i]

        if sig == 1 and position == 0:  # Enter long
            position = 1
            entry_price = price * (1 + slippage)
        elif sig == 0 and position == 1:  # Exit long
            # Calculate return
            exit_price = price * (1 - slippage)
            ret = (exit_price / entry_price) - 1
            capital *= (1 + ret * bet_size)
            trades.append(ret)
            position = 0

        positions.append(position)
        equity.append(capital)

    # If still in position at end, close at last price
    if position == 1:
        exit_price = prices.iloc[-1] * (1 - slippage)
        ret = (exit_price / entry_price) - 1
        capital *= (1 + ret * bet_size)
        trades.append(ret)

    equity = pd.Series(equity[1:], index=prices.index)

    # Count bets (entries)
    entries = (signals.diff() == 1).sum() + (signals.iloc[0] == 1)
    bets_per_month = entries / (len(signals) / (4 * 24 * 30))  # 15m candles -> approx months

    # Calculate accuracy
    if trades:
        winning_trades = sum(1 for t in trades if t > 0)
        accuracy = winning_trades / len(trades)
    else:
        accuracy = 0.0

    # Sharpe ratio (daily-like using all periods)
    returns_series = equity.pct_change().dropna()
    if returns_series.std() > 0:
        sharpe = (returns_series.mean() / returns_series.std()) * np.sqrt(365 * 24 * 4)  # annualized (15m)
    else:
        sharpe = 0.0

    # Max drawdown
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    # Profit factor
    gross_profit = sum(t for t in trades if t > 0)
    gross_loss = abs(sum(t for t in trades if t < 0))
    profit_factor = gross_profit / (gross_loss + 1e-10)

    # Total return
    total_return = (capital - initial_capital) / initial_capital

    metrics = {
        "accuracy": accuracy,
        "bets_per_month": bets_per_month,
        "sharpe": sharpe,
        "max_drawdown": abs(max_drawdown),
        "total_return": total_return,
        "profit_factor": profit_factor,
        "n_trades": len(trades),
        "final_equity": capital,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss
    }

    return metrics


def evaluate_predictions(
    predictions: pd.Series,
    actuals: pd.Series,
    signals: pd.Series = None
) -> Dict[str, float]:
    """
    Evaluate directional accuracy and betting performance.
    """
    preds = predictions.reindex(actuals.index)
    mask = preds.notna() & actuals.notna()

    if not mask.any():
        return {"accuracy": 0.0, "bets": 0}

    aligned_preds = preds[mask]
    aligned_actuals = actuals[mask]

    correct = (aligned_preds == aligned_actuals).sum()
    accuracy = correct / len(aligned_preds)

    # Count bets (non -1 signals)
    if signals is not None:
        bets = (signals.reindex(actuals.index) != -1).sum()
    else:
        bets = len(aligned_preds)

    return {
        "accuracy": accuracy,
        "bets": bets,
        "n_samples": len(aligned_preds)
    }
