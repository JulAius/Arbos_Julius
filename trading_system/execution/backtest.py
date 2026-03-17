"""
Backtesting Engine - Simulate trading with historical data.

Provides realistic simulation with transaction costs, slippage, and
position management. Tracks performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Trade:
    """Represents a single executed trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    position_size: float  # Positive for long, negative for short
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    trades: list[Trade]
    equity_curve: pd.Series
    positions: pd.Series
    signals: pd.Series
    metrics: Dict[str, float]
    final_equity: float
    total_return: float

    def summary(self) -> str:
        """Generate a summary string."""
        summary_lines = [
            "=== Backtest Results ===",
            f"Total Trades: {len(self.trades)}",
            f"Final Equity: ${self.final_equity:.2f}",
            f"Total Return: {self.total_return:.2%}",
            f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}",
            f"Max Drawdown: {self.metrics.get('max_drawdown', 0):.2%}",
            f"Win Rate: {self.metrics.get('win_rate', 0):.2%}",
            f"Profit Factor: {self.metrics.get('profit_factor', 0):.2f}",
            f"Calmar Ratio: {self.metrics.get('calmar_ratio', 0):.2f}",
        ]
        return "\n".join(summary_lines)


class BacktestEngine:
    """
    Event-driven backtesting engine for trading strategies.

    Simulates trading with:
    - Signal-based position entries/exits
    - Fixed position sizing (percentage of equity)
    - Transaction costs (commission + slippage)
    - No partial fills (for simplicity)
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_rate: float = 0.0006,  # 0.06% (Binance taker fee)
        slippage_rate: float = 0.0005,     # 0.05% slippage per trade
        max_position_size: float = 1.0,    # 100% of equity
        allow_short: bool = True
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital
            commission_rate: Commission as decimal per trade
            slippage_rate: Estimated slippage as decimal per trade
            max_position_size: Maximum position size as fraction of equity
            allow_short: Whether to allow short positions
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position_size = max_position_size
        self.allow_short = allow_short

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        prices: Optional[pd.Series] = None,
        close_col: str = 'close',
        signal_threshold: float = 0.5
    ) -> BacktestResult:
        """
        Run backtest with given signals.

        Args:
            data: OHLCV DataFrame with datetime index
            signals: Series of trading signals (-1 to 1), aligned with data index
            prices: Optional price series to use (default: close price)
            close_col: Column name for close price in data
            signal_threshold: Minimum absolute signal to trigger trade

        Returns:
            BacktestResult with trades and metrics
        """
        if prices is None:
            prices = data[close_col]

        # Align signals and prices
        common_idx = signals.index.intersection(prices.index)
        prices = prices.loc[common_idx]
        signals = signals.loc[common_idx]

        equity = pd.Series(index=common_idx, dtype=float)
        positions = pd.Series(index=common_idx, dtype=float)  # Net position (-1 to 1)
        equity.iloc[0] = self.initial_capital
        current_position = 0.0
        cash = self.initial_capital
        current_price = prices.iloc[0]

        trades = []
        entry_trade = None  # Track open trade

        for i in range(1, len(common_idx)):
            idx = common_idx[i]
            price = prices.iloc[i]
            signal = signals.iloc[i]

            # Check for exit signal (opposite position or zero)
            target_position = 0.0
            if abs(signal) >= signal_threshold:
                target_position = np.clip(signal, -1, 1) * self.max_position_size
                if not self.allow_short and target_position < 0:
                    target_position = 0.0

            # Position change
            position_change = target_position - current_position

            if abs(position_change) > 1e-6:
                # Execute trade at current price with slippage
                trade_price = price * (1 + self.slippage_rate * np.sign(position_change))

                # Calculate trade size in cash
                trade_value = abs(position_change) * cash
                commission = trade_value * self.commission_rate

                if current_position != 0:
                    # Closing existing position
                    entry_price = entry_trade.entry_price if entry_trade else current_price
                    exit_value = position_change * trade_value * (trade_price / current_price)
                    # PnL from closing
                    if current_position > 0:
                        pnl = (trade_price - entry_price) * (current_position * self.initial_capital)
                    else:
                        pnl = (entry_price - trade_price) * (abs(current_position) * self.initial_capital)

                    # Record trade
                    trade = Trade(
                        entry_time=entry_trade.entry_time if entry_trade else common_idx[i-1],
                        exit_time=idx,
                        entry_price=entry_price,
                        exit_price=trade_price,
                        position_size=current_position,
                        pnl=float(pnl - commission),
                        pnl_pct=float((trade_price - entry_price) / entry_price * np.sign(current_position)),
                        commission=float(commission),
                        slippage=float(abs(position_change) * trade_value * (abs(trade_price - price) / price))
                    )
                    trades.append(trade)
                    cash += pnl - commission

                # Update position
                current_position = target_position

                if current_position != 0:
                    # Opening new trade
                    entry_trade = Trade(
                        entry_time=idx,
                        exit_time=None,  # Will be set on close
                        entry_price=trade_price,
                        exit_price=None,
                        position_size=current_position,
                        pnl=0.0,
                        pnl_pct=0.0,
                        commission=commission,
                        slippage=abs(position_change) * trade_value * (abs(trade_price - price) / price)
                    )

            # Update equity
            if current_position != 0 and entry_trade is not None:
                # Mark to market
                mark_pnl = (price - entry_trade.entry_price) * (current_position * self.initial_capital)
                equity.iloc[i] = self.initial_capital + mark_pnl
            else:
                equity.iloc[i] = cash

            positions.iloc[i] = current_position

        # Close any open position at the end
        if entry_trade is not None and current_position != 0:
            final_price = prices.iloc[-1]
            if current_position > 0:
                pnl = (final_price - entry_trade.entry_price) * (current_position * self.initial_capital)
            else:
                pnl = (entry_trade.entry_price - final_price) * (abs(current_position) * self.initial_capital)
            commission = abs(current_position * cash) * self.commission_rate
            cash += pnl - commission
            equity.iloc[-1] = cash

        # Calculate metrics
        returns = equity.pct_change().dropna()
        from validation.metrics import compute_all_metrics
        metrics = compute_all_metrics(returns, equity)

        return BacktestResult(
            trades=trades,
            equity_curve=equity,
            positions=positions,
            signals=signals,
            metrics=metrics,
            final_equity=equity.iloc[-1],
            total_return=(equity.iloc[-1] / equity.iloc[0]) - 1
        )


def simple_backtest(
    signals: pd.Series,
    prices: pd.Series,
    initial_capital: float = 10000.0,
    commission: float = 0.0006,
    threshold: float = 0.5
) -> BacktestResult:
    """
    Simple one-liner backtest.

    Example:
        signals = model.predict(X_test)
        result = simple_backtest(signals, df['close'])
        print(result.summary())
    """
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=commission,
        allow_short=True
    )

    result = engine.run(
        data=pd.DataFrame({'close': prices}),
        signals=signals,
        prices=prices,
        signal_threshold=threshold
    )

    return result
