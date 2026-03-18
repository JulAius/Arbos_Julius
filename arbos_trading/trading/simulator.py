"""
Paper trading simulator.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from .metrics import calculate_metrics


class SimulatedTrader:
    """
    Simulates paper trading given a sequence of signals and price data.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.001,
        slippage: float = 0.0005,
        bet_size: float = 1.0,
        stop_loss_pct: float = 0.0
    ):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.bet_size = bet_size
        self.stop_loss_pct = stop_loss_pct

        self.equity_curve = None
        self.trades = []
        self.metrics = {}

    def run(
        self,
        signals: pd.Series,
        prices,
        entry_price_col: str = "close"
    ) -> Dict[str, float]:
        """
        Run simulation with fixed 1-period holding, optional stop-loss.

        Args:
            signals: Series of signals (1=long, 0=short, -1=no trade)
            prices: Either a Series of close prices, or a DataFrame with columns ['close','high','low'].
                    If DataFrame provided and stop_loss_pct > 0, stop-loss will be tested against the high/low of the next period.
            entry_price_col: ignored (we use close)

        Returns:
            dict of performance metrics
        """
        self.trades = []

        capital = self.initial_capital
        # Combined transaction cost: fee + slippage
        entry_cost = self.fee_rate + self.slippage
        exit_cost = self.fee_rate + self.slippage

        # Determine if we have high/low for stop-loss
        use_stop = self.stop_loss_pct > 0
        if isinstance(prices, pd.DataFrame):
            # Expect columns: 'close', optionally 'high', 'low'
            if 'high' in prices.columns and 'low' in prices.columns:
                close_series = prices['close']
                high_series = prices['high']
                low_series = prices['low']
            else:
                close_series = prices['close']
                high_series = low_series = close_series
        else:
            close_series = prices
            high_series = low_series = prices

        # Align signals and prices
        signals = signals.reindex(close_series.index).fillna(-1)

        # Build equity curve indexed by signals.index
        equity = pd.Series(index=signals.index, dtype=float)
        equity.iloc[0] = capital

        # Iterate through each timestamp, use signal to open a trade that closes at next timestamp
        for i in range(len(signals) - 1):
            sig = signals.iloc[i]
            current_price = close_series.iloc[i]
            next_close = close_series.iloc[i+1]
            next_high = high_series.iloc[i+1]
            next_low = low_series.iloc[i+1]

            if sig == -1:
                # No trade, carry forward capital
                equity.iloc[i+1] = capital
                continue

            # Determine entry price with cost
            if sig == 1:  # long
                entry = current_price * (1 + entry_cost)
                # Determine exit price considering stop-loss
                stop_price = entry * (1 - self.stop_loss_pct) if use_stop else None
                exit_price = next_close * (1 - exit_cost)
                if use_stop and next_low <= stop_price:
                    # Stop triggered: exit at stop price (with exit cost)
                    exit_price = stop_price * (1 - exit_cost)
                ret = (exit_price / entry) - 1
            else:  # short
                entry = current_price * (1 - entry_cost)
                stop_price = entry * (1 + self.stop_loss_pct) if use_stop else None
                exit_price = next_close * (1 + exit_cost)
                if use_stop and next_high >= stop_price:
                    exit_price = stop_price * (1 + exit_cost)
                ret = (entry / exit_price) - 1

            # Apply bet size
            capital *= (1 + ret * self.bet_size)

            self.trades.append({
                "entry_time": signals.index[i],
                "exit_time": signals.index[i+1],
                "entry_price": entry,
                "exit_price": exit_price,
                "return": ret,
                "pnl": capital - self.initial_capital,
                "stopped": use_stop and ((sig == 1 and next_low <= stop_price) or (sig == 0 and next_high >= stop_price))
            })

            equity.iloc[i+1] = capital

        # Fill any remaining NaNs in equity with last value
        equity = equity.ffill()

        self.equity_curve = equity

        # Calculate metrics
        self.metrics = self._compute_metrics()

        return self.metrics

    def _compute_metrics(self) -> Dict[str, float]:
        """Compute metrics from simulation."""
        if not self.trades:
            return {
                "accuracy": 0.0,
                "bets_per_month": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "total_return": 0.0,
                "n_trades": 0,
                "profit_factor": 0.0,
                "final_equity": self.initial_capital
            }

        returns = [t["return"] for t in self.trades]
        winning = sum(1 for r in returns if r > 0)
        accuracy = winning / len(returns)

        # Bets per month (entries)
        n_entries = len(self.trades)
        if self.equity_curve is not None and len(self.equity_curve) > 0:
            period_days = len(self.equity_curve) / (24 * 4)  # 15-min candles to days
        else:
            period_days = 0
        months = period_days / 30.44
        bets_per_month = n_entries / months if months > 0 else 0

        # Sharpe on per-trade returns only (avoids inflation from zero-return no-trade periods)
        if returns and len(returns) > 1:
            ret_arr = np.array(returns)
            trades_per_year = (n_entries / max(period_days, 1)) * 365
            sharpe = (ret_arr.mean() / (ret_arr.std() + 1e-10)) * np.sqrt(trades_per_year)
        else:
            sharpe = 0.0

        # Max drawdown
        if self.equity_curve is not None:
            running_max = self.equity_curve.expanding().max()
            drawdown = (self.equity_curve - running_max) / running_max
            max_drawdown = abs(drawdown.min())
        else:
            max_drawdown = 0.0

        total_return = (self.equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital

        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        profit_factor = gross_profit / (gross_loss + 1e-10)

        return {
            "accuracy": accuracy,
            "bets_per_month": bets_per_month,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "total_return": total_return,
            "n_trades": len(self.trades),
            "profit_factor": profit_factor,
            "final_equity": self.equity_curve.iloc[-1] if self.equity_curve is not None else self.initial_capital,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss
        }

    def get_equity_curve(self) -> pd.Series:
        return self.equity_curve

    def get_trades(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)
