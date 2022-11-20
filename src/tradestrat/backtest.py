# Imports
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import numpy as np
import pandas as pd  # type: ignore[import]
from matplotlib import pyplot as plt  # type: ignore[import]
from numpy.typing import NDArray

from .strategies import Strategy


class Backtest:
    def __init__(
        self,
        strategy: Strategy,
        rebal_period: int | None = None,
        transaction_cost: float = 0,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> None:
        """
        Initialize Backtest class

        Args:
            strategy:         computed strategy results
            rebal_period:     rebalance period in days (if None, never rebalance) [default None]
            transaction_cost: cost per transaction as a fraction of total spending [default 0]
            start_date:       date to start backtest period (if None, start at beginning of available data) [default None]
            end_date:         date to end backtest period (if None, end at end of available data) [default None]

        Return:
            None
        """

        self.rebal_period = rebal_period
        self.transaction_cost = transaction_cost
        self.start_date = start_date
        self.end_date = end_date

        # Set start and end date dataframes
        if start_date == None:
            start_date = strategy.data.index[0]

        if end_date == None:
            end_date = strategy.data.index[-1]

        self.price_data = strategy.data.loc[start_date:end_date]  # type: ignore[misc]
        self.strat_weights = strategy.weights.loc[start_date:end_date]  # type: ignore[misc]

    def get_return(self, cumulative: bool = False) -> NDArray[float]:
        """
        Get backtested returns

        Args:
            cumulative: if True, return cumulative return [default False]

        Return:
            Array of returns over the backtest period

        """

        # Initialize returns
        r = []

        # Initialize variables
        prev_portfolio_val = 1
        tcost_loss = 0

        # Initialize portfolio weights for asset i (x_i), using strategy weights (w_i) and prices (p_i)
        # x_i = w_i / p_i
        p = self.price_data.iloc[0]
        w = self.strat_weights.iloc[0]
        x = w / p

        for t in range(len(self.price_data)):

            # Get portfolio value and corresponding return for current period
            p = self.price_data.iloc[t]
            portfolio_val = (x * p).sum() - tcost_loss
            r.append(portfolio_val / prev_portfolio_val - 1)

            # Update previous portfolio value
            prev_portfolio_val = portfolio_val

            # Check if current period is a rebalancing period
            if self.rebal_period and t % self.rebal_period == 0:
                # Save previous portfolio weights
                x_prev: pd.Series = x

                # Get updated portfolio weights
                w = self.strat_weights.iloc[t]
                x = w / p

                # Compute amount lost to transaction costs
                tcost_loss += self.transaction_cost * abs(x - x_prev).sum()

        if cumulative:
            return np.cumprod(1 + np.array(r)) - 1

        else:
            return np.array(r)

    def plot_returns(self, cumulative: int = 1) -> None:
        """
        Plot the cumulative returns of the backtest returns

        Args:
            cumulative: if 0, plot daily return
                        if 1, plot cumulative return [default 1]
                        if 2, plot both daily and cumulative return

        Return:
            None
        """

        n_days = len(self.strat_weights)
        n_ticks = 10
        date_labels = [
            self.strat_weights.index[i * (n_days // n_ticks)] for i in range(n_ticks)
        ]

        if cumulative not in [0, 1, 2]:
            raise ValueError("cumulative must be 0, 1 or 2")

        if cumulative < 2:
            # Plot either the cumulative or daily returns
            ret = self.get_return(cumulative=bool(cumulative))
            if cumulative:
                return_type = "Cumulative"

            else:
                return_type = "Daily"

            # Create the plot
            plt.plot(self.price_data.index, ret)
            plt.xticks(date_labels, rotation=45)
            plt.title(f"{return_type} Returns over Backtest Period")
            plt.xlabel("Date")
            plt.ylabel(f"{return_type} Return")
            plt.show()

        else:
            # Plot both the cumulative and daily returns
            cum_ret = self.get_return(cumulative=True)
            day_ret = self.get_return(cumulative=False)

            # Create the plot
            fig, ax = plt.subplots()

            ax2 = ax.twinx()
            ax.plot(
                self.price_data.index,
                day_ret,
                label="Daily Return",
                color="orange",
            )
            ax2.plot(
                self.price_data.index, cum_ret, label="Cumulative Return", color="blue"
            )

            ax.set_xticks(date_labels)
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
            ax.set_xlabel("Date")
            ax.set_ylabel(f"Daily Return")
            ax2.set_ylabel(f"Cumulative Return")

            plt.title(f"Returns over Backtest Period")
            ax.legend()
            ax2.legend()
            plt.show()

    def get_sharpe(self, rf_rate: float = 0.01) -> float:
        """
        Get Sharpe ratio

        Args:
            rf_rate: risk free rate [default 0.01]

        Return:
            float of the Sharpe ratio over the backtest period

        """

        excess_return = self.get_return(cumulative=True)[-1] - rf_rate
        std_return = np.std(self.get_return())

        return excess_return / std_return
