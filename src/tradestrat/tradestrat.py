# Imports
from __future__ import annotations

import abc
from collections.abc import Callable
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.typing import NDArray


class Strategy(abc.ABC):
    def __init__(self, data: list[str] | dict[str, pd.DataFrame]) -> None:
        """
        Initialize Strategy class

        Args:
            data: list of tickers to be considered in universe OR
                  dictionary of DataFrames, each containing dates along rows and tickers along columns,
                  with one DataFrame per value (e.g. data = {'returns': ..., 'PE': ...})

        Return:
            None
        """

        self.weights: pd.DataFrame | None = None
        self.data: pd.DataFrame

        if type(data) == list:
            # TODO: CURRENTLY USING DUMMY DATA
            self.data = data

    @abc.abstractmethod
    def get_weights(self) -> pd.DataFrame:
        """
        Get strategy weights over time

        Args:
            ...

        Return:
            DataFrame containing dates along rows and tickers along columns, with values being the strategy weights

        """

        ...


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

        self.price_data: pd.DataFrame = strategy.data.loc[start_date:end_date]  # type: ignore[misc]
        self.strat_weights: pd.DataFrame = strategy.weights.loc[start_date:end_date]  # type: ignore[misc]

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

    def plot_returns(self, cumulative: bool = True) -> None:
        """
        Plot the cumulative returns of the backtest returns

        Args:
            cumulative: if True, plot cumulative return, otherwise plot daily return [default True]

        Return:
            None
        """

        if cumulative:
            # Get cumulative returns
            ret = self.get_return(cumulative=True)
            return_type = "Cumulative"
        else:
            ret = self.get_return()
            return_type = "Daily"

        # Create the plot
        plt.plot(self.price_data.index, ret)
        plt.xticks(rotation=45)
        plt.title(f"{return_type} Returns over Backtest Period")
        plt.xlabel("Date")
        plt.ylabel(f"{return_type} Return")

    def get_sharpe(self, rf_rate: float = 0.01) -> float:
        """
        Get Sharpe ratio

        Args:
            rf_rate: risk free rate [default 0.01]

        Return:
            float of the Sharpe ratio over the backtest period

        """

        excess_return: float = self.get_return(cumulative=True)[-1] - rf_rate
        std_return: float = np.std(self.get_return())

        return excess_return / std_return
