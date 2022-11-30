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

        if transaction_cost < 0:
            raise ValueError("transaction cost must be nonnegative")

        if rebal_period is not None:
            if rebal_period <= 0:
                raise ValueError("rebalance period must be positive or None")

            if type(rebal_period) != int:
                raise TypeError("rebalance period must be an integer or None")

        # Set start and end date dataframes
        if start_date == None:
            start_date = strategy.data.index[0]

        if end_date == None:
            end_date = strategy.data.index[-1]

        if start_date > end_date:  # type: ignore[operator]
            raise ValueError("start date must not be after end date")

        self.start_date = start_date
        self.end_date = end_date

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

    def get_annualized_return(self, daily_returns: NDArray | None = None) -> float:
        """
        Get annualized return
        Args:
            daily_returns: if given, returns to calculate annualized return over [default None]
        Return:
            annualized return over the backtest period or over returns passed in
        """
        if daily_returns is None:
            daily_returns = self.get_return(cumulative=False)

        annualized_return: float = (np.mean(daily_returns) + 1) ** 252 - 1

        return annualized_return

    def get_annualized_vol(self, returns: NDArray | None = None) -> float:
        """
        Get annualized volatility
        Args:
            returns: if given, returns to calculate volatility over [default None]
        Return:
            annualized volatility over the backtest period or over returns passed in
        """

        if returns is None:
            returns = self.get_return()

        annualized_vol: float = np.sqrt(252) * np.std(returns)

        return annualized_vol

    def get_annualized_downside_vol(
        self, rf_rate: float = 0.01, returns: NDArray | None = None
    ) -> float:
        """
        Get downside volatility
        Args:
            rf_rate: risk free rate [default 0.01]
            returns: if given, returns to calculate volatility over [default None]
        Return:
            downside volatility over the backtest period
        """

        if returns is None:
            returns = self.get_return()

        daily_rf = (1 + rf_rate) ** (1 / 252) - 1

        temp = np.minimum(0, returns - daily_rf)
        downside_vol: float = np.sqrt(252) * np.std(temp)

        return downside_vol

    def get_sharpe(self, rf_rate: float = 0.01) -> float:
        """
        Get Sharpe ratio
        Args:
            rf_rate: risk free rate [default 0.01]
        Return:
            Sharpe ratio over the backtest period
        """

        excess_return: float = self.get_annualized_return() - rf_rate
        std_return: float = self.get_annualized_vol()

        return excess_return / std_return

    def get_rolling_sharpe(self, rf_rate: float = 0.01, lookback: int = 20) -> NDArray:
        """
        Get rolling Sharpe ratio
        Args:
            rf_rate: risk free rate [default 0.01]
            lookback: lookback period to calculate Sharpe ratio [default 10]
        Return:
            Sharpe ratio over the backtest period
        """

        if lookback < 5:
            raise ValueError("lookback is too short, must be >= 5")

        if lookback > len(self.strat_weights) - lookback:
            raise ValueError("lookback is longer than backtest period")

        if type(lookback) != int:
            raise TypeError("lookback must be an integer")

        stds = []
        rets = []
        returns = self.get_return(cumulative=False)

        for i in range(len(self.strat_weights) - lookback):
            excess_returns = returns[i : i + lookback] - rf_rate
            rets.append(self.get_annualized_return(returns[i : i + lookback]))
            stds.append(self.get_annualized_vol(excess_returns))

        return np.array(rets) / np.array(stds)

    def get_sortino(self, rf_rate: float = 0.01) -> float:
        """
        Get Sortino ratio
        Args:
            rf_rate: risk free rate [default 0.01]
        Return:
            float of the Sortino ratio over the backtest period
        """

        excess_return: float = self.get_annualized_return() - rf_rate
        downside_vol: float = self.get_annualized_downside_vol()

        return excess_return / downside_vol

    def get_rolling_sortino(self, rf_rate: float = 0.01, lookback: int = 20) -> NDArray:
        """
        Get rolling Sortino ratio
        Args:
            rf_rate: risk free rate [default 0.01]
            lookback: lookback period to calculate Sortino ratio [default 10]
        Return:
            Sharpe ratio over the backtest period
        """

        if lookback < 5:
            raise ValueError("lookback is too short, must be >= 5")

        if lookback > len(self.strat_weights) - lookback:
            raise ValueError("lookback is longer than backtest period")

        if type(lookback) != int:
            raise TypeError("lookback must be an integer")

        stds = []
        rets = []
        returns = self.get_return(cumulative=False)

        for i in range(len(self.strat_weights) - lookback):
            excess_returns = returns[i : i + lookback] - rf_rate
            rets.append(self.get_annualized_return(returns[i : i + lookback]))
            stds.append(self.get_annualized_downside_vol(returns=excess_returns))

        return np.array(rets) / np.array(stds)

    def get_max_drawdown(self) -> float:
        """
        Get maximum drawdown
        Args:
            None
        Return:
            max drawdown over the backtest period
        """

        returns = self.get_return(cumulative=True)

        numerator = max(returns) - min(returns)
        denominator = max(returns)

        max_dd: float = numerator / denominator

        return max_dd

    def get_max_consecutive_positive(self) -> int:
        """
        Get maximum number of consecutive days of positive returns
        Args:
            None
        Return:
            maximum number of consecutive days of positive returns
        """

        returns = self.get_return()

        count = 0
        max_count = 0

        for i in range(len(returns)):
            if returns[i] > 0:
                count += 1
            else:
                max_count = max(max_count, count)
                count = 0

        max_count = max(max_count, count)

        return max_count

    def get_max_consecutive_negative(self) -> int:
        """
        Get maximum number of consecutive days of negative returns
        Args:
            None
        Return:
            maximum number of consecutive days of negative returns
        """

        returns = self.get_return()

        count = 0
        max_count = 0

        for i in range(len(returns)):
            if returns[i] < 0:
                count += 1
            else:
                max_count = max(max_count, count)
                count = 0

        max_count = max(max_count, count)

        return max_count

    def get_underwater_time(self, threshold_days: int = 10) -> tuple[int, float]:
        """
        Get maximum and average underwater times.
        Underwater time is defined as the number of days it takes an investor to recover its money at the start of the maximum drawdown period.
        Args:
            threshold_days: number of subsequent days the return must beat in order to be considered a maximum
        Return:
            tuple containing maximum underwater time and mean underwater time, respectively
        """

        if type(threshold_days) != int:
            raise TypeError("threshold_days must be an integer")

        if threshold_days < 3:
            raise ValueError("threshold_days is too short, must be >= 3")

        returns = self.get_return(cumulative=True)

        curr_max = returns[0]
        curr_count = 0
        max_drawdown_i = None
        underwater_times = []

        for i in range(1, len(returns)):
            if returns[i] > curr_max:
                curr_max = returns[i]
                curr_count = 0
            else:
                curr_count += 1

            if curr_count >= threshold_days:
                max_drawdown_i = i
                curr_count = 0
                continue

            if max_drawdown_i:
                if returns[i] >= returns[max_drawdown_i]:
                    underwater_time = i - max_drawdown_i
                    underwater_times.append(underwater_time)
                    max_drawdown_i = None

        return max(underwater_times), np.mean(underwater_times)

    def get_max_underwater_time(self, threshold_days: int = 10) -> int:
        """
        Get maximum underwater time.
        Args:
            threshold_days: number of subsequent days the return must beat in order to be considered a maximum
        Return:
            maximum underwater time
        """

        if type(threshold_days) != int:
            raise TypeError("threshold_days must be an integer")

        if threshold_days < 3:
            raise ValueError("threshold_days is too short, must be >= 3")

        max_underwater_time = self.get_underwater_time(threshold_days)[0]
        return max_underwater_time

    def get_avg_underwater_time(self, threshold_days: int = 10) -> float:
        """
        Get mean underwater time.
        Args:
            threshold_days: number of subsequent days the return must beat in order to be considered a maximum
        Return:
            mean underwater time
        """

        if type(threshold_days) != int:
            raise TypeError("threshold_days must be an integer")

        if threshold_days < 3:
            raise ValueError("threshold_days is too short, must be >= 3")

        mean_underwater_time = self.get_underwater_time(threshold_days)[1]
        return mean_underwater_time

    def plot_returns(self, cumulative: int = 1) -> None:
        """
        Plot the backtest returns
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
            plt.xticks(date_labels, rotation=90)
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
                tick.set_rotation(90)
            ax.set_xlabel("Date")
            ax.set_ylabel(f"Daily Return")
            ax2.set_ylabel(f"Cumulative Return")

            plt.title(f"Returns over Backtest Period")
            ax.legend()
            ax2.legend()
            plt.show()

    def plot_sharpe(self, rf_rate: float = 0.01, lookback: int = 20) -> None:
        """
        Plot the rolling Sharpe ratio
        Args:
            rf_rate: risk free rate [default 0.01]
            lookback: lookback period to calculate sharpe ratio [default 10]
        Return:
            None
        """

        sharpes = self.get_rolling_sharpe(rf_rate, lookback)
        dates = self.strat_weights.index[lookback:]

        n_days = len(self.strat_weights)
        n_ticks = 10
        date_labels = [dates[i * (n_days // n_ticks)] for i in range(n_ticks)]

        # Create the plot
        plt.plot(dates, sharpes)
        plt.xticks(date_labels, rotation=90)
        plt.title(f"{lookback}-Day Rolling Sharpe Ratio over Backtest Period")
        plt.xlabel("Date")
        plt.ylabel(f"Sharpe Ratio")
        plt.show()

    def plot_sortino(self, rf_rate: float = 0.01, lookback: int = 20) -> None:
        """
        Plot the rolling Sharpe ratio
        Args:
            rf_rate: risk free rate [default 0.01]
            lookback: lookback period to calculate sharpe ratio [default 10]
        Return:
            None
        """

        sortinos = self.get_rolling_sortino(rf_rate, lookback)
        dates = self.strat_weights.index[lookback:]

        n_days = len(self.strat_weights)
        n_ticks = 10
        date_labels = [dates[i * (n_days // n_ticks)] for i in range(n_ticks)]

        # Create the plot
        plt.plot(dates, sortinos)
        plt.xticks(date_labels, rotation=90)
        plt.title(f"{lookback}-Day Rolling Sortino Ratio over Backtest Period")
        plt.xlabel("Date")
        plt.ylabel(f"Sortino Ratio")
        plt.show()
