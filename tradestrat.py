# Imports
from __future__ import annotations

import abc
from collections.abc import Callable

import numpy as np
import pandas as pd
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

        if type(data) == list:
            # TODO: CURRENTLY USING DUMMY DATA
            self.data: pd.DataFrame = data

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

        self.rebal_period: int = rebal_period
        self.transaction_cost: float = transaction_cost
        self.start_date: datetime = start_date
        self.end_date: datetime = end_date

        # Set start and end date dataframes
        if start_date == None:
            start_date = strategy.data.index[0]

        if end_date == None:
            end_date = strategy.data.index[-1]

        self.price_data: pd.DataFrame = strategy.data.loc[start_date:end_date]
        self.strat_weights: pd.DataFrame = strategy.weights.loc[start_date:end_date]

    def get_return(self, cumulative: bool = False) -> NDArray[float]:
        """
        Get backtested returns

        Args:
            cumulative: if True, return cumulative return [default False]

        Return:
            Array of returns over the backtest period

        """

        # Initialize returns
        r: list[float] = []

        # Initialize variables
        prev_portfolio_val: float = 1
        tcost_loss: float = 0

        # Initialize portfolio weights for asset i (x_i), using strategy weights (w_i) and prices (p_i)
        # x_i = w_i / p_i
        p: pd.Series = self.price_data.iloc[0]
        w: pd.Series = self.strat_weights.iloc[0]
        x: pd.Series = w / p

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

    def plot(self) -> None:
        """
        Plot the cumulative returns of the backtest returns

        Args:
            None

        Return:
            None
        """

        # Get cumulative returns
        cum_ret: NDArray[float] = self.get_return(cumulative=True)

        # Create the plot
        plt.plot(self.price_data.index, cum_ret)
        plt.xticks(rotation=45)
        plt.title("Cumulative Returns over Backtest Period")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        
    def get_annualized_vol(self) -> float:
        """
        Get annualized volatility

        Args:
            None

        Return:
            float of the annualized volatility over the backtest period

        """
        
        annualized_vol = np.sqrt(252) * np.std(self.get_return())
        
        return annualized_vol
    
    def get_downside_vol(self) -> float:    
        """
        Get downside volatility

        Args:
            None

        Return:
            float of the downside volatility over the backtest period

        """
        
        temp = np.minimum(0, self.get_return() - rf_rate)**2
        temp_expectation = np.mean(temp)
        downside_vol = np.sqrt(temp_expectation)
        
        return downside_vol

    def get_sharpe(self, rf_rate: float = 0.01) -> float:
        """
        Get Sharpe ratio

        Args:
            rf_rate: risk free rate [default 0.01]

        Return:
            float of the Sharpe ratio over the backtest period

        """

        excess_return = self.get_return(cumulative=True)[-1] - rf_rate
        std_return = self.get_annualized_vol()

        return excess_return / std_return
    
    def get_sortino(self, rf_rate: float = 0.01) -> float:
        """
        Get Sortino ratio

        Args:
            rf_rate: risk free rate [default 0.01]

        Return:
            float of the Sortino ratio over the backtest period

        """
        
        downside_vol = self.get_downside_vol()
        
        sortino_ratio = np.mean(self.get_return() - rf_rate) / downside_vol

        return sortino_ratio
    
    def get_max_drawdown(self) -> float:
        """
        Get maximum drawdown

        Args:
            None

        Return:
            float of max drawdown over the backtest period

        """        
       
        returns = self.get_return()
        
        numerator = max(returns) - min(returns)
        denominator = max(returns)
        
        max_dd = numerator/denominator
        
        return max_dd   
    
    def get_max_consecutive_positive(self) -> int:
        """
        Get maximum number of consecutive days of positive returns
        
        Args:
            None
            
        Return:
            integer of maximum number of consecutive days of positive returns
            
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
            integer of maximum number of consecutive days of negative returns
            
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
    
    def get_max_underwater_time(self) -> int:
        """
        Get maximum underwater time.
        This is defined as the maximum number of days it takes an investor to recover its money at the start of the maximum drawdown period
        
        Args:
            None
            
        Return: 
            integer of maximum underwater time
            
        """
        pass
        
        
