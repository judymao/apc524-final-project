import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))

import pytest

import pandas as pd
import numpy as np
from datetime import date

from tradestrat import Strategy, Backtest


class StrategyTest(Strategy):
    def get_weights(self) -> pd.DataFrame:
        """
        Get strategy weights over time
        Args:
            None
        Return:
            DataFrame containing dates along rows and tickers along columns, with values being the strategy weights
        """
        weights_df = self.data.copy()
        weights_df[:] = 1/len(self.data.columns)
        return weights_df

def test_tcost():
    strat = StrategyTest(data = ["AAPL", "IBM"])

    # Check transaction cost >= 0
    with pytest.raises(ValueError):
        Backtest(strategy = strat, transaction_cost = -1)

    with pytest.raises(ValueError):
        Backtest(strategy = strat, transaction_cost = -1.5)

    assert Backtest(strategy = strat, transaction_cost = 0).transaction_cost == 0
    assert Backtest(strategy = strat, transaction_cost = 1).transaction_cost == 1
    assert Backtest(strategy = strat, transaction_cost = 1.6).transaction_cost == 1.6

    assert Backtest(strategy = strat).transaction_cost == 0

def test_rebal_period():
    strat = StrategyTest(data = ["AAPL", "IBM"])

    # Check rebalance period > 0
    with pytest.raises(ValueError):
        Backtest(strat, rebal_period = -1)

    with pytest.raises(ValueError):
        Backtest(strat, rebal_period = 0)

    with pytest.raises(ValueError):
        Backtest(strat, rebal_period = -1.5)

    with pytest.raises(TypeError):
        Backtest(strat, rebal_period = 1.5)

    assert Backtest(strat, rebal_period = 1).rebal_period == 1
    assert Backtest(strat, rebal_period = 5).rebal_period == 5

    assert Backtest(strat).rebal_period == None

def test_dates():
    strat = StrategyTest(data = ["AAPL", "IBM"])

    # Check start_date <= end_date
    with pytest.raises(ValueError):
        start_date = date.fromisoformat('2015-12-31')
        end_date = date.fromisoformat('2015-12-01')
        Backtest(strat, start_date = start_date, end_date = end_date)

    with pytest.raises(ValueError):
        start_date = date.fromisoformat('2013-03-04')
        end_date = date.fromisoformat('2010-05-29')
        Backtest(strat, start_date = start_date, end_date = end_date)

    start_date = date.fromisoformat('2015-12-01')
    end_date = date.fromisoformat('2015-12-31')
    backtest = Backtest(strat, start_date = start_date, end_date = end_date)
    assert backtest.start_date == start_date
    assert backtest.end_date == end_date

    start_date = date.fromisoformat('2011-03-01')
    end_date = date.fromisoformat('2017-11-18')
    backtest = Backtest(strat, start_date = start_date, end_date = end_date)
    assert backtest.start_date == start_date
    assert backtest.end_date == end_date

    start_date = date.fromisoformat('2015-12-01')
    end_date = date.fromisoformat('2015-12-01')
    backtest = Backtest(strat, start_date = start_date, end_date = end_date)
    assert backtest.start_date == backtest.end_date

    backtest = Backtest(strat)
    assert backtest.start_date == backtest.price_data.index[0]
    assert backtest.end_date == backtest.price_data.index[-1]

def test_get_return():
    strat = StrategyTest(data = ["AAPL", "IBM"])
    backtest = Backtest(strat)

    assert type(backtest.get_return()) == np.ndarray
    assert type(backtest.get_return(cumulative = True)) == np.ndarray

    assert len(backtest.get_return()) == len(backtest.price_data)
    assert len(backtest.get_return(cumulative = True)) == len(backtest.price_data)

    returns = backtest.get_return()
    cum_returns = backtest.get_return(cumulative = True)
    assert list(np.cumprod([1 + ret for ret in returns]) - 1) == list(cum_returns)

def test_get_annualized_return():
    strat = StrategyTest(data = ["AAPL", "IBM"])
    backtest = Backtest(strat)

    assert backtest.get_annualized_return() >= -1
    assert type(backtest.get_annualized_return()) == np.float64

    assert backtest.get_annualized_return(daily_returns = np.array([0.001, 0.002, 0.005])) == pytest.approx(0.95, abs=0.01)
    assert backtest.get_annualized_return(daily_returns = np.array([-0.001, -0.002, -0.005])) == pytest.approx(-0.49, abs=0.01)
    assert backtest.get_annualized_return(daily_returns = np.array([-0.001, 0.002, -0.005])) == pytest.approx(-0.29, abs=0.01)

def test_vol():
    strat = StrategyTest(data = ["AAPL", "IBM"])
    backtest = Backtest(strat)

    # test get_annualized_vol
    assert type(backtest.get_annualized_vol()) == np.float64
    assert backtest.get_annualized_vol() >= 0

    assert backtest.get_annualized_vol(returns = np.array([0.001, 0.002, 0.005])) == pytest.approx(0.027, abs=0.001)
    assert backtest.get_annualized_vol(returns = np.array([-0.001, -0.002, -0.005])) == pytest.approx(0.027, abs=0.001)
    assert backtest.get_annualized_vol(returns = np.array([-0.001, 0.002, -0.005])) == pytest.approx(0.046, abs=0.001)

    vol_original = backtest.get_annualized_vol(returns = np.array([0.003, 0.008, -0.009]))
    vol_shifted_up = backtest.get_annualized_vol(returns = np.array([0.003, 0.008, -0.009]) + 0.001)
    vol_shifted_down = backtest.get_annualized_vol(returns = np.array([0.003, 0.008, -0.009]) - 0.001)
    vol_reflected = backtest.get_annualized_vol(returns = -np.array([0.003, 0.008, -0.009]))
    assert vol_original == pytest.approx(vol_shifted_up, rel=0.001)
    assert vol_original == pytest.approx(vol_shifted_down, rel=0.001)
    assert vol_original == pytest.approx(vol_reflected, rel=0.001)

    vol_scaled = backtest.get_annualized_vol(returns = 2*np.array([0.003, 0.008, -0.009]))
    assert vol_scaled == pytest.approx(2*vol_original, rel=0.001)

    vol_neg_scaled = backtest.get_annualized_vol(returns = -2*np.array([0.003, 0.008, -0.009]))
    assert vol_neg_scaled == pytest.approx(2*vol_original, rel=0.001)

    # test get_annualized_downside_vol
    assert type(backtest.get_annualized_downside_vol()) == np.float64
    assert backtest.get_annualized_downside_vol() >= 0

    assert backtest.get_annualized_downside_vol(returns = np.array([0.001, 0.002, 0.005])) == 0
    assert backtest.get_annualized_downside_vol(returns = np.array([-0.001, -0.002, -0.005])) == pytest.approx(0.027, abs=0.001)
    assert backtest.get_annualized_downside_vol(returns = np.array([-0.001, 0.002, -0.005])) == pytest.approx(0.034, abs=0.001)

    downside_vol = backtest.get_annualized_downside_vol(returns = np.array([-0.004, -0.003, -0.007]))
    regular_vol = backtest.get_annualized_vol(returns = np.array([-0.004, -0.003, -0.007]))
    assert downside_vol == regular_vol

    vol_original = backtest.get_annualized_downside_vol(returns = np.array([0.003, 0.008, -0.009]))
    vol_shifted_up = backtest.get_annualized_downside_vol(returns = np.array([0.003, 0.008, -0.009]) + 0.001)
    vol_shifted_down = backtest.get_annualized_downside_vol(returns = np.array([0.003, 0.008, -0.009]) - 0.001)
    assert vol_original >= vol_shifted_up
    assert vol_original <= vol_shifted_down

def test_sharpe():
    strat = StrategyTest(data = ["AAPL", "IBM"])
    backtest = Backtest(strat)

    # test sortino
    assert type(backtest.get_sharpe()) == np.float64

    # test rolling_sortino
    with pytest.raises(ValueError):
        backtest.get_rolling_sharpe(lookback = 2)

    with pytest.raises(ValueError):
        backtest.get_rolling_sharpe(lookback = 1e5)

    with pytest.raises(TypeError):
        backtest.get_rolling_sharpe(lookback = 10.5)

    assert len(backtest.get_rolling_sharpe(lookback = 10)) == len(backtest.price_data) - 10

def test_sortino():
    strat = StrategyTest(data = ["AAPL", "IBM"])
    backtest = Backtest(strat)

    # test sortino
    assert type(backtest.get_sortino()) == np.float64

    # test rolling_sortino
    with pytest.raises(ValueError):
        backtest.get_rolling_sortino(lookback = 2)

    with pytest.raises(ValueError):
        backtest.get_rolling_sortino(lookback = 1e5)

    with pytest.raises(TypeError):
        backtest.get_rolling_sortino(lookback = 10.5)

    assert len(backtest.get_rolling_sortino(lookback = 10)) == len(backtest.price_data) - 10

def test_get_avg_underwater_time():
    strat = StrategyTest(data = ["AAPL", "IBM"])
    backtest = Backtest(strat)

    with pytest.raises(TypeError):
        backtest.get_avg_underwater_time(threshold_days = 10.5)

    with pytest.raises(ValueError):
        backtest.get_avg_underwater_time(threshold_days = 1)

    assert backtest.get_avg_underwater_time() >= 0
    assert backtest.get_avg_underwater_time() < len(strat.data.index)
    assert type(backtest.get_avg_underwater_time()) == np.float64

def test_get_max_underwater_time():
    strat = StrategyTest(data = ["AAPL", "IBM"])
    backtest = Backtest(strat)

    with pytest.raises(TypeError):
        backtest.get_max_underwater_time(threshold_days = 10.5)

    with pytest.raises(ValueError):
        backtest.get_max_underwater_time(threshold_days = 1)

    assert backtest.get_max_underwater_time() >= 0
    assert backtest.get_max_underwater_time() < len(strat.data.index)
    assert type(backtest.get_max_underwater_time()) == int

def test_get_underwater_time():
    strat = StrategyTest(data = ["AAPL", "IBM"])
    backtest = Backtest(strat)

    with pytest.raises(TypeError):
        backtest.get_underwater_time(threshold_days = 10.5)

    with pytest.raises(ValueError):
        backtest.get_underwater_time(threshold_days = 1)

    assert type(backtest.get_underwater_time()) == tuple

def test_get_max_consecutive():
    strat = StrategyTest(data = ["AAPL", "IBM"])
    backtest = Backtest(strat)

    assert backtest.get_max_consecutive_positive() >= 0
    assert backtest.get_max_consecutive_positive() <= len(strat.data.index)
    assert type(backtest.get_max_consecutive_positive()) == int

    assert backtest.get_max_consecutive_negative() >= 0
    assert backtest.get_max_consecutive_negative() <= len(strat.data.index)
    assert type(backtest.get_max_consecutive_negative()) == int

def test_get_max_drawdown():
    strat = StrategyTest(data = ["AAPL", "IBM"])
    backtest = Backtest(strat)

    assert backtest.get_max_drawdown() >= 0
    assert type(backtest.get_max_drawdown()) == np.float64