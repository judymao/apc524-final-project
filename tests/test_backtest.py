# type: ignore

import os
from datetime import date

import numpy as np
import pytest

from src.tradestrat import Backtest, Strategy


class StrategyTest(Strategy):
    def get_weights(self):
        """
        Get strategy weights over time
        Args:
            None
        Return:
            DataFrame containing dates along rows and tickers along columns, with values being the strategy weights
        """
        weights_df = self.data.copy()
        weights_df[:] = 1 / len(self.data.columns)
        return weights_df


def test_tcost():
    strat = StrategyTest(data=["AAPL", "IBM"])

    # Check transaction cost >= 0
    with pytest.raises(ValueError):
        Backtest(strategy=strat, transaction_cost=-1)

    with pytest.raises(ValueError):
        Backtest(strategy=strat, transaction_cost=-1.5)

    assert Backtest(strategy=strat, transaction_cost=0).transaction_cost == 0
    assert Backtest(strategy=strat, transaction_cost=1).transaction_cost == 1
    assert Backtest(strategy=strat, transaction_cost=1.6).transaction_cost == 1.6

    assert Backtest(strategy=strat).transaction_cost == 0


def test_rebal_period():
    strat = StrategyTest(data=["AAPL", "IBM"])

    # Check rebalance period > 0
    with pytest.raises(ValueError):
        Backtest(strat, rebal_period=-1)

    with pytest.raises(ValueError):
        Backtest(strat, rebal_period=0)

    with pytest.raises(ValueError):
        Backtest(strat, rebal_period=-1.5)

    with pytest.raises(TypeError):
        Backtest(strat, rebal_period=1.5)

    assert Backtest(strat, rebal_period=1).rebal_period == 1
    assert Backtest(strat, rebal_period=5).rebal_period == 5

    assert Backtest(strat).rebal_period == None


def test_dates():
    strat = StrategyTest(data=["AAPL", "IBM"])

    # Check start_date <= end_date
    with pytest.raises(ValueError):
        start_date = date.fromisoformat("2015-12-31")
        end_date = date.fromisoformat("2015-12-01")
        Backtest(strat, start_date=start_date, end_date=end_date)

    with pytest.raises(ValueError):
        start_date = date.fromisoformat("2013-03-04")
        end_date = date.fromisoformat("2010-05-29")
        Backtest(strat, start_date=start_date, end_date=end_date)

    start_date = date.fromisoformat("2015-12-01")
    end_date = date.fromisoformat("2015-12-31")
    backtest = Backtest(strat, start_date=start_date, end_date=end_date)
    assert backtest.start_date == start_date
    assert backtest.end_date == end_date

    start_date = date.fromisoformat("2011-03-01")
    end_date = date.fromisoformat("2017-11-18")
    backtest = Backtest(strat, start_date=start_date, end_date=end_date)
    assert backtest.start_date == start_date
    assert backtest.end_date == end_date

    start_date = date.fromisoformat("2015-12-01")
    end_date = date.fromisoformat("2015-12-01")
    backtest = Backtest(strat, start_date=start_date, end_date=end_date)
    assert backtest.start_date == backtest.end_date

    backtest = Backtest(strat)
    assert backtest.start_date == backtest.price_data.index[0]
    assert backtest.end_date == backtest.price_data.index[-1]


def test_get_return():
    strat = StrategyTest(data=["AAPL", "IBM"])
    backtest = Backtest(strat)

    returns = backtest.get_return()
    cum_returns = backtest.get_return(cumulative=True)

    assert type(returns) == np.ndarray
    assert type(cum_returns) == np.ndarray

    assert len(returns) == len(backtest.price_data)
    assert len(cum_returns) == len(backtest.price_data)

    assert list(np.cumprod([1 + ret for ret in returns]) - 1) == list(cum_returns)

    assert not np.isnan(returns).any()
    assert not np.isnan(cum_returns).any()

    assert not np.isinf(returns).any()
    assert not np.isinf(cum_returns).any()


def test_get_annualized_return():
    strat = StrategyTest(data=["AAPL", "IBM"])
    backtest = Backtest(strat)

    annualized_return = backtest.get_annualized_return()
    assert annualized_return >= -1
    assert type(annualized_return) == np.float64

    assert backtest.get_annualized_return(
        daily_returns=np.array([0.001, 0.002, 0.005])
    ) == pytest.approx(0.95, abs=0.01)
    assert backtest.get_annualized_return(
        daily_returns=np.array([-0.001, -0.002, -0.005])
    ) == pytest.approx(-0.49, abs=0.01)
    assert backtest.get_annualized_return(
        daily_returns=np.array([-0.001, 0.002, -0.005])
    ) == pytest.approx(-0.29, abs=0.01)

    assert not np.isnan(annualized_return)
    assert not np.isinf(annualized_return)


def test_vol():
    strat = StrategyTest(data=["AAPL", "IBM"])
    backtest = Backtest(strat)

    # test get_annualized_vol
    annual_vol = backtest.get_annualized_vol()
    assert type(annual_vol) == np.float64
    assert annual_vol >= 0
    assert not np.isinf(annual_vol)
    assert not np.isnan(annual_vol)

    assert backtest.get_annualized_vol(
        returns=np.array([0.001, 0.002, 0.005])
    ) == pytest.approx(0.027, abs=0.001)
    assert backtest.get_annualized_vol(
        returns=np.array([-0.001, -0.002, -0.005])
    ) == pytest.approx(0.027, abs=0.001)
    assert backtest.get_annualized_vol(
        returns=np.array([-0.001, 0.002, -0.005])
    ) == pytest.approx(0.046, abs=0.001)

    vol_original = backtest.get_annualized_vol(returns=np.array([0.003, 0.008, -0.009]))
    vol_shifted_up = backtest.get_annualized_vol(
        returns=np.array([0.003, 0.008, -0.009]) + 0.001
    )
    vol_shifted_down = backtest.get_annualized_vol(
        returns=np.array([0.003, 0.008, -0.009]) - 0.001
    )
    vol_reflected = backtest.get_annualized_vol(
        returns=-np.array([0.003, 0.008, -0.009])
    )
    assert vol_original == pytest.approx(vol_shifted_up, rel=0.001)
    assert vol_original == pytest.approx(vol_shifted_down, rel=0.001)
    assert vol_original == pytest.approx(vol_reflected, rel=0.001)

    vol_scaled = backtest.get_annualized_vol(
        returns=2 * np.array([0.003, 0.008, -0.009])
    )
    assert vol_scaled == pytest.approx(2 * vol_original, rel=0.001)

    vol_neg_scaled = backtest.get_annualized_vol(
        returns=-2 * np.array([0.003, 0.008, -0.009])
    )
    assert vol_neg_scaled == pytest.approx(2 * vol_original, rel=0.001)

    # test get_annualized_downside_vol
    annual_downside_vol = backtest.get_annualized_downside_vol()
    assert type(annual_downside_vol) == np.float64
    assert annual_downside_vol >= 0
    assert not np.isinf(annual_downside_vol)
    assert not np.isnan(annual_downside_vol)

    assert (
        backtest.get_annualized_downside_vol(returns=np.array([0.001, 0.002, 0.005]))
        == 0
    )
    assert backtest.get_annualized_downside_vol(
        returns=np.array([-0.001, -0.002, -0.005])
    ) == pytest.approx(0.027, abs=0.001)
    assert backtest.get_annualized_downside_vol(
        returns=np.array([-0.001, 0.002, -0.005])
    ) == pytest.approx(0.034, abs=0.001)

    downside_vol = backtest.get_annualized_downside_vol(
        returns=np.array([-0.004, -0.003, -0.007])
    )
    regular_vol = backtest.get_annualized_vol(
        returns=np.array([-0.004, -0.003, -0.007])
    )
    assert downside_vol == regular_vol

    vol_original = backtest.get_annualized_downside_vol(
        returns=np.array([0.003, 0.008, -0.009])
    )
    vol_shifted_up = backtest.get_annualized_downside_vol(
        returns=np.array([0.003, 0.008, -0.009]) + 0.001
    )
    vol_shifted_down = backtest.get_annualized_downside_vol(
        returns=np.array([0.003, 0.008, -0.009]) - 0.001
    )
    assert vol_original >= vol_shifted_up
    assert vol_original <= vol_shifted_down


def test_sharpe():
    strat = StrategyTest(data=["AAPL", "IBM"])
    backtest = Backtest(strat)

    # test sortino
    sharpe = backtest.get_sharpe()
    assert type(sharpe) == np.float64

    # test rolling_sortino
    with pytest.raises(ValueError):
        backtest.get_rolling_sharpe(lookback=2)

    with pytest.raises(ValueError):
        backtest.get_rolling_sharpe(lookback=1e5)

    with pytest.raises(TypeError):
        backtest.get_rolling_sharpe(lookback=10.5)

    assert (
        len(backtest.get_rolling_sharpe(lookback=10)) == len(backtest.price_data) - 10
    )

    assert not np.isnan(sharpe)
    assert not np.isinf(sharpe)


def test_sortino():
    strat = StrategyTest(data=["AAPL", "IBM"])
    backtest = Backtest(strat)

    # test sortino
    sortino = backtest.get_sortino()
    assert type(sortino) == np.float64

    # test rolling_sortino
    with pytest.raises(ValueError):
        backtest.get_rolling_sortino(lookback=2)

    with pytest.raises(ValueError):
        backtest.get_rolling_sortino(lookback=1e5)

    with pytest.raises(TypeError):
        backtest.get_rolling_sortino(lookback=10.5)

    assert (
        len(backtest.get_rolling_sortino(lookback=10)) == len(backtest.price_data) - 10
    )

    assert not np.isnan(sortino)
    assert not np.isinf(sortino)


def test_get_avg_underwater_time():
    strat = StrategyTest(data=["AAPL", "IBM"])
    backtest = Backtest(strat)

    with pytest.raises(TypeError):
        backtest.get_avg_underwater_time(threshold_days=10.5)

    with pytest.raises(ValueError):
        backtest.get_avg_underwater_time(threshold_days=1)

    avg_underwater_time = backtest.get_avg_underwater_time()
    assert avg_underwater_time >= 0
    assert avg_underwater_time < len(strat.data.index)
    assert type(avg_underwater_time) == np.float64

    assert not np.isnan(avg_underwater_time)
    assert not np.isinf(avg_underwater_time)


def test_get_max_underwater_time():
    strat = StrategyTest(data=["AAPL", "IBM"])
    backtest = Backtest(strat)

    with pytest.raises(TypeError):
        backtest.get_max_underwater_time(threshold_days=10.5)

    with pytest.raises(ValueError):
        backtest.get_max_underwater_time(threshold_days=1)

    max_underwater_time = backtest.get_max_underwater_time()
    assert max_underwater_time >= 0
    assert max_underwater_time < len(strat.data.index)
    assert type(max_underwater_time) == int

    assert not np.isnan(max_underwater_time)
    assert not np.isinf(max_underwater_time)


def test_get_underwater_time():
    strat = StrategyTest(data=["AAPL", "IBM"])
    backtest = Backtest(strat)

    with pytest.raises(TypeError):
        backtest.get_underwater_time(threshold_days=10.5)

    with pytest.raises(ValueError):
        backtest.get_underwater_time(threshold_days=1)

    underwater_time = backtest.get_underwater_time()
    assert type(underwater_time) == tuple

    assert not np.isnan(underwater_time).any()
    assert not np.isinf(underwater_time).any()


def test_get_max_consecutive():
    strat = StrategyTest(data=["AAPL", "IBM"])
    backtest = Backtest(strat)

    assert backtest.get_max_consecutive_positive() >= 0
    assert backtest.get_max_consecutive_positive() <= len(strat.data.index)
    assert type(backtest.get_max_consecutive_positive()) == int

    assert backtest.get_max_consecutive_negative() >= 0
    assert backtest.get_max_consecutive_negative() <= len(strat.data.index)
    assert type(backtest.get_max_consecutive_negative()) == int

    assert not np.isnan(backtest.get_max_consecutive_positive())
    assert not np.isinf(backtest.get_max_consecutive_positive())

    assert not np.isnan(backtest.get_max_consecutive_negative())
    assert not np.isinf(backtest.get_max_consecutive_negative())


def test_get_max_drawdown():
    strat = StrategyTest(data=["AAPL", "IBM"])
    backtest = Backtest(strat)

    max_drawdown = backtest.get_max_drawdown()
    assert max_drawdown >= 0
    assert type(max_drawdown) == np.float64

    assert not np.isnan(max_drawdown)
    assert not np.isinf(max_drawdown)
