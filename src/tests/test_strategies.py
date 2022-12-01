# type: ignore
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from src.tradestrat.strategies import Momentum, Value

warnings.filterwarnings("ignore")

DATA = pd.read_csv("tradestrat/data/sp500_prices.csv")

px_last = pd.read_csv(
    "tradestrat/data/sp500_prices.csv",
    index_col="Date",
)
ret_data = np.log(px_last) - np.log(px_last.shift(1))

mom_data_dict = {"returns": ret_data, "price": DATA}


mom_test = Momentum(
    data=mom_data_dict, lookback_period=6, min_periods=120, skip_period=21, perc=10
)


def test_mom_weight_sum():
    result = []
    for v in mom_test.weights.sum(axis=1):
        result.append(v == pytest.approx(0, abs=10e-10))

    assert all(v == True for v in result)


def test_mom_weight_legs():
    result = []
    for v in mom_test.weights[mom_test.weights > 0].sum(axis=1):
        result.append(v == pytest.approx(1, abs=10e-10))

    assert all(v == True for v in result)

    result = []
    for v in mom_test.weights[mom_test.weights < 0].sum(axis=1):
        result.append(v == pytest.approx(-1, abs=10e-10))

    assert all(v == True for v in result)


def test_mom_intraleg_weights():
    result = []
    for v in mom_test.weights[mom_test.weights > 0].apply(pd.Series.nunique, axis=1):
        result.append(v == pytest.approx(1, abs=10e-10))

    assert all(v == True for v in result)

    result = []
    for v in mom_test.weights[mom_test.weights < 0].apply(pd.Series.nunique, axis=1):
        result.append(v == pytest.approx(1, abs=10e-10))

    assert all(v == True for v in result)
