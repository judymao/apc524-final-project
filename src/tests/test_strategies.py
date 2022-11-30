import numpy as np
import pandas as pd

from tradestrat.strategies import Momentum, Strategy, Value

DATA = pd.read_csv(
    "/Users/lipe/Documents/Mestrado/3 Semester/Software Engineering/PROJECT/apc524-final-project/tradestrat/data/sp500_prices.csv"
)

px_last = pd.read_csv(
    "/Users/lipe/Documents/Mestrado/3 Semester/Software Engineering/PROJECT/apc524-final-project/"
    "tradestrat/data/sp500_prices.csv",
    index_col="Date",
)
ret_data = np.log(px_last) - np.log(px_last.shift(1))

mom_data_dict = {"returns": ret_data, "price": DATA}


mom_test = Momentum(
    data=mom_data_dict, lookback_period=6, min_periods=120, skip_period=21, perc=10
)


def test_mom_weight_sum():
    assert (mom_test.weights.sum(axis=1) == pytest.approx(0)).all()


def test_mom_weight_legs():
    assert (
        mom_test.weights[mom_test.weights > 0].sum(axis=1) == pytest.approx(1)
    ).all()
    assert (
        mom_test.weights[mom_test.weights < 0].sum(axis=1) == pytest.approx(-1)
    ).all()


def test_mom_intraleg_weights():
    assert (
        mom_test.weights[mom_test.weights > 0].apply(pd.Series.nunique, axis=1) == 1
    ).all()
    assert (
        mom_test.weights[mom_test.weights < 0].apply(pd.Series.nunique, axis=1) == -1
    ).all()
