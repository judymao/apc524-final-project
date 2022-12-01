# type: ignore

import numpy as np
import pandas as pd
import pytest

from src.tradestrat.strategies import MachineLearningMethod, Momentum, Value

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

ml_test = MachineLearningMethod(data=["AAPL", "IBM", "A", "MSFT", "AMZN"])


def test_mom_weight_sum():
    assert mom_test.weights.sum(axis=1).values == pytest.approx(0, abs=10e-10)


def test_mom_weight_legs():
    assert mom_test.weights[mom_test.weights > 0].sum(axis=1).values == pytest.approx(
        1, abs=10e-10
    )

    assert mom_test.weights[mom_test.weights < 0].sum(axis=1).values == pytest.approx(
        -1, abs=10e-10
    )


def test_mom_intraleg_weights():
    assert mom_test.weights[mom_test.weights > 0].apply(
        pd.Series.nunique, axis=1
    ).values == pytest.approx(1, abs=10e-10)

    assert mom_test.weights[mom_test.weights < 0].apply(
        pd.Series.nunique, axis=1
    ).values == pytest.approx(1, abs=10e-10)


def test_ML_predict_returns():
    with pytest.raises(ValueError):
        ml_test.predict_returns(model="")

    with pytest.raises(ValueError):
        ml_test.predict_returns(lookahead=0)

    with pytest.raises(ValueError):
        ml_test.predict_returns(lookahead=1e5)

    with pytest.raises(ValueError):
        ml_test.predict_returns(max_lag=0)

    with pytest.raises(ValueError):
        ml_test.predict_returns(max_lag=1e5)

    with pytest.raises(TypeError):
        ml_test.predict_returns(daily=0)

    # Test ML for default parameters for first two stocks
    prediction = ml_test.predict_returns()["pred_return"].values
    assert prediction[0] == pytest.approx(0.00122, 0.1)
    assert prediction[1] == pytest.approx(0.00022, 0.1)

    # Test ML for annual return for first two stocks
    prediction = ml_test.predict_returns(daily=False)["pred_return"].values
    assert prediction[0] == pytest.approx(0.479, 0.1)
    assert prediction[1] == pytest.approx(0.116, 0.1)
