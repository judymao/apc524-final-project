# type: ignore
import os

import numpy as np
import pandas as pd
import pytest

from src.tradestrat import MachineLearningMethod, Momentum, Value

DATA = pd.read_csv("src/tradestrat/data/sp500_prices.csv")

MOM_TEST = Momentum(
    data={
        "returns": (
            np.log(DATA.set_index("Date")) - np.log(DATA.set_index("Date").shift(1))
        ),
        "price": DATA,
    },
    lookback_period=6,
    min_periods=120,
    skip_period=21,
    perc=10,
)

ML_TEST = MachineLearningMethod(data=["AAPL", "IBM", "A", "MSFT", "AMZN"])


@pytest.mark.parametrize("strategy", [MOM_TEST, ML_TEST])
def test_weights(strategy):
    assert strategy.weights.sum(axis=1).values == pytest.approx(0, abs=10e-10)

    assert strategy.weights[strategy.weights > 0].sum(axis=1).values == pytest.approx(
        1, abs=10e-10
    )

    assert strategy.weights[strategy.weights < 0].sum(axis=1).values == pytest.approx(
        -1, abs=10e-10
    )


@pytest.mark.parametrize("strategy", [MOM_TEST])
def test_weights(strategy):
    assert strategy.weights[strategy.weights > 0].apply(
        pd.Series.nunique, axis=1
    ).values == pytest.approx(1, abs=10e-10)

    assert strategy.weights[strategy.weights < 0].apply(
        pd.Series.nunique, axis=1
    ).values == pytest.approx(1, abs=10e-10)


def test_ML_predict_returns():
    with pytest.raises(ValueError):
        ML_TEST.predict_returns(model="")

    with pytest.raises(ValueError):
        ML_TEST.predict_returns(lookahead=0)

    with pytest.raises(ValueError):
        ML_TEST.predict_returns(lookahead=1e5)

    with pytest.raises(ValueError):
        ML_TEST.predict_returns(max_lag=0)

    with pytest.raises(ValueError):
        ML_TEST.predict_returns(max_lag=1e5)

    with pytest.raises(TypeError):
        ML_TEST.predict_returns(daily=0)

    # Test ML for default parameters for first two stocks
    prediction = ML_TEST.predict_returns()["pred_return"].values
    assert prediction[0] == pytest.approx(0.00122, 0.1)
    assert prediction[1] == pytest.approx(0.00022, 0.1)

    # Test ML for annual return for first two stocks
    prediction = ML_TEST.predict_returns(daily=False)["pred_return"].values
    assert prediction[0] == pytest.approx(0.479, 0.1)
    assert prediction[1] == pytest.approx(0.116, 0.1)
