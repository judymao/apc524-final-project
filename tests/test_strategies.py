# type: ignore
import os

import numpy as np
import openpyxl
import pandas as pd
import pytest

from src.tradestrat.strategies import LO2MA, MLStrt, Momentum, TrendFollowing, Value

DATA = pd.read_csv(
    "/Users/lipe/Documents/Mestrado/3 Semester/Software Engineering/PROJECT/apc524-final-project/src/tradestrat/data/sp500_prices.csv"
)
# DATA = pd.read_csv("src/tradestrat/data/sp500_prices.csv")

pe_data = pd.read_excel(
    "/Users/lipe/Documents/Mestrado/3 Semester/Software Engineering/PROJECT/apc524-final-project/src/tradestrat/data/pe_data.xlsx",
    index_col=0,
)
pe_data.columns = pe_data.columns.str.replace(" UN Equity", "")
pe_data.columns = pe_data.columns.str.replace(" UW Equity", "")
pe_data = pe_data.loc[:, DATA.columns.values[1:]]
pe_data = pe_data.dropna(axis=1, how="any")

VALUE_TEST_PE = Value({"price": DATA, "P_E": pe_data}, signal_name="P_E", perc=0.1)
MOM_TEST = Momentum(
    ["all"], lookback_period=6, min_periods=120, skip_period=21, perc=0.1
)
TF_TEST = TrendFollowing(["all"])
ML_TEST = MLStrt(data=["AAPL", "IBM", "A", "MSFT", "AMZN"])
LO2MA_TEST = LO2MA(data=["all"])

A = TF_TEST.weights


@pytest.mark.parametrize("strategy", [Momentum])
def test_mom_init(strategy):
    with pytest.raises(ValueError):
        strategy(
            ["A", "AAPL", "ABC", "ABMD", "ABT", "ADBE", "ADI", "ADM", "ADP", "ADSK"],
            perc=0.01,
        )

    with pytest.raises(ValueError):
        strategy(
            ["A", "AAPL", "ABC", "ABMD", "ABT", "ADBE", "ADI", "ADM", "ADP", "ADSK"],
            lookback_period=10000,
        )

    with pytest.raises(ValueError):
        strategy(
            ["A", "AAPL", "ABC", "ABMD", "ABT", "ADBE", "ADI", "ADM", "ADP", "ADSK"],
            lookback_period=-1,
        )

    with pytest.raises(ValueError):
        strategy(
            ["A", "AAPL", "ABC", "ABMD", "ABT", "ADBE", "ADI", "ADM", "ADP", "ADSK"],
            perc="1",
        )

    with pytest.raises(ValueError):
        strategy(
            ["A", "AAPL", "ABC", "ABMD", "ABT", "ADBE", "ADI", "ADM", "ADP", "ADSK"],
            perc=5,
        )

    with pytest.raises(ValueError):
        strategy(
            ["A", "AAPL", "ABC", "ABMD", "ABT", "ADBE", "ADI", "ADM", "ADP", "ADSK"],
            perc=-0.5,
        )

    with pytest.raises(ValueError):
        strategy(
            ["A", "AAPL", "ABC", "ABMD", "ABT", "ADBE", "ADI", "ADM", "ADP", "ADSK"],
            skip_period="1",
        )

    with pytest.raises(ValueError):
        strategy(
            ["A", "AAPL", "ABC", "ABMD", "ABT", "ADBE", "ADI", "ADM", "ADP", "ADSK"],
            skip_period=-1,
        )

    with pytest.raises(ValueError):
        strategy(
            ["A", "AAPL", "ABC", "ABMD", "ABT", "ADBE", "ADI", "ADM", "ADP", "ADSK"],
            skip_period=50000,
        )

    with pytest.raises(ValueError):
        strategy(
            ["A", "AAPL", "ABC", "ABMD", "ABT", "ADBE", "ADI", "ADM", "ADP", "ADSK"],
            min_periods=-1,
        )

    with pytest.raises(ValueError):
        strategy(
            ["A", "AAPL", "ABC", "ABMD", "ABT", "ADBE", "ADI", "ADM", "ADP", "ADSK"],
            min_periods=50,
            lookback_period=1,
        )


@pytest.mark.parametrize("strategy", [Value])
def test_Value_init(strategy):
    with pytest.raises(ValueError):
        strategy({"price": DATA, "P_E": pe_data}, perc="1")

    with pytest.raises(ValueError):
        strategy({"price": DATA, "P_E": pe_data}, perc=5)

    with pytest.raises(ValueError):
        strategy({"price": DATA, "P_E": pe_data}, perc=-0.5)

    with pytest.raises(ValueError):
        strategy({"price": DATA}, perc=-0.5)


@pytest.mark.parametrize("strategy", [TrendFollowing])
def test_TrendFollowing_init(strategy):

    with pytest.raises(ValueError):
        strategy(
            ["A", "AAPL", "ABC", "ABMD", "ABT", "ADBE", "ADI", "ADM", "ADP", "ADSK"],
            skip_period="1",
        )

    with pytest.raises(ValueError):
        strategy(
            ["A", "AAPL", "ABC", "ABMD", "ABT", "ADBE", "ADI", "ADM", "ADP", "ADSK"],
            min_periods=-1,
        )

    with pytest.raises(ValueError):
        strategy(
            ["A", "AAPL", "ABC", "ABMD", "ABT", "ADBE", "ADI", "ADM", "ADP", "ADSK"],
            min_periods=50,
            wind=1,
        )

    with pytest.raises(ValueError):
        strategy(
            ["A", "AAPL", "ABC", "ABMD", "ABT", "ADBE", "ADI", "ADM", "ADP", "ADSK"],
            risk_free="True",
        )

    with pytest.raises(ValueError):
        strategy(
            ["A", "AAPL", "ABC", "ABMD", "ABT", "ADBE", "ADI", "ADM", "ADP", "ADSK"],
            wind=-1,
        )

    with pytest.raises(ValueError):
        strategy(
            ["A", "AAPL", "ABC", "ABMD", "ABT", "ADBE", "ADI", "ADM", "ADP", "ADSK"],
            wind=50000,
        )

    with pytest.raises(ValueError):
        strategy(
            ["A", "AAPL", "ABC", "ABMD", "ABT", "ADBE", "ADI", "ADM", "ADP", "ADSK"],
            wind="1",
        )


@pytest.mark.parametrize("strategy", [LO2MA])
def test_LO2MA_init(strategy):
    with pytest.raises(ValueError):
        strategy(["all"], MA_long_wind=20, MA_short_wind=50)

    with pytest.raises(ValueError):
        strategy(["all"], MA_long_wind=50.5)

    with pytest.raises(ValueError):
        strategy(["all"], MA_short_wind=20.5)

    with pytest.raises(ValueError):
        strategy(["all"], MA_long_wind=-50)

    with pytest.raises(ValueError):
        strategy(["all"], MA_short_wind=-10)

    with pytest.raises(ValueError):
        strategy(["all"], MA_long_wind=50000)

    with pytest.raises(ValueError):
        strategy(["all"], skip_period="1")

    with pytest.raises(ValueError):
        strategy(["all"], skip_period=-1)

    with pytest.raises(ValueError):
        strategy(["all"], skip_period=50000)

    with pytest.raises(ValueError):
        strategy(["all"], min_periods=-1)


@pytest.mark.parametrize("strategy", [MLStrt])
def test_MLStrt_init(strategy):
    with pytest.raises(ValueError):
        strategy(["all"], model="")

    with pytest.raises(ValueError):
        strategy(["all"], lookahead=0)

    with pytest.raises(ValueError):
        strategy(["all"], lookahead=1e5)

    with pytest.raises(ValueError):
        strategy(["all"], max_lag=0)

    with pytest.raises(ValueError):
        strategy(["all"], max_lag=1e5)


@pytest.mark.parametrize("strategy", [MOM_TEST, VALUE_TEST_PE, TF_TEST, ML_TEST])
def test_weights_ls(strategy):
    assert strategy.weights.sum(axis=1).values == pytest.approx(0, abs=10e-10)

    assert strategy.weights[strategy.weights > 0].sum(axis=1).values == pytest.approx(
        1, abs=10e-10
    )

    assert strategy.weights[strategy.weights < 0].sum(axis=1).values == pytest.approx(
        -1, abs=10e-10
    )


@pytest.mark.parametrize("strategy", [MOM_TEST, VALUE_TEST_PE])
def test_weights_intraleg(strategy):
    assert strategy.weights[strategy.weights > 0].apply(
        pd.Series.nunique, axis=1
    ).values == pytest.approx(1, abs=10e-10)

    assert strategy.weights[strategy.weights < 0].apply(
        pd.Series.nunique, axis=1
    ).values == pytest.approx(1, abs=10e-10)


@pytest.mark.parametrize("strategy", [LO2MA_TEST])
def test_weights_lo(strategy):
    assert strategy.weights.sum(axis=1).values == pytest.approx(1, abs=10e-10)


def test_ML_predict_returns():

    # Test ML for default parameters for first two stocks
    prediction = ML_TEST.predict_returns()["pred_return"].values
    assert prediction[0] == pytest.approx(0.00122, 0.1)
    assert prediction[1] == pytest.approx(0.00022, 0.1)

    # Test ML for annual return for first two stocks
    prediction = ML_TEST.predict_returns(daily=False)["pred_return"].values
    assert prediction[0] == pytest.approx(0.479, 0.1)
    assert prediction[1] == pytest.approx(0.116, 0.1)
