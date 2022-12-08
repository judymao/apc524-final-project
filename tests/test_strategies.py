# type: ignore
import os

import pandas as pd
import pytest

from src.tradestrat.strategies import LO2MA, MLStrat, Momentum, TrendFollowing, Value

if os.getcwd()[-20:] == "apc524-final-project":
    DATA_PATH = os.path.join(os.getcwd(), "src/tradestrat/data/sp500_prices.csv")
else:
    DATA_PATH = os.path.join(
        os.path.relpath(os.path.dirname(__file__)), "data/sp500_prices.csv"
    )

DATA = pd.read_csv(DATA_PATH)

if os.getcwd()[-20:] == "apc524-final-project":
    PE_PATH = os.path.join(os.getcwd(), "src/tradestrat/data/pe_data.xlsx")
else:
    PE_PATH = os.path.join(
        os.path.relpath(os.path.dirname(__file__)), "data/pe_data.xlsx"
    )

PE_DATA = pd.read_excel(
    PE_PATH,
    index_col=0,
)

VALUE_TEST_PE = Value({"price": DATA, "P_E": PE_DATA}, signal_name="P_E", perc=0.1)
MOM_TEST = Momentum(
    ["all"], lookback_period=6, min_periods=120, skip_period=21, perc=0.1
)
TF_TEST = TrendFollowing(["all"])
ML_TEST = MLStrat(data=["AAPL", "IBM", "A", "MSFT", "AMZN"])
LO2MA_TEST = LO2MA(data=["all"])


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

    with pytest.raises(TypeError):
        strategy(
            ["A", "AAPL", "ABC", "ABMD", "ABT", "ADBE", "ADI", "ADM", "ADP", "ADSK"],
            perc="1",
        )

    with pytest.raises(ValueError):
        strategy(
            ["A", "AAPL", "ABC", "ABMD", "ABT", "ADBE", "ADI", "ADM", "ADP", "ADSK"],
            perc=5.1,
        )

    with pytest.raises(ValueError):
        strategy(
            ["A", "AAPL", "ABC", "ABMD", "ABT", "ADBE", "ADI", "ADM", "ADP", "ADSK"],
            perc=-0.5,
        )

    with pytest.raises(TypeError):
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
    with pytest.raises(TypeError):
        strategy({"price": DATA, "P_E": PE_DATA}, perc="1")

    with pytest.raises(ValueError):
        strategy({"price": DATA, "P_E": PE_DATA}, perc=5.1)

    with pytest.raises(ValueError):
        strategy({"price": DATA, "P_E": PE_DATA}, perc=-0.5)

    with pytest.raises(ValueError):
        strategy({"price": DATA}, perc=-0.5)


@pytest.mark.parametrize("strategy", [TrendFollowing])
def test_TrendFollowing_init(strategy):

    with pytest.raises(TypeError):
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

    with pytest.raises(TypeError):
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

    with pytest.raises(TypeError):
        strategy(
            ["A", "AAPL", "ABC", "ABMD", "ABT", "ADBE", "ADI", "ADM", "ADP", "ADSK"],
            wind="1",
        )


@pytest.mark.parametrize("strategy", [LO2MA])
def test_LO2MA_init(strategy):
    with pytest.raises(ValueError):
        strategy(["all"], MA_long_wind=20, MA_short_wind=50)

    with pytest.raises(TypeError):
        strategy(["all"], MA_long_wind=50.5)

    with pytest.raises(TypeError):
        strategy(["all"], MA_short_wind=20.5)

    with pytest.raises(ValueError):
        strategy(["all"], MA_long_wind=-50)

    with pytest.raises(ValueError):
        strategy(["all"], MA_short_wind=-10)

    with pytest.raises(ValueError):
        strategy(["all"], MA_long_wind=50000)

    with pytest.raises(TypeError):
        strategy(["all"], skip_period="1")

    with pytest.raises(ValueError):
        strategy(["all"], skip_period=-1)

    with pytest.raises(ValueError):
        strategy(["all"], skip_period=50000)

    with pytest.raises(ValueError):
        strategy(["all"], min_periods=-1)


@pytest.mark.parametrize("strategy", [MLStrat])
def test_MLStrat_init(strategy):
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


@pytest.mark.parametrize("strategy", [ML_TEST])
def test_ML_predict_returns(strategy):
    with pytest.raises(ValueError):
        strategy.predict_returns(model="")

    with pytest.raises(ValueError):
        strategy.predict_returns(lookahead=0)

    with pytest.raises(ValueError):
        strategy.predict_returns(lookahead=1e5)

    with pytest.raises(ValueError):
        strategy.predict_returns(max_lag=0)

    with pytest.raises(ValueError):
        strategy.predict_returns(max_lag=1e5)

    with pytest.raises(TypeError):
        strategy.predict_returns(daily=0)

    # Test ML for default parameters for first two stocks
    prediction = strategy.predict_returns()["pred_return"].values
    assert prediction[0] == pytest.approx(0.00122, 0.1)
    assert prediction[1] == pytest.approx(0.00022, 0.1)

    # Test ML for annual return for first two stocks
    prediction = strategy.predict_returns(daily=False)["pred_return"].values
    assert prediction[0] == pytest.approx(0.479, 0.1)
    assert prediction[1] == pytest.approx(0.116, 0.1)
