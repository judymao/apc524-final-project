from __future__ import annotations

import abc

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

DATA_PATH = "tradestrat/data/sp500_prices.csv"
DATA = pd.read_csv(DATA_PATH)


class Strategy(abc.ABC):
    def __init__(self, data: list[str] | dict[str, pd.DataFrame]) -> None:
        """
        Initialize Strategy class

        Args:
            data: list of tickers to be considered in universe OR
                  dictionary of DataFrames, each containing dates along rows and tickers along columns,
                  with one DataFrame per value (e.g. data = {'price': ..., 'PE': ...})

        Return:
            None
        """

        if type(data) == list:
            # TODO: CURRENTLY USING DUMMY DATA
            self.data = DATA[data].copy()
            self.data["Date"] = pd.to_datetime(DATA["Date"])
            self.data.set_index("Date", inplace=True)
            # TODO: raise errors
        elif type(data) == dict:
            self.data = data["price"]
        else:
            raise TypeError(
                "data must either be a list of tickers or a dictionary of dataframes"
            )
        self.weights = self.get_weights()

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


class Momentum(Strategy):
    def __init__(
        self,
        data: list[str] | dict[str, pd.DataFrame],
        lookback_period: int,
        min_periods: int | None,
        skip_period: int,
        perc: float,
    ) -> None:
        super().__init__(data)
        self.data = data
        self.lookback_period = lookback_period
        self.min_periods = min_periods
        self.skip_period = skip_period
        self.perc = perc

        if self.min_periods == None:
            self.min_periods = self.lookback_period - 1

        self.weights = self.get_weights()

    def equal_weights_ls(self, portfolio: pd.DataFrame) -> pd.DataFrame:
        """
        Equally Weighted long and short Porfolios

        :param portfolio: pd.DataFrame; dataframe with 1 in assets you want to buy and -1 in stocks you want to sell.
        :return: pd.DataFrame
        """
        port_long = portfolio.where(portfolio > 0, 0)
        port_short = portfolio.where(portfolio < 0, 0)
        n_long = (port_long.fillna(0) != 0).astype(int).sum(axis=1)
        n_short = (port_long.fillna(0) != 0).astype(int).sum(axis=1)
        port_long = port_long.div(n_long, axis=0)
        port_short = port_short.div(n_long, axis=0)
        port_long[np.isnan(port_long)] = 0
        port_short[np.isnan(port_short)] = 0

        final_df = port_long + port_short
        return final_df

    def get_weights(self) -> pd.DataFrame:
        """
        Calculates the desired weights of this strategy per day per asset.
        :return:
            pd.DataFrame
        """
        # Signal
        ret_data = self.data["returns"]
        self.lookback = (
            np.exp(
                ret_data.shift(self.skip_period)
                .rolling(21 * self.lookback_period)
                .sum()
            )
            - 1
        )
        self.rsd = np.exp(ret_data).rolling(21 * self.lookback_period).std()
        self._signal = self.lookback / self.rsd

        # Weights
        final_ret = self._signal.copy()
        rank = final_ret.rank(axis=1, pct=True)
        final_ret[:] = np.where(
            rank > (1 - self.perc / 100), 1, np.where(rank < self.perc / 100, -1, 0)
        )

        final_weights = self.equal_weights_ls(final_ret)

        return final_weights


class Value(Strategy):
    def __init__(
        self,
        data: list[str] | dict[str, pd.DataFrame],
        signal_name: str,
        perc: float,
    ) -> None:
        super().__init__(data)
        self.data = data
        self.perc = perc
        self.signal_name = signal_name

        self.weights = self.get_weights()

    def equal_weights_ls(self, portfolio: pd.DataFrame) -> pd.DataFrame:
        """
        Equally Weighted long and short Porfolios

        :param portfolio: pd.DataFrame; dataframe with 1 in assets you want to buy and -1 in stocks you want to sell.
        :return: pd.DataFrame
        """
        port_long = portfolio.where(portfolio > 0, 0)
        port_short = portfolio.where(portfolio < 0, 0)
        n_long = (port_long.fillna(0) != 0).astype(int).sum(axis=1)
        n_short = (port_long.fillna(0) != 0).astype(int).sum(axis=1)
        port_long = port_long.div(n_long, axis=0)
        port_short = port_short.div(n_long, axis=0)
        port_long[np.isnan(port_long)] = 0
        port_short[np.isnan(port_short)] = 0

        final_df = port_long + port_short
        return final_df

    def get_weights(self) -> pd.DataFrame:
        """
        Calculates the desired weights of this strategy per day per asset.
        :return:
            pd.DataFrame
        """
        # Signal
        self._signal = self.data[self.signal_name]

        # Weights
        final_ret = self._signal.copy()
        rank = final_ret.rank(axis=1, pct=True)
        final_ret[:] = np.where(
            rank > (1 - self.perc / 100), -1, np.where(rank < self.perc / 100, 1, 0)
        )

        final_weights = self.equal_weights_ls(final_ret)

        return final_weights


class MachineLearningMethod(Strategy):
    def __init__(self, data: list[str] | dict[str, pd.DataFrame]) -> None:
        """
        Initialize MachineLearningMethod class
        Args:
            list of tickers to be considered in universe OR
                  dictionary of DataFrames, each containing dates along rows and tickers along columns,
                  with one DataFrame per value (e.g. data = {'price': ..., 'PE': ...})
        Return:
            None
        """
        super().__init__(data)

        self.weights = self.get_weights()

    def predict_returns(
        self,
        model: str = "lr",
        lookahead: int = 1,
        max_lag: int = 10,
        daily: bool = True,
    ) -> pd.DataFrame:
        """
        Get predicted daily returns lookahead days into the future
        Args:
            lookahead: number of days ahead to predict return for [default 1]
            max_lag: number of lagged returns to use as features for predicting future daily return
            daily: if True, return daily return, else return annual return [default True]
        Return:
            average predicted daily return over the test period for each stock
        """
        if model not in ["lr", "rf"]:
            raise ValueError("model must be one of: ['lr', 'rf']")

        if lookahead < 1:
            raise ValueError("lookahead must be at least 1")

        if lookahead > len(self.data):
            raise ValueError("lookahead is too large")

        if max_lag < 1:
            raise ValueError("max_lag must be at least 1")

        if max_lag > len(self.data):
            raise ValueError("max_lag is too large")

        if type(daily) != bool:
            raise TypeError("daily must be a Boolean")

        # get daily returns
        returns = self.data.pct_change(1)
        returns = returns.iloc[1:]
        stocks = self.data.columns

        # define training set size
        train_size = 0.7
        n = len(returns)
        n_train = int(n * train_size)

        # variables to store the return information
        avg_pred_returns = []

        for stock in stocks:
            # get features
            stock_data = returns[[stock]].copy()
            for i in range(max_lag):
                lag = i + 1
                stock_data[f"lag{lag}"] = stock_data[stock].shift(lag)

            stock_data["y"] = stock_data[stock].shift(-lookahead)

            # remove missing rows of data
            stock_data = stock_data[max_lag:]
            train_data = stock_data.iloc[:n_train]
            test_data = stock_data.iloc[n_train:]

            # separate dataset into training and testing
            X_train = train_data.drop(columns="y")
            y_train = train_data["y"]

            X_test = test_data.drop(columns="y")

            # build machine learning model
            if model == "lr":
                lr = LinearRegression(fit_intercept=True).fit(X_train, y_train)
                y_pred = lr.predict(X_test)

            else:
                rf = RandomForestRegressor().fit(X_train, y_train)
                y_pred = rf.predict(X_test)

            if not daily:
                y_pred = (1 + y_pred) ** 252 - 1

            # save this stock's average predicted return
            avg_pred_return = np.mean(y_pred)
            avg_pred_returns.append(avg_pred_return)

        predicted_returns = pd.DataFrame(
            {"stock": stocks, "pred_return": avg_pred_returns}
        )

        return predicted_returns

    def get_weights(self) -> pd.DataFrame:
        """
        Get strategy weights over time

        Args:
            ...

        Return:
            DataFrame containing dates along rows and tickers along columns, with values being the strategy weights

        """
        ...
