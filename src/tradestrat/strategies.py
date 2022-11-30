from __future__ import annotations

import abc

import numpy as np
import pandas as pd  # type: ignore[import]

DATA = pd.read_csv("tradestrat/data/sp500_prices.csv")


class Strategy(abc.ABC):
    def __init__(self, data: list[str] | dict[str, pd.DataFrame]) -> None:
        """
        Initialize Strategy class

        Args:
            data: list of tickers to be considered in universe (if you are only using price data) OR
                  dictionary of DataFrames, each containing dates along rows and tickers along columns,
                  with one DataFrame per value (e.g. data = {'price': ..., 'PE': ...})

        Return:
            None
        """

        if type(data) == list:
            # TODO: CURRENTLY USING DUMMY DATA
            if data == ["all"]:
                self.data = DATA.copy()
            else:
                self.data = DATA[data].copy()

            self.data["Date"] = pd.to_datetime(DATA["Date"])
            self.data.set_index("Date", inplace=True)
            # TODO: raise errors
        elif type(data) == dict:
            self.data = data["price"]
            self.strategy_data = data
        else:
            raise TypeError(
                "data must either be a list of tickers or a dictionary of dataframes"
            )
        try:
            self.weights = self.get_weights()
        except:
            pass

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

    """
    An asset's recent relative performance tends to continue in the near future.

    See Jegadeesh and Titman (1991) and Carhart (1997) for academic references.
    """

    def __init__(
        self,
        data: list[str] | dict[str, pd.DataFrame],
        lookback_period: int,
        min_periods: int | None,
        skip_period: int = 0,
        perc: float = 0.1,
    ) -> None:
        """
        Initialize Strategy class

        Args:
            data: list of tickers to be considered in universe OR
                  dictionary of DataFrames, each containing dates along rows and tickers along columns,
                  with one DataFrame per value (e.g. data = {'price': ..., 'PE': ...})
            lookback_period: Number of months used to calculate returns of assets.
            min_periods: Minimal periods of data necessary for weights to be non-empty while calculating rolling
                         returns
            skip_period: Number of days that should be skipped for returns to start being calculated.
            perc: percentage of assets you would like to be long and short. It must be written in percentage form.
                  If 0.1 is the input, the strategy will buy the signal's highest 10% assets and short the signal's 10%
                  lowest 10% assets.

            Return:
                None
        """

        super().__init__(data)
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

        Args:
            portfolio: dataframe with equal positive values in assets you want to buy and equal negative
             in stocks you want to sell.

        Return:
            Dataframe of equally weighted long and short portfolios. Long positions sums to 100%, short positions
            sum to -100%

        """

        port_long = portfolio.where(portfolio > 0, 0)
        port_short = portfolio.where(portfolio < 0, 0)
        n_long = (port_long.fillna(0) != 0).astype(int).sum(axis=1)
        n_short = (port_short.fillna(0) != 0).astype(int).sum(axis=1)
        port_long = port_long.div(n_long, axis=0)
        port_short = port_short.div(n_short, axis=0)
        port_long[np.isnan(port_long)] = 0
        port_short[np.isnan(port_short)] = 0

        final_df = port_long + port_short
        return final_df

    def get_weights(self) -> pd.DataFrame:

        """
        Calculate desired weights of strategy

        Args:
            None

        Return:
            Calculate the desired weights of this strategy per day per asset. Each day is a row and each asset is a
            column

        """
        # Signal
        try:
            ret_data = self.strategy_data["returns"]
        except:
            ret_data = np.log(self.data) - np.log(self.data.shift(1))

        lookback = (
            np.exp(
                ret_data.shift(self.skip_period)
                .rolling(21 * self.lookback_period)
                .sum()
            )
            - 1
        )
        rsd = np.exp(ret_data).rolling(21 * self.lookback_period).std()
        _signal = lookback / rsd

        # Weights
        final_ret = _signal.copy()
        rank = final_ret.rank(axis=1, pct=True)
        final_ret[:] = np.where(
            rank > (1 - self.perc / 100), 1, np.where(rank < self.perc / 100, -1, 0)
        )

        final_weights = self.equal_weights_ls(final_ret)

        return final_weights


class Value(Strategy):
    """Stocks with cheap fundamentals tend to outperform stocks with expensive fundamentals

    See Fama,French (1992,2012) and Asness,Moskowitz,Pedersen(2013) for academic references.
    """

    def __init__(
        self,
        data: list[str] | dict[str, pd.DataFrame],
        signal_name: str,
        perc: float,
    ) -> None:
        """
        Initialize Strategy class

        Args:
             data: list of tickers to be considered in universe OR
                  dictionary of DataFrames, each containing dates along rows and tickers along columns,
                  with one DataFrame per value (e.g. data = {'price': ..., 'PE': ...})
             signal_name: Name of the signal being used. Must be the same as the name of data. Ex: "PE","EVEBITDA".
             perc: percentage of assets you would like to be long and short. It must be written in percentage form.
                   If 0.1 is the input, the strategy will buy the signal's highest 10% assets and short the signal's 10%
                   lowest 10% assets.

        Return:
            None
        """

        super().__init__(data)
        self.perc = perc
        self.signal_name = signal_name

        self.weights = self.get_weights()

    def equal_weights_ls(self, portfolio: pd.DataFrame) -> pd.DataFrame:
        """
        Equally Weighted long and short Porfolios

        Args:
            portfolio: dataframe with equal positive values in assets you want to buy and equal negative
             in stocks you want to sell.

        Return:
            Dataframe of equally weighted long and short portfolios. Long positions sums to 100%, short positions
            sum to -100%
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
        Calculate desired weights of strategy

        Args:
            None

        Return:
            Calculate the desired weights of this strategy per day per asset. Each day is a row and each asset is a
            column

        """

        # Signal
        _signal = self.data[self.signal_name]

        # Weights
        final_ret = _signal.copy()
        rank = final_ret.rank(axis=1, pct=True)
        final_ret[:] = np.where(
            rank > (1 - self.perc / 100), -1, np.where(rank < self.perc / 100, 1, 0)
        )

        final_weights = self.equal_weights_ls(final_ret)

        return final_weights


class trend_following(Strategy):
    """
    Stocks that performed poorly in the past tend to continue performing poorly and viceversa.

    See Moskowitz,Ooi,Pedersen (2012) for academic reference.
    """

    def __init__(
        self,
        data: list[str] | dict[str, pd.DataFrame],
        lookback_period: int,
        min_periods: int | None,
        skip_period: int,
        wind: int,
        risk_free: bool = False,
    ) -> None:

        """
        Initialize Strategy class

        Args:
            data: list of tickers to be considered in universe OR
                  dictionary of DataFrames, each containing dates along rows and tickers along columns,
                  with one DataFrame per value (e.g. data = {'price': ..., 'PE': ...})
            lookback_period: Number of months used to calculate returns of assets.
            min_periods: Minimal periods of data necessary for weights to be non-empty while calculating rolling
                         returns
            skip_period: Number of days that should be skipped for returns to start being calculated.
            wind: Size of window (in months) used to determine an asset's performance
            perc: percentage of assets you would like to be long and short. It must be written in percentage form.
                  If 0.1 is the input, the strategy will buy the signal's highest 10% assets and short the signal's 10%
                  lowest 10% assets.

        Return:
            None
        """
        super().__init__(data)
        self.lookback_period = lookback_period
        self.min_periods = min_periods
        self.skip_period = skip_period
        self.wind = wind
        self.risk_free = risk_free

        if self.min_periods == None:
            self.min_periods = self.wind - 20

        self.weights = self.get_weights()

    def get_weights(self) -> pd.DataFrame:
        """
        Calculate desired weights of strategy

        Args:
            None

        Return:
            Calculate the desired weights of this strategy per day per asset. Each day is a row and each asset is a
            column

        """
        ret_data = self.data["returns"]

        aux = self.skip_period
        lookback = (
            (ret_data + 1)
            .shift(aux)
            .rolling(self.wind, min_periods=self.min_periods)
            .prod()
        ) - 1

        vol = ret_data.rolling(self.wind, min_periods=self.min_periods).std()

        if self.risk_free == False:

            signal = lookback / vol

        else:
            rf = self.data["risk_free"]
            rf_lookback = (
                (rf + 1)
                .shift(aux)
                .rolling(self.wind, min_periods=self.min_periods)
                .prod()
            ) - 1
            lookback_adj = lookback.sub(rf_lookback, axis=0)
            signal = lookback_adj / vol

        neg = signal[signal < 0].div(signal[signal < 0].abs().sum(axis=1), axis=0)
        neg = neg.fillna(0)
        pos = signal[signal > 0].div(signal[signal > 0].abs().sum(axis=1), axis=0)
        pos = pos.fillna(0)

        weights = neg + pos
        final_weights = weights.dropna(axis=0, how="all")

        return final_weights


class LO_2MA(Strategy):
    """Trading strategy based on the moving averages of returns. Moving Average signals are very flexible and the
    version implemented here buys an asset when the short term average crosses above the long term moving average and
    vice versa.

    While the academic treatment of Moving averages is less theoretical than the other strategies here presented one
    can look at Brock et al. (1992) for an academic work on moving average trading rules
    """

    def __init__(
        self,
        data: list[str] | dict[str, pd.DataFrame],
        MA_long_wind: int,
        MA_short_wind: int,
        min_periods: int | None,
        skip_period: int = 0,
    ) -> None:

        """
        Initialize Strategy class

        Args:
            data: list of tickers to be considered in universe OR
                  dictionary of DataFrames, each containing dates along rows and tickers along columns,
                  with one DataFrame per value (e.g. data = {'price': ..., 'PE': ...})
            MA_long_wind: number of days to calculate the long Moving Average. Must be bigger than MA_short_wind
            MA_short_wind: number of days to calculate the short Moving Average.
            min_periods: Minimal periods of data necessary for weights to be non-empty while calculating rolling
                         returns
            skip_period: Number of days that should be skipped for returns to start being calculated.
            perc: percentage of assets you would like to be long and short. It must be written in percentage form.
                  If 0.1 is the input, the strategy will buy the signal's highest 10% assets and short the signal's 10%
                  lowest 10% assets.

        Return:
            None
        """

        super().__init__(data)
        self.MA_long_wind = MA_long_wind
        self.MA_short_wind = MA_short_wind
        self.min_periods = min_periods
        self.skip_period = skip_period

        if self.min_periods == None:
            self.min_periods = np.floor(np.min([MA_long_wind, MA_short_wind]) / 2)

        if MA_long_wind < MA_short_wind:
            raise ValueError("MA_long_wind must be bigger than MA_short_wind")

        self.weights = self.get_weights()

    def equal_weights_long(self, portfolio: pd.DataFrame) -> pd.DataFrame:

        """
        Equally Weighted long Porfolios

        Args:
            portfolio: dataframe with equal positive values in assets you want to buy.

        Return:
            Dataframe of equally weighted long portfolios. Long positions sums to 100%

        """

        n_long = (portfolio.fillna(0) != 0).astype(int).sum(axis=1)
        port_long = portfolio.div(n_long, axis=0)
        port_long[np.isnan(port_long)] = 0

        return port_long

    def get_weights(self) -> pd.DataFrame:
        """
        Calculate desired weights of strategy

        Args:
            None

        Return:
            Calculate the desired weights of this strategy per day per asset. Each day is a row and each asset is a
            column
        """

        ret_data = self.data["returns"]
        ret_MA_long = ret_data.rolling(
            window=self.MA_long_wind, min_periods=self.min_periods
        ).mean()
        ret_MA_short = ret_data.rolling(
            window=self.MA_short_wind, min_periods=self.min_periods
        ).mean()

        signal = np.where(ret_MA_short > ret_MA_long, 1.0, 0.0)
        final_weights = self.equal_weights_long(signal)

        return final_weights
