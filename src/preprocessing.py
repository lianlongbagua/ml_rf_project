from datetime import datetime

import pandas as pd
import numpy as np
import talib

from sklearn.preprocessing import robust_scale

import statsmodels.api as sm
from tuneta.tune_ta import TuneTA
from pandas_ta import log_return


def create_up_down_dataframe(
    csv_filepath,
    lookback_minutes=30,
    lookforward_minutes=5,
    up_down_factor=2.0,
    percent_factor=0.01,
    start=None,
    end=None
):
    ts = pd.read_csv(csv_filepath, index_col='datetime', parse_dates=True)
    ts.drop(
        columns=["symbol", "exchange", "turnover", "open_interest"],
        inplace=True
    )

    if start is not None:
        ts = ts.loc[start:]
    if end is not None:
        ts = ts.loc[:end]

    ts['adj_close'] = talib.MEDPRICE(ts.high, ts.low)
    ts.drop(columns=['open', 'close', 'high', 'low', 'volume'], inplace=True)

    for i in range(0, lookback_minutes):
        ts[f"lookback{i+1}"] = ts.adj_close.shift(i + 1)
    for i in range(0, lookforward_minutes):
        ts[f"lookforward{i+1}"] = ts.adj_close.shift(-i - 1)
    ts.dropna(inplace=True)

    ts['lookback0'] = ts.adj_close.pct_change() * 100
    for i in range(0, lookback_minutes):
        ts[f"lookback{i+1}"] = ts[f"lookback{i+1}"].pct_change() * 100
    for i in range(0, lookforward_minutes):
        ts[f"lookforward{i+1}"] = ts[f"lookforward{i+1}"].pct_change() * 100
    ts.dropna(inplace=True)

    up = up_down_factor * percent_factor
    down = percent_factor

    down_cols = [
        ts[f"lookforward{i+1}"] > -down for i in range(0, lookforward_minutes)
    ]
    up_cols = [
        ts[f"lookforward{i+1}"] > up for i in range(0, lookforward_minutes)
    ]

    down_tot = down_cols[0]
    for c in down_cols[1:]:
        down_tot = down_tot & c
    up_tot = up_cols[0]
    for c in up_cols[1:]:
        up_tot = up_tot & c
    ts['UpDownSignal'] = down_tot and up_tot

    ts['UpdownSignal'] = ts['UpDownSignal'].astype(int).replace(to_replace=0, value=-1)
    return ts


class Preprocessor:

    def __init__(self, df: pd.DataFrame, future_period: int):
        self.df = df
        self.future_period = future_period
        self.X = pd.DataFrame()
        self.y = pd.DataFrame()

    def prepare_target(self):
        print('Generating targets')
        self.df[f"target_{self.future_period}m_ret"] = log_return(
            self.df.close,
            length=self.future_period,
            offset=-self.future_period
        )
        self.df.dropna(inplace=True)
        self.X = self.df.drop(columns=[f"target_{self.future_period}m_ret"])
        self.y = self.df[f"target_{self.future_period}m_ret"]

    def generate_features(self):
        print("Generating features...")

        print("Generating TA features...")
        tuner = TuneTA(n_jobs=4, verbose=True)
        tuner.fit(
            self.X,
            self.y,
            indicators=[
                'tta.RSI', 'tta.ATR', 'tta.ADX', 'tta.LINEARREG_SLOPE',
                'tta.MOM'
            ],
            ranges=[(4, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 100),
                    (100, 200), (200, 300)],
            trials=300,
            early_stop=50,
        )
        tuner.prune(max_inter_correlation=0.95)
        features = tuner.transform(self.X)
        print(f"Features selected are {features.columns.tolist()}")
        self.X = pd.concat([self.X, features], axis=1)

        print("Calculating ACF...")
        acf_arr = np.argsort(
            np.abs(
                sm.tsa.stattools.acf(
                    robust_scale(self.y)[::self.future_period], missing='drop'
                )
            )
        )[::-1][1:6]
        print("ACF calculated, top 5 lags are: ", acf_arr)
        for i in acf_arr:
            self.X[f'lagged_target_{i*self.future_period}'] = log_return(
                self.X.close,
                length=self.future_period,
                offset=i * self.future_period
            )

        self.X.dropna(inplace=True)
        self.y = self.y.loc[self.X.index]
        print("Features generated")

    def generate_cossin_time_features(self):
        print("Generating cossin time features...")

        self.X['time_hour_sin'] = talib.SIN(
            (self.X.index.hour / 24 * 2 * np.pi).to_numpy()
        )
        self.X['time_hour_cos'] = talib.COS(
            (self.X.index.hour / 24 * 2 * np.pi).to_numpy()
        )

        self.X['time_minute_sin'] = talib.SIN(
            (self.X.index.minute / 60 * 2 * np.pi).to_numpy()
        )
        self.X['time_minute_cos'] = talib.COS(
            (self.X.index.minute / 60 * 2 * np.pi).to_numpy()
        )

        self.X['time_day_of_week_sin'] = talib.SIN(
            (self.X.index.dayofweek / 7 * 2 * np.pi).to_numpy()
        )
        self.X['time_day_of_week_cos'] = talib.COS(
            (self.X.index.dayofweek / 7 * 2 * np.pi).to_numpy()
        )

        self.X['time_day_of_month_sin'] = talib.SIN(
            (self.X.index.day / 30 * 2 * np.pi).to_numpy()
        )
        self.X['time_day_of_month_cos'] = talib.COS(
            (self.X.index.day / 30 * 2 * np.pi).to_numpy()
        )

        print("Cossin time features generated")

    def prep(self):
        self.prepare_target()
        self.generate_features()
        self.generate_cossin_time_features()
        self.X.drop(
            columns=['open', 'high', 'low', 'close', 'volume'], inplace=True
        )
        return self.X, self.y

    def save(self):
        self.X.to_hdf(f'{datetime.now().date()}.h5', key='X')
        self.y.to_hdf(f'{datetime.now().date()}.h5', key='y')
