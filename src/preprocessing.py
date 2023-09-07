from datetime import datetime

import pandas as pd
import numpy as np
import talib

from sklearn.preprocessing import robust_scale
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline

import statsmodels.api as sm
from tuneta.tune_ta import TuneTA
from pandas_ta import log_return


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
                self.X.close, length=self.future_period, offset=i * self.future_period
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
        self.generate_cossin_time_feaures()
        self.X.drop(
            columns=['open', 'high', 'low', 'close', 'volume'], inplace=True
        )
        return self.X, self.y

    def save(self):
        self.X.to_hdf(f'{datetime.now().date()}.h5', key='X')
        self.y.to_hdf(f'{datetime.now().date()}.h5', key='y')
