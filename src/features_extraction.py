import pandas as pd
import numpy as np
import talib
import warnings

def generate_og_features_df(df: pd.DataFrame, lags: list):
    for lag in lags:
        df["ADOSC" + str(lag)] = talib.ADOSC(
            df["high"], df["low"], df["close"], df["volume"], lag, lag * 3
        )
        df["MFI" + str(lag)] = talib.MFI(
            df["high"], df["low"], df["close"], df["volume"], lag
        )


def generate_mom_features_df(df: pd.DataFrame, lags: list):
    for lag in lags:
        df["ROC" + str(lag)] = talib.ROC(df["close"], lag)
        df["MOM" + str(lag)] = talib.MOM(df["close"], lag)
        df["PLUSDM" + str(lag)] = talib.PLUS_DM(df["high"], df["low"], lag)
        df["MINUSDM" + str(lag)] = talib.MINUS_DM(df["high"], df["low"], lag)
        df["ADX" + str(lag)] = talib.ADX(df["high"], df["low"], df["close"], lag)
        df["ADXR" + str(lag)] = talib.ADXR(df["high"], df["low"], df["close"], lag)
        df["APO" + str(lag)] = talib.APO(df["close"], lag, lag * 2)
        df["AROONOSC" + str(lag)] = talib.AROONOSC(df["high"], df["low"], lag)

        df["CCI" + str(lag)] = talib.CCI(df["high"], df["low"], df["close"], lag)
        df["CMO" + str(lag)] = talib.CMO(df["close"], lag)
        df["DX" + str(lag)] = talib.DX(df["high"], df["low"], df["close"], lag)
        (
            df["MACD" + str(lag)],
            df["MACDSIGNAL" + str(lag)],
            df["MACDHIST" + str(lag)],
        ) = talib.MACD(df["close"], lag, lag * 2, lag * 3)
        (
            df["MACDFIX" + str(lag)],
            df["MACDSIGNALFIX" + str(lag)],
            df["MACDHISTFIX" + str(lag)],
        ) = talib.MACDFIX(df["close"], lag)
        df["PPO" + str(lag)] = talib.PPO(df["close"], lag, lag * 2)
        df["RSI" + str(lag)] = talib.RSI(df["close"], lag)
        df["ULTOSC" + str(lag)] = talib.ULTOSC(
            df["high"], df["low"], df["close"], lag, lag * 2, lag * 3
        )
        df["WILLR" + str(lag)] = talib.WILLR(df["high"], df["low"], df["close"], lag)
        (
            df["STOCHRSI" + str(lag) + "k"],
            df["STOCHRSI" + str(lag) + "d"],
        ) = talib.STOCHRSI(df["close"], lag, 3, 3)
        df["NATR" + str(lag)] = talib.NATR(df["high"], df["low"], df["close"], lag)
        df["ATR" + str(lag)] = talib.ATR(df["high"], df["low"], df["close"], lag)
        df["KELTNER" + str(lag)] = (df["close"] - talib.SMA(df["close"], lag)) / df[
            "ATR" + str(lag)
        ]

    df["HT_TRENDLINE"] = talib.HT_TRENDLINE(df["close"])
    df["HT_TRENDMODE"] = talib.HT_TRENDMODE(df["close"])
    df["HT_DCPERIOD"] = talib.HT_DCPERIOD(df["close"])
    df["HT_DCPHASE"] = talib.HT_DCPHASE(df["close"])
    df["HT_PHASORinphase"], df["HT_PHASORquadrature"] = talib.HT_PHASOR(df["close"])
    df["HT_SINEsine"], df["HT_SINEleadsine"] = talib.HT_SINE(df["close"])
    df["BOP"] = talib.BOP(df["open"], df["high"], df["low"], df["close"])


def generate_math_features_df(df: pd.DataFrame, lags: list):
    for lag in lags:
        df["BETA" + str(lag)] = talib.BETA(df["high"], df["low"], lag)
        df["CORREL" + str(lag)] = talib.CORREL(df["high"], df["low"], lag)
        df["LINEARREG" + str(lag)] = talib.LINEARREG(df["close"], lag)
        df["LINEARREG_ANGLE" + str(lag)] = talib.LINEARREG_ANGLE(df["close"], lag)
        df["LINEARREG_INTERCEPT" + str(lag)] = talib.LINEARREG_INTERCEPT(
            df["close"], lag
        )
        df["LINEARREG_SLOPE" + str(lag)] = talib.LINEARREG_SLOPE(df["close"], lag)
        df["STDDEV" + str(lag)] = talib.STDDEV(df["close"], lag)
        df["TSF" + str(lag)] = talib.TSF(df["close"], lag)
        df["VAR" + str(lag)] = talib.VAR(df["close"], lag)


def generate_pattern_features_df(df: pd.DataFrame):
    df["CDL2CROWS"] = talib.WMA(
        talib.CDL2CROWS(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDL3BLACKCROWS"] = talib.WMA(
        talib.CDL3BLACKCROWS(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDL3INSIDE"] = talib.WMA(
        talib.CDL3INSIDE(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDL3LINESTRIKE"] = talib.WMA(
        talib.CDL3LINESTRIKE(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDL3OUTSIDE"] = talib.WMA(
        talib.CDL3OUTSIDE(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDL3STARSINSOUTH"] = talib.WMA(
        talib.CDL3STARSINSOUTH(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDL3WHITESOLDIERS"] = talib.WMA(
        talib.CDL3WHITESOLDIERS(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLABANDONEDBABY"] = talib.WMA(
        talib.CDLABANDONEDBABY(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLADVANCEBLOCK"] = talib.WMA(
        talib.CDLADVANCEBLOCK(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLBELTHOLD"] = talib.WMA(
        talib.CDLBELTHOLD(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLBREAKAWAY"] = talib.WMA(
        talib.CDLBREAKAWAY(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLCLOSINGMARUBOZU"] = talib.WMA(
        talib.CDLCLOSINGMARUBOZU(df["open"], df["high"], df["low"], df["close"]), 300
    )

    df["CDLCONCEALBABYSWALL"] = talib.WMA(
        talib.CDLCONCEALBABYSWALL(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLCOUNTERATTACK"] = talib.WMA(
        talib.CDLCOUNTERATTACK(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLDARKCLOUDCOVER"] = talib.WMA(
        talib.CDLDARKCLOUDCOVER(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLDOJI"] = talib.WMA(
        talib.CDLDOJI(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLDOJISTAR"] = talib.WMA(
        talib.CDLDOJISTAR(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLDRAGONFLYDOJI"] = talib.WMA(
        talib.CDLDRAGONFLYDOJI(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLENGULFING"] = talib.WMA(
        talib.CDLENGULFING(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLEVENINGDOJISTAR"] = talib.WMA(
        talib.CDLEVENINGDOJISTAR(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLEVENINGSTAR"] = talib.WMA(
        talib.CDLEVENINGSTAR(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLGAPSIDESIDEWHITE"] = talib.WMA(
        talib.CDLGAPSIDESIDEWHITE(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLGRAVESTONEDOJI"] = talib.WMA(
        talib.CDLGRAVESTONEDOJI(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLHAMMER"] = talib.WMA(
        talib.CDLHAMMER(df["open"], df["high"], df["low"], df["close"]), 300
    )

    df["CDLHANGINGMAN"] = talib.WMA(
        talib.CDLHANGINGMAN(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLHARAMI"] = talib.WMA(
        talib.CDLHARAMI(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLHARAMICROSS"] = talib.WMA(
        talib.CDLHARAMICROSS(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLHIGHWAVE"] = talib.WMA(
        talib.CDLHIGHWAVE(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLHIKKAKE"] = talib.WMA(
        talib.CDLHIKKAKE(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLHIKKAKEMOD"] = talib.WMA(
        talib.CDLHIKKAKEMOD(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLHOMINGPIGEON"] = talib.WMA(
        talib.CDLHOMINGPIGEON(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLIDENTICAL3CROWS"] = talib.WMA(
        talib.CDLIDENTICAL3CROWS(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLINNECK"] = talib.WMA(
        talib.CDLINNECK(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLINVERTEDHAMMER"] = talib.WMA(
        talib.CDLINVERTEDHAMMER(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLKICKING"] = talib.WMA(
        talib.CDLKICKING(df["open"], df["high"], df["low"], df["close"]), 300
    )

    df["CDLKICKINGBYLENGTH"] = talib.WMA(
        talib.CDLKICKINGBYLENGTH(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLLADDERBOTTOM"] = talib.WMA(
        talib.CDLLADDERBOTTOM(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLLONGLEGGEDDOJI"] = talib.WMA(
        talib.CDLLONGLEGGEDDOJI(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLLONGLINE"] = talib.WMA(
        talib.CDLLONGLINE(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLMARUBOZU"] = talib.WMA(
        talib.CDLMARUBOZU(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLMATCHINGLOW"] = talib.WMA(
        talib.CDLMATCHINGLOW(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLMATHOLD"] = talib.WMA(
        talib.CDLMATHOLD(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLMORNINGDOJISTAR"] = talib.WMA(
        talib.CDLMORNINGDOJISTAR(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLMORNINGSTAR"] = talib.WMA(
        talib.CDLMORNINGSTAR(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLONNECK"] = talib.WMA(
        talib.CDLONNECK(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLPIERCING"] = talib.WMA(
        talib.CDLPIERCING(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLRICKSHAWMAN"] = talib.WMA(
        talib.CDLRICKSHAWMAN(df["open"], df["high"], df["low"], df["close"]), 300
    )

    df["CDLRISEFALL3METHODS"] = talib.WMA(
        talib.CDLRISEFALL3METHODS(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLSEPARATINGLINES"] = talib.WMA(
        talib.CDLSEPARATINGLINES(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLSHOOTINGSTAR"] = talib.WMA(
        talib.CDLSHOOTINGSTAR(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLSHORTLINE"] = talib.WMA(
        talib.CDLSHORTLINE(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLSPINNINGTOP"] = talib.WMA(
        talib.CDLSPINNINGTOP(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLSTALLEDPATTERN"] = talib.WMA(
        talib.CDLSTALLEDPATTERN(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLSTICKSANDWICH"] = talib.WMA(
        talib.CDLSTICKSANDWICH(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLTAKURI"] = talib.WMA(
        talib.CDLTAKURI(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLTASUKIGAP"] = talib.WMA(
        talib.CDLTASUKIGAP(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLTHRUSTING"] = talib.WMA(
        talib.CDLTHRUSTING(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLTRISTAR"] = talib.WMA(
        talib.CDLTRISTAR(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLUNIQUE3RIVER"] = talib.WMA(
        talib.CDLUNIQUE3RIVER(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLUPSIDEGAP2CROWS"] = talib.WMA(
        talib.CDLUPSIDEGAP2CROWS(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLXSIDEGAP3METHODS"] = talib.WMA(
        talib.CDLXSIDEGAP3METHODS(df["open"], df["high"], df["low"], df["close"]), 300
    )

def generate_time_features(df: pd.DataFrame):
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["dayofweek"] = df.index.dayofweek
    df["dayofmonth"] = df.index.day
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df['is_within_15_mins_of_open'] = np.where((df['hour'] == 21 & (df['minute'] >= 0) & (df['minute'] <= 15), 1, 0))
    df['is_within_15_mins_of_close'] = np.where((df['hour'] == 14) & (df['minute'] >= 45) & (df['minute'] <= 59), 1, 0)
    df['is_within_7_mins_of_open'] = np.where((df['hour'] == 21 & (df['minute'] >= 0) & (df['minute'] <= 7), 1, 0))
    df['is_within_7_mins_of_close'] = np.where((df['hour'] == 14) & (df['minute'] >= 53) & (df['minute'] <= 59), 1, 0)
    df['is_within_5_mins_of_open'] = np.where((df['hour'] == 9) & (df['minute'] >= 25) & (df['minute'] <= 30), 1, 0)
    df['is_within_5_mins_of_close'] = np.where((df['hour'] == 15) & (df['minute'] >= 55) & (df['minute'] <= 59), 1, 0)
    df['is_within_3_mins_of_open'] = np.where((df['hour'] == 9) & (df['minute'] >= 27) & (df['minute'] <= 30), 1, 0)
    df['is_within_3_mins_of_close'] = np.where((df['hour'] == 15) & (df['minute'] >= 57) & (df['minute'] <= 59), 1, 0)
    df['is_within_1_mins_of_open'] = np.where((df['hour'] == 9) & (df['minute'] >= 29) & (df['minute'] <= 30), 1, 0)
    df['is_within_1_mins_of_close'] = np.where((df['hour'] == 15) & (df['minute'] >= 59) & (df['minute'] <= 59), 1, 0)


def generate_all_features_df(df: pd.DataFrame, lags: list):
    warnings.filterwarnings("ignore")
    generate_og_features_df(df, lags)
    generate_mom_features_df(df, lags)
    generate_math_features_df(df, lags)
    generate_pattern_features_df(df)
    # generate_time_features(df)
    df.dropna(inplace=True)

    # sort by name
    df = df.reindex(sorted(df.columns), axis=1)
    return df
