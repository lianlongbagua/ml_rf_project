from sklearn.preprocessing import StandardScaler
from pandas_ta import log_return
import pandas as pd
import numpy as np
import talib
import warnings


def renaming(df: pd.DataFrame):
    """rename columns"""
    df.rename(
        columns={
            "close_price": "close",
            "high_price": "high",
            "low_price": "low",
            "open_price": "open",
        },
        inplace=True,
    )
    return df


def keep_essentials(df: pd.DataFrame):
    """Keep only OHLCVT"""
    df.drop(
        columns=["exchange", "turnover", "symbol"],
        axis=1,
        inplace=True,
    )
    return renaming(df)


def prepare_target(df, future_period):
    print('Generating targets')
    df = df.copy()
    scaler = StandardScaler()
    df[f"{future_period}m_ret"] = scaler.fit_transform(
        log_return(df.close, length=future_period, offset=-future_period).values.reshape(-1, 1)
    )
    df.dropna(inplace=True)
    df["target_position_change"] = (df[f"{future_period}m_ret"] * 10).apply(int)
    df['target_position_change'] = df['target_position_change'] - df['target_position_change'].shift(future_period).fillna(0)
    df["target_total_position"] = df['target_position_change'].cumsum()
    df["target_position_change_quartile"] = pd.cut(
        df["target_position_change"], 5,
        labels=["strong_sell", "meh", "meh", "meh", "strong_buy"],
        ordered=False
    )
    df["target_total_position_quartile"] = pd.cut(
        df['target_total_position'], 7,
        labels=['max_short_pos', 'short_hold', 'short_hold', 'no_trade', 'long_hold', 'long_hold', 'max_long_pos'],
        ordered=False
    )
    df.drop(columns=[f"{future_period}m_ret"], inplace=True)
    print("Targets generated")

    return df


def rle(df, plot=False):
    """Run length encoding"""
    mask = df["pos_change_signal"].ne(df["pos_change_signal"].shift())
    groups = mask.cumsum()
    rle_result = df.groupby(groups)["pos_change_signal"].agg([
        ("value", "first"), ("count", "size")
    ])
    if plot:
        rle_result.groupby("value").mean().plot(
            kind="bar", title="Average count of consecutive same values"
        )
    return rle_result


def generate_og_features_df(df: pd.DataFrame, lags: list):
    print("Generating original features...")
    for future_period in lags:
        df["ADOSC_" + str(future_period)] = talib.ADOSC(
            df["high"], df["low"], df["close"], df["volume"], future_period, future_period * 3
        )
        df["MFI_" + str(future_period)] = talib.MFI(
            df["high"], df["low"], df["close"], df["volume"], future_period
        )


def generate_mom_features_df(df: pd.DataFrame, lags: list):
    print("Generating momentum features...")
    for future_period in lags:
        df["ROC_" + str(future_period)] = talib.ROC(df["close"], future_period)
        df["MOM_" + str(future_period)] = talib.MOM(df["close"], future_period)
        df["PLUS_DM_" + str(future_period)] = talib.PLUS_DM(df["high"], df["low"], future_period)
        df["MINUS_DM_" + str(future_period)] = talib.MINUS_DM(df["high"], df["low"], future_period)
        df["ADX_" +
           str(future_period)] = talib.ADX(df["high"], df["low"], df["close"], future_period)
        df["ADXR_" +
           str(future_period)] = talib.ADXR(df["high"], df["low"], df["close"], future_period)
        df["APO_" + str(future_period)] = talib.APO(df["close"], future_period, future_period * 2)
        df["AROONOSC_" + str(future_period)] = talib.AROONOSC(df["high"], df["low"], future_period)

        df["CCI_" +
           str(future_period)] = talib.CCI(df["high"], df["low"], df["close"], future_period)
        df["CMO_" + str(future_period)] = talib.CMO(df["close"], future_period)
        df["DX_" +
           str(future_period)] = talib.DX(df["high"], df["low"], df["close"], future_period)
        df["STOCH_" + str(future_period) + "slowk"], _ = talib.STOCH(
            df["high"],
            df["low"],
            df["close"],
            fastk_period=future_period,
            slowk_period=int(future_period / 2),
            slowk_matype=0,
            slowd_period=int(future_period / 2),
            slowd_matype=0,
        )
        df["STOCHF_" + str(future_period) + "fastk"], _ = talib.STOCHF(
            df["high"], df["low"], df["close"], future_period, int(future_period / 2), 0
        )
        (_, df["MACDSIGNAL_" + str(future_period)],
         _) = talib.MACD(df["close"], future_period, future_period * 2, int(future_period / 2))
        _, df["MACDSIGNALFIX_" + str(future_period)], _ = talib.MACDFIX(df["close"], future_period)
        df["PPO_" + str(future_period)] = talib.PPO(df["close"], future_period, future_period * 2)
        df["RSI_" + str(future_period)] = talib.RSI(df["close"], future_period)
        df["ULTOSC_" + str(future_period)] = talib.ULTOSC(
            df["high"], df["low"], df["close"], future_period, future_period * 2, future_period * 3
        )
        df["WILLR_" +
           str(future_period)] = talib.WILLR(df["high"], df["low"], df["close"], future_period)
        df["STOCHRSI_" + str(future_period) +
           "k"], _ = talib.STOCHRSI(df["close"], future_period, 3, 3)
        df["NATR_" +
           str(future_period)] = talib.NATR(df["high"], df["low"], df["close"], future_period)
        df["ATR_" +
           str(future_period)] = talib.ATR(df["high"], df["low"], df["close"], future_period)
        df["TRANGE_" +
           str(future_period)] = talib.TRANGE(df["high"], df["low"], df["close"])

    df["HT_TRENDLINE"] = talib.HT_TRENDLINE(df["close"])
    df["HT_TRENDMODE"] = talib.HT_TRENDMODE(df["close"])
    df["HT_DCPERIOD"] = talib.HT_DCPERIOD(df["close"])
    df["HT_DCPHASE"] = talib.HT_DCPHASE(df["close"])
    df["HT_PHASORinphase"], _ = talib.HT_PHASOR(df["close"])
    df["HT_SINEsine"], _ = talib.HT_SINE(df["close"])


def generate_math_features_df(df: pd.DataFrame, lags: list):
    print("Generating math features...")
    for future_period in lags:
        df["BETA_" + str(future_period)] = talib.BETA(df["high"], df["low"], future_period)
        df["CORREL_" + str(future_period)] = talib.CORREL(df["high"], df["low"], future_period)
        df["LINEARREG_" + str(future_period)] = talib.LINEARREG(df["close"], future_period)
        df["LINEARREG_ANGLE_" +
           str(future_period)] = talib.LINEARREG_ANGLE(df["close"], future_period)
        df["LINEARREG_INTERCEPT_" +
           str(future_period)] = talib.LINEARREG_INTERCEPT(df["close"], future_period)
        df["LINEARREG_SLOPE_" +
           str(future_period)] = talib.LINEARREG_SLOPE(df["close"], future_period)
        df["STDDEV_" + str(future_period)] = talib.STDDEV(df["close"], future_period)
        df["TSF_" + str(future_period)] = talib.TSF(df["close"], future_period)
        df["VAR_" + str(future_period)] = talib.VAR(df["close"], future_period)


def generate_pattern_features_df(df: pd.DataFrame):
    print("Generating pattern features...")
    df["CDL2CROWS"] = talib.WMA(
        talib.CDL2CROWS(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDL3BLACKCROWS"] = talib.WMA(
        talib.CDL3BLACKCROWS(df["open"], df["high"], df["low"], df["close"]),
        300
    )
    df["CDL3INSIDE"] = talib.WMA(
        talib.CDL3INSIDE(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDL3LINESTRIKE"] = talib.WMA(
        talib.CDL3LINESTRIKE(df["open"], df["high"], df["low"], df["close"]),
        300
    )
    df["CDL3OUTSIDE"] = talib.WMA(
        talib.CDL3OUTSIDE(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDL3STARSINSOUTH"] = talib.WMA(
        talib.CDL3STARSINSOUTH(df["open"], df["high"], df["low"], df["close"]),
        300
    )
    df["CDL3WHITESOLDIERS"] = talib.WMA(
        talib.CDL3WHITESOLDIERS(
            df["open"], df["high"], df["low"], df["close"]
        ), 300
    )
    df["CDLABANDONEDBABY"] = talib.WMA(
        talib.CDLABANDONEDBABY(df["open"], df["high"], df["low"], df["close"]),
        300
    )
    df["CDLADVANCEBLOCK"] = talib.WMA(
        talib.CDLADVANCEBLOCK(df["open"], df["high"], df["low"], df["close"]),
        300
    )
    df["CDLBELTHOLD"] = talib.WMA(
        talib.CDLBELTHOLD(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLBREAKAWAY"] = talib.WMA(
        talib.CDLBREAKAWAY(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLCLOSINGMARUBOZU"] = talib.WMA(
        talib.CDLCLOSINGMARUBOZU(
            df["open"], df["high"], df["low"], df["close"]
        ), 300
    )

    df["CDLCONCEALBABYSWALL"] = talib.WMA(
        talib.CDLCONCEALBABYSWALL(
            df["open"], df["high"], df["low"], df["close"]
        ), 300
    )
    df["CDLCOUNTERATTACK"] = talib.WMA(
        talib.CDLCOUNTERATTACK(df["open"], df["high"], df["low"], df["close"]),
        300
    )
    df["CDLDARKCLOUDCOVER"] = talib.WMA(
        talib.CDLDARKCLOUDCOVER(
            df["open"], df["high"], df["low"], df["close"]
        ), 300
    )
    df["CDLDOJI"] = talib.WMA(
        talib.CDLDOJI(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLDOJISTAR"] = talib.WMA(
        talib.CDLDOJISTAR(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLDRAGONFLYDOJI"] = talib.WMA(
        talib.CDLDRAGONFLYDOJI(df["open"], df["high"], df["low"], df["close"]),
        300
    )
    df["CDLENGULFING"] = talib.WMA(
        talib.CDLENGULFING(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLEVENINGDOJISTAR"] = talib.WMA(
        talib.CDLEVENINGDOJISTAR(
            df["open"], df["high"], df["low"], df["close"]
        ), 300
    )
    df["CDLEVENINGSTAR"] = talib.WMA(
        talib.CDLEVENINGSTAR(df["open"], df["high"], df["low"], df["close"]),
        300
    )
    df["CDLGAPSIDESIDEWHITE"] = talib.WMA(
        talib.CDLGAPSIDESIDEWHITE(
            df["open"], df["high"], df["low"], df["close"]
        ), 300
    )
    df["CDLGRAVESTONEDOJI"] = talib.WMA(
        talib.CDLGRAVESTONEDOJI(
            df["open"], df["high"], df["low"], df["close"]
        ), 300
    )
    df["CDLHAMMER"] = talib.WMA(
        talib.CDLHAMMER(df["open"], df["high"], df["low"], df["close"]), 300
    )

    df["CDLHANGINGMAN"] = talib.WMA(
        talib.CDLHANGINGMAN(df["open"], df["high"], df["low"], df["close"]),
        300
    )
    df["CDLHARAMI"] = talib.WMA(
        talib.CDLHARAMI(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLHARAMICROSS"] = talib.WMA(
        talib.CDLHARAMICROSS(df["open"], df["high"], df["low"], df["close"]),
        300
    )
    df["CDLHIGHWAVE"] = talib.WMA(
        talib.CDLHIGHWAVE(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLHIKKAKE"] = talib.WMA(
        talib.CDLHIKKAKE(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLHIKKAKEMOD"] = talib.WMA(
        talib.CDLHIKKAKEMOD(df["open"], df["high"], df["low"], df["close"]),
        300
    )
    df["CDLHOMINGPIGEON"] = talib.WMA(
        talib.CDLHOMINGPIGEON(df["open"], df["high"], df["low"], df["close"]),
        300
    )
    df["CDLIDENTICAL3CROWS"] = talib.WMA(
        talib.CDLIDENTICAL3CROWS(
            df["open"], df["high"], df["low"], df["close"]
        ), 300
    )
    df["CDLINNECK"] = talib.WMA(
        talib.CDLINNECK(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLINVERTEDHAMMER"] = talib.WMA(
        talib.CDLINVERTEDHAMMER(
            df["open"], df["high"], df["low"], df["close"]
        ), 300
    )
    df["CDLKICKING"] = talib.WMA(
        talib.CDLKICKING(df["open"], df["high"], df["low"], df["close"]), 300
    )

    df["CDLKICKINGBYLENGTH"] = talib.WMA(
        talib.CDLKICKINGBYLENGTH(
            df["open"], df["high"], df["low"], df["close"]
        ), 300
    )
    df["CDLLADDERBOTTOM"] = talib.WMA(
        talib.CDLLADDERBOTTOM(df["open"], df["high"], df["low"], df["close"]),
        300
    )
    df["CDLLONGLEGGEDDOJI"] = talib.WMA(
        talib.CDLLONGLEGGEDDOJI(
            df["open"], df["high"], df["low"], df["close"]
        ), 300
    )
    df["CDLLONGLINE"] = talib.WMA(
        talib.CDLLONGLINE(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLMARUBOZU"] = talib.WMA(
        talib.CDLMARUBOZU(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLMATCHINGLOW"] = talib.WMA(
        talib.CDLMATCHINGLOW(df["open"], df["high"], df["low"], df["close"]),
        300
    )
    df["CDLMATHOLD"] = talib.WMA(
        talib.CDLMATHOLD(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLMORNINGDOJISTAR"] = talib.WMA(
        talib.CDLMORNINGDOJISTAR(
            df["open"], df["high"], df["low"], df["close"]
        ), 300
    )
    df["CDLMORNINGSTAR"] = talib.WMA(
        talib.CDLMORNINGSTAR(df["open"], df["high"], df["low"], df["close"]),
        300
    )
    df["CDLONNECK"] = talib.WMA(
        talib.CDLONNECK(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLPIERCING"] = talib.WMA(
        talib.CDLPIERCING(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLRICKSHAWMAN"] = talib.WMA(
        talib.CDLRICKSHAWMAN(df["open"], df["high"], df["low"], df["close"]),
        300
    )

    df["CDLRISEFALL3METHODS"] = talib.WMA(
        talib.CDLRISEFALL3METHODS(
            df["open"], df["high"], df["low"], df["close"]
        ), 300
    )
    df["CDLSEPARATINGLINES"] = talib.WMA(
        talib.CDLSEPARATINGLINES(
            df["open"], df["high"], df["low"], df["close"]
        ), 300
    )
    df["CDLSHOOTINGSTAR"] = talib.WMA(
        talib.CDLSHOOTINGSTAR(df["open"], df["high"], df["low"], df["close"]),
        300
    )
    df["CDLSHORTLINE"] = talib.WMA(
        talib.CDLSHORTLINE(df["open"], df["high"], df["low"], df["close"]), 300
    )
    df["CDLSPINNINGTOP"] = talib.WMA(
        talib.CDLSPINNINGTOP(df["open"], df["high"], df["low"], df["close"]),
        300
    )
    df["CDLSTALLEDPATTERN"] = talib.WMA(
        talib.CDLSTALLEDPATTERN(
            df["open"], df["high"], df["low"], df["close"]
        ), 300
    )
    df["CDLSTICKSANDWICH"] = talib.WMA(
        talib.CDLSTICKSANDWICH(df["open"], df["high"], df["low"], df["close"]),
        300
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
        talib.CDLUNIQUE3RIVER(df["open"], df["high"], df["low"], df["close"]),
        300
    )
    df["CDLUPSIDEGAP2CROWS"] = talib.WMA(
        talib.CDLUPSIDEGAP2CROWS(
            df["open"], df["high"], df["low"], df["close"]
        ), 300
    )
    df["CDLXSIDEGAP3METHODS"] = talib.WMA(
        talib.CDLXSIDEGAP3METHODS(
            df["open"], df["high"], df["low"], df["close"]
        ), 300
    )


def generate_time_features(df: pd.DataFrame):
    print("Generating time features...")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["time_hour"] = df["datetime"].dt.hour
    df["time_minute"] = df["datetime"].dt.minute
    df["time_day_of_week"] = df["datetime"].dt.dayofweek
    df["time_day_of_month"] = df["datetime"].dt.day
    df.drop(columns=["datetime"], inplace=True)


def generate_all_features_df(df: pd.DataFrame, lags: list):

    warnings.filterwarnings("ignore")
    generate_og_features_df(df, lags)
    generate_mom_features_df(df, lags)
    generate_math_features_df(df, lags)
    # generate_pattern_features_df(df)
    generate_time_features(df)
    df.dropna(inplace=True)

    # sort by name
    df = df.reindex(sorted(df.columns), axis=1)
    print("All features generated")
    return df


def drop_ohlcv_cols(df: pd.DataFrame):
    """drop ohlcv columns"""
    return df.drop(
        columns=["open", "high", "low", "close", "volume", "open_interest"],
        axis=1
    )


def split_features_target(df: pd.DataFrame):
    """split features and target"""
    target_cols = df.columns.str.contains(r'target')

    return df.loc[:, ~target_cols], df.loc[:, target_cols]


def prep_data(df: pd.DataFrame, lags: list, future_period: int, multiplier: int):
    """prep data for training"""
    df = keep_essentials(df)
    df = prepare_desired_pos(df, future_period, multiplier)
    df = generate_all_features_df(df, lags)
    df = drop_ohlcv_cols(df)
    X, y = split_features_target(df)
    return X, y