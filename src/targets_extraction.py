from sklearn.preprocessing import StandardScaler
from pandas_ta import log_return
import pandas as pd
import numpy as np


def prepare_desired_pos(df, lag=50, multiplier=10):
    df = df.copy()
    scaler = StandardScaler()
    df[f"{lag}m_ret"] = scaler.fit_transform(
        log_return(df.close, length=lag, offset=-lag).values.reshape(-1, 1)
    )
    df.dropna(inplace=True)
    df["desired_pos_change"] = (df[f"{lag}m_ret"] * multiplier).apply(int)
    df["pos_change_signal"] = pd.qcut(
        df["desired_pos_change"], 5, ["strong sell", "sell", "meh", "buy", "strong buy"]
    )
    df["desired_pos_rolling"] = (
        df["desired_pos_change"].rolling(lag, min_periods=1).sum().apply(int)
    )
    df["net_pos_signal"] = np.where(
        df["desired_pos_rolling"] > 0, "long hold", "short hold"
    )
    df.drop(columns=[f"{lag}m_ret"], inplace=True)

    return df


def rle(df, plot=False):
    """Run length encoding"""
    mask = df["pos_change_signal"].ne(df["pos_change_signal"].shift())
    groups = mask.cumsum()
    rle_result = df.groupby(groups)["pos_change_signal"].agg(
        [("value", "first"), ("count", "size")]
    )
    if plot:
        rle_result.groupby("value").mean().plot(
            kind="bar", title="Average count of consecutive same values"
        )
    return rle_result


def get_desired_pos(df, lag=50, multiplier=10):
    df = df.copy()
    scaler = StandardScaler()
    df[f"{lag}m_ret"] = scaler.fit_transform(
        log_return(df.close, length=lag, offset=-lag).values.reshape(-1, 1)
    )
    df.dropna(inplace=True)
    desired_pos_change = (df[f"{lag}m_ret"] * multiplier).apply(int)
    desired_pos_change_signal = pd.qcut(
        df["desired_pos_change"], 5, ["strong sell", "sell", "meh", "buy", "strong buy"]
    )
    desired_pos_rolling = (
        df["desired_pos_change"].rolling(lag, min_periods=1).sum().apply(int)
    )
    net_pos_signal = np.where(df["desired_pos_rolling"] > 0, "long hold", "short hold")
    df.drop(columns=[f"{lag}m_ret"], inplace=True)

    return (
        desired_pos_change,
        desired_pos_change_signal,
        desired_pos_rolling,
        net_pos_signal,
    )
