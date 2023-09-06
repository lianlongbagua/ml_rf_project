import numpy as np
import warnings
warnings.filterwarnings("ignore")

from elite_ctastrategy import (
    EliteCtaTemplate,
    HistoryManager,
    Parameter,
    Variable,
)

import talib as ta
import joblib


class HGBRStrategy(EliteCtaTemplate):
    """提升树策略"""

    author = "czl"

    # 基础参数（必填）
    bar_window: int = Parameter(1)  # K线窗口
    bar_interval: int = Parameter("1m")  # K线级别
    bar_buffer: int = Parameter(500)  # K线缓存

    # 策略参数（可选）
    price_add: int = Parameter(5)  # 委托下单超价
    N: int = Parameter(0.03)  # 信号阈值
    max_holding: int = Parameter(20)  # 最大持仓时间

    # 策略变量
    trading_size: int = Variable(1)  # 下单手数

    def on_init(self) -> None:
        """初始化"""
        self.write_log("策略初始化")
        self.load_bar(10)
        self.enter_pos_classifier_model = joblib.load("model/HGBR9-5.pkl")

    def on_start(self) -> None:
        """启动"""
        self.write_log("策略启动")

    def on_stop(self) -> None:
        """停止"""
        self.write_log("策略停止")

    def on_history(self, hm: HistoryManager) -> None:
        """K线推送"""
        # 获取特征矩阵
        ATR7 = ta.ATR(hm.high, hm.low, hm.close, timeperiod=7)[-1]
        ATR22 = ta.ATR(hm.high, hm.low, hm.close, timeperiod=22)[-1]
        ATR61 = ta.ATR(hm.high, hm.low, hm.close, timeperiod=61)[-1]
        MOM16 = ta.MOM(hm.close, timeperiod=16)[-1]
        MOM22 = ta.MOM(hm.close, timeperiod=22)[-1]
        MOM7 = ta.MOM(hm.close, timeperiod=7)[-1]
        MOM33 = ta.MOM(hm.close, timeperiod=33)[-1]
        MOM41 = ta.MOM(hm.close, timeperiod=41)[-1]
        MOM57 = ta.MOM(hm.close, timeperiod=57)[-1]
        SLOPE7 = ta.LINEARREG_SLOPE(hm.close, timeperiod=7)[-1]
        SLOPE11 = ta.LINEARREG_SLOPE(hm.close, timeperiod=11)[-1]
        SLOPE22 = ta.LINEARREG_SLOPE(hm.close, timeperiod=22)[-1]
        SLOPE38 = ta.LINEARREG_SLOPE(hm.close, timeperiod=38)[-1]
        MOM103 = ta.MOM(hm.close, timeperiod=103)[-1]
        SLOPE54 = ta.LINEARREG_SLOPE(hm.close, timeperiod=54)[-1]
        MOM213 = ta.MOM(hm.close, timeperiod=213)[-1]
        SLOPE103 = ta.LINEARREG_SLOPE(hm.close, timeperiod=103)[-1]
        RSI23 = ta.RSI(hm.close, timeperiod=23)[-1]
        RSI7 = ta.RSI(hm.close, timeperiod=7)[-1]
        SLOPE204 = ta.LINEARREG_SLOPE(hm.close, timeperiod=204)[-1]
        RSI105 = ta.RSI(hm.close, timeperiod=105)[-1]
        ADX16 = ta.ADX(hm.high, hm.low, hm.close, timeperiod=16)[-1]
        ADX23 = ta.ADX(hm.high, hm.low, hm.close, timeperiod=23)[-1]
        RSI205 = ta.RSI(hm.close, timeperiod=205)[-1]
        ADX32 = ta.ADX(hm.high, hm.low, hm.close, timeperiod=32)[-1]
        ADX42 = ta.ADX(hm.high, hm.low, hm.close, timeperiod=42)[-1]
        ADX7 = ta.ADX(hm.high, hm.low, hm.close, timeperiod=7)[-1]
        ADX61 = ta.ADX(hm.high, hm.low, hm.close, timeperiod=61)[-1]
        ADX104 = ta.ADX(hm.high, hm.low, hm.close, timeperiod=104)[-1]
        ADX240 = ta.ADX(hm.high, hm.low, hm.close, timeperiod=240)[-1]
        lagged30 = (np.diff(hm.close, 10) / hm.close[10:])[-30]
        lagged380 = (np.diff(hm.close, 10) / hm.close[10:])[-380]
        lagged120 = (np.diff(hm.close, 10) / hm.close[10:])[-120]
        lagged240 = (np.diff(hm.close, 10) / hm.close[10:])[-240]
        lagged400 = (np.diff(hm.close, 10) / hm.close[10:])[-400]
        time_hour_sin = np.sin(hm.datetime[-1].hour / 24 * 2 * np.pi)
        time_hour_cos = np.cos(hm.datetime[-1].hour / 24 * 2 * np.pi)
        time_minute_sin = np.sin(hm.datetime[-1].minute / 60 * 2 * np.pi)
        time_minute_cos = np.cos(hm.datetime[-1].minute / 60 * 2 * np.pi)
        time_day_of_wk_sin = np.sin(hm.datetime[-1].dayofweek / 7 * 2 * np.pi)
        time_day_of_wk_cos = np.cos(hm.datetime[-1].dayofweek / 7 * 2 * np.pi)
        time_day_of_mth_si = np.sin(hm.datetime[-1].day / 30 * 2 * np.pi)
        time_day_of_mth_co = np.cos(hm.datetime[-1].day / 30 * 2 * np.pi)

        feature_matrix = np.array([
            ATR7, ATR22, ATR61, MOM16, MOM22, MOM7, MOM33, MOM41, MOM57,
            SLOPE7, SLOPE11, SLOPE22, SLOPE38, MOM103, SLOPE54, MOM213,
            SLOPE103, RSI23, RSI7, SLOPE204, RSI105, ADX16, ADX23, RSI205,
            ADX32, ADX42, ADX7, ADX61, ADX104, ADX240, lagged30, lagged380,
            lagged120, lagged240, lagged400, time_hour_sin, time_hour_cos,
            time_minute_sin, time_minute_cos, time_day_of_wk_sin,
            time_day_of_wk_cos, time_day_of_mth_si, time_day_of_mth_co
        ]).reshape(1, -1)

        # 模型推理
        enter_pos_signal = self.enter_pos_classifier_model.predict(
            feature_matrix
        )
        if enter_pos_signal > self.N:
            long_signal = True
        elif enter_pos_signal < -self.N:
            short_signal = True
        else:
            long_signal = False
            short_signal = False

        # 获取当前目标
        last_target: int = self.get_target()

        # 初始化新一轮目标（默认不变）
        new_target: int = last_target

        # 执行开仓信号
        if long_signal:
            new_target = self.trading_size
        elif short_signal:
            new_target = -self.trading_size

        # 持仓时间平仓
        if self.bar_since_entry() >= self.max_holding:
            new_target = 0

        # 保护止损平仓

        # 设置新一轮目标
        self.set_target(new_target)

        # 执行目标交易
        self.execute_trading(self.price_add)

        # 推送UI更新
        self.put_event()
