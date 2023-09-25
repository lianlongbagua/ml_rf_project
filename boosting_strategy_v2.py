import numpy as np
import warnings

from elite_ctastrategy import (
    EliteCtaTemplate,
    HistoryManager,
    Parameter,
    Variable,
)

import joblib
import talib


class HGBRStrategy(EliteCtaTemplate):
    """提升树策略"""

    author = "czl"

    # 基础参数（必填）
    bar_window: int = Parameter(1)  # K线窗口
    bar_interval: int = Parameter("1m")  # K线级别
    bar_buffer: int = Parameter(100)  # K线缓存

    # 策略参数（可选）
    price_add: int = Parameter(5)  # 委托下单超价
    invested: bool = Parameter(False)  # 是否持仓
    model_path: str = Parameter("model/HGBC_SP_0925.pkl")  # 模型路径
    lags = 30

    # 策略变量
    trading_size: int = Variable(1)  # 下单手数

    def on_init(self) -> None:
        """初始化"""
        self.write_log("策略初始化")
        self.load_bar(10)
        warnings.filterwarnings("ignore")
        self.invested = False
        self.lags = 30
        self.model = joblib.load(self.model_path)

    def on_start(self) -> None:
        """启动"""
        self.write_log("策略启动")

    def on_stop(self) -> None:
        """停止"""
        self.write_log("策略停止")

    def on_history(self, hm: HistoryManager) -> None:
        """K线推送"""
        # 获取特征矩阵
        adj_close = talib.MEDPRICE(hm.high, hm.low)
        shifted_adj_close = np.roll(adj_close, 1)
        returns = (adj_close/shifted_adj_close - 1) * 100

        feature_matrix = returns[-self.lags:].reshape(1, -1)

        # 模型推理
        signal = self.model.predict(feature_matrix)

        # 设置新一轮目标
        if signal == 1:
            self.set_target(10)
        elif signal == -1:
            self.set_target(0)

        # 执行目标交易
        self.execute_trading(self.price_add)

        # 推送UI更新
        self.put_event()
