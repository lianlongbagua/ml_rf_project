from numpy import ndarray

from elite_ctastrategy import (
    EliteCtaTemplate,
    HistoryManager,
    Parameter,
    Variable,
)
from src.feature_dispatcher import FeatureDispatcher
import joblib


class RfStrategy(EliteCtaTemplate):
    """RandomForest策略"""

    author = "czl"

    # 基础参数（必填）
    bar_window: int = Parameter(1)             # K线窗口
    bar_interval: int = Parameter("1m")         # K线级别
    bar_buffer: int = Parameter(300)            # K线缓存

    # 策略参数（可选）
    price_add: int = Parameter(5)               # 委托下单超价

    # 策略变量
    enter_pos_signal: str = Variable("")          
    enter_pos_signal_proba: float = Variable(0)   
    hold_pos_signal: str = Variable("")
    hold_pos_signal_proba: float = Variable(0)         
    change_pos: int = Variable(0)
    change_pos_proba: float = Variable(0)
    total_pos: int = Variable(0)               
    total_pos_proba: float = Variable(0)       

    def on_init(self) -> None:
        """初始化"""
        self.write_log("策略初始化")
        self.load_bar(10)
        self.dispatcher = FeatureDispatcher()
        self.enter_pos_classifier_model = joblib.load("model/enter_pos_classifier.pkl")
        self.hold_pos_classifier_model = joblib.load("model/hold_pos_classifier.pkl")
        self.change_pos_regressor_model = joblib.load("model/change_pos_regressor.pkl")
        self.total_pos_regressor_model = joblib.load("model/total_pos_regressor.pkl")

    def on_start(self) -> None:
        """启动"""
        self.write_log("策略启动")

    def on_stop(self) -> None:
        """停止"""
        self.write_log("策略停止")

    def on_history(self, hm: HistoryManager) -> None:
        """K线推送"""
        # 获取特征矩阵
        feature_matrix = dispatcher.feed(hm)

        # 模型推理
        enter_pos_signal = self.enter_pos_classifier_model.predict(feature_matrix)
        enter_pos_signal_proba = self.enter_pos_classifier_model.predict_proba(feature_matrix)
        hold_pos_signal = self.hold_pos_classifier_model.predict(feature_matrix)
        hold_pos_signal_proba = self.hold_pos_classifier_model.predict_proba(feature_matrix)
        change_pos = self.change_pos_regressor_model.predict(feature_matrix)
        change_pos_proba = self.change_pos_regressor_model.predict_proba(feature_matrix)
        total_pos = self.total_pos_regressor_model.predict(feature_matrix)
        total_pos_proba = self.total_pos_regressor_model.predict_proba(feature_matrix)

        # 简单处理：一共有八种组合，外加置信区间，可以通过组合优化调整
        if enter_pos_signal == "strong buy" and enter_pos_signal_proba > 0.8:
            if hold_pos_signal == "long hold" and hold_pos_signal_proba > 0.8:
                long_signal = True
                short_signal = False
        
        if enter_pos_signal == "strong sell" and enter_pos_signal_proba > 0.8:
            if hold_pos_signal == "short hold" and hold_pos_signal_proba > 0.8:
                long_signal = False
                short_signal = True

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
        close_price = hm.close[-1]

        if last_target > 0:
            stop_price: float = self.long_average_price() * (1 - self.stop_percent)
            if close_price <= stop_price:
                new_target = 0
        elif last_target < 0:
            stop_price: float = self.short_average_price() * (1 + self.stop_percent)
            if close_price >= stop_price:
                new_target = 0

        # 设置新一轮目标
        self.set_target(new_target)

        # 执行目标交易
        self.execute_trading(self.price_add)

        # 推送UI更新
        self.put_event()
