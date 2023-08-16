import numpy as np
import re
from functools import partial
from talib_map import ta_map, func_map
from elite_ctastrategy import HistoryManager


class FeatureDispatcher:
    """Function dispatcher for feature generation."""

    def __init__(self):
        with open("../src/feature_names.txt", "r") as f:
            self.features = f.read()
        interval_feats = re.findall(r"(\d+)_(\w+)_(\d+)", features)
        pattern_feats = re.findall(r"(\d+)_(CDL\w+)", features)
        time_feats = re.findall((r"(\d+)_(ti\w+)"), features)
        ht_feats = re.findall(r"(\d+)_(HT_\w+)", features)

        self.func_pool = [None] * len(interval_feats + pattern_feats + ht_feats + time_feats)
    
    def _map_funcs(self):
        for func in self.interval_feats:
            if func[1] == "MACDHIST" or func[1] == "MACDSIGNAL":
                function = partial(
                    ta_map["MACD"],
                    fastperiod=int(func[2]),
                    slowperiod=int(int(func[2]) * 2),
                    signalperiod=int(int(func[2]) / 2),
                )
                self.func_pool[int(func[0])] = function
                continue
            elif func[1] == "MACDSIGNALFIX" or func[1] == "MACDHISTFIX":
                function = partial(ta_map["MACDFIX"], signalperiod=int(func[2]))
                self.func_pool[int(func[0])] = function
                continue
            function = ta_map[func[1]]
            if "timeperiod" in function.parameters:
                function = partial(function, timeperiod=int(func[2]))
            elif "fastperiod" in function.parameters:
                function = partial(
                    function, fastperiod=int(func[2]), slowperiod=int(int(func[2])) * 3
                )
            elif func[1] == "STOCH":
                function = partial(
                    function,
                    fastk_period=int(func[2]),
                    slowk_period=int(int(func[2]) / 2),
                    slowd_period=int(int(func[2]) / 2),
                )
            elif func[1] == "STOCHF":
                function = partial(
                    function, fastk_period=int(func[2]), fastd_period=int(int(func[2]) / 2)
                )

            self.func_pool[int(func[0])] = function

        for func in self.pattern_feats:
            function = ta_map[func[1]]
            self.func_pool[int(func[0])] = function

        for func in self.ht_feats:
            if func[1] == "HT_PHASORinphase" or func[1] == "HT_PHASORquadrature":
                function = ta_map["HT_PHASOR"]
            elif func[1] == "HT_SINEsine" or func[1] == "HT_SINEleadsine":
                function = ta_map["HT_SINE"]
            else:
                function = ta_map[func[1]]
            self.func_pool[int(func[0])] = function

        for func in self.time_feats:
            function = func_map[func[1]]
            self.func_pool[int(func[0])] = function

    def _generate_features(self, hm: HistoryManager):
        cache = np.zeros(len(self.func_pool))
        inputs = {
            "open": hm.open,
            "high": hm.high,
            "low": hm.low,
            "close": hm.close,
            "volume": hm.volume,
            "datetime": hm.close.datetime,
        }
        for i, func in enumerate(self.func_pool):
            cache[i] = func(inputs)[-1]
        return cache
    
    def feed(self, hm: HistoryManager):
        return self._generate_features(hm)