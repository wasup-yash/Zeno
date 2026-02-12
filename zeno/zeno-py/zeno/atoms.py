"""
Atomic operations for time series transformations
"""
from typing import List, Optional
from ._zeno import WindowOp as _WindowOp
from ._zeno import ArrowPipeline as _ArrowCore

class Window:
    """Create lag and rolling window features with zero-copy"""
    
    def __init__(self, lags: List[int], rolling: Optional[List[int]] = None):
        self._core = _WindowOp(lags, rolling)
        self.lags = lags
        self.rolling = rolling or []
    
    def transform(self, values: List[float]):
        """Create lag features"""
        return self._core.create_lags(values)
    
    def rolling_mean(self, values: List[float], window: int):
        """Create rolling mean feature"""
        return self._core.rolling_mean(values, window)
    
    def __repr__(self):
        return f"Window(lags={self.lags}, rolling={self.rolling})"

class EMA:
    """Exponential Moving Average Atom"""
    
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._core = _ArrowCore()
    
    def transform(self, values: List[float]):
        return self._core.ema_simple(values, self.alpha)

class Scale:
    """Placeholder for scaling operations"""
    
    def __init__(self, method: str = "standard"):
        self.method = method
    
    def fit(self, data):
        pass
    
    def transform(self, data):
        return data
