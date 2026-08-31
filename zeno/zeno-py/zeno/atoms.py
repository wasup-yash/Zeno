"""
Atomic operations for time series transformations
"""
from typing import List, Optional
from zeno._zeno import WindowOp as _WindowOp

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


class Scale:
    """Placeholder for scaling operations"""
    
    def __init__(self, method: str = "standard"):
        self.method = method
    
    def fit(self, data):
        pass
    
    def transform(self, data):
        return data
