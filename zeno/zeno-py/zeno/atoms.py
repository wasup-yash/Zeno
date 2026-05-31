"""Atomic operations for time series transformations."""

from typing import List, Optional

import polars as pl
import pyarrow as pa

from .advanced import ArrowWindow, PolarsWindow
from ._zeno import WindowOp as _WindowOp
from ._zeno import ArrowPipeline as _ArrowCore

class Window:
    """Create lag and rolling window features.

    Polars and Arrow inputs use zero-copy data paths. Plain Python lists are
    kept as a small compatibility path for quick experiments and tests.
    """
    
    def __init__(self, lags: List[int], rolling: Optional[List[int]] = None):
        self._core = _WindowOp(lags, rolling)
        self._arrow = ArrowWindow(lags, rolling)
        self._polars = PolarsWindow(lags, rolling)
        self.lags = lags
        self.rolling = rolling or []
    
    def transform(self, values, column: Optional[str] = None, *, include_original: bool = True):
        """Create lag features."""
        if isinstance(values, pl.DataFrame):
            return self._polars.transform(values, self._resolve_column(values, column), include_original=include_original)
        if isinstance(values, pa.Table):
            return self._arrow.transform(values, self._resolve_column(values, column), include_original=include_original)
        return self._core.create_lags(values)
    
    def rolling_mean(self, values, window: int, column: Optional[str] = None, *, include_original: bool = True):
        """Create rolling mean feature."""
        if isinstance(values, pl.DataFrame):
            column = self._resolve_column(values, column)
            return values.with_columns(pl.col(column).rolling_mean(window).alias(f"{column}_rolling_{window}"))
        if isinstance(values, pa.Table):
            return self._arrow.rolling_mean(values, self._resolve_column(values, column), window, include_original=include_original)
        return self._core.rolling_mean(values, window)

    @staticmethod
    def _resolve_column(data, column: Optional[str]) -> str:
        if column is not None:
            return column
        names = data.columns if isinstance(data, pl.DataFrame) else data.column_names
        numeric_names = [
            name for name in names
            if (
                isinstance(data, pl.DataFrame)
                and data.schema[name].is_numeric()
            ) or (
                isinstance(data, pa.Table)
                and pa.types.is_floating(data.schema.field(name).type)
            ) or (
                isinstance(data, pa.Table)
                and pa.types.is_integer(data.schema.field(name).type)
            )
        ]
        if len(numeric_names) != 1:
            raise ValueError("Pass column=... when the input has zero or multiple numeric columns")
        return numeric_names[0]
    
    def __repr__(self):
        return f"Window(lags={self.lags}, rolling={self.rolling})"

class EMA:
    """Exponential moving average atom."""
    
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._core = _ArrowCore()
    
    def transform(self, values, column: Optional[str] = None, *, include_original: bool = True):
        if isinstance(values, pa.Table):
            if column is None:
                column = Window._resolve_column(values, column)
            return ArrowWindow().ema(values, column, self.alpha, include_original=include_original)
        if isinstance(values, pl.DataFrame):
            if column is None:
                column = Window._resolve_column(values, column)
            return values.with_columns(
                pl.col(column).ewm_mean(alpha=self.alpha, adjust=False).alias(f"{column}_ema_{self.alpha:g}")
            )
        return self._core.ema_simple(values, self.alpha)

class Scale:
    """Placeholder for scaling operations."""
    
    def __init__(self, method: str = "standard"):
        self.method = method
    
    def fit(self, data):
        pass
    
    def transform(self, data):
        return data
