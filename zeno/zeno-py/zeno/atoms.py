"""Atomic operations for time series transformations."""

from typing import List, Optional

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc

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
    """Standard/robust scaling that keeps source columns untouched."""

    def __init__(self, method: str = "standard", column: Optional[str] = None):
        if method not in {"standard", "robust"}:
            raise ValueError("method must be 'standard' or 'robust'")
        self.method = method
        self.column = column
        self.center_ = None
        self.scale_ = None

    def fit(self, data, column: Optional[str] = None):
        column = column or self.column
        if isinstance(data, pl.DataFrame):
            column = Window._resolve_column(data, column)
            if self.method == "standard":
                stats = data.select(
                    pl.col(column).mean().alias("center"),
                    pl.col(column).std().alias("scale"),
                )
            else:
                stats = data.select(
                    pl.col(column).median().alias("center"),
                    (pl.col(column).quantile(0.75) - pl.col(column).quantile(0.25)).alias("scale"),
                )
            self.center_ = float(stats["center"][0])
            self.scale_ = float(stats["scale"][0] or 1.0)
            self.column = column
            return self

        if isinstance(data, pa.Table):
            column = Window._resolve_column(data, column)
            values = data.column(column)
            if self.method == "standard":
                self.center_ = pc.mean(values).as_py()
                self.scale_ = pc.stddev(values).as_py() or 1.0
            else:
                quantiles = pc.quantile(values, q=[0.25, 0.5, 0.75]).as_py()
                self.center_ = quantiles[1]
                self.scale_ = (quantiles[2] - quantiles[0]) or 1.0
            self.column = column
            return self

        values = [float(value) for value in data]
        sorted_values = sorted(values)
        if self.method == "standard":
            self.center_ = sum(values) / len(values)
            variance = sum((value - self.center_) ** 2 for value in values) / len(values)
            self.scale_ = variance**0.5 or 1.0
        else:
            mid = len(sorted_values) // 2
            self.center_ = sorted_values[mid]
            q1 = sorted_values[len(sorted_values) // 4]
            q3 = sorted_values[(3 * len(sorted_values)) // 4]
            self.scale_ = (q3 - q1) or 1.0
        return self

    def transform(self, data, column: Optional[str] = None, *, include_original: bool = True):
        if self.center_ is None or self.scale_ is None:
            self.fit(data, column)
        column = column or self.column

        if isinstance(data, pl.DataFrame):
            column = Window._resolve_column(data, column)
            expr = ((pl.col(column) - self.center_) / self.scale_).alias(f"{column}_scaled")
            return data.with_columns(expr) if include_original else data.select(expr)

        if isinstance(data, pa.Table):
            column = Window._resolve_column(data, column)
            scaled = pc.divide(pc.subtract(data.column(column), self.center_), self.scale_)
            if include_original:
                return data.append_column(f"{column}_scaled", scaled)
            return pa.Table.from_arrays([scaled], names=[f"{column}_scaled"])

        return [(float(value) - self.center_) / self.scale_ for value in data]
