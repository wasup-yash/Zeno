from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import pyarrow as pa

from ._zeno import (
    ArrowPipeline,
    AuditReport,
    GPUAccelerator,
    LeakageDetector,
    ManagedPipeline,
    PolarsValidator,
    PolarsWindowOp,
    RollingHashValidator,
)
from .zero_copy import (
    append_arrow_columns,
    arrow_lag,
    ensure_arrow_table,
    normalize_lags,
    validate_temporal_coverage,
    zero_copy_temporal_split,
)


class ArrowWindow:
    """Arrow-native feature generation with zero-copy input handling."""

    def __init__(
        self,
        lags: Optional[List[int]] = None,
        rolling: Optional[List[int]] = None,
    ):
        self._pipeline = ArrowPipeline()
        self.lags = normalize_lags(lags)
        self.rolling = [int(window) for window in (rolling or [])]

    def create_lags(
        self,
        table: pa.Table,
        column: str,
        lags: Optional[List[int]] = None,
        *,
        include_original: bool = True,
    ) -> pa.Table:
        table = ensure_arrow_table(table)
        selected_lags = normalize_lags(self.lags if lags is None else lags)
        source = table.column(column)

        columns = [
            (f"{column}_lag_{lag}", arrow_lag(source, lag))
            for lag in selected_lags
        ]
        return append_arrow_columns(table, columns, include_original=include_original)

    def rolling_mean(
        self,
        table: pa.Table,
        column: str,
        window: int,
        *,
        include_original: bool = True,
    ) -> pa.Table:
        table = ensure_arrow_table(table)
        if window <= 0:
            raise ValueError("window must be positive")

        name = f"{column}_rolling_{window}"
        df = pl.from_arrow(table, rechunk=False).with_columns(
            pl.col(column).rolling_mean(window).alias(name)
        )
        result = df.to_arrow()
        return result if include_original else result.select([name])

    def ema(
        self,
        table: pa.Table,
        column: str,
        alpha: float = 0.3,
        *,
        include_original: bool = True,
    ) -> pa.Table:
        table = ensure_arrow_table(table)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the interval (0, 1]")

        name = f"{column}_ema_{alpha:g}"
        df = pl.from_arrow(table, rechunk=False).with_columns(
            pl.col(column).ewm_mean(alpha=alpha, adjust=False).alias(name)
        )
        result = df.to_arrow()
        return result if include_original else result.select([name])

    def transform(
        self,
        table: pa.Table,
        column: str,
        *,
        include_original: bool = True,
    ) -> pa.Table:
        result = self.create_lags(
            table,
            column,
            self.lags,
            include_original=include_original,
        )
        for window in self.rolling:
            result = self.rolling_mean(result, column, window, include_original=True)
        return result


class AdvancedLeakageDetector:
    def __init__(self, threshold: float = 0.1):
        self._detector = LeakageDetector(threshold)
        self._hash_validator = RollingHashValidator(hash_size=100)

    def register_training_feature(
        self,
        timestamps: List[int],
        values: List[float],
        feature_name: str,
    ):
        self._detector.register_train_window(timestamps, values, feature_name)
        self._hash_validator.add_window(values)

    def check_test_feature(
        self,
        timestamps: List[int],
        values: List[float],
        feature_name: str,
    ) -> Dict[str, float]:
        if self._hash_validator.check_window(values):
            raise ValueError(f"EXACT MATCH: Feature '{feature_name}' appears in training data.")
        return self._detector.check_test_window(timestamps, values, feature_name)

    def get_report(self):
        return self._detector.get_leakage_report()


class PolarsWindow:
    """Polars-native feature generation using expression plans."""

    def __init__(self, lags: List[int], rolling: Optional[List[int]] = None):
        self._op = PolarsWindowOp(lags, rolling or [])
        self.lags = normalize_lags(lags)
        self.rolling = [int(window) for window in (rolling or [])]

    def _expressions_for(self, column: str) -> list[pl.Expr]:
        exprs: list[pl.Expr] = [
            pl.col(column).shift(lag).alias(f"{column}_lag_{lag}")
            for lag in self.lags
        ]
        exprs.extend(
            pl.col(column).rolling_mean(window).alias(f"{column}_rolling_{window}")
            for window in self.rolling
        )
        return exprs

    def transform(
        self,
        df: pl.DataFrame,
        column: str,
        *,
        include_original: bool = True,
    ) -> pl.DataFrame:
        exprs = self._expressions_for(column)
        return df.with_columns(exprs) if include_original else df.select(exprs)

    def transform_parallel(
        self,
        df: pl.DataFrame,
        columns: List[str],
        *,
        include_original: bool = True,
    ) -> pl.DataFrame:
        exprs: list[pl.Expr] = []
        for column in columns:
            exprs.extend(self._expressions_for(column))
        return df.with_columns(exprs) if include_original else df.select(exprs)

    def transform_lazy(self, lf: pl.LazyFrame, columns: List[str]) -> pl.LazyFrame:
        exprs: list[pl.Expr] = []
        for column in columns:
            exprs.extend(self._expressions_for(column))
        return lf.with_columns(exprs)


class PolarsTemporalValidator:
    """Temporal validation and zero-copy slicing for sorted Polars DataFrames."""

    def __init__(self):
        self._validator = PolarsValidator()

    def validate_split(
        self,
        df: pl.DataFrame,
        time_col: str,
        train_end: datetime,
        test_start: datetime,
    ) -> bool:
        return validate_temporal_coverage(df, time_col, train_end, test_start)

    def split(
        self,
        df: pl.DataFrame,
        time_col: str,
        train_end: datetime,
        test_start: Optional[datetime] = None,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        return zero_copy_temporal_split(df, time_col, train_end, test_start)


class ExpandingWindowValidator:
    def __init__(self, min_train_size: int, test_size: int, step_size: int = 1):
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.step_size = step_size

    def split(self, df: pl.DataFrame, time_col: str):
        from .zero_copy import assert_sorted_by

        assert_sorted_by(df, time_col)
        n_rows = len(df)
        splits = []
        for i in range(self.min_train_size, n_rows - self.test_size + 1, self.step_size):
            train = df.slice(0, i)
            test = df.slice(i, self.test_size)
            splits.append((train, test))
        return splits

    def validate(
        self,
        df: pl.DataFrame,
        time_col: str,
        feature_cols: List[str],
    ) -> Dict[str, List[float]]:
        splits = self.split(df, time_col)
        results = {
            "train_sizes": [],
            "test_sizes": [],
            "gaps": [],
        }

        for train, test in splits:
            results["train_sizes"].append(len(train))
            results["test_sizes"].append(len(test))
            train_max = train[time_col].max()
            test_min = test[time_col].min()
            results["gaps"].append(test_min - train_max)

        return results


def create_arrow_pipeline(
    lags: List[int],
    rolling_windows: Optional[List[int]] = None,
) -> ArrowWindow:
    return ArrowWindow(lags, rolling_windows)


def create_polars_pipeline(
    lags: List[int],
    rolling_windows: Optional[List[int]] = None,
) -> PolarsWindow:
    return PolarsWindow(lags, rolling_windows)


class GPUManager:
    """Managed GPU resources."""

    def __init__(self, device: str = "cuda:0"):
        self._accel = GPUAccelerator(device=device)

    def stats(self):
        if self._accel.is_available():
            alloc, res = self._accel.get_memory_info()
            return {"allocated_mb": alloc / 1024**2, "reserved_mb": res / 1024**2}
        return {"status": "GPU not available"}


class AuditManager:
    """Generate compliance reports."""

    def __init__(self, report_id: str):
        self.report = AuditReport(report_id)

    def log_metric(self, name: str, value: float):
        self.report.add_metric(name, value)

    def finalize(self) -> Dict[str, str]:
        return self.report.summary


class ManagedExecutor:
    """Execute validation pipelines with automatic scheduling."""

    def __init__(self, pipeline_id: str):
        self._pipe = ManagedPipeline(pipeline_id)

    def add_validation_step(self, name: str, step_type: str = "temporal_split"):
        self._pipe.add_step(name, step_type)

    def run(self, data: Any):
        return self._pipe.execute(data)
