"""
Shared zero-copy helpers for Arrow and Polars data paths.

Derived features still need new output buffers. The contract here is stricter:
never materialize the input series through Python lists, NumPy copies, or dict
round-trips, and prefer views/slices for existing data.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import polars as pl
import pyarrow as pa


def ensure_arrow_table(table: pa.Table) -> pa.Table:
    if not isinstance(table, pa.Table):
        raise TypeError("Expected a pyarrow.Table")
    return table


def ensure_polars_frame(df: pl.DataFrame) -> pl.DataFrame:
    if not isinstance(df, pl.DataFrame):
        raise TypeError("Expected a polars.DataFrame")
    return df


def normalize_lags(lags: Optional[Iterable[int]]) -> List[int]:
    clean = [int(lag) for lag in (lags or [])]
    if any(lag < 0 for lag in clean):
        raise ValueError("Lags must be non-negative")
    return clean


def arrow_lag(column: pa.ChunkedArray, lag: int) -> pa.ChunkedArray:
    """
    Build a lagged Arrow column without copying the source value buffers.

    The lagged column is represented as chunks: a newly allocated null prefix
    followed by sliced chunks from the original column.
    """
    lag = int(lag)
    if lag < 0:
        raise ValueError("Lag must be non-negative")

    n_rows = len(column)
    if lag == 0:
        return column
    if n_rows == 0:
        return pa.chunked_array([], type=column.type)
    if lag >= n_rows:
        return pa.chunked_array([pa.nulls(n_rows, type=column.type)])

    shifted = column.slice(0, n_rows - lag)
    chunks = [pa.nulls(lag, type=column.type), *shifted.chunks]
    return pa.chunked_array(chunks, type=column.type)


def append_arrow_columns(
    table: pa.Table,
    columns: Sequence[Tuple[str, Union[pa.Array, pa.ChunkedArray]]],
    *,
    include_original: bool = True,
) -> pa.Table:
    ensure_arrow_table(table)

    if include_original:
        result = table
        for name, array in columns:
            result = result.append_column(name, array)
        return result

    arrays = [array for _, array in columns]
    names = [name for name, _ in columns]
    return pa.Table.from_arrays(arrays, names=names)


def datetime_after(value: datetime) -> datetime:
    return value + timedelta(microseconds=1)


def validate_temporal_bounds(train_end, test_start) -> None:
    if test_start <= train_end:
        raise ValueError("Test start must be after train end (temporal leakage detected!)")


def assert_sorted_by(df: pl.DataFrame, time_col: str) -> None:
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found")
    if df.is_empty():
        raise ValueError("Cannot split an empty DataFrame")
    if not df.get_column(time_col).is_sorted():
        raise ValueError(
            f"Column '{time_col}' must be sorted ascending for zero-copy slicing"
        )


def count_leq(df: pl.DataFrame, time_col: str, value) -> int:
    return int(df.select((pl.col(time_col) <= value).sum().alias("__n")).item())


def count_lt(df: pl.DataFrame, time_col: str, value) -> int:
    return int(df.select((pl.col(time_col) < value).sum().alias("__n")).item())


def zero_copy_temporal_split(
    df: pl.DataFrame,
    time_col: str,
    train_end,
    test_start=None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split a sorted Polars DataFrame with contiguous slices.

    This avoids `filter`, so existing column buffers can be shared by the train
    and test views. If `test_start` is omitted, the test slice starts at the
    first row after `train_end`.
    """
    ensure_polars_frame(df)
    assert_sorted_by(df, time_col)

    if test_start is not None:
        validate_temporal_bounds(train_end, test_start)
        test_start_idx = count_lt(df, time_col, test_start)
    else:
        test_start_idx = count_leq(df, time_col, train_end)

    train_end_idx = count_leq(df, time_col, train_end)
    train = df.slice(0, train_end_idx)
    test = df.slice(test_start_idx)
    return train, test


def validate_temporal_coverage(
    df: pl.DataFrame,
    time_col: str,
    train_end,
    test_start,
) -> bool:
    ensure_polars_frame(df)
    assert_sorted_by(df, time_col)
    validate_temporal_bounds(train_end, test_start)

    bounds = df.select(
        pl.col(time_col).min().alias("__min"),
        pl.col(time_col).max().alias("__max"),
    )
    min_time = bounds["__min"][0]
    max_time = bounds["__max"][0]

    if min_time > train_end or max_time < test_start:
        raise ValueError("Time range does not cover the requested split")
    return True
