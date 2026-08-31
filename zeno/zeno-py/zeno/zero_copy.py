"""
Shared zero-copy helpers for Arrow and Polars data paths.

Derived features still need new output buffers. The contract here is stricter:
never materialize the input series through Python lists, NumPy copies, or dict
round-trips, and prefer views/slices for existing data.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from hashlib import blake2b
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc

from ._zeno import create_lags_chunked


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


def generate_lag(series: pl.Series, lag: int) -> pl.Series:
    """Construct a lagged Series from a null chunk and a zero-copy Arrow slice.

    The source Series must fit in one Arrow array, as returned by Polars'
    ``Series.to_arrow``. The Rust layer creates only the null-prefix metadata;
    the non-null values retain their original Arrow buffer.
    """
    if not isinstance(series, pl.Series):
        raise TypeError("Expected a polars.Series")

    lag = int(lag)
    if lag < 0:
        raise ValueError("Lag must be non-negative")

    arrow_array = series.to_arrow()
    rust_chunks = create_lags_chunked(arrow_array, lag)
    return pl.from_arrow(pa.chunked_array(rust_chunks, type=arrow_array.type)).rename(series.name)


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


def arrow_window_view(table: pa.Table, offset: int, length: Optional[int] = None) -> pa.Table:
    ensure_arrow_table(table)
    return table.slice(offset, length)


def polars_window_view(
    df: pl.DataFrame,
    offset: int,
    length: Optional[int] = None,
) -> pl.DataFrame:
    ensure_polars_frame(df)
    return df.slice(offset, length)


def _primitive_byte_width(dtype: pa.DataType) -> int:
    bit_width = getattr(dtype, "bit_width", None)
    if bit_width is None or bit_width % 8 != 0:
        raise TypeError(f"Cannot create a zero-copy byte view for Arrow type {dtype}")
    return bit_width // 8


def arrow_array_value_view(array: pa.Array) -> memoryview:
    """
    Return a zero-copy byte view over a primitive Arrow array's visible values.

    Chunked arrays must be handled chunk-by-chunk by callers. Variable-width
    arrays are deliberately rejected because their logical values are split
    across offset and data buffers.
    """
    if not (
        pa.types.is_integer(array.type)
        or pa.types.is_floating(array.type)
        or pa.types.is_temporal(array.type)
    ):
        raise TypeError(f"Only fixed-width numeric/temporal arrays are supported, got {array.type}")

    value_buffer = array.buffers()[1]
    if value_buffer is None:
        return memoryview(b"")

    width = _primitive_byte_width(array.type)
    start = array.offset * width
    end = start + len(array) * width
    return memoryview(value_buffer)[start:end]


def arrow_chunked_value_hash(column: pa.ChunkedArray) -> int:
    """
    Hash visible Arrow values without materializing them into Python objects.
    """
    hasher = blake2b(digest_size=16)
    hasher.update(str(column.type).encode("utf-8"))
    hasher.update(len(column).to_bytes(8, "little", signed=False))

    for chunk in column.chunks:
        hasher.update(len(chunk).to_bytes(8, "little", signed=False))
        null_buffer = chunk.buffers()[0]
        if null_buffer is not None:
            hasher.update(memoryview(null_buffer))
        hasher.update(arrow_array_value_view(chunk))

    return int.from_bytes(hasher.digest(), "little", signed=False)


def arrow_feature_fingerprint(
    table: pa.Table,
    time_col: str,
    value_col: str,
    feature_name: str,
) -> Dict[str, object]:
    """
    Fingerprint a time-series feature window using Arrow buffers directly.
    """
    table = ensure_arrow_table(table)
    if time_col not in table.column_names:
        raise ValueError(f"Column '{time_col}' not found")
    if value_col not in table.column_names:
        raise ValueError(f"Column '{value_col}' not found")

    time_array = table.column(time_col)
    value_array = table.column(value_col)
    bounds = pc.min_max(time_array)

    return {
        "feature_name": feature_name,
        "n_rows": table.num_rows,
        "time_min": bounds["min"].as_py(),
        "time_max": bounds["max"].as_py(),
        "time_hash": arrow_chunked_value_hash(time_array),
        "value_hash": arrow_chunked_value_hash(value_array),
    }


def polars_feature_fingerprint(
    df: pl.DataFrame,
    time_col: str,
    value_col: str,
    feature_name: str,
) -> Dict[str, object]:
    """
    Fingerprint a Polars feature window through Arrow-backed Series views.
    """
    df = ensure_polars_frame(df)
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found")
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found")

    time_arrow = pa.chunked_array([df.get_column(time_col).to_arrow()])
    value_arrow = pa.chunked_array([df.get_column(value_col).to_arrow()])
    bounds = df.select(
        pl.col(time_col).min().alias("__min"),
        pl.col(time_col).max().alias("__max"),
    )

    return {
        "feature_name": feature_name,
        "n_rows": len(df),
        "time_min": bounds["__min"][0],
        "time_max": bounds["__max"][0],
        "time_hash": arrow_chunked_value_hash(time_arrow),
        "value_hash": arrow_chunked_value_hash(value_arrow),
    }


def zero_copy_numpy_view(table: pa.Table, column: str):
    """
    Return a NumPy view over one single-chunk Arrow column.

    This intentionally raises when Arrow would need to combine chunks or copy.
    """
    table = ensure_arrow_table(table)
    chunked = table.column(column)
    if len(chunked.chunks) != 1:
        raise ValueError("zero_copy_numpy_view requires a single Arrow chunk")
    return chunked.chunk(0).to_numpy(zero_copy_only=True)


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
    series = df.get_column(time_col)
    if hasattr(series, "search_sorted"):
        return int(series.search_sorted(value, side="right"))
    return int(df.select((pl.col(time_col) <= value).sum().alias("__n")).item())


def count_lt(df: pl.DataFrame, time_col: str, value) -> int:
    series = df.get_column(time_col)
    if hasattr(series, "search_sorted"):
        return int(series.search_sorted(value, side="left"))
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

    train_end_idx = count_leq(df, time_col, train_end)

    if test_start is not None:
        validate_temporal_bounds(train_end, test_start)
        test_start_idx = count_lt(df, time_col, test_start)
    else:
        test_start_idx = train_end_idx

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
