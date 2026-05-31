from datetime import datetime, timedelta

import polars as pl
import pyarrow as pa
import pytest

from zeno.advanced import ArrowWindow, PolarsTemporalValidator, PolarsWindow


def test_arrow_lag_reuses_source_value_buffer():
    table = pa.table(
        {
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    result = ArrowWindow().create_lags(table, "value", [1])

    source_buffer = table.column("value").chunk(0).buffers()[1]
    lag_buffer = result.column("value_lag_1").chunk(1).buffers()[1]

    assert result["value_lag_1"].to_pylist() == [None, 1.0, 2.0, 3.0, 4.0]
    assert lag_buffer.address == source_buffer.address


def test_polars_features_are_expression_based():
    df = pl.DataFrame({"t": [1, 2, 3, 4], "value": [10.0, 11.0, 13.0, 16.0]})

    result = PolarsWindow(lags=[1], rolling=[2]).transform(df, "value")

    assert result["value_lag_1"].to_list() == [None, 10.0, 11.0, 13.0]
    assert result["value_rolling_2"].to_list() == [None, 10.5, 12.0, 14.5]


def test_polars_temporal_split_uses_sorted_slices():
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(6)],
            "value": [float(i) for i in range(6)],
        }
    )

    train, test = PolarsTemporalValidator().split(
        df,
        "timestamp",
        datetime(2024, 1, 3),
        datetime(2024, 1, 5),
    )

    assert train["value"].to_list() == [0.0, 1.0, 2.0]
    assert test["value"].to_list() == [4.0, 5.0]


def test_polars_temporal_split_rejects_unsorted_data():
    df = pl.DataFrame({"timestamp": [3, 1, 2], "value": [3.0, 1.0, 2.0]})

    with pytest.raises(ValueError, match="sorted ascending"):
        PolarsTemporalValidator().split(df, "timestamp", 1, 2)
