import pytest
import polars as pl
import pyarrow as pa
from datetime import datetime, timedelta
from zeno.advanced import (
    ArrowWindow,
    PolarsWindow,
    AdvancedLeakageDetector,
)

def test_arrow_pipeline():
    # Create Arrow table
    table = pa.table({
        'timestamp': [datetime(2024, 1, i) for i in range(1, 11)],
        'value': [float(i) for i in range(10)],
    })
    
    # Create lags
    window = ArrowWindow()
    result = window.create_lags(table, 'value', [1, 2])
    
    assert result.num_columns == 4  # original 2 + 2 lags
    assert 'value_lag_1' in result.column_names
    assert 'value_lag_2' in result.column_names

def test_polars_pipeline():
    df = pl.DataFrame({
        'timestamp': [datetime(2024, 1, i) for i in range(1, 11)],
        'value': [float(i) for i in range(10)],
    })
    
    window = PolarsWindow(lags=[1, 2])
    result = window.transform(df, 'value')
    
    assert 'value_lag_1' in result.columns
    assert 'value_lag_2' in result.columns

def test_leakage_detection():
    detector = AdvancedLeakageDetector(threshold=0.1)
    
    # Register training
    detector.register_training_feature(
        list(range(100)),
        list(range(100)),
        "train_feature"
    )
    
    # Test with non-overlapping data (should pass)
    result = detector.check_test_feature(
        list(range(100, 200)),
        list(range(100, 200)),
        "test_feature"
    )
    
    assert len(result) == 0  # No leakage
    
    # Test with overlapping data (should fail)
    with pytest.raises(ValueError, match="LEAKAGE"):
        detector.check_test_feature(
            list(range(50, 150)),
            list(range(50, 150)),
            "test_leak"
        )