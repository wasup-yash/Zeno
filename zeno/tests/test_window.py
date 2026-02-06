import pytest
from zeno import Window

def test_lag_creation():
    window = Window(lags=[1, 2])
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    lags = window.transform(values)
    
    assert len(lags) == 2  # Two lag features
    assert lags[0][0] is None  # First lag-1 value
    assert lags[0][1] == 1.0  # Second lag-1 value
    assert lags[1][2] == 1.0  # lag-2 at position 2

def test_rolling_mean():
    window = Window(lags=[1])
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    rolling = window.rolling_mean(values, window=3)
    
    assert rolling[0] is None
    assert rolling[1] is None
    assert rolling[2] == 2.0  # (1+2+3)/3
    assert rolling[4] == 4.0  # (3+4+5)/3
