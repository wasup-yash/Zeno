import pytest
from zeno import TemporalSplitter
from datetime import datetime

def test_valid_split():
    splitter = TemporalSplitter()
    dates = [datetime(2024, 1, i) for i in range(1, 11)]
    
    train_mask, test_mask = splitter.split(
        dates,
        datetime(2024, 1, 5),
        datetime(2024, 1, 6)
    )
    
    assert sum(train_mask) == 5
    assert sum(test_mask) == 5

def test_leakage_detection():
    splitter = TemporalSplitter()
    dates = [datetime(2024, 1, i) for i in range(1, 11)]
    
    splitter.split(
        dates,
        datetime(2024, 1, 5),
        datetime(2024, 1, 6)
    )
    
    # Should pass
    splitter.validate_feature(datetime(2024, 1, 3))
    
    # Should fail
    with pytest.raises(ValueError, match="TEMPORAL LEAKAGE"):
        splitter.validate_feature(datetime(2024, 1, 7))
