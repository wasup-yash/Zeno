"""
Temporal validation and leakage detection
"""
from zeno._zeno import TemporalValidator as _TemporalValidator
from datetime import datetime

class TemporalSplitter:
    """Enforce temporal ordering in train/test splits"""
    
    def __init__(self):
        self._core = _TemporalValidator()
    
    def split(self, timestamps, train_end_date: datetime, test_start_date: datetime):
        """
        Create a temporal train/test split with validation
        """
        train_ts = int(train_end_date.timestamp())
        test_ts = int(test_start_date.timestamp())
        
        self._core.set_split(train_ts, test_ts)
        
        train_mask = [ts <= train_end_date for ts in timestamps]
        test_mask = [ts >= test_start_date for ts in timestamps]
        
        return train_mask, test_mask
    
    def validate_feature(self, feature_date: datetime):
        """Check if a feature uses future data"""
        ts = int(feature_date.timestamp())
        return self._core.check_feature_window(ts)
