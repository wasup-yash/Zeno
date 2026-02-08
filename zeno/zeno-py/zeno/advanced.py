"""
Phase 2: Advanced Features Python Interface
zeno/advanced.py
"""

from typing import List, Optional, Dict, Tuple
from datetime import datetime
import polars as pl
import pyarrow as pa

# Import Phase 2 Rust modules
from zeno._zeno import (
    ArrowPipeline,
    LeakageDetector,
    RollingHashValidator,
    PolarsWindowOp,
    PolarsValidator,
)


class ArrowWindow:
    """Zero-copy windowing operations using Apache Arrow"""
    
    def __init__(self):
        self._pipeline = ArrowPipeline()
    
    def create_lags(
        self,
        table: pa.Table,
        column: str,
        lags: List[int],
    ) -> pa.Table:
        """
        Create lag features with true zero-copy using Arrow
        
        Args:
            table: PyArrow Table
            column: Column name to create lags from
            lags: List of lag periods
            
        Returns:
            PyArrow Table with lag features added
        """
        # Convert to RecordBatch
        batch = table.to_batches()[0]
        
        # Create lags using Rust
        result_batch = self._pipeline.create_lags_arrow(column, lags, batch)
        
        # Convert back to Table
        return pa.Table.from_batches([result_batch])
    
    def rolling_mean(
        self,
        table: pa.Table,
        column: str,
        window: int,
    ) -> pa.Table:
        """Create rolling mean feature"""
        batch = table.to_batches()[0]
        result_batch = self._pipeline.rolling_mean_arrow(column, window, batch)
        return pa.Table.from_batches([result_batch])
    
    def ema(
        self,
        table: pa.Table,
        column: str,
        alpha: float = 0.3,
    ) -> pa.Table:
        """Create exponential moving average feature"""
        batch = table.to_batches()[0]
        result_batch = self._pipeline.ema_arrow(column, alpha, batch)
        return pa.Table.from_batches([result_batch])


class AdvancedLeakageDetector:
    """Feature fingerprinting and advanced leakage detection"""
    
    def __init__(self, threshold: float = 0.1):
        """
        Args:
            threshold: Similarity threshold for leakage detection (0.0 to 1.0)
        """
        self._detector = LeakageDetector(threshold)
        self._hash_validator = RollingHashValidator(hash_size=100)
    
    def register_training_feature(
        self,
        timestamps: List[int],
        values: List[float],
        feature_name: str,
    ):
        """Register a training feature for leakage detection"""
        self._detector.register_train_window(timestamps, values, feature_name)
        self._hash_validator.add_window(values)
    
    def check_test_feature(
        self,
        timestamps: List[int],
        values: List[float],
        feature_name: str,
    ) -> Dict[str, float]:
        """
        Check if test feature leaks training data
        
        Returns:
            Dictionary of {train_feature: similarity_score} for detected leaks
            
        Raises:
            ValueError: If leakage detected
        """
        # First check with rolling hash (fast)
        if self._hash_validator.check_window(values):
            raise ValueError(
                f"EXACT MATCH DETECTED: Feature '{feature_name}' "
                f"appears in training data (rolling hash match)"
            )
        
        # Then do detailed fingerprint check
        return self._detector.check_test_window(timestamps, values, feature_name)
    
    def get_report(self) -> Dict[str, List[Tuple[str, float]]]:
        """Get detailed leakage report"""
        return self._detector.get_leakage_report()
    
    def reset(self):
        """Clear all registered features"""
        self._detector.reset()


class PolarsWindow:
    """Polars-native window operations (fastest for Polars DataFrames)"""
    
    def __init__(self, lags: List[int], rolling: Optional[List[int]] = None):
        self._op = PolarsWindowOp(lags, rolling or [])
        self.lags = lags
        self.rolling = rolling or []
    
    def transform(self, df: pl.DataFrame, column: str) -> pl.DataFrame:
        """
        Create lag features on Polars DataFrame
        
        Args:
            df: Polars DataFrame
            column: Column to create lags from
            
        Returns:
            DataFrame with lag columns added
        """
        # Use Polars native operations (faster than Rust FFI)
        result = df
        
        for lag in self.lags:
            result = result.with_columns(
                pl.col(column).shift(lag).alias(f"{column}_lag_{lag}")
            )
        
        for window in self.rolling:
            result = result.with_columns(
                pl.col(column).rolling_mean(window).alias(f"{column}_rolling_{window}")
            )
        
        return result
    
    def transform_parallel(
        self,
        df: pl.DataFrame,
        columns: List[str],
    ) -> pl.DataFrame:
        """Create features for multiple columns in parallel"""
        return self._op.create_features_parallel(df, columns)


class PolarsTemporalValidator:
    """Temporal validation for Polars DataFrames"""
    
    def __init__(self):
        self._validator = PolarsValidator()
    
    def validate_split(
        self,
        df: pl.DataFrame,
        time_col: str,
        train_end: datetime,
        test_start: datetime,
    ) -> bool:
        """
        Validate temporal split in Polars DataFrame
        
        Args:
            df: Polars DataFrame
            time_col: Name of timestamp column
            train_end: Last timestamp in training set
            test_start: First timestamp in test set
            
        Returns:
            True if split is valid
            
        Raises:
            ValueError: If split is invalid or has temporal leakage
        """
        train_ts = int(train_end.timestamp())
        test_ts = int(test_start.timestamp())
        
        return self._validator.validate_temporal_split(
            df, time_col, train_ts, test_ts
        )
    
    def split(
        self,
        df: pl.DataFrame,
        time_col: str,
        cutoff: datetime,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split DataFrame into train and test
        
        Args:
            df: Polars DataFrame
            time_col: Name of timestamp column
            cutoff: Split point (last train timestamp)
            
        Returns:
            (train_df, test_df) tuple
        """
        cutoff_ts = int(cutoff.timestamp())
        return self._validator.split_dataframe(df, time_col, cutoff_ts)


class ExpandingWindowValidator:
    """Expanding window cross-validation"""
    
    def __init__(
        self,
        min_train_size: int,
        test_size: int,
        step_size: int = 1,
    ):
        """
        Args:
            min_train_size: Minimum training samples
            test_size: Number of test samples per fold
            step_size: Step between folds
        """
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.step_size = step_size
    
    def split(
        self,
        df: pl.DataFrame,
        time_col: str,
    ) -> List[Tuple[pl.DataFrame, pl.DataFrame]]:
        """
        Generate expanding window splits
        
        Returns:
            List of (train, test) DataFrame tuples
        """
        df_sorted = df.sort(time_col)
        n = len(df_sorted)
        
        splits = []
        
        for i in range(self.min_train_size, n - self.test_size + 1, self.step_size):
            train = df_sorted[:i]
            test = df_sorted[i:i + self.test_size]
            splits.append((train, test))
        
        return splits
    
    def validate(
        self,
        df: pl.DataFrame,
        time_col: str,
        feature_cols: List[str],
    ) -> Dict[str, List[float]]:
        """
        Validate features across expanding windows
        
        Returns:
            Dictionary of {metric_name: [scores_per_fold]}
        """
        splits = self.split(df, time_col)
        
        results = {
            'train_sizes': [],
            'test_sizes': [],
            'gaps': [],
        }
        
        for train, test in splits:
            results['train_sizes'].append(len(train))
            results['test_sizes'].append(len(test))
            
            # Check temporal gap
            train_max = train[time_col].max()
            test_min = test[time_col].min()
            gap = test_min - train_max
            results['gaps'].append(gap)
        
        return results


# Convenience function for quick setup
def create_arrow_pipeline(
    lags: List[int],
    rolling_windows: Optional[List[int]] = None,
) -> ArrowWindow:
    """Create Arrow-based window pipeline"""
    return ArrowWindow()


def create_polars_pipeline(
    lags: List[int],
    rolling_windows: Optional[List[int]] = None,
) -> PolarsWindow:
    """Create Polars-native window pipeline"""
    return PolarsWindow(lags, rolling_windows)