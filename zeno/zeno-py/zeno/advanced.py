from typing import List, Optional, Dict, Any
from datetime import datetime
import polars as pl
import pyarrow as pa

# Import raw classes from the Rust binary extension
from ._zeno import (
    ArrowPipeline,
    LeakageDetector,
    RollingHashValidator,
    PolarsWindowOp,
    PolarsValidator,
    GPUAccelerator,
    ManagedPipeline,
    AuditReport,
)

class ArrowWindow:
    """Zero-copy windowing operations using Apache Arrow"""
    
    def __init__(self):
        self._pipeline = ArrowPipeline()
    
    def create_lags(self, table: pa.Table, column: str, lags: List[int]) -> pa.Table:
        values = table.column(column).to_pylist()
        result_data = self._pipeline.create_lags_simple(values, lags)
        
        arrays = [pa.array(col, type=pa.float64()) for col in result_data]
        names = [f"{column}_lag_{l}" for l in lags]
        return pa.Table.from_arrays(arrays, names=names)

    def rolling_mean(self, table: pa.Table, column: str, window: int) -> pa.Table:
        values = table.column(column).to_pylist()
        result_data = self._pipeline.rolling_mean_simple(values, window)
        return pa.Table.from_arrays([pa.array(result_data)], names=[f"{column}_rolling_{window}"])
    
    def ema(self, table: pa.Table, column: str, alpha: float = 0.3) -> pa.Table:
        """Create exponential moving average feature"""
        values = table.column(column).to_pylist()
        # Updated to use ema_simple to match the Vec<f64> pattern
        result_data = self._pipeline.ema_simple(values, alpha)
        return pa.Table.from_arrays([pa.array(result_data)], names=[f"{column}_ema_{alpha}"])

class AdvancedLeakageDetector:
    def __init__(self, threshold: float = 0.1):
        self._detector = LeakageDetector(threshold)
        self._hash_validator = RollingHashValidator(hash_size=100)
    
    def register_training_feature(self, timestamps: List[int], values: List[float], feature_name: str):
        self._detector.register_train_window(timestamps, values, feature_name)
        self._hash_validator.add_window(values)
    
    def check_test_feature(self, timestamps: List[int], values: List[float], feature_name: str) -> Dict[str, float]:
        if self._hash_validator.check_window(values):
            raise ValueError(f"EXACT MATCH: Feature '{feature_name}' appears in training data.")
        return self._detector.check_test_window(timestamps, values, feature_name)

class PolarsWindow:
    def __init__(self, lags: List[int], rolling: Optional[List[int]] = None):
        self._op = PolarsWindowOp(lags, rolling or [])
        self.lags = lags
        self.rolling = rolling or []
    
    def transform(self, df: pl.DataFrame, column: str) -> pl.DataFrame:
        result = df
        for lag in self.lags:
            result = result.with_columns(pl.col(column).shift(lag).alias(f"{column}_lag_{lag}"))
        for window in self.rolling:
            result = result.with_columns(pl.col(column).rolling_mean(window).alias(f"{column}_rolling_{window}"))
        return result

class PolarsTemporalValidator:
    def __init__(self):
        self._validator = PolarsValidator()
    
    def validate_split(self, df: pl.DataFrame, time_col: str, train_end: datetime, test_start: datetime) -> bool:
        return self._validator.validate_temporal_split(df, time_col, int(train_end.timestamp()), int(test_start.timestamp()))

class ExpandingWindowValidator:
    def __init__(self, min_train_size: int, test_size: int, step_size: int = 1):
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.step_size = step_size
    
    def split(self, df: pl.DataFrame, time_col: str):
        df_sorted = df.sort(time_col)
        n = len(df_sorted)
        splits = []
        for i in range(self.min_train_size, n - self.test_size + 1, self.step_size):
            splits.append((df_sorted[:i], df_sorted[i:i + self.test_size]))
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

class GPUManager:
    """Managed GPU Resources"""
    def __init__(self, device: str = "cuda:0"):
        self._accel = GPUAccelerator(device=device)
    
    def stats(self):
        if self._accel.is_available():
            alloc, res = self._accel.get_memory_info()
            return {"allocated_mb": alloc / 1024**2, "reserved_mb": res / 1024**2}
        return {"status": "GPU not available"}

class AuditManager:
    """Generate Compliance Reports"""
    def __init__(self, report_id: str):
        self.report = AuditReport(report_id)
    
    def log_metric(self, name: str, value: float):
        self.report.add_metric(name, value)
    
    def finalize(self) -> str:
        return self.report.summary

class ManagedExecutor:
    """Execute validation pipelines with automatic scheduling"""
    def __init__(self, pipeline_id: str):
        self._pipe = ManagedPipeline(pipeline_id)
    
    def add_validation_step(self, name: str, step_type: str = "temporal"):
        self._pipe.add_step(name, step_type)
    
    def run(self, data: Any):
        return self._pipe.execute(data)