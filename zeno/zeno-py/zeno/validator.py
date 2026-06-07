"""Temporal validation and leakage detection."""
from typing import List, Tuple
from datetime import datetime
import polars as pl
from ._zeno import PolarsValidator as _PolarsValidator
from .zero_copy import validate_temporal_coverage, zero_copy_temporal_split


class TemporalSplitter:
    """
    Enforce temporal ordering in train/test splits.
    Supports both mask‑based (legacy) and DataFrame‑based (zero‑copy) splitting.
    """

    def __init__(self):
        self._core = _PolarsValidator()
        self._train_end_ts = None
        self._test_start_ts = None

    # ---------- Mask‑based API (compatible with quickstart.py) ----------
    def split(
        self,
        timestamps: List[datetime],
        train_end_date: datetime,
        test_start_date: datetime,
    ) -> Tuple[List[bool], List[bool]]:
        """
        Create train/test masks from a list of datetime objects.
        Raises ValueError if test_start <= train_end (temporal leakage).
        """
        train_ts = int(train_end_date.timestamp())
        test_ts = int(test_start_date.timestamp())

        # Validate (this will raise if leakage)
        self._core.set_split(train_ts, test_ts)

        self._train_end_ts = train_ts
        self._test_start_ts = test_ts

        train_mask = [ts <= train_end_date for ts in timestamps]
        test_mask = [ts >= test_start_date for ts in timestamps]
        return train_mask, test_mask

    def validate_feature(self, feature_date: datetime) -> bool:
        """Check that a feature does not use data after train cutoff."""
        if self._train_end_ts is None:
            raise ValueError("Call split() before validate_feature()")
        return self._core.check_feature_window(int(feature_date.timestamp()))

    # ---------- Zero‑copy DataFrame API (recommended) ----------
    def split_dataframe(
        self,
        df: pl.DataFrame,
        time_col: str,
        train_end: datetime,
        test_start: datetime,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Zero‑copy split of a Polars DataFrame.
        Returns (train_df, test_df) without copying data.
        """
        validate_temporal_coverage(df, time_col, train_end, test_start)
        return zero_copy_temporal_split(df, time_col, train_end, test_start)
