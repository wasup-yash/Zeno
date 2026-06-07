"""Foundation-model bridges that preserve Arrow/Polars views until inference."""
from __future__ import annotations
from typing import List, Optional
import polars as pl
import pyarrow as pa
import torch

from .zero_copy import arrow_window_view, polars_window_view, zero_copy_numpy_view


class FoundationModelBridge:
    """
    Prepare context windows for foundation models without list materialization.

    The returned contexts are Arrow tables or Polars DataFrames produced with
    `slice`, so existing source buffers stay shared until the model boundary.
    """

    def __init__(self, context_length: int = 512, prediction_length: int = 64, stride: int = 1):
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if prediction_length <= 0:
            raise ValueError("prediction_length must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride

    def contexts(self, data) -> List[object]:
        n_rows = len(data)
        if n_rows < self.context_length:
            raise ValueError(
                f"Need at least {self.context_length} rows, got {n_rows}"
            )

        windows: List[object] = []
        for start in range(0, n_rows - self.context_length + 1, self.stride):
            windows.append(self.context_view(data, start))
        return windows

    def context_view(self, data, start: int = 0):
        if isinstance(data, pa.Table):
            return arrow_window_view(data, start, self.context_length)
        if isinstance(data, pl.DataFrame):
            return polars_window_view(data, start, self.context_length)
        raise TypeError("Expected a pyarrow.Table or polars.DataFrame")

    def latest_context(self, data):
        start = len(data) - self.context_length
        if start < 0:
            raise ValueError(
                f"Need at least {self.context_length} rows, got {len(data)}"
            )
        return self.context_view(data, start)

    def predict(self, model, data, *, horizon: Optional[int] = None):
        context = self.latest_context(data)
        horizon = horizon or self.prediction_length
        return model.predict(context, horizon)

    def predict_batch(self, model, data, *, horizon: Optional[int] = None):
        horizon = horizon or self.prediction_length
        return [model.predict(context, horizon) for context in self.contexts(data)]


class TensorBridge:
    """Zero-copy CPU tensor helpers for Arrow-backed model inputs."""

    def __init__(self, dtype: Optional[str] = None):
        self.dtype = dtype

    def arrow_numpy_view(self, table: pa.Table, column: str):
        return zero_copy_numpy_view(table, column)

    def arrow_torch_tensor(self, table: pa.Table, column: str, *, device: str = "cpu"):
        

        array = self.arrow_numpy_view(table, column)
        tensor = torch.from_numpy(array)
        if self.dtype is not None:
            tensor = tensor.to(getattr(torch, self.dtype))
        if device != "cpu":
            tensor = tensor.to(device)
        return tensor

    def polars_torch_tensor(self, df: pl.DataFrame, column: str, *, device: str = "cpu"):
        table = df.select(column).to_arrow()
        return self.arrow_torch_tensor(table, column, device=device)
