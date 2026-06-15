"""Cloud-ready orchestration primitives built around zero-copy slices."""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import polars as pl

from ._zeno import AuditReport, ServerlessConfig
from .zero_copy import assert_sorted_by, zero_copy_temporal_split


class ZeroCopyBacktestRunner:
    """Expanding-window backtests using Polars slice views."""

    def __init__(self, test_size: int = 30, step_size: int = 1, n_splits: Optional[int] = None):
        if test_size <= 0:
            raise ValueError("test_size must be positive")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        self.test_size = test_size
        self.step_size = step_size
        self.n_splits = n_splits

    def expanding_splits(self, df: pl.DataFrame, time_col: str, min_train_size: int):
        assert_sorted_by(df, time_col)
        splits = []
        split_count = 0

        for train_end in range(min_train_size, len(df) - self.test_size + 1, self.step_size):
            if self.n_splits is not None and split_count >= self.n_splits:
                break
            train = df.slice(0, train_end)
            test = df.slice(train_end, self.test_size)
            splits.append((train, test))
            split_count += 1

        return splits

    def run_expanding_window(
        self,
        model,
        df: pl.DataFrame,
        time_col: str,
        target_col: str,
        min_train_size: int,
    ) -> List[Dict[str, float]]:
        results: List[Dict[str, float]] = []
        for fold, (train, test) in enumerate(
            self.expanding_splits(df, time_col, min_train_size)
        ):
            if hasattr(model, "fit"):
                model.fit(train)
            predictions = model.predict(train, len(test))
            metrics = self.compute_metrics(predictions, test.get_column(target_col))
            metrics["fold"] = float(fold)
            metrics["train_rows"] = float(len(train))
            metrics["test_rows"] = float(len(test))
            results.append(metrics)
        return results

    @staticmethod
    def compute_metrics(predictions, actuals: pl.Series) -> Dict[str, float]:
        pred = pl.Series("__pred", predictions)
        if len(pred) != len(actuals):
            raise ValueError(
                f"Prediction length {len(pred)} does not match actual length {len(actuals)}"
            )
        if len(pred) == 0:
            raise ValueError("Cannot compute metrics for an empty prediction window")

        actual = actuals.alias("__actual")
        frame = pl.DataFrame([pred, actual])
        metrics = frame.select(
            ((pl.col("__actual") - pl.col("__pred")) ** 2).mean().alias("mse"),
            (pl.col("__actual") - pl.col("__pred")).abs().mean().alias("mae"),
        )
        mse = float(metrics["mse"][0])
        mae = float(metrics["mae"][0])
        return {"mse": mse, "mae": mae, "rmse": mse**0.5}


class ManagedValidationPipeline:
    """Small in-process validation pipeline for sorted Polars datasets."""

    def __init__(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        self.steps: List[Dict[str, object]] = []

    def add_temporal_split(self, time_col: str, train_end, test_start=None):
        self.steps.append(
            {
                "type": "temporal_split",
                "time_col": time_col,
                "train_end": train_end,
                "test_start": test_start,
            }
        )
        return self

    def run(self, df: pl.DataFrame) -> Dict[str, object]:
        artifacts: Dict[str, object] = {"input": df}
        report = AuditReport(self.pipeline_id)

        for step in self.steps:
            if step["type"] == "temporal_split":
                train, test = zero_copy_temporal_split(
                    df,
                    str(step["time_col"]),
                    step["train_end"],
                    step["test_start"],
                )
                artifacts["train"] = train
                artifacts["test"] = test
                report.add_validation("temporal_split", True)

        return {
            "pipeline_id": self.pipeline_id,
            "artifacts": artifacts,
            "audit": report,
        }


class ServerlessBacktestJob:
    """Serializable serverless job descriptor plus optional AWS submission."""

    def __init__(
        self,
        dataset_uri: str,
        target_col: str,
        time_col: str,
        config: Optional[ServerlessConfig] = None,
        lambda_function: str = "zeno-backtest",
    ):
        self.dataset_uri = dataset_uri
        self.target_col = target_col
        self.time_col = time_col
        self.config = config or ServerlessConfig()
        self.lambda_function = lambda_function

    def payload(self) -> Dict[str, object]:
        return {
            "dataset_uri": self.dataset_uri,
            "target_col": self.target_col,
            "time_col": self.time_col,
            "zero_copy_required": True,
            "format": "arrow_or_parquet",
        }

    def submit(self):
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("Install the cloud extra to submit serverless jobs") from exc

        client = boto3.client("lambda")
        response = client.invoke(
            FunctionName=self.lambda_function,
            InvocationType="Event",
            Payload=json.dumps(self.payload()).encode("utf-8"),
        )
        return response["ResponseMetadata"]["RequestId"]
