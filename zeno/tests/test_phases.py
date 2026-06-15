from datetime import datetime, timedelta

import polars as pl
import pyarrow as pa
import pytest

from zeno.advanced import AdvancedLeakageDetector
from zeno.atoms import Scale
from zeno.cloud import ManagedValidationPipeline, ServerlessBacktestJob, ZeroCopyBacktestRunner
from zeno.foundation import FoundationModelBridge
from zeno.zero_copy import arrow_chunked_value_hash, zero_copy_numpy_view


def test_phase2_frame_fingerprints_detect_leakage_without_lists():
    table = pa.table(
        {
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(8)],
            "value": [float(i) for i in range(8)],
        }
    )
    detector = AdvancedLeakageDetector(threshold=0.1)
    detector.register_training_frame(table.slice(0, 4), "timestamp", "value", "train")

    assert detector.check_test_frame(table.slice(4, 4), "timestamp", "value", "clean") == {}

    with pytest.raises(ValueError, match="LEAKAGE DETECTED"):
        detector.check_test_frame(table.slice(2, 4), "timestamp", "value", "overlap")


def test_phase3_context_views_reuse_arrow_buffers():
    table = pa.table(
        {
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(6)],
            "value": [float(i) for i in range(6)],
        }
    )
    bridge = FoundationModelBridge(context_length=3, prediction_length=2)

    context = bridge.latest_context(table)

    assert context.num_rows == 3
    assert (
        context.column("value").chunk(0).buffers()[1].address
        == table.column("value").chunk(0).buffers()[1].address
    )


def test_phase3_numpy_view_points_at_arrow_buffer():
    table = pa.table({"value": [1.0, 2.0, 3.0]})
    view = zero_copy_numpy_view(table, "value")

    assert view.__array_interface__["data"][0] == table.column("value").chunk(0).buffers()[1].address


def test_arrow_hash_uses_visible_values_not_slice_offset():
    base = pa.chunked_array([pa.array([10.0, 20.0, 30.0, 40.0])])
    sliced = base.slice(1, 2)
    equivalent = pa.chunked_array([pa.array([20.0, 30.0])])

    assert arrow_chunked_value_hash(sliced) == arrow_chunked_value_hash(equivalent)


def test_robust_scale_uses_interpolated_quantiles_and_rejects_empty():
    scaler = Scale(method="robust").fit([1.0, 2.0, 3.0, 4.0])

    assert scaler.center_ == 2.5
    assert scaler.scale_ == 1.5
    with pytest.raises(ValueError, match="empty"):
        Scale().fit([])


def test_phase4_backtest_uses_slice_based_folds():
    class LastValueModel:
        def fit(self, train):
            self.last = float(train["value"][-1])

        def predict(self, train, horizon):
            return [self.last] * horizon

    df = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(8)],
            "value": [float(i) for i in range(8)],
        }
    )
    runner = ZeroCopyBacktestRunner(test_size=2, step_size=2, n_splits=2)

    results = runner.run_expanding_window(
        LastValueModel(),
        df,
        "timestamp",
        "value",
        min_train_size=4,
    )

    assert len(results) == 2
    assert results[0]["train_rows"] == 4.0
    assert "mse" in results[0]


def test_phase4_metrics_reject_prediction_length_mismatch():
    with pytest.raises(ValueError, match="Prediction length"):
        ZeroCopyBacktestRunner.compute_metrics([1.0], pl.Series("actual", [1.0, 2.0]))


def test_phase4_managed_pipeline_and_serverless_payload():
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)],
            "value": [float(i) for i in range(5)],
        }
    )
    pipeline = ManagedValidationPipeline("phase4").add_temporal_split(
        "timestamp",
        datetime(2024, 1, 3),
        datetime(2024, 1, 4),
    )

    result = pipeline.run(df)
    payload = ServerlessBacktestJob("s3://bucket/data.parquet", "value", "timestamp").payload()

    assert len(result["artifacts"]["train"]) == 3
    assert result["audit"].summary["passed"] == "true"
    assert payload["zero_copy_required"] is True
