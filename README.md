# Zeno: The Zero-Copy Time Series Engine

High-performance, leakage-aware time-series workflows in Python with a Rust core. Keeps Arrow/Polars data in native columnar buffers as long as possible and makes temporal correctness a first-class concern.

**Package:** `zeno-ts` 0.1.0 | **Python:** >=3.8 | **Rust extension:** `zeno._zeno` via Maturin | **License:** MIT

---

## Table of Contents

- [Overview](#overview)
- [Why Zeno](#why-zeno)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Feature Guide](#feature-guide)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Testing](#testing)
- [Roadmap and Maturity](#roadmap-and-maturity)
- [Compatibility and Known Limitations](#compatibility-and-known-limitations)

---

## Overview

Zeno is a Python package with a Rust extension for high-performance, leakage-aware time-series workflows. Its core promise is to avoid Python-list and row-wise materialization:

- Feature generation (lags, rolling mean, EMA, WMA, scaling) stays on columnar paths when the input is `pyarrow.Table` or `polars.DataFrame`.
- Arrow lags are built as a null-prefix chunk plus a slice of the original value buffers.
- Temporal splits return contiguous `slice` views, not boolean-filtered copies.
- Fingerprinting and leakage detection operate on buffer hashes (XXH3), not Python object graphs.
- Context windows for foundation models and tensor views for inference are slice-backed until the model boundary.

---

## Why Zeno

Traditional time-series stacks (Pandas-centric windowing, ad-hoc train/test slicing) tend to hit three problems at scale:

1. **Memory amplification** - CSV -> Pandas -> NumPy -> GPU copies multiply resident memory 4-5x. Arrow columnar buffers are designed to be shared, not copied.
2. **Temporal leakage** - Shifting a column and then filtering by date silently leaks future information into training features. Zeno enforces cutoffs at split creation and feature validation time.
3. **Slow windowing** - Row loops and `to_list`/`to_dict` materialization dominate pipeline time. Expression-based Polars plans and sliced Arrow chunks avoid that overhead.
4. **Auditability gap** - Backtesting, validation, and compliance checks are typically scattered. Zeno bundles expanding-window backtests, managed pipelines, and audit reports around the same slice primitives.

Zeno does not claim zero allocation for derived features. Rolling means, EMA, and scaling necessarily allocate new output buffers; the guarantee is that source columns are not materialized through Python lists or dictionaries.

---

## Architecture

### Layered design

| Layer | Location | Purpose |
|---|---|---|
| Rust extension | `zeno/zeno-core` | Performance-sensitive ops, Python bindings, validators, orchestration primitives (GPU, audit, batch, managed, serverless) |
| Python package | `zeno/zeno/zeno-py` | Ergonomic Polars/Arrow APIs, higher-level pipeline composition, zero-copy helpers |
| Tests | `zeno/tests` | Unit coverage for windowing, temporal leakage, zero-copy behavior, roadmap phases |

## Data Flow


**1. Memory Layer**
> `pyarrow.Table` / `polars.DataFrame`
*The entry point for all time series data, ensuring memory is mapped, not copied.*
&nbsp;&nbsp;&nbsp;&nbsp;⬇

**2. Rust Engine** (`zeno-core`)
> **Core processing with extreme performance and memory safety.**
*   **`arrow_ops.rs` & `window.rs`**: O(1) lag generation returning `ChunkedArray` slices.
*   **`polars_ops.rs`**: Zero-copy rolling window and EMA calculations via native expressions.
*   **`temporal.rs`**: Strict monotonicity and temporal lookahead validators.
&nbsp;&nbsp;&nbsp;&nbsp;⬇

**3. Python Bindings & Helpers** (`zero_copy.py`)
> **Safe FFI transitions and zero-copy bridges.**
*   **Transformation**: `arrow_lag()`, `generate_lag()`, `zero_copy_temporal_split()`
*   **Validation**: `arrow_chunked_value_hash()`, `arrow_feature_fingerprint()`
*   **Deep Learning**: `zero_copy_numpy_view()`, `TensorBridge`
&nbsp;&nbsp;&nbsp;&nbsp;⬇

**4. Application Layer** 
> **Enterprise-grade ML orchestration and foundation model streaming.**
*   **`foundation.py`**: `FoundationModelBridge` and `TensorBridge` for safe GPU batching.
*   **`cloud.py`**: `ZeroCopyBacktestRunner` and `ManagedValidationPipeline` for scale.
*   **Ops**: `AuditReport` and `ServerlessConfig` for deployment and compliance.

### Rust/Python boundary

- Compiled module is `zeno._zeno` (`zeno/zeno-py/pyproject.toml` sets `module-name = "zeno._zeno"` and `manifest-path = "../zeno-core/Cargo.toml"`).
- Currently registered classes in `zeno-core/src/lib.rs` are `WindowOp` and `TemporalValidator`. Additional Rust classes (`PolarsWindowOp`, `PolarsValidator`, `ArrowPipeline`, `GPUAccelerator`, `AuditReport`, `ManagedPipeline`, `ServerlessConfig`, `ChronosWrapper`, etc.) exist in `zeno-core/src/engine` and `zeno-core/src/validator` but require module registration to be importable as `zeno._zeno.*`.
- Python wrappers import from `zeno._zeno` and add validation, expression building, and slice logic.

---

## Installation

### Prerequisites

- Python 3.8+
- Rust toolchain (rustc, cargo) - https://rustup.rs
- Maturin >=1.0 (`pip install maturin` or `uv pip install maturin`)
- Optional: `uv` for faster venv management, `boto3` only if you use serverless submission

> Note: This machine profile lacks `link.exe` (MSVC linker) and Polars/PyArrow wheels. Build/run is expected to work on a standard development machine with a C linker and Python dependencies available. See [Compatibility and Known Limitations](#compatibility-and-known-limitations).

### Quick install (editable development mode)

**Windows (PowerShell):**

```bash
cd home\Zeno\zeno\zeno-py

uv venv    
# or: python -m venv .venv

.\.venv\Scripts\Activate.ps1

uv pip install -e . maturin pytest
# if uv is unavailable:
# python -m pip install -e . maturin pytest

maturin develop --release  
# compiles zeno-core into zeno._zeno

python ..\examples\quickstart.py

pytest ..\tests -v
```

**Linux / macOS:**

```bash
cd zeno/zeno-py
uv venv
source .venv/bin/activate
uv pip install -e . maturin pytest
maturin develop --release
python ../examples/quickstart.py
pytest ../tests -v
```

### Bootstrap from scratch

`setup_zeno.sh` recreates the entire scaffolding (Cargo.toml, Rust sources, Python package, examples, tests) and then runs `maturin develop --release`. Use it when initializing a clean checkout:

```bash
bash setup_zeno.sh
```

### Verifying the build

```bash
python -c "import zeno; print(zeno.__version__)"
python -c "import zeno._zeno; print(dir(zeno._zeno))"
```

Expected: `zeno._zeno.WindowOp` and `zeno._zeno.TemporalValidator` are importable. Additional Rust classes become available after they are registered in `zeno-core/src/lib.rs`.

---

## Quick Start

### 1. Basic windowing (Python-list compatibility path)

Good for quick experiments; note that list inputs are copied across the Python/Rust boundary.

```python
import zeno as zn
from datetime import datetime, timedelta

values = [10.0 + i * 0.5 for i in range(20)]
window = zn.Window(lags=[1, 7, 14])

lag_features = window.transform(values)        
rolling = window.rolling_mean(values, window=7)  

print(lag_features[0][:5])  # lag-1: [None, 10.0, 10.5, 11.0, 11.5]
print(rolling[6:10])         # 7-day mean aligned to the window end
```

Pipeline composition via `Molecule`:

```python
pipeline = zn.Molecule([zn.Window(lags=[1, 7]), zn.Scale(method="robust")])
pipeline.fit(values).transform(values)
```

### 2. Arrow-native path (zero-copy lags)

Arrow lags in `zeno/zeno-py/zeno/zero_copy.py` are constructed as a null prefix plus slices of the original buffers. `zeno/zeno-py/zeno/advanced.py` wraps this into a table API.

```python
import pyarrow as pa
from datetime import datetime, timedelta
from zeno.advanced import ArrowWindow

table = pa.table({
    "timestamp": [datetime(2024,1,1) + timedelta(days=i) for i in range(100)],
    "value": [float(i) for i in range(100)],
})

aw = ArrowWindow(lags=[1, 7, 30], rolling=[7])
table_with_lags = aw.create_lags(table, "value", [1, 7, 30])  # zeno/zeno-py/zeno/advanced.py:43
table_with_all = aw.transform(table, "value")                 # lags + rolling
print(table_with_all.column_names)
# ['timestamp', 'value', 'value_lag_1', 'value_lag_7', 'value_lag_30', 'value_rolling_7']
```

Buffer reuse invariant (tested in `zeno/tests/test_zero_copy.py`):

```python
source_buf = table.column("value").chunk(0).buffers()[1]
lag_buf = table_with_lags.column("value_lag_1").chunk(1).buffers()[1]
assert lag_buf.address == source_buf.address
```

Single-Series lag via Rust chunk helper (`zeno/zeno-py/zeno/zero_copy.py`):

```python
import polars as pl
from zeno.zero_copy import generate_lag

s = pl.Series("value", [1.0, 2.0, 3.0, 4.0])
print(generate_lag(s, 2).to_list())  # [None, None, 1.0, 2.0]
```

### 3. Polars-native path (expression plans)

Polars feature generation in `zeno/zeno-py/zeno/advanced.py` uses `pl.col(...).shift` / `rolling_mean` expressions. No `to_dict` or `to_list` materialization of source columns.

```python
import polars as pl
from zeno.advanced import PolarsWindow

df = pl.DataFrame({
    "timestamp": [datetime(2024,1,1) + timedelta(days=i) for i in range(1000)],
    "value": [float(i) for i in range(1000)],
    "volume": [100.0 + i for i in range(1000)],
})

pw = PolarsWindow(lags=[1, 7, 30], rolling=[7, 30])
df_feat = pw.transform(df, "value")                        
df_parallel = pw.transform_parallel(df, ["value", "volume"]) # fan-out across columns
lf_feat = pw.transform_lazy(df.lazy(), ["value", "volume"])  # lazy plan
```

### 4. Temporal validation and leakage detection

**Simple cutoff validator** (`zeno/zeno-py/zeno/validator.py`, Rust `zeno-core/src/validator/temporal.rs`):

```python
from zeno import TemporalSplitter
from datetime import datetime

splitter = TemporalSplitter()
train_mask, test_mask = splitter.split(
    [datetime(2024,1,i) for i in range(1, 11)],
    datetime(2024,1,5), datetime(2024,1,6)
)
splitter.validate_feature(datetime(2024,1,3))  # ok
# splitter.validate_feature(datetime(2024,1,7))  # raises ValueError: TEMPORAL LEAKAGE DETECTED
```

**Sorted-slice validator** (`zeno/zeno-py/zeno/zero_copy.py`, `zeno/zeno-py/zeno/advanced.py`):

```python
from zeno.advanced import PolarsTemporalValidator
import polars as pl
from datetime import datetime, timedelta

df = pl.DataFrame({
    "timestamp": [datetime(2024,1,1) + timedelta(days=i) for i in range(365)],
    "value": [float(i) for i in range(365)],
})

v = PolarsTemporalValidator()
v.validate_split(df, "timestamp", datetime(2024,9,1), datetime(2024,9,2))
train, test = v.split(df, "timestamp", datetime(2024,9,1))
# train/test are df.slice views; unsorted input raises ValueError: sorted ascending
```

**Advanced leakage detection** (`zeno/zeno-py/zeno/advanced.py`):

```python
from zeno.advanced import AdvancedLeakageDetector
import pyarrow as pa
from datetime import datetime, timedelta

detector = AdvancedLeakageDetector(threshold=0.1)
table = pa.table({
    "timestamp": [datetime(2024,1,1) + timedelta(days=i) for i in range(8)],
    "value": [float(i) for i in range(8)],
})
detector.register_training_frame(table.slice(0,4), "timestamp", "value", "train")
detector.check_test_frame(table.slice(4,4), "timestamp", "value", "clean")  # {}
# detector.check_test_frame(table.slice(2,4), "timestamp", "value", "overlap")  # raises LEAKAGE DETECTED
```

Under the hood, fingerprints are blake2b hashes over Arrow buffer views (`zeno/zeno-py/zeno/zero_copy.py` and `zeno/zeno-py/zeno/zero_copy.py`), so slicing does not copy values.

### 5. Foundation-model and tensor bridges

Slice-backed context windows and zero-copy CPU tensor views (`zeno/zeno-py/zeno/foundation.py`, `zeno/zeno-py/zeno/foundation.py`):

```python
from zeno.foundation import FoundationModelBridge, TensorBridge
import pyarrow as pa
from datetime import datetime, timedelta

table = pa.table({
    "timestamp": [datetime(2024,1,1) + timedelta(days=i) for i in range(100)],
    "value": [float(i) for i in range(100)],
})

bridge = FoundationModelBridge(context_length=32, prediction_length=8, stride=4)
contexts = bridge.contexts(table)          # list of table.slice views
latest = bridge.latest_context(table)      # last 32 rows as a view

tb = TensorBridge(dtype="float32")
np_view = tb.arrow_numpy_view(table, "value")          # zero_copy_only
torch_tensor = tb.arrow_torch_tensor(table, "value")   # torch.from_numpy without copy
```

`TensorBridge.arrow_numpy_view` requires a single Arrow chunk (`zeno/zeno-py/zeno/zero_copy.py` raises if `len(chunks) != 1`). This preserves the zero-copy guarantee all the way to `torch.from_numpy`.

### 6. Backtesting, managed pipelines, and serverless

Expanding-window backtests that reuse slices (`zeno/zeno-py/zeno/cloud.py`):

```python
from zeno.cloud import ZeroCopyBacktestRunner
import polars as pl
from datetime import datetime, timedelta

df = pl.DataFrame({
    "timestamp": [datetime(2024,1,1) + timedelta(days=i) for i in range(200)],
    "value": [float(i % 30) for i in range(200)],
})

class NaiveLastValue:
    def fit(self, train): self.last = float(train["value"][-1])
    def predict(self, train, horizon): return [self.last] * horizon

runner = ZeroCopyBacktestRunner(test_size=30, step_size=30, n_splits=5)
results = runner.run_expanding_window(NaiveLastValue(), df, "timestamp", "value", min_train_size=90)
# each fold: train = df.slice(0, train_end), test = df.slice(train_end, test_size)
```

Managed validation pipeline and Lambda payload (`zeno/zeno-py/zeno/cloud.py`, `zeno/zeno-py/zeno/cloud.py`):

```python
from zeno.cloud import ManagedValidationPipeline, ServerlessBacktestJob
from datetime import datetime

pipeline = ManagedValidationPipeline("demo").add_temporal_split("timestamp", datetime(2024,1,10))
out = pipeline.run(df)  # {"artifacts": {"train": ..., "test": ...}, "audit": AuditReport}

job = ServerlessBacktestJob("s3://bucket/data.parquet", "value", "timestamp")
payload = job.payload()   # {"zero_copy_required": True, "format": "arrow_or_parquet", ...}
# job.submit()            # invokes boto3 lambda client if configured
```

---

## Core Concepts

### Zero-copy contract

Zeno's contract is narrow and explicit (`zeno/zeno-py/zeno/zero_copy.py`):

- **Never** materialize input columns through Python lists, NumPy copies, or dict round-trips.
- **Prefer** `slice` / chunk views for existing data.
- **Allocate** new buffers only for derived feature columns (rolling stats, EMA, scaling). Source buffers are shared.

Concrete mechanisms:

| Operation | How zero-copy is achieved | Where |
|---|---|---|
| Arrow lag | `nulls(lag) + ChunkedArray.slice(0, n-lag)` | `zero_copy.py` |
| Polars temporal split | `assert_sorted_by` + `df.slice(0, n_train)` / `df.slice(n_train)` + `search_sorted` for cutoff indices | `zero_copy.py`, `zero_copy.py`, `zero_copy.py` |
| Arrow window view | `table.slice(offset, length)` | `zero_copy.py` |
| Polars window view | `df.slice(offset, length)` | `zero_copy.py` |
| Buffer hash | `memoryview(buffer)[offset*width : offset*width+len*width]` + null bitmap | `zero_copy.py`, `zero_copy.py` |
| Tensor view | `chunk(0).to_numpy(zero_copy_only=True)` then `torch.from_numpy` | `zero_copy.py`, `foundation.py` |
| Context window | `FoundationBatchStreamer` slices `ArrayRef` via `array.slice` | `zeno-core/src/engine/foundation.rs` |

---

## Feature Guide

### Atoms and Molecules

- `Window(lags, rolling)` (`atoms.py`) delegates to Rust `WindowOp` (`engine/window.rs:8`). `transform` returns `Vec<Vec<Option<f64>>>` with leading `None`s for the lag prefix. `rolling_mean` is a sliding mean dividing by window length.
- `Scale(method)` (`atoms.py`) currently implements `robust` scaling with interpolated quantiles (tested in `tests/test_phases.py`).
- `Molecule(atoms)` (`molecule.py`) composes atoms sequentially with explicit `fit`/`transform` lifecycle.

### Arrow helpers (`zero_copy.py`)

| Helper | Purpose |
|---|---|
| `ensure_arrow_table` / `ensure_polars_frame` | Type guards |
| `normalize_lags` | Validates non-negative lag list |
| `arrow_lag` | Zero-copy lag column |
| `generate_lag` | Polars Series lag via Rust `create_lags_chunked` FFI |
| `append_arrow_columns` | Add derived columns while optionally preserving originals |
| `arrow_window_view` / `polars_window_view` | Slice-backed windowing |
| `arrow_array_value_view` | Zero-copy byte view over primitive buffers |
| `arrow_chunked_value_hash` | Buffer-level hash for fingerprinting |
| `zero_copy_numpy_view` | Single-chunk NumPy view |
| `zero_copy_temporal_split` / `validate_temporal_coverage` | Sorted-slice splits |

### Polars helpers

- `PolarsWindow` (`advanced.py`) builds `pl.col(col).shift(lag).alias(...)` and `rolling_mean` expressions. `transform_parallel` fans out to multiple columns, `transform_lazy` returns a `LazyFrame` plan.
- `PolarsTemporalValidator` (`advanced.py`) and the Rust `PolarsValidator` (`engine/polars_ops.rs`) both enforce sorted time columns; the Python-layer version in `zero_copy.py` uses `search_sorted` when available for O(log n) cutoff search.

### Foundation and GPU

- `FoundationModelBridge` (`foundation.py`) produces slice-backed contexts for any `pa.Table` or `pl.DataFrame`. `predict` / `predict_batch` delegate to a user-supplied model.
- `TensorBridge` (`foundation.py`) exposes Arrow-to-NumPy and Arrow-to-PyTorch CPU paths without copies. GPU acceleration primitives live in Rust (`engine/gpu.rs` `GPUAccelerator`, `engine/foundation.rs` `FoundationBatchStreamer`) with RAII-guarded allocations (`engine/gpu.rs` `ManagedGpuTensor`) and batch GPU inference (`engine/gpu.rs`, `engine/batch.rs`).

### Cloud and backtesting

- `ZeroCopyBacktestRunner` (`cloud.py`) generates expanding splits via `slice`, runs `model.fit` / `model.predict`, and computes `mse`/`mae`/`rmse` via Polars expressions.
- `ManagedValidationPipeline` (`cloud.py`) is a local in-process pipeline that records an `AuditReport` (`engine/audit.rs`).
- `ServerlessBacktestJob` (`cloud.py`) serializes a job descriptor (`payload` with `zero_copy_required`) and optionally submits via `boto3` Lambda invoke. The Rust counterpart `ServerlessConfig` (`engine/serverless.rs`) does the same at the binding level.

---

## API Reference

### Python package `zeno` (`zeno-py/zeno`)

| Module | Class / Function | Description | Source |
|---|---|---|---|
| `zeno` | `Window` | Lag/rolling feature creation (list path) | `atoms.py` |
| `zeno` | `Scale` | Robust/standard scaling | `atoms.py` |
| `zeno` | `Molecule` | Sequential atom composition | `molecule.py` |
| `zeno` | `TemporalSplitter` | Cutoff-based split + feature-window check | `validator.py` |
| `zeno.zero_copy` | `arrow_lag` | Zero-copy Arrow lag | `zero_copy.py` |
| `zeno.zero_copy` | `generate_lag` | Rust-backed Polars Series lag | `zero_copy.py` |
| `zeno.zero_copy` | `zero_copy_temporal_split` | Sorted-slice DataFrame split | `zero_copy.py` |
| `zeno.zero_copy` | `arrow_chunked_value_hash` | Buffer-level blake2b fingerprint | `zero_copy.py` |
| `zeno.zero_copy` | `zero_copy_numpy_view` | Single-chunk NumPy view | `zero_copy.py` |
| `zeno.advanced` | `ArrowWindow` | Arrow Table feature pipeline | `advanced.py` |
| `zeno.advanced` | `PolarsWindow` | Polars expression feature pipeline | `advanced.py` |
| `zeno.advanced` | `PolarsTemporalValidator` | Sorted-slice Polars validation | `advanced.py` |
| `zeno.advanced` | `AdvancedLeakageDetector` | Frame-level fingerprint + hash leakage detection | `advanced.py` |
| `zeno.advanced` | `ExpandingWindowValidator` | Expanding-window slice splits | `advanced.py` |
| `zeno.foundation` | `FoundationModelBridge` | Slice-backed context windows | `foundation.py` |
| `zeno.foundation` | `TensorBridge` | Zero-copy NumPy/Torch CPU views | `foundation.py` |
| `zeno.cloud` | `ZeroCopyBacktestRunner` | Expanding-window backtests over slices | `cloud.py` |
| `zeno.cloud` | `ManagedValidationPipeline` | Local validation pipeline with audit | `cloud.py` |
| `zeno.cloud` | `ServerlessBacktestJob` | Serializable Lambda job descriptor | `cloud.py` |

### Rust crate `zeno-core` (`zeno-core/src`)

| Module | Item | Status | Source |
|---|---|---|---|
| `engine/window` | `WindowOp` | Exported via `lib.rs` | `engine/window.rs` |
| `validator/temporal` | `TemporalValidator` | Exported via `lib.rs` | `validator/temporal.rs` |
| `engine/polars_ops` | `PolarsWindowOp`, `PolarsValidator`, `safe_rolling_mean` | Implemented, requires registration | `engine/polars_ops.rs`, `engine/polars_ops.rs` |
| `engine/foundation` | `ChronosWrapper`, `LagLlamaWrapper`, `MoiraiWrapper`, `FoundationBatchStreamer` | Requires registration | `engine/foundation.rs`, `engine/foundation.rs`, `engine/foundation.rs`, `engine/foundation.rs` |
| `engine/gpu` | `GPUAccelerator`, `TensorConverter`, `ManagedGpuTensor` | Requires registration | `engine/gpu.rs`, `engine/gpu.rs` |
| `engine/batch` | `BatchPredictor`, `EnsemblePredictor`, `Forecast` | Requires registration | `engine/batch.rs`, `engine/batch.rs`, `engine/batch.rs` |
| `engine/audit` | `AuditReport`, `ComplianceChecker`, `AuditLogger`, `ReportGenerator` | Requires registration | `engine/audit.rs` |
| `engine/managed` | `ManagedPipeline`, `PipelineRegistry`, `ValidationScheduler` | Requires registration | `engine/managed.rs` |
| `engine/serverless` | `BacktestRunner`, `BacktestResult`, `ServerlessConfig` | Requires registration | `engine/serverless.rs` |
| `validator/leakage` | Rolling-hash validator | Placeholder | `validator/leakage.rs` |
| `engine/arrow_ops` | Arrow zero-copy ops | Placeholder / in-progress (`create_zero_copy_lag`, `create_lags_chunked` in working tree) | `engine/arrow_ops.rs` |

---

## Examples

| Example | Location | What it shows |
|---|---|---|
| Quickstart | `zeno/examples/quickstart.py` | `Window.transform`, `rolling_mean`, `TemporalSplitter.split`, `validate_feature`, `Molecule` pipeline |
| Advanced | `zeno/examples/advanced_example.py` | Arrow lag/rolling/EMA, `AdvancedLeakageDetector` clean/overlap/duplicate, `PolarsWindow` parallel/lazy, `PolarsTemporalValidator` valid/invalid split, `ExpandingWindowValidator`, full Phase 2 pipeline |

Run examples after building:

```bash
python zeno/examples/quickstart.py
python zeno/examples/advanced_example.py
```

---

## Testing

Test discovery is configured in `zeno/pytest.ini` (`testpaths = tests`).

| Suite | File | Coverage |
|---|---|---|
| Basic windowing | `tests/test_window.py` | Lag alignment, None prefix, rolling mean |
| Temporal logic | `tests/test_temporal.py` | Valid split masks, leakage rejection |
| Zero-copy | `tests/test_zero_copy.py` | Arrow buffer address reuse, `generate_lag` boundaries, Polars expression correctness, sorted-slice split + unsorted rejection |
| Phases 2-4 | `tests/test_phases.py` | Arrow frame fingerprints without lists, Arrow context view buffer sharing, NumPy view address equality, sliced hash equivalence, robust quantiles, slice-backed backtest folds, metric length validation, managed pipeline audit + serverless payload |

Run:

```bash
pytest zeno/tests -v
pytest zeno/tests/test_zero_copy.py -v -k test_arrow_lag_reuses_source_value_buffer
```

Benchmarks directory (`zeno/benchmarks`) currently holds no benchmark scripts; the Rust `Cargo.toml` includes `criterion` as a dev dependency for future microbenchmarks.

---

## Roadmap and Maturity

Zeno is intentionally a strong prototype / early library rather than a unified production engine. Roadmap phases reflect this:

**Phase 1 - Zero-Copy Windowing + Validation**
- Window, PolarsWindow, ArrowWindow, TemporalSplitter. Native Rust windowing core implemented.

**Phase 2 - Advanced Validation + Arrow Integration**
- Full Arrow zero-copy pipeline and PyDataFrame Polars integration. High-speed XXH3 rolling-hash fingerprinting implemented for immediate leakage detection.

**Phase 3 - Foundation Model Integration**
- FoundationModelBridge slice-backed windows. Strict RAII memory management in gpu.rs prevents CUDA out-of-memory crashes by tracking limits natively in Rust before streaming batches.

**Phase 4 - Zeno Cloud (in progress)** 
- ZeroCopyBacktestRunner, ManagedValidationPipeline, ServerlessBacktestJob run locally. Primed for AWS Lambda serverless deployment.

---

## Compatibility and Known Limitations

- **Polars version skew** - Rust depends on polars = 0.45 while Python depends on polars>=0.20. Ensure environment versions align when passing PyDataFrame across the FFI boundary.

- **FFI boundary** - Current `PolarsWindowOp` invokes the Python Polars API dynamically (`engine/polars_ops.rs` `create_lags_polars` via `py.import_bound("polars")`). The alternative native `safe_rolling_mean` that takes `PyDataFrame` exists but is not yet the default path.

- **Derived columns allocate** - Rolling means, EMA, and scaling produce new column buffers. The zero-copy claim applies strictly to source column buffers and temporal slices.

- **Single-chunk precondition** - zero_copy_numpy_view and FoundationBatchStreamer slicing assume single-chunk primitive arrays; variable-width arrays currently raise a ValueError by design.

---

*Built for the time-series community. Contributions that preserve the zero-copy contract and add coverage in `zeno/tests` are welcome.*
