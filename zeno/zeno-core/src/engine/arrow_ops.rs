use arrow::array::{Array, Float64Array, ArrayRef};
use arrow::datatypes::{Schema, Field, DataType};
use arrow::record_batch::RecordBatch;
use pyo3::prelude::*;
use std::sync::Arc;
use rayon::prelude::*;

#[pyclass]
pub struct ArrowPipeline {
    schema: Arc<Schema>,
}

#[pymethods]
impl ArrowPipeline {
    #[new]
    pub fn new() -> Self {
        Self {
            schema: Arc::new(Schema::empty()),
        }
    }

    // ── Phase 2: Python-facing API (Vec-based for PyO3 compatibility) ──────

    pub fn create_lags_simple(
        &self,
        values: Vec<f64>,
        lags: Vec<usize>,
    ) -> PyResult<Vec<Vec<Option<f64>>>> {
        let n = values.len();
        let mut result: Vec<Vec<Option<f64>>> = Vec::with_capacity(lags.len());
        for &lag in &lags {
            let mut lagged: Vec<Option<f64>> = vec![None; n];
            for i in lag..n {
                lagged[i] = Some(values[i - lag]);
            }
            result.push(lagged);
        }
        Ok(result)
    }

    pub fn rolling_mean_simple(
        &self,
        values: Vec<f64>,
        window: usize,
    ) -> PyResult<Vec<Option<f64>>> {
        let n = values.len();
        let mut result: Vec<Option<f64>> = vec![None; n];
        for i in window..=n {
            let sum: f64 = values[i - window..i].iter().sum();
            result[i - 1] = Some(sum / window as f64);
        }
        Ok(result)
    }

    pub fn ema_simple(
        &self,
        values: Vec<f64>,
        alpha: f64,
    ) -> PyResult<Vec<Option<f64>>> {
        let mut result: Vec<Option<f64>> = Vec::with_capacity(values.len());
        let mut ema: Option<f64> = None;
        for &value in &values {
            ema = Some(match ema {
                None       => value,
                Some(prev) => alpha * value + (1.0 - alpha) * prev,
            });
            result.push(ema);
        }
        Ok(result)
    }

    // ── Phase 2: Advanced features ──────────────────────────────────────────

    /// Double exponential smoothing (Holt's method)
    pub fn double_exponential_smoothing(
        &self,
        values: Vec<f64>,
        alpha: f64,
        beta: f64,
    ) -> PyResult<Vec<Option<f64>>> {
        if values.is_empty() {
            return Ok(Vec::new());
        }

        let mut result: Vec<Option<f64>> = Vec::with_capacity(values.len());
        let mut level = values[0];
        let mut trend = if values.len() > 1 { values[1] - values[0] } else { 0.0 };

        result.push(Some(level));

        for &value in &values[1..] {
            let last_level = level;
            level = alpha * value + (1.0 - alpha) * (level + trend);
            trend = beta * (level - last_level) + (1.0 - beta) * trend;
            result.push(Some(level + trend));
        }

        Ok(result)
    }

    /// Parallel lag creation with Rayon
    pub fn create_lags_parallel(
        &self,
        values: Vec<f64>,
        lags: Vec<usize>,
    ) -> PyResult<Vec<Vec<Option<f64>>>> {
        let n = values.len();
        
        let result: Vec<Vec<Option<f64>>> = lags
            .par_iter()
            .map(|&lag| {
                let mut lagged = vec![None; n];
                for i in lag..n {
                    lagged[i] = Some(values[i - lag]);
                }
                lagged
            })
            .collect();

        Ok(result)
    }

    fn __repr__(&self) -> String {
        "ArrowPipeline()".to_string()
    }
}

// ── Phase 2: Internal Arrow RecordBatch operations ──────────────────────────
// These are used internally for Rust-to-Rust pipelines
// Not exposed to Python due to RecordBatch FFI limitations

/// Create lag features on RecordBatch with parallel execution
pub fn create_lags_record_batch(
    batch: &RecordBatch,
    column_name: &str,
    lags: Vec<usize>,
) -> Result<RecordBatch, arrow::error::ArrowError> {
    let source_array = batch
        .column_by_name(column_name)
        .ok_or_else(|| arrow::error::ArrowError::InvalidArgumentError(
            format!("Column '{}' not found", column_name)
        ))?;

    let float_array = source_array
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| arrow::error::ArrowError::InvalidArgumentError(
            "Column must be Float64".to_string()
        ))?;

    let lag_arrays: Vec<ArrayRef> = lags
        .par_iter()
        .map(|&lag| {
            let mut builder = arrow::array::Float64Builder::new();
            for _ in 0..lag {
                builder.append_null();
            }
            for i in 0..float_array.len().saturating_sub(lag) {
                if float_array.is_null(i) {
                    builder.append_null();
                } else {
                    builder.append_value(float_array.value(i));
                }
            }
            Arc::new(builder.finish()) as ArrayRef
        })
        .collect();

    let mut fields: Vec<Arc<Field>> = batch.schema().fields().to_vec();
    for &lag in lags.iter() {
        fields.push(Arc::new(Field::new(
            format!("{}_lag_{}", column_name, lag),
            DataType::Float64,
            true,
        )));
    }

    let mut all_columns: Vec<ArrayRef> = batch.columns().to_vec();
    all_columns.extend(lag_arrays);

    RecordBatch::try_new(Arc::new(Schema::new(fields)), all_columns)
}

/// Rolling mean on RecordBatch
pub fn rolling_mean_record_batch(
    batch: &RecordBatch,
    column_name: &str,
    window: usize,
) -> Result<RecordBatch, arrow::error::ArrowError> {
    let source_array = batch
        .column_by_name(column_name)
        .ok_or_else(|| arrow::error::ArrowError::InvalidArgumentError(
            format!("Column '{}' not found", column_name)
        ))?;

    let float_array = source_array
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| arrow::error::ArrowError::InvalidArgumentError(
            "Column must be Float64".to_string()
        ))?;

    let mut builder = arrow::array::Float64Builder::new();

    for i in 0..float_array.len() {
        if i < window - 1 {
            builder.append_null();
        } else {
            let start = i - window + 1;
            let mut sum = 0.0_f64;
            let mut count = 0_usize;
            for j in start..=i {
                if !float_array.is_null(j) {
                    sum += float_array.value(j);
                    count += 1;
                }
            }
            if count > 0 {
                builder.append_value(sum / count as f64);
            } else {
                builder.append_null();
            }
        }
    }

    let rolling_array = Arc::new(builder.finish()) as ArrayRef;

    let mut fields: Vec<Arc<Field>> = batch.schema().fields().to_vec();
    fields.push(Arc::new(Field::new(
        format!("{}_rolling_mean_{}", column_name, window),
        DataType::Float64,
        true,
    )));

    let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
    columns.push(rolling_array);

    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
}

/// EMA on RecordBatch
pub fn ema_record_batch(
    batch: &RecordBatch,
    column_name: &str,
    alpha: f64,
) -> Result<RecordBatch, arrow::error::ArrowError> {
    let source_array = batch
        .column_by_name(column_name)
        .ok_or_else(|| arrow::error::ArrowError::InvalidArgumentError(
            format!("Column '{}' not found", column_name)
        ))?;

    let float_array = source_array
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| arrow::error::ArrowError::InvalidArgumentError(
            "Column must be Float64".to_string()
        ))?;

    let mut builder = arrow::array::Float64Builder::new();
    let mut ema: Option<f64> = None;

    for i in 0..float_array.len() {
        if float_array.is_null(i) {
            builder.append_null();
        } else {
            let value = float_array.value(i);
            ema = Some(match ema {
                None       => value,
                Some(prev) => alpha * value + (1.0 - alpha) * prev,
            });
            builder.append_value(ema.unwrap());
        }
    }

    let ema_array = Arc::new(builder.finish()) as ArrayRef;
    let mut fields: Vec<Arc<Field>> = batch.schema().fields().to_vec();
    fields.push(Arc::new(Field::new(
        format!("{}_ema_{:.2}", column_name, alpha),
        DataType::Float64,
        true,
    )));

    let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
    columns.push(ema_array);

    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
}

// ── Phase 3: Foundation Model Integration (Placeholder) ─────────────────────

/// Prepare batch for Foundation Model input
/// Will be implemented in Phase 3 with GPU tensors
pub fn prepare_for_model_inference(
    _batch: &RecordBatch,
    _lookback_window: usize,
) -> Result<Vec<Vec<f64>>, arrow::error::ArrowError> {
    // TODO Phase 3: Convert to tensor format for Chronos/Lag-Llama
    Err(arrow::error::ArrowError::NotYetImplemented(
        "Foundation model integration planned for Phase 3".to_string()
    ))
}