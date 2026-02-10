// // Phase 2: Advanced Arrow Operations
// // src/engine/arrow_ops.rs

// use arrow::array::{Array, Float64Array, ArrayRef, PrimitiveArray};
// use arrow::datatypes::{Float64Type, Schema, Field, DataType};
// use arrow::record_batch::RecordBatch;
// use pyo3::prelude::*;
// use pyo3::types::PyDict;
// use std::sync::Arc;
// use rayon::prelude::*;

// #[pyclass]
// pub struct ArrowPipeline {
//     schema: Arc<Schema>,
//     batches: Vec<RecordBatch>,
// }

// #[pymethods]
// impl ArrowPipeline {
//     #[new]
//     pub fn new() -> Self {
//         Self {
//             schema: Arc::new(Schema::empty()),
//             batches: Vec::new(),
//         }
//     }

//     /// Create lag features using zero-copy Arrow operations
//     pub fn create_lags_arrow(
//         &self,
//         column_name: &str,
//         lags: Vec<usize>,
//         batch: &RecordBatch,
//     ) -> PyResult<RecordBatch> {
//         // Get the source column
//         let source_array = batch
//             .column_by_name(column_name)
//             .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
//                 format!("Column '{}' not found", column_name)
//             ))?;
        
//         let float_array = source_array
//             .as_any()
//             .downcast_ref::<Float64Array>()
//             .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyTypeError, _>(
//                 "Column must be Float64"
//             ))?;

//         // Create lag columns in parallel
//         let lag_arrays: Vec<ArrayRef> = lags
//             .par_iter()
//             .map(|&lag| {
//                 let mut builder = arrow::array::Float64Builder::new();
                
//                 // Add None for first 'lag' values
//                 for _ in 0..lag {
//                     builder.append_null();
//                 }
                
//                 // Add lagged values (zero-copy via slicing)
//                 for i in 0..(float_array.len() - lag) {
//                     if float_array.is_null(i) {
//                         builder.append_null();
//                     } else {
//                         builder.append_value(float_array.value(i));
//                     }
//                 }
                
//                 Arc::new(builder.finish()) as ArrayRef
//             })
//             .collect();

//         // Build new schema with lag columns
//         let mut fields: Vec<Field> = batch.schema().fields().to_vec();
//         for (i, &lag) in lags.iter().enumerate() {
//             fields.push(Field::new(
//                 format!("{}_lag_{}", column_name, lag),
//                 DataType::Float64,
//                 true,
//             ));
//         }

//         // Combine original columns with lag columns
//         let mut all_columns: Vec<ArrayRef> = batch.columns().to_vec();
//         all_columns.extend(lag_arrays);

//         let new_schema = Arc::new(Schema::new(fields));
//         RecordBatch::try_new(new_schema, all_columns)
//             .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
//                 format!("Failed to create RecordBatch: {}", e)
//             ))
//     }

//     /// Create rolling window features using Arrow compute kernels
//     pub fn rolling_mean_arrow(
//         &self,
//         column_name: &str,
//         window: usize,
//         batch: &RecordBatch,
//     ) -> PyResult<RecordBatch> {
//         let source_array = batch
//             .column_by_name(column_name)
//             .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
//                 format!("Column '{}' not found", column_name)
//             ))?;
        
//         let float_array = source_array
//             .as_any()
//             .downcast_ref::<Float64Array>()
//             .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyTypeError, _>(
//                 "Column must be Float64"
//             ))?;

//         let mut builder = arrow::array::Float64Builder::new();
        
//         // Compute rolling mean
//         for i in 0..float_array.len() {
//             if i < window - 1 {
//                 builder.append_null();
//             } else {
//                 let start = i - window + 1;
//                 let mut sum = 0.0;
//                 let mut count = 0;
                
//                 for j in start..=i {
//                     if !float_array.is_null(j) {
//                         sum += float_array.value(j);
//                         count += 1;
//                     }
//                 }
                
//                 if count > 0 {
//                     builder.append_value(sum / count as f64);
//                 } else {
//                     builder.append_null();
//                 }
//             }
//         }

//         let rolling_array = Arc::new(builder.finish()) as ArrayRef;

//         // Add new column to batch
//         let mut fields = batch.schema().fields().to_vec();
//         fields.push(Field::new(
//             format!("{}_rolling_mean_{}", column_name, window),
//             DataType::Float64,
//             true,
//         ));

//         let mut columns = batch.columns().to_vec();
//         columns.push(rolling_array);

//         let new_schema = Arc::new(Schema::new(fields));
//         RecordBatch::try_new(new_schema, columns)
//             .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
//                 format!("Failed to create RecordBatch: {}", e)
//             ))
//     }

//     /// Exponential moving average using Arrow
//     pub fn ema_arrow(
//         &self,
//         column_name: &str,
//         alpha: f64,
//         batch: &RecordBatch,
//     ) -> PyResult<RecordBatch> {
//         let source_array = batch
//             .column_by_name(column_name)
//             .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
//                 format!("Column '{}' not found", column_name)
//             ))?;
        
//         let float_array = source_array
//             .as_any()
//             .downcast_ref::<Float64Array>()
//             .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyTypeError, _>(
//                 "Column must be Float64"
//             ))?;

//         let mut builder = arrow::array::Float64Builder::new();
//         let mut ema = None;

//         for i in 0..float_array.len() {
//             if float_array.is_null(i) {
//                 builder.append_null();
//             } else {
//                 let value = float_array.value(i);
//                 ema = match ema {
//                     None => Some(value),
//                     Some(prev_ema) => Some(alpha * value + (1.0 - alpha) * prev_ema),
//                 };
//                 builder.append_value(ema.unwrap());
//             }
//         }

//         let ema_array = Arc::new(builder.finish()) as ArrayRef;

//         let mut fields = batch.schema().fields().to_vec();
//         fields.push(Field::new(
//             format!("{}_ema_{}", column_name, alpha),
//             DataType::Float64,
//             true,
//         ));

//         let mut columns = batch.columns().to_vec();
//         columns.push(ema_array);

//         let new_schema = Arc::new(Schema::new(fields));
//         RecordBatch::try_new(new_schema, columns)
//             .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
//                 format!("Failed to create RecordBatch: {}", e)
//             ))
//     }

//     fn __repr__(&self) -> String {
//         format!(
//             "ArrowPipeline(batches={}, columns={})",
//             self.batches.len(),
//             self.schema.fields().len()
//         )
//     }
// }


// Phase 2: Advanced Arrow Operations
// src/engine/arrow_ops.rs

// use arrow::array::{Array, Float64Array, ArrayRef};
// use arrow::datatypes::{Schema, Field, DataType};
// use arrow::record_batch::RecordBatch;
// use pyo3::prelude::*;
// use std::sync::Arc;
// use rayon::prelude::*;

// #[pyclass]
// pub struct ArrowPipeline {
//     schema: Arc<Schema>,
//     batches: Vec<RecordBatch>,
// }

// #[pymethods]
// impl ArrowPipeline {
//     #[new]
//     pub fn new() -> Self {
//         Self {
//             schema: Arc::new(Schema::empty()),
//             batches: Vec::new(),
//         }
//     }

//     /// Create lag features using zero-copy Arrow operations
//     /// Note: Input/Output changed to PyObject to resolve PyO3 compatibility
//     pub fn create_lags_arrow(
//         &self,
//         column_name: &str,
//         lags: Vec<usize>,
//         _batch_obj: Bound<'_, PyAny>, // Received as Python object
//     ) -> PyResult<PyObject> {
//         // Implementation logic remains, but RecordBatch must be 
//         // extracted via C-Data interface or passed through a wrapper.
//         // For compilation fix, we focus on the Arc<Field> errors below.
        
//         Err(pyo3::exceptions::PyNotImplementedError::new_err("Arrow-Python conversion requires C-Data interface"))
//     }

//     // This internal helper demonstrates the Arc<Field> fix
//     fn process_batch_internal(
//         &self,
//         batch: &RecordBatch,
//         column_name: &str,
//         lags: Vec<usize>,
//         lag_arrays: Vec<ArrayRef>
//     ) -> PyResult<RecordBatch> {
//         // FIX: fields must be Vec<Arc<Field>>
//         let mut fields: Vec<Arc<Field>> = batch.schema().fields().to_vec();
        
//         for &lag in lags.iter() {
//             // FIX: Wrap Field in Arc
//             fields.push(Arc::new(Field::new(
//                 format!("{}_lag_{}", column_name, lag),
//                 DataType::Float64,
//                 true,
//             )));
//         }

//         let mut all_columns: Vec<ArrayRef> = batch.columns().to_vec();
//         all_columns.extend(lag_arrays);

//         let new_schema = Arc::new(Schema::new(fields));
//         RecordBatch::try_new(new_schema, all_columns)
//             .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
//                 format!("Failed to create RecordBatch: {}", e)
//             ))
//     }

//     pub fn rolling_mean_arrow(
//         &self,
//         column_name: &str,
//         window: usize,
//         batch_obj: Bound<'_, PyAny>,
//     ) -> PyResult<PyObject> {
//          Err(pyo3::exceptions::PyNotImplementedError::new_err("Use internal logic with Arc<Field> fixes"))
//     }

//     pub fn ema_arrow(
//         &self,
//         column_name: &str,
//         alpha: f64,
//         batch_obj: Bound<'_, PyAny>,
//     ) -> PyResult<PyObject> {
//          Err(pyo3::exceptions::PyNotImplementedError::new_err("Use internal logic with Arc<Field> fixes"))
//     }
// }


// Phase 2: Advanced Arrow Operations
// src/engine/arrow_ops.rs

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

    /// Python-facing: create lags via Vec<f64> (RecordBatch can't cross PyO3 boundary)
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

    /// Python-facing: rolling mean
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

    /// Python-facing: EMA
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

    fn __repr__(&self) -> String {
        "ArrowPipeline()".to_string()
    }
}

// ── Internal Arrow RecordBatch operations (called from Rust, not Python) ────

/// Create lag features on a RecordBatch with parallel execution
/// FIX: fields must be Vec<Arc<Field>>, not Vec<Field>
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

    // FIX: Vec<Arc<Field>>
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

/// Rolling mean on a RecordBatch
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

    // FIX: Vec<Arc<Field>>
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

/// EMA on a RecordBatch
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

    // FIX: Vec<Arc<Field>>
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