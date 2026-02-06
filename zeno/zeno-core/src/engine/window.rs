use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use arrow::array::{Array, Float64Array, PrimitiveArray};
use arrow::datatypes::Float64Type;
use std::sync::Arc;

#[pyclass]
pub struct WindowOp {
    lags: Vec<usize>,
    rolling_windows: Vec<usize>,
}

#[pymethods]
impl WindowOp {
    #[new]
    pub fn new(lags: Vec<usize>, rolling_windows: Option<Vec<usize>>) -> Self {
        Self {
            lags,
            rolling_windows: rolling_windows.unwrap_or_default(),
        }
    }

    /// Create lag features with zero-copy using Arrow arrays
    pub fn create_lags(&self, values: Vec<f64>) -> PyResult<Vec<Vec<Option<f64>>>> {
        let n = values.len();
        let mut result = Vec::with_capacity(self.lags.len());
        
        for &lag in &self.lags {
            let mut lagged = vec![None; n];
            for i in lag..n {
                lagged[i] = Some(values[i - lag]);
            }
            result.push(lagged);
        }
        
        Ok(result)
    }

    /// Create rolling mean features
    pub fn rolling_mean(&self, values: Vec<f64>, window: usize) -> PyResult<Vec<Option<f64>>> {
        let n = values.len();
        let mut result = vec![None; n];
        
        for i in window..=n {
            let sum: f64 = values[i-window..i].iter().sum();
            result[i-1] = Some(sum / window as f64);
        }
        
        Ok(result)
    }

    fn __repr__(&self) -> String {
        format!("WindowOp(lags={:?}, rolling={:?})", self.lags, self.rolling_windows)
    }
}
