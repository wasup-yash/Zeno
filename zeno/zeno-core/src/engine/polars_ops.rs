// // Phase 2: Polars-Native Integration
// // src/engine/polars_ops.rs

// use pyo3::prelude::*;
// use pyo3::types::{PyDict, PyList};
// use polars::prelude::*;
// use std::sync::Arc;

// #[pyclass]
// pub struct PolarsWindowOp {
//     lags: Vec<usize>,
//     rolling_windows: Vec<usize>,
// }

// #[pymethods]
// impl PolarsWindowOp {
//     #[new]
//     pub fn new(lags: Vec<usize>, rolling_windows: Option<Vec<usize>>) -> Self {
//         Self {
//             lags,
//             rolling_windows: rolling_windows.unwrap_or_default(),
//         }
//     }

//     /// Create lag features directly on Polars DataFrame (zero-copy)
//     pub fn create_lags_polars(
//         &self,
//         df: &PyAny,
//         column: &str,
//     ) -> PyResult<PyObject> {
//         Python::with_gil(|py| {
//             // Get the Polars DataFrame from Python
//             let df_dict = df.call_method0("to_dict")?;
//             let columns: &PyDict = df_dict.downcast()?;
            
//             // Get the target column
//             let col_data: &PyList = columns.get_item(column)?
//                 .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
//                     format!("Column '{}' not found", column)
//                 ))?
//                 .downcast()?;
            
//             // Convert to Vec<f64>
//             let values: Vec<f64> = col_data
//                 .iter()
//                 .map(|v| v.extract::<f64>())
//                 .collect::<PyResult<Vec<_>>>()?;

//             // Create lag features
//             let mut new_columns = vec![(column.to_string(), values.clone())];
            
//             for &lag in &self.lags {
//                 let mut lagged = vec![f64::NAN; lag];
//                 lagged.extend(&values[..values.len().saturating_sub(lag)]);
                
//                 new_columns.push((format!("{}_lag_{}", column, lag), lagged));
//             }

//             // Convert back to Python dict
//             let result = PyDict::new(py);
//             for (name, data) in new_columns {
//                 result.set_item(name, data)?;
//             }

//             // Create Polars DataFrame from dict
//             let pl = py.import("polars")?;
//             let df_new = pl.call_method1("DataFrame", (result,))?;
            
//             Ok(df_new.into())
//         })
//     }

//     /// Create rolling features on Polars DataFrame
//     pub fn rolling_mean_polars(
//         &self,
//         df: &PyAny,
//         column: &str,
//         window: usize,
//     ) -> PyResult<PyObject> {
//         Python::with_gil(|py| {
//             // Use Polars native rolling operations
//             let result = df.call_method1(
//                 "with_columns",
//                 (vec![
//                     df.call_method1("col", (column,))?
//                         .call_method1("rolling_mean", (window,))?
//                         .call_method1("alias", (format!("{}_rolling_mean_{}", column, window),))?
//                 ],)
//             )?;
            
//             Ok(result.into())
//         })
//     }

//     /// Create expanding window mean
//     pub fn expanding_mean_polars(
//         &self,
//         df: &PyAny,
//         column: &str,
//     ) -> PyResult<PyObject> {
//         Python::with_gil(|py| {
//             let result = df.call_method1(
//                 "with_columns",
//                 (vec![
//                     df.call_method1("col", (column,))?
//                         .call_method("cum_sum", (), None)?
//                         .call_method1("truediv", (
//                             df.call_method("select", (column,), None)?
//                                 .call_method("count", (), None)?,
//                         ))?
//                         .call_method1("alias", (format!("{}_expanding_mean", column),))?
//                 ],)
//             )?;
            
//             Ok(result.into())
//         })
//     }

//     /// Parallel window operations across multiple columns
//     pub fn create_features_parallel(
//         &self,
//         df: &PyAny,
//         columns: Vec<String>,
//     ) -> PyResult<PyObject> {
//         Python::with_gil(|py| {
//             let mut df_result = df.clone().into();
            
//             for column in columns {
//                 // Create lags
//                 for &lag in &self.lags {
//                     df_result = df_result
//                         .call_method1(
//                             "with_columns",
//                             (vec![
//                                 df_result
//                                     .call_method1("col", (&column,))?
//                                     .call_method1("shift", (lag,))?
//                                     .call_method1("alias", (format!("{}_lag_{}", column, lag),))?
//                             ],)
//                         )?
//                         .into();
//                 }
                
//                 // Create rolling windows
//                 for &window in &self.rolling_windows {
//                     df_result = df_result
//                         .call_method1(
//                             "with_columns",
//                             (vec![
//                                 df_result
//                                     .call_method1("col", (&column,))?
//                                     .call_method1("rolling_mean", (window,))?
//                                     .call_method1("alias", (format!("{}_rolling_{}", column, window),))?
//                             ],)
//                         )?
//                         .into();
//                 }
//             }
            
//             Ok(df_result)
//         })
//     }

//     fn __repr__(&self) -> String {
//         format!(
//             "PolarsWindowOp(lags={:?}, rolling={:?})",
//             self.lags, self.rolling_windows
//         )
//     }
// }

// #[pyclass]
// pub struct PolarsValidator {
//     train_end_col: Option<String>,
//     test_start_col: Option<String>,
// }

// #[pymethods]
// impl PolarsValidator {
//     #[new]
//     pub fn new() -> Self {
//         Self {
//             train_end_col: None,
//             test_start_col: None,
//         }
//     }

//     /// Validate temporal ordering in Polars DataFrame
//     pub fn validate_temporal_split(
//         &mut self,
//         df: &PyAny,
//         time_col: &str,
//         train_end: i64,
//         test_start: i64,
//     ) -> PyResult<bool> {
//         if test_start <= train_end {
//             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
//                 "Test start must be after train end"
//             ));
//         }

//         Python::with_gil(|py| {
//             // Get min and max timestamps
//             let time_series = df.call_method1("col", (time_col,))?;
//             let min_time: i64 = time_series.call_method0("min")?.extract()?;
//             let max_time: i64 = time_series.call_method0("max")?.extract()?;

//             // Check for gaps
//             if min_time > train_end || max_time < test_start {
//                 return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
//                     "Time range doesn't cover train/test split"
//                 ));
//             }

//             Ok(true)
//         })
//     }

//     /// Split Polars DataFrame into train and test
//     pub fn split_dataframe(
//         &self,
//         df: &PyAny,
//         time_col: &str,
//         cutoff: i64,
//     ) -> PyResult<(PyObject, PyObject)> {
//         Python::with_gil(|py| {
//             // Filter for train
//             let train = df.call_method1(
//                 "filter",
//                 (df.call_method1("col", (time_col,))?
//                     .call_method1("le", (cutoff,))?,)
//             )?;

//             // Filter for test
//             let test = df.call_method1(
//                 "filter",
//                 (df.call_method1("col", (time_col,))?
//                     .call_method1("gt", (cutoff,))?,)
//             )?;

//             Ok((train.into(), test.into()))
//         })
//     }

//     fn __repr__(&self) -> String {
//         "PolarsValidator()".to_string()
//     }
// }


// Phase 2: Polars-Native Integration
// src/engine/polars_ops.rs

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use polars::prelude::*;

#[pyclass]
pub struct PolarsWindowOp {
    lags: Vec<usize>,
    rolling_windows: Vec<usize>,
}

#[pymethods]
impl PolarsWindowOp {
    #[new]
    // FIX: Add explicit signature for Option arguments
    #[pyo3(signature = (lags, rolling_windows=None))]
    pub fn new(lags: Vec<usize>, rolling_windows: Option<Vec<usize>>) -> Self {
        Self {
            lags,
            rolling_windows: rolling_windows.unwrap_or_default(),
        }
    }

    pub fn create_lags_polars(
        &self,
        df: Bound<'_, PyAny>, // FIX: Use Bound API
        column: &str,
    ) -> PyResult<PyObject> {
        let py = df.py();
        
        let df_dict = df.call_method0("to_dict")?;
        let columns = df_dict.downcast::<PyDict>()?;
        
        let col_data = columns.get_item(column)?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Column '{}' not found", column)
            ))?;
        let col_list = col_data.downcast::<PyList>()?;
        
        // FIX: Explicit type for closure parameter
        let values: Vec<f64> = col_list
            .iter()
            .map(|v: Bound<'_, PyAny>| v.extract::<f64>())
            .collect::<PyResult<Vec<_>>>()?;

        let mut new_columns = vec![(column.to_string(), values.clone())];
        
        for &lag in &self.lags {
            let mut lagged = vec![f64::NAN; lag];
            lagged.extend(&values[..values.len().saturating_sub(lag)]);
            new_columns.push((format!("{}_lag_{}", column, lag), lagged));
        }

        // FIX: Use new_bound
        let result = PyDict::new_bound(py);
        for (name, data) in new_columns {
            result.set_item(name, data)?;
        }

        // FIX: Use import_bound
        let pl = py.import_bound("polars")?;
        let df_new = pl.call_method1("DataFrame", (result,))?;
        
        Ok(df_new.into())
    }

    pub fn rolling_mean_polars(
        &self,
        df: Bound<'_, PyAny>,
        column: &str,
        window: usize,
    ) -> PyResult<PyObject> {
        let col_expr = df.call_method1("col", (column,))?
            .call_method1("rolling_mean", (window,))?
            .call_method1("alias", (format!("{}_rolling_mean_{}", column, window),))?;
            
        let result = df.call_method1("with_columns", (vec![col_expr],))?;
        Ok(result.into())
    }

    pub fn expanding_mean_polars(
        &self,
        df: Bound<'_, PyAny>,
        column: &str,
    ) -> PyResult<PyObject> {
        let count_val = df.call_method("select", (column,), None)?
            .call_method("count", (), None)?;

        let col_expr = df.call_method1("col", (column,))?
            .call_method("cum_sum", (), None)?
            .call_method1("truediv", (count_val,))?
            .call_method1("alias", (format!("{}_expanding_mean", column),))?;
            
        let result = df.call_method1("with_columns", (vec![col_expr],))?;
        Ok(result.into())
    }

    pub fn create_features_parallel(
        &self,
        df: Bound<'_, PyAny>,
        columns: Vec<String>,
    ) -> PyResult<PyObject> {
        let mut df_result = df.clone();
        
        for column in columns {
            for &lag in &self.lags {
                let lag_expr = df_result.call_method1("col", (&column,))?
                    .call_method1("shift", (lag,))?
                    .call_method1("alias", (format!("{}_lag_{}", column, lag),))?;
                
                df_result = df_result.call_method1("with_columns", (vec![lag_expr],))?;
            }
        }
        
        Ok(df_result.into())
    }
}

#[pyclass]
pub struct PolarsValidator {
    train_end_col: Option<String>,
    test_start_col: Option<String>,
}

#[pymethods]
impl PolarsValidator {
    #[new]
    pub fn new() -> Self {
        Self { train_end_col: None, test_start_col: None }
    }

    pub fn validate_temporal_split(
        &mut self,
        df: Bound<'_, PyAny>,
        time_col: &str,
        train_end: i64,
        test_start: i64,
    ) -> PyResult<bool> {
        if test_start <= train_end {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Test start must be after train end"));
        }

        let time_series = df.call_method1("col", (time_col,))?;
        let min_time: i64 = time_series.call_method0("min")?.extract()?;
        let max_time: i64 = time_series.call_method0("max")?.extract()?;

        if min_time > train_end || max_time < test_start {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Time range doesn't cover split"));
        }

        Ok(true)
    }

    pub fn split_dataframe(
        &self,
        df: Bound<'_, PyAny>,
        time_col: &str,
        cutoff: i64,
    ) -> PyResult<(PyObject, PyObject)> {
        let train_cond = df.call_method1("col", (time_col,))?.call_method1("le", (cutoff,))?;
        let test_cond = df.call_method1("col", (time_col,))?.call_method1("gt", (cutoff,))?;

        let train = df.call_method1("filter", (train_cond,))?;
        let test = df.call_method1("filter", (test_cond,))?;

        Ok((train.into(), test.into()))
    }
}