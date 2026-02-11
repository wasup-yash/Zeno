use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
#[pyclass]
pub struct PolarsWindowOp {
    lags: Vec<usize>,
    rolling_windows: Vec<usize>,
}

#[pymethods]
impl PolarsWindowOp {
    #[new]
    //Add explicit signature for Option arguments
    #[pyo3(signature = (lags, rolling_windows=None))]
    pub fn new(lags: Vec<usize>, rolling_windows: Option<Vec<usize>>) -> Self {
        Self {
            lags,
            rolling_windows: rolling_windows.unwrap_or_default(),
        }
    }

    pub fn create_lags_polars(
        &self,
        df: Bound<'_, PyAny>, //Use Bound API
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
        
        //Explicit type for closure parameter
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
        let result = PyDict::new_bound(py);
        for (name, data) in new_columns {
            result.set_item(name, data)?;
        }
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