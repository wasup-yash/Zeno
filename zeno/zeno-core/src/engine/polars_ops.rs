use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3_polars::PyDataFrame;
use polars::prelude::*;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PolarsFFIError {
    #[error("Polars execution failed: {0}")]
    Polars(#[from] PolarsError),
}

impl From<PolarsFFIError> for PyErr {
    fn from(err: PolarsFFIError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

#[pyfunction]
pub fn safe_rolling_mean(
    pydf: PyDataFrame,
    time_col: &str,
    value_col: &str,
    window_size: &str,
) -> PyResult<PyDataFrame> {
    // Safely unwrap the PyDataFrame into a native Rust DataFrame
    let df: DataFrame = pydf.into();

    // Construct and execute the LazyFrame expression natively in Rust
    let result = df.lazy()
        .with_columns([
            col(value_col)
                .rolling_mean(RollingOptions {
                    window_size: Duration::parse(window_size),
                    window_by: Some(time_col.into()),
                    ..Default::default()
                })
                .alias("rolling_mean")
        ])
        .collect()
        .map_err(PolarsFFIError::from)?; // Safe error propagation

    // Safely wrap back into PyDataFrame for memory-safe transfer
    Ok(PyDataFrame(result))
}

#[pyclass]
pub struct PolarsWindowOp {
    lags: Vec<usize>,
    rolling_windows: Vec<usize>,
}

#[pymethods]
impl PolarsWindowOp {
    #[new]
    #[pyo3(signature = (lags, rolling_windows=None))]
    pub fn new(lags: Vec<usize>, rolling_windows: Option<Vec<usize>>) -> Self {
        Self {
            lags,
            rolling_windows: rolling_windows.unwrap_or_default(),
        }
    }

    pub fn create_lags_polars(&self, df: Bound<'_, PyAny>, column: &str) -> PyResult<PyObject> {
        let py = df.py();
        let pl = py.import_bound("polars")?;
        let col = pl.getattr("col")?;

        let exprs: Vec<PyObject> = self
            .lags
            .iter()
            .map(|lag| {
                col.call1((column,))
                    .and_then(|expr| expr.call_method1("shift", (*lag,)))
                    .and_then(|expr| {
                        expr.call_method1("alias", (format!("{}_lag_{}", column, lag),))
                    })
                    .map(|expr| expr.into())
            })
            .collect::<PyResult<Vec<_>>>()?;

        let result = df.call_method1("with_columns", (exprs,))?;
        Ok(result.into())
    }

    pub fn rolling_mean_polars(
        &self,
        df: Bound<'_, PyAny>,
        column: &str,
        window: usize,
    ) -> PyResult<PyObject> {
        let py = df.py();
        let pl = py.import_bound("polars")?;
        let expr = pl
            .getattr("col")?
            .call1((column,))?
            .call_method1("rolling_mean", (window,))?
            .call_method1("alias", (format!("{}_rolling_mean_{}", column, window),))?;

        let result = df.call_method1("with_columns", (vec![expr],))?;
        Ok(result.into())
    }

    pub fn expanding_mean_polars(&self, df: Bound<'_, PyAny>, column: &str) -> PyResult<PyObject> {
        let py = df.py();
        let pl = py.import_bound("polars")?;
        let expr = pl
            .getattr("col")?
            .call1((column,))?
            .call_method0("cum_sum")?
            .call_method1("truediv", (pl.getattr("len")?.call0()?,))?
            .call_method1("alias", (format!("{}_expanding_mean", column),))?;

        let result = df.call_method1("with_columns", (vec![expr],))?;
        Ok(result.into())
    }

    pub fn create_features_parallel(
        &self,
        df: Bound<'_, PyAny>,
        columns: Vec<String>,
    ) -> PyResult<PyObject> {
        let py = df.py();
        let pl = py.import_bound("polars")?;
        let col = pl.getattr("col")?;
        let mut exprs: Vec<PyObject> = Vec::new();

        for column in &columns {
            for lag in &self.lags {
                let expr = col
                    .call1((column.as_str(),))?
                    .call_method1("shift", (*lag,))?
                    .call_method1("alias", (format!("{}_lag_{}", column, lag),))?;
                exprs.push(expr.into());
            }

            for window in &self.rolling_windows {
                let expr = col
                    .call1((column.as_str(),))?
                    .call_method1("rolling_mean", (*window,))?
                    .call_method1("alias", (format!("{}_rolling_{}", column, window),))?;
                exprs.push(expr.into());
            }
        }

        let result = df.call_method1("with_columns", (exprs,))?;
        Ok(result.into())
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
        Self {
            train_end_col: None,
            test_start_col: None,
        }
    }

    pub fn validate_temporal_split(
        &mut self,
        df: Bound<'_, PyAny>,
        time_col: &str,
        train_end: i64,
        test_start: i64,
    ) -> PyResult<bool> {
        if test_start <= train_end {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Test start must be after train end",
            ));
        }

        let time_series = df.call_method1("get_column", (time_col,))?;
        let is_sorted: bool = time_series.call_method0("is_sorted")?.extract()?;
        if !is_sorted {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Column '{}' must be sorted ascending for zero-copy slicing",
                time_col
            )));
        }

        let min_time: i64 = time_series.call_method0("min")?.extract()?;
        let max_time: i64 = time_series.call_method0("max")?.extract()?;

        if min_time > train_end || max_time < test_start {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Time range doesn't cover split",
            ));
        }

        Ok(true)
    }

    pub fn split_dataframe(
        &self,
        df: Bound<'_, PyAny>,
        time_col: &str,
        cutoff: i64,
    ) -> PyResult<(PyObject, PyObject)> {
        let py = df.py();
        let pl = py.import_bound("polars")?;
        let time_series = df.call_method1("get_column", (time_col,))?;
        let is_sorted: bool = time_series.call_method0("is_sorted")?.extract()?;
        if !is_sorted {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Column '{}' must be sorted ascending for zero-copy slicing",
                time_col
            )));
        }

        let train_len_expr = pl
            .getattr("col")?
            .call1((time_col,))?
            .call_method1("le", (cutoff,))?
            .call_method0("sum")?
            .call_method1("alias", ("__train_len",))?;
        let train_len: usize = df
            .call_method1("select", (vec![train_len_expr],))?
            .call_method0("item")?
            .extract()?;

        let train = df.call_method1("slice", (0_i64, train_len))?;
        let test = df.call_method1("slice", (train_len as i64,))?;

        Ok((train.into(), test.into()))
    }

    pub fn set_split(&mut self, train_end: i64, test_start: i64) -> PyResult<()> {
        if test_start <= train_end {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Test start must be after train end (temporal leakage detected!)",
            ));
        }
        self.train_end_col = Some(train_end.to_string());
        self.test_start_col = Some(test_start.to_string());
        Ok(())
    }

    pub fn check_feature_window(&self, feature_ts: i64) -> PyResult<bool> {
        if let Some(ref train_end_str) = self.train_end_col {
            let train_end: i64 = train_end_str.parse().unwrap_or(i64::MAX);
            if feature_ts > train_end {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!(
                        "Feature uses data from {} which is after train cutoff! TEMPORAL LEAKAGE DETECTED.",
                        feature_ts
                    ),
                ));
            }
        }
        Ok(true)
    }
}
