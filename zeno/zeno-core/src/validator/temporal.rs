use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass]
pub struct TemporalValidator {
    train_end: Option<i64>,
    test_start: Option<i64>,
}

#[pymethods]
impl TemporalValidator {
    #[new]
    pub fn new() -> Self {
        Self {
            train_end: None,
            test_start: None,
        }
    }

    /// Set the train/test split boundary
    pub fn set_split(
        &mut self,
        train_end_timestamp: i64,
        test_start_timestamp: i64,
    ) -> PyResult<()> {
        if test_start_timestamp <= train_end_timestamp {
            return Err(PyValueError::new_err(
                "Test start must be after train end (temporal leakage detected!)",
            ));
        }

        self.train_end = Some(train_end_timestamp);
        self.test_start = Some(test_start_timestamp);
        Ok(())
    }

    /// Validate that a feature doesn't use future data
    pub fn check_feature_window(&self, feature_timestamp: i64) -> PyResult<bool> {
        if let (Some(train_end), Some(_test_start)) = (self.train_end, self.test_start) {
            if feature_timestamp > train_end {
                return Err(PyValueError::new_err(
                    format!("Feature uses data from {} which is after train cutoff! TEMPORAL LEAKAGE DETECTED.", feature_timestamp)
                ));
            }
        }
        Ok(true)
    }

    fn __repr__(&self) -> String {
        format!(
            "TemporalValidator(train_end={:?}, test_start={:?})",
            self.train_end, self.test_start
        )
    }
}
