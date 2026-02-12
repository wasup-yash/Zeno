use pyo3::prelude::*;

#[pyclass]
pub struct WindowOp {
    lags: Vec<usize>,
    rolling_windows: Vec<usize>,
}

#[pymethods]
impl WindowOp {
    #[new]
    #[pyo3(signature = (lags, rolling_windows=None))]
    pub fn new(lags: Vec<usize>, rolling_windows: Option<Vec<usize>>) -> Self {
        Self {
            lags,
            rolling_windows: rolling_windows.unwrap_or_default(),
        }
    }

    /// Phase 1: Basic lag creation
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

    /// Phase 1: Basic rolling mean
    pub fn rolling_mean(&self, values: Vec<f64>, window: usize) -> PyResult<Vec<Option<f64>>> {
        let n = values.len();
        let mut result = vec![None; n];
        
        for i in window..=n {
            let sum: f64 = values[i-window..i].iter().sum();
            result[i-1] = Some(sum / window as f64);
        }
        
        Ok(result)
    }

    /// Phase 2: Rolling standard deviation (added)
    pub fn rolling_std(&self, values: Vec<f64>, window: usize) -> PyResult<Vec<Option<f64>>> {
        let n = values.len();
        let mut result = vec![None; n];
        
        for i in window..=n {
            let slice = &values[i-window..i];
            let mean: f64 = slice.iter().sum::<f64>() / window as f64;
            let variance: f64 = slice.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / window as f64;
            result[i-1] = Some(variance.sqrt());
        }
        
        Ok(result)
    }

    /// Phase 2: Weighted moving average
    pub fn weighted_moving_average(
        &self,
        values: Vec<f64>,
        weights: Vec<f64>,
    ) -> PyResult<Vec<Option<f64>>> {
        if values.len() != weights.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Values and weights must have same length"
            ));
        }

        let window = weights.len();
        let n = values.len();
        let mut result = vec![None; n];
        let weight_sum: f64 = weights.iter().sum();

        for i in window..=n {
            let weighted_sum: f64 = values[i-window..i]
                .iter()
                .zip(&weights)
                .map(|(&v, &w)| v * w)
                .sum();
            result[i-1] = Some(weighted_sum / weight_sum);
        }

        Ok(result)
    }

    fn __repr__(&self) -> String {
        format!("WindowOp(lags={:?}, rolling={:?})", self.lags, self.rolling_windows)
    }
}

// ── Phase 1: Internal helper functions ──────────────────────────────────────

/// Compute lag for a single vector (used internally)
pub fn compute_single_lag(values: &[f64], lag: usize) -> Vec<Option<f64>> {
    let n = values.len();
    let mut result = vec![None; n];
    for i in lag..n {
        result[i] = Some(values[i - lag]);
    }
    result
}

/// Compute rolling window statistic (generic)
pub fn compute_rolling_stat<F>(
    values: &[f64],
    window: usize,
    stat_fn: F,
) -> Vec<Option<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    let n = values.len();
    let mut result = vec![None; n];
    
    for i in window..=n {
        let slice = &values[i-window..i];
        result[i-1] = Some(stat_fn(slice));
    }
    
    result
}