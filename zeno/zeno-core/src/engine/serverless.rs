use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

#[pyclass]
#[derive(Clone)]
pub struct BacktestResult {
    metrics: HashMap<String, f64>,
    predictions: Vec<f64>,
    actuals: Vec<f64>,
    residuals: Vec<f64>,
}

#[pymethods]
impl BacktestResult {
    #[new]
    pub fn new(metrics: HashMap<String, f64>, predictions: Vec<f64>, actuals: Vec<f64>) -> Self {
        let residuals: Vec<f64> = predictions
            .iter()
            .zip(&actuals)
            .map(|(&p, &a)| a - p)
            .collect();

        Self {
            metrics,
            predictions,
            actuals,
            residuals,
        }
    }

    #[getter]
    pub fn get_metrics(&self) -> HashMap<String, f64> {
        self.metrics.clone()
    }

    #[getter]
    pub fn mse(&self) -> f64 {
        self.metrics.get("mse").copied().unwrap_or(0.0)
    }

    #[getter]
    pub fn mae(&self) -> f64 {
        self.metrics.get("mae").copied().unwrap_or(0.0)
    }

    #[getter]
    pub fn mape(&self) -> f64 {
        self.metrics.get("mape").copied().unwrap_or(0.0)
    }

    fn __repr__(&self) -> String {
        format!(
            "BacktestResult(samples={}, MSE={:.4}, MAE={:.4})",
            self.predictions.len(),
            self.mse(),
            self.mae()
        )
    }
}

#[pyclass]
pub struct BacktestRunner {
    n_splits: usize,
    test_size: usize,
    step_size: usize,
}

#[pymethods]
impl BacktestRunner {
    #[new]
    #[pyo3(signature = (n_splits=5, test_size=30, step_size=1))]
    pub fn new(
        n_splits: Option<usize>,
        test_size: Option<usize>,
        step_size: Option<usize>,
    ) -> Self {
        Self {
            n_splits: n_splits.unwrap_or(5),
            test_size: test_size.unwrap_or(30),
            step_size: step_size.unwrap_or(1),
        }
    }

    /// Run expanding window backtest
    pub fn run_expanding_window(
        &self,
        _py: Python<'_>,
        model: Bound<'_, PyAny>,
        data: Vec<f64>,
        min_train: usize,
    ) -> PyResult<Vec<BacktestResult>> {
        let mut results = Vec::new();

        for i in 0..self.n_splits {
            let train_end = min_train + i * self.step_size;
            let test_end = train_end + self.test_size;

            if test_end > data.len() {
                break;
            }

            let train_data = data[..train_end].to_vec();
            let test_data = &data[train_end..test_end];

            // Fit model - pass Vec instead of slice
            model.call_method1("fit", (train_data,))?;

            // Predict
            let preds_obj = model.call_method1("predict", (self.test_size,))?;
            let predictions: Vec<f64> = preds_obj.extract()?;

            // Compute metrics
            let metrics = self.compute_metrics(&predictions, test_data);

            results.push(BacktestResult::new(
                metrics,
                predictions,
                test_data.to_vec(),
            ));
        }

        Ok(results)
    }

    /// Run rolling window backtest
    pub fn run_rolling_window(
        &self,
        _py: Python<'_>,
        model: Bound<'_, PyAny>,
        data: Vec<f64>,
        window_size: usize,
    ) -> PyResult<Vec<BacktestResult>> {
        let mut results = Vec::new();

        for i in 0..self.n_splits {
            let train_start = i * self.step_size;
            let train_end = train_start + window_size;
            let test_end = train_end + self.test_size;

            if test_end > data.len() {
                break;
            }

            let train_data = data[train_start..train_end].to_vec();
            let test_data = &data[train_end..test_end];

            // Fit model - pass Vec instead of slice
            model.call_method1("fit", (train_data,))?;
            let preds_obj = model.call_method1("predict", (self.test_size,))?;
            let predictions: Vec<f64> = preds_obj.extract()?;

            let metrics = self.compute_metrics(&predictions, test_data);
            results.push(BacktestResult::new(
                metrics,
                predictions,
                test_data.to_vec(),
            ));
        }

        Ok(results)
    }

    fn __repr__(&self) -> String {
        format!(
            "BacktestRunner(n_splits={}, test_size={}, step={})",
            self.n_splits, self.test_size, self.step_size
        )
    }
}

// Move compute_metrics outside of #[pymethods] block
impl BacktestRunner {
    /// Compute common metrics
    fn compute_metrics(&self, predictions: &[f64], actuals: &[f64]) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        // MSE
        let mse: f64 = predictions
            .iter()
            .zip(actuals)
            .map(|(&p, &a)| (a - p).powi(2))
            .sum::<f64>()
            / predictions.len() as f64;
        metrics.insert("mse".to_string(), mse);

        // MAE
        let mae: f64 = predictions
            .iter()
            .zip(actuals)
            .map(|(&p, &a)| (a - p).abs())
            .sum::<f64>()
            / predictions.len() as f64;
        metrics.insert("mae".to_string(), mae);

        // RMSE
        metrics.insert("rmse".to_string(), mse.sqrt());

        // MAPE
        let mape: f64 = predictions
            .iter()
            .zip(actuals)
            .filter(|(_, &a)| a != 0.0)
            .map(|(&p, &a)| ((a - p) / a).abs())
            .sum::<f64>()
            / predictions.len() as f64
            * 100.0;
        metrics.insert("mape".to_string(), mape);

        metrics
    }
}

#[pyclass]
pub struct ServerlessConfig {
    pub lambda_function: String,
    pub timeout: u32,
    pub memory_mb: u32,
}

#[pymethods]
impl ServerlessConfig {
    #[new]
    #[pyo3(signature = (lambda_function=None, timeout=None, memory_mb=None))]
    pub fn new(
        lambda_function: Option<String>,
        timeout: Option<u32>,
        memory_mb: Option<u32>,
    ) -> Self {
        Self {
            lambda_function: lambda_function.unwrap_or_else(|| "zeno-backtest".to_string()),
            timeout: timeout.unwrap_or(300),
            memory_mb: memory_mb.unwrap_or(3008),
        }
    }

    pub fn submit_job(&self, py: Python<'_>, job_config: Bound<'_, PyDict>) -> PyResult<String> {
        let boto3 = py.import_bound("boto3")?;
        let lambda_client = boto3.call_method1("client", ("lambda",))?;

        let json = py.import_bound("json")?;
        let payload_str = json.call_method1("dumps", (job_config,))?;

        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("FunctionName", &self.lambda_function)?;
        kwargs.set_item("InvocationType", "Event")?;
        kwargs.set_item("Payload", payload_str)?;

        let response = lambda_client.call_method("invoke", (), Some(&kwargs))?;

        // Fix for Bound API item access
        let metadata = response.get_item("ResponseMetadata")?;
        let request_id: String = metadata.get_item("RequestId")?.extract()?;

        Ok(request_id)
    }
}
