use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct Forecast {
    point_forecast: Vec<f64>,
    lower_bound: Vec<f64>,
    upper_bound: Vec<f64>,
    confidence_level: f64,
}

#[pymethods]
impl Forecast {
    #[new]
    pub fn new(
        point_forecast: Vec<f64>,
        lower_bound: Vec<f64>,
        upper_bound: Vec<f64>,
        confidence_level: f64,
    ) -> Self {
        Self {
            point_forecast,
            lower_bound,
            upper_bound,
            confidence_level,
        }
    }

    #[getter]
    pub fn point(&self) -> Vec<f64> {
        self.point_forecast.clone()
    }

    #[getter]
    pub fn lower(&self) -> Vec<f64> {
        self.lower_bound.clone()
    }

    #[getter]
    pub fn upper(&self) -> Vec<f64> {
        self.upper_bound.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Forecast(horizon={}, confidence={:.1}%)",
            self.point_forecast.len(),
            self.confidence_level * 100.0
        )
    }
}

#[pyclass]
pub struct BatchPredictor {
    max_parallel: usize,
    use_gpu: bool,
}

#[pymethods]
impl BatchPredictor {
    #[new]
    #[pyo3(signature = (max_parallel=4, use_gpu=false))]
    pub fn new(max_parallel: Option<usize>, use_gpu: Option<bool>) -> Self {
        Self {
            max_parallel: max_parallel.unwrap_or(4),
            use_gpu: use_gpu.unwrap_or(false),
        }
    }

    /// Predict multiple time series in parallel
    pub fn predict_batch(
        &self,
        py: Python<'_>,
        model: Bound<'_, PyAny>,
        series_list: Vec<Vec<f64>>,
        horizon: usize,
    ) -> PyResult<Vec<Forecast>> {
        if self.use_gpu {
            self.predict_batch_gpu(py, model, series_list, horizon)
        } else {
            self.predict_batch_cpu(py, model, series_list, horizon)
        }
    }

    /// CPU-based batch prediction
    fn predict_batch_cpu(
        &self,
        py: Python<'_>,
        model: Bound<'_, PyAny>,
        series_list: Vec<Vec<f64>>,
        horizon: usize,
    ) -> PyResult<Vec<Forecast>> {
        let mut forecasts = Vec::new();

        for series in series_list {
            let pred = model.call_method1("predict", (series, horizon))?;
            let point: Vec<f64> = pred.get_item("point")?.extract()?;
            let lower: Vec<f64> = pred.get_item("lower")?.extract()?;
            let upper: Vec<f64> = pred.get_item("upper")?.extract()?;

            forecasts.push(Forecast::new(point, lower, upper, 0.95));
        }

        Ok(forecasts)
    }

    /// GPU-based batch prediction
    fn predict_batch_gpu(
        &self,
        py: Python<'_>,
        model: Bound<'_, PyAny>,
        series_list: Vec<Vec<f64>>,
        horizon: usize,
    ) -> PyResult<Vec<Forecast>> {
        let torch = py.import_bound("torch")?;

        // Stack all series into one tensor
        let batch_tensor = torch.call_method1("tensor", (series_list.clone(),))?;
        let batch_gpu = batch_tensor.call_method1("to", ("cuda",))?;

        // Single forward pass for entire batch
        let output = model.call_method1("forward", (batch_gpu, horizon))?;
        let cpu_output = output.call_method0("cpu")?;

        // Extract forecasts
        let point_forecasts: Vec<Vec<f64>> = cpu_output
            .get_item("mean")?
            .call_method0("tolist")?
            .extract()?;
        let lower_bounds: Vec<Vec<f64>> = cpu_output
            .get_item("lower")?
            .call_method0("tolist")?
            .extract()?;
        let upper_bounds: Vec<Vec<f64>> = cpu_output
            .get_item("upper")?
            .call_method0("tolist")?
            .extract()?;

        let forecasts: Vec<Forecast> = point_forecasts
            .into_iter()
            .zip(lower_bounds)
            .zip(upper_bounds)
            .map(|((p, l), u)| Forecast::new(p, l, u, 0.95))
            .collect();

        Ok(forecasts)
    }

    /// Streaming predictions for large datasets
    pub fn predict_stream(
        &self,
        py: Python<'_>,
        model: Bound<'_, PyAny>,
        series_iter: Vec<Vec<f64>>,
        horizon: usize,
        chunk_size: usize,
    ) -> PyResult<Vec<Forecast>> {
        let mut all_forecasts = Vec::new();

        for chunk in series_iter.chunks(chunk_size) {
            let chunk_forecasts = self.predict_batch(py, model.clone(), chunk.to_vec(), horizon)?;
            all_forecasts.extend(chunk_forecasts);
        }

        Ok(all_forecasts)
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchPredictor(parallel={}, gpu={})",
            self.max_parallel, self.use_gpu
        )
    }
}

#[pyclass]
pub struct EnsemblePredictor {
    models: Vec<String>,
    weights: Vec<f64>,
}

#[pymethods]

impl EnsemblePredictor {
    #[new]
    #[pyo3(signature = (models, weights=None))]
    pub fn new(models: Vec<String>, weights: Option<Vec<f64>>) -> PyResult<Self> {
        let w = weights.unwrap_or_else(|| vec![1.0 / models.len() as f64; models.len()]);

        if w.len() != models.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Weights must match number of models",
            ));
        }

        Ok(Self { models, weights: w })
    }

    /// Weighted ensemble prediction
    pub fn predict_ensemble(
        &self,
        py: Python<'_>,
        model_dict: Bound<'_, PyDict>,
        series: Vec<f64>,
        horizon: usize,
    ) -> PyResult<Forecast> {
        let mut weighted_forecasts = vec![0.0; horizon];

        for (model_name, weight) in self.models.iter().zip(&self.weights) {
            let model = model_dict.get_item(model_name)?.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                    "Model '{}' not found",
                    model_name
                ))
            })?;

            let pred = model.call_method1("predict", (series.clone(), horizon))?;
            let forecast: Vec<f64> = pred.extract()?;

            for (i, &val) in forecast.iter().enumerate() {
                weighted_forecasts[i] += val * weight;
            }
        }

        // Simple confidence bounds (±10% of forecast)
        let lower: Vec<f64> = weighted_forecasts.iter().map(|&x| x * 0.9).collect();
        let upper: Vec<f64> = weighted_forecasts.iter().map(|&x| x * 1.1).collect();

        Ok(Forecast::new(weighted_forecasts, lower, upper, 0.95))
    }

    fn __repr__(&self) -> String {
        format!("EnsemblePredictor(models={})", self.models.len())
    }
}
