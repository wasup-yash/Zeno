use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass]
pub struct ChronosWrapper {
    model_name: String,
    context_length: usize,
    prediction_length: usize,
}

#[pymethods]
impl ChronosWrapper {
    #[new]
    #[pyo3(signature = (model_name="chronos-t5-small", context_length=512, prediction_length=64))]
    pub fn new(
        model_name: Option<&str>,
        context_length: Option<usize>,
        prediction_length: Option<usize>,
    ) -> Self {
        Self {
            model_name: model_name.unwrap_or("chronos-t5-small").to_string(),
            context_length: context_length.unwrap_or(512),
            prediction_length: prediction_length.unwrap_or(64),
        }
    }

    /// Load model from HuggingFace (calls Python transformers library)
    pub fn load_model(&self, py: Python<'_>) -> PyResult<PyObject> {
        let transformers = py.import_bound("transformers")?;
        let model = transformers.call_method1(
            "AutoModelForSeq2SeqLM.from_pretrained",
            (&self.model_name,)
        )?;
        Ok(model.into())
    }

    /// Prepare time series data for Chronos input format
    pub fn prepare_input(
        &self,
        values: Vec<f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        if values.len() < self.context_length {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Need at least {} values, got {}", self.context_length, values.len())
            ));
        }

        // Create sliding windows
        let mut windows = Vec::new();
        for i in 0..=values.len().saturating_sub(self.context_length) {
            windows.push(values[i..i + self.context_length].to_vec());
        }

        Ok(windows)
    }

    /// Forecast future values
    
    pub fn forecast(
        &self,
        py: Python<'_>,
        model: Bound<'_, PyAny>,
        history: Vec<f64>,
        frequency: &str,
    ) -> PyResult<Vec<Vec<f64>>> {
        let gluonts_data = py.import_bound("gluonts.dataset.common")?;
        
        let entry = PyDict::new_bound(py);
        // Convert Vec<f64> to Python list to satisfy IntoPy bound
        entry.set_item("target", history.to_object(py))?;
        entry.set_item("start", "2024-01-01")?;
        entry.set_item("freq", frequency)?;

        let dataset = gluonts_data.call_method1("ListDataset", (vec![entry], frequency))?;
        let forecasts = model.call_method1("predict", (dataset,))?;
        
        // Correctly extract from iterator in Bound API
        let forecast_iter = forecasts.call_method0("__iter__")?;
        let mut samples: Vec<Vec<f64>> = Vec::new();

        if let Ok(first_forecast) = forecast_iter.call_method0("__next__") {
            samples = first_forecast.getattr("samples")?.extract()?;
        }
        
        Ok(samples)
    }

    fn __repr__(&self) -> String {
        format!(
            "ChronosWrapper(model='{}', context={}, horizon={})",
            self.model_name, self.context_length, self.prediction_length
        )
    }
}

#[pyclass]
pub struct LagLlamaWrapper {
    model_path: String,
    context_length: usize,
    prediction_length: usize,
}

#[pymethods]
impl LagLlamaWrapper {
    #[new]
    #[pyo3(signature = (model_path="time-series-foundation-models/Lag-Llama", context_length=32, prediction_length=1))]
    pub fn new(
        model_path: Option<&str>,
        context_length: Option<usize>,
        prediction_length: Option<usize>,
    ) -> Self {
        Self {
            model_path: model_path.unwrap_or("time-series-foundation-models/Lag-Llama").to_string(),
            context_length: context_length.unwrap_or(32),
            prediction_length: prediction_length.unwrap_or(1),
        }
    }

    /// Load Lag-Llama model
    pub fn load_model(&self, py: Python<'_>) -> PyResult<PyObject> {
        let gluonts = py.import_bound("gluonts.torch")?;
        let model = gluonts.call_method1("LagLlamaEstimator.from_pretrained", (&self.model_path,))?;
        Ok(model.into())
    }

    /// Zero-shot forecast
    pub fn zero_shot_forecast(
        &self,
        py: Python<'_>,
        model: Bound<'_, PyAny>,
        history: Vec<f64>,
        num_samples: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let gluonts = py.import_bound("gluonts.dataset.common")?;
        
        // Create dataset entry
        let entry = PyDict::new_bound(py);
        entry.set_item("target", history)?;
        entry.set_item("start", "2024-01-01")?;

        // Run inference
        let forecasts = model.call_method(
            "predict",
            (vec![entry], num_samples, self.prediction_length),
            None
        )?;

        let samples: Vec<Vec<f64>> = forecasts.extract()?;
        Ok(samples)
    }

    fn __repr__(&self) -> String {
        format!(
            "LagLlamaWrapper(path='{}', context={}, horizon={})",
            self.model_path, self.context_length, self.prediction_length
        )
    }
}

#[pyclass]
pub struct MoiraiWrapper {
    model_size: String,
    context_length: usize,
    prediction_length: usize,
    patch_size: usize,
}

#[pymethods]
impl MoiraiWrapper {
    #[new]
    #[pyo3(signature = (model_size="small", context_length=512, prediction_length=128, patch_size=16))]
    pub fn new(
        model_size: Option<&str>,
        context_length: Option<usize>,
        prediction_length: Option<usize>,
        patch_size: Option<usize>,
    ) -> Self {
        Self {
            model_size: model_size.unwrap_or("small").to_string(),
            context_length: context_length.unwrap_or(512),
            prediction_length: prediction_length.unwrap_or(128),
            patch_size: patch_size.unwrap_or(16),
        }
    }

    /// Load MOIRAI model from HuggingFace via uni2ts/gluonts
    pub fn load_model(&self, py: Python<'_>) -> PyResult<PyObject> {
        let uni2ts = py.import_bound("uni2ts.model.moirai")?;
        
        // MOIRAI models are often loaded via MoiraiForecastPredictor or MoiraiModule
        let model = uni2ts.call_method(
            "MoiraiForecastPredictor.from_pretrained",
            (format!("Salesforce/moirai-1.0-R-{}", self.model_size),),
            Some(&{
                let kwargs = PyDict::new_bound(py);
                kwargs.set_item("prediction_length", self.prediction_length)?;
                kwargs.set_item("context_length", self.context_length)?;
                kwargs.set_item("patch_size", self.patch_size)?;
                kwargs
            })
        )?;
        Ok(model.into())
    }

    /// Zero-shot forecast with MOIRAI
    pub fn forecast(
        &self,
        py: Python<'_>,
        model: Bound<'_, PyAny>,
        history: Vec<f64>,
        frequency: &str,
    ) -> PyResult<Vec<Vec<f64>>> {
        let gluonts_data = py.import_bound("gluonts.dataset.common")?;
        
        // Create the dataset entry required by uni2ts/gluonts
        let entry = PyDict::new_bound(py);
        entry.set_item("target", history)?;
        entry.set_item("start", "2024-01-01")?; // Placeholder start
        entry.set_item("freq", frequency)?;

        // Wrap in a PandasDataset or simple ListDataset
        let dataset = gluonts_data.call_method1("ListDataset", (vec![entry], frequency))?;

        // Run prediction
        let forecasts = model.call_method1("predict", (&dataset,))?;
        
        // Extract samples from the Forecast object
        
        let forecast_iter = forecasts.call_method0("__iter__")?;
        let first_forecast = forecast_iter.call_method0("__next__")?;
        let samples: Vec<Vec<f64>> = first_forecast.getattr("samples")?.extract()?;
        Ok(samples)
    }

    fn __repr__(&self) -> String {
        format!(
            "MoiraiWrapper(size='{}', context={}, horizon={}, patch={})",
            self.model_size, self.context_length, self.prediction_length, self.patch_size
        )
    }
}


/// Normalize time series for model input
pub fn normalize_for_model(values: &[f64]) -> (Vec<f64>, f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let std = (values.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / values.len() as f64)
        .sqrt();

    let normalized: Vec<f64> = values.iter()
        .map(|&x| (x - mean) / std)
        .collect();

    (normalized, mean, std)
}

/// Denormalize model output
pub fn denormalize_forecast(forecast: &[f64], mean: f64, std: f64) -> Vec<f64> {
    forecast.iter()
        .map(|&x| x * std + mean)
        .collect()
}