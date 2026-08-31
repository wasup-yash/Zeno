use pyo3::prelude::*;
use pyo3::exceptions::{PyMemoryError, PyRuntimeError};
use thiserror::Error;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("GPU Out Of Memory: Requested {requested} bytes, Available {available} bytes")]
    OutOfMemory { requested: usize, available: usize },
    #[error("Device execution failed: {0}")]
    ExecutionFailed(String),
}

impl From<GpuError> for PyErr {
    fn from(err: GpuError) -> PyErr {
        match err {
            GpuError::OutOfMemory { .. } => PyMemoryError::new_err(err.to_string()),
            _ => PyRuntimeError::new_err(err.to_string()),
        }
    }
}

// Global GPU Memory tracker to prevent CUDA/WGPU OOM panics
static GPU_ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
const MAX_GPU_MEMORY: usize = 16 * 1024 * 1024 * 1024; // e.g., 16GB limit

/// Represents a safely managed GPU memory allocation utilizing RAII
pub struct ManagedGpuTensor {
    pub size_bytes: usize,
    // ptr: *mut std::ffi::c_void, // In production, this holds the device pointer (CUDA/WGPU)
}

impl ManagedGpuTensor {
    /// Attempts to allocate GPU memory. Returns a clean error if limits are exceeded.
    pub fn allocate(size_bytes: usize) -> Result<Self, GpuError> {
        let current = GPU_ALLOCATED_BYTES.load(Ordering::SeqCst);
        if current + size_bytes > MAX_GPU_MEMORY {
            return Err(GpuError::OutOfMemory {
                requested: size_bytes,
                available: MAX_GPU_MEMORY.saturating_sub(current),
            });
        }
        
        GPU_ALLOCATED_BYTES.fetch_add(size_bytes, Ordering::SeqCst);
        
        Ok(Self { size_bytes })
    }
}

// The core of production GPU safety: deterministic memory freeing
impl Drop for ManagedGpuTensor {
    fn drop(&mut self) {
        GPU_ALLOCATED_BYTES.fetch_sub(self.size_bytes, Ordering::SeqCst);
        // unsafe { cudaFree(self.ptr); } // Explicit release of C-device pointers
    }
}

#[pyclass]
pub struct GPUAccelerator {
    device: String,
    batch_size: usize,
}

#[pymethods]
impl GPUAccelerator {
    #[new]
    #[pyo3(signature = (device="cuda:0", batch_size=32))]
    pub fn new(device: Option<&str>, batch_size: Option<usize>) -> Self {
        Self {
            device: device.unwrap_or("cuda:0").to_string(),
            batch_size: batch_size.unwrap_or(32),
        }
    }

    /// Check GPU availability
    pub fn is_available(&self, py: Python<'_>) -> PyResult<bool> {
        let torch = py.import_bound("torch")?;
        let available: bool = torch
            .getattr("cuda")?
            .call_method0("is_available")?
            .extract()?;
        Ok(available)
    }

    /// Get GPU memory info
    pub fn get_memory_info(&self, py: Python<'_>) -> PyResult<(u64, u64)> {
        let torch = py.import_bound("torch")?;
        let cuda = torch.getattr("cuda")?;

        let allocated: u64 = cuda.call_method0("memory_allocated")?.extract()?;
        let reserved: u64 = cuda.call_method0("memory_reserved")?.extract()?;

        Ok((allocated, reserved))
    }

    /// Move tensor to GPU
    pub fn to_gpu(&self, _py: Python<'_>, tensor: Bound<'_, PyAny>) -> PyResult<PyObject> {
        let moved = tensor.call_method1("to", (&self.device,))?;
        Ok(moved.into())
    }

    /// Batch inference on GPU
    pub fn batch_inference_gpu(
        &self,
        py: Python<'_>,
        model: Bound<'_, PyAny>,
        inputs: Vec<Vec<f64>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let torch = py.import_bound("torch")?;

        // Convert to tensor
        let input_tensor = torch.call_method1("tensor", (inputs.clone(),))?;
        let input_gpu = input_tensor.call_method1("to", (&self.device,))?;

        // Run inference in batches
        let mut results = Vec::new();
        let num_batches = (inputs.len() + self.batch_size - 1) / self.batch_size;

        for i in 0..num_batches {
            let start = i * self.batch_size;
            let end = ((i + 1) * self.batch_size).min(inputs.len());

            let batch = input_gpu.get_item(pyo3::types::PySlice::new_bound(
                py,
                start as isize,
                end as isize,
                1,
            ))?;

            // Inference
            let output = model.call_method1("forward", (batch,))?;
            let cpu_output = output.call_method0("cpu")?;
            let batch_result: Vec<Vec<f64>> = cpu_output.call_method0("tolist")?.extract()?;

            results.extend(batch_result);
        }

        Ok(results)
    }

    /// Clear GPU cache
    pub fn clear_cache(&self, py: Python<'_>) -> PyResult<()> {
        let torch = py.import_bound("torch")?;
        torch.getattr("cuda")?.call_method0("empty_cache")?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "GPUAccelerator(device='{}', batch_size={})",
            self.device, self.batch_size
        )
    }
}

#[pyclass]
pub struct TensorConverter {
    dtype: String,
}

#[pymethods]
impl TensorConverter {
    #[new]
    #[pyo3(signature = (dtype="float32"))]
    pub fn new(dtype: Option<&str>) -> Self {
        Self {
            dtype: dtype.unwrap_or("float32").to_string(),
        }
    }

    /// Convert Vec<f64> to PyTorch tensor
    pub fn to_tensor(&self, py: Python<'_>, values: Vec<f64>) -> PyResult<PyObject> {
        let torch = py.import_bound("torch")?;
        let tensor = torch.call_method1("tensor", (values,))?;

        // Set dtype
        let dtype_obj = torch.getattr(self.dtype.as_str())?;
        let typed_tensor = tensor.call_method1("to", (dtype_obj,))?;

        Ok(typed_tensor.into())
    }

    /// Convert 2D array to tensor
    pub fn to_tensor_2d(&self, py: Python<'_>, values: Vec<Vec<f64>>) -> PyResult<PyObject> {
        let torch = py.import_bound("torch")?;
        let tensor = torch.call_method1("tensor", (values,))?;
        let dtype_obj = torch.getattr(self.dtype.as_str())?;
        let typed_tensor = tensor.call_method1("to", (dtype_obj,))?;
        Ok(typed_tensor.into())
    }

    /// Reshape tensor for model input
    pub fn reshape(&self, tensor: Bound<'_, PyAny>, shape: Vec<i64>) -> PyResult<PyObject> {
        let reshaped = tensor.call_method1("reshape", (shape,))?;
        Ok(reshaped.into())
    }

    fn __repr__(&self) -> String {
        format!("TensorConverter(dtype='{}')", self.dtype)
    }
}
