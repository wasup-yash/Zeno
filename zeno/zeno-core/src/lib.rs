use pyo3::prelude::*;

mod engine;
mod validator;
mod types;

use engine::window::WindowOp;
use engine::arrow_ops::ArrowPipeline;
use engine::polars_ops::{PolarsWindowOp, PolarsValidator};
use validator::temporal::TemporalValidator;
use validator::leakage::{LeakageDetector, RollingHashValidator};

#[pymodule]
fn _zeno(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ArrowPipeline>()?;
    m.add_class::<WindowOp>()?;
    m.add_class::<TemporalValidator>()?;
    m.add_class::<PolarsWindowOp>()?;
    m.add_class::<PolarsValidator>()?;
    m.add_class::<LeakageDetector>()?;
    m.add_class::<RollingHashValidator>()?;
    Ok(())
}