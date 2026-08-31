use pyo3::prelude::*;

mod engine;
mod validator;
mod types;

use engine::window::WindowOp;
use validator::temporal::TemporalValidator;

#[pymodule]
fn _zeno(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WindowOp>()?;
    m.add_class::<TemporalValidator>()?;
    Ok(())
}
