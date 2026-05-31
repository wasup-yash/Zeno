use pyo3::prelude::*;

mod engine;
mod types;
mod validator;

use engine::arrow_ops::ArrowPipeline;
use engine::audit::{AuditLogger, AuditReport, ComplianceChecker, ReportGenerator};
use engine::batch::{BatchPredictor, EnsemblePredictor, Forecast};
use engine::foundation::{ChronosWrapper, LagLlamaWrapper, MoiraiWrapper};
use engine::gpu::{GPUAccelerator, TensorConverter};
use engine::managed::{ManagedPipeline, PipelineRegistry, ValidationScheduler};
use engine::polars_ops::{PolarsValidator, PolarsWindowOp};
use engine::serverless::{BacktestResult, BacktestRunner, ServerlessConfig};
use engine::window::WindowOp;
use validator::leakage::{LeakageDetector, RollingHashValidator};
use validator::temporal::TemporalValidator;

#[pymodule]
fn _zeno(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ArrowPipeline>()?;
    m.add_class::<WindowOp>()?;
    m.add_class::<TemporalValidator>()?;
    m.add_class::<PolarsWindowOp>()?;
    m.add_class::<PolarsValidator>()?;
    m.add_class::<LeakageDetector>()?;
    m.add_class::<RollingHashValidator>()?;
    m.add_class::<ChronosWrapper>()?;
    m.add_class::<LagLlamaWrapper>()?;
    m.add_class::<MoiraiWrapper>()?;
    m.add_class::<GPUAccelerator>()?;
    m.add_class::<TensorConverter>()?;
    m.add_class::<Forecast>()?;
    m.add_class::<BatchPredictor>()?;
    m.add_class::<EnsemblePredictor>()?;
    m.add_class::<BacktestResult>()?;
    m.add_class::<BacktestRunner>()?;
    m.add_class::<ServerlessConfig>()?;
    m.add_class::<ManagedPipeline>()?;
    m.add_class::<PipelineRegistry>()?;
    m.add_class::<ValidationScheduler>()?;
    m.add_class::<AuditReport>()?;
    m.add_class::<ComplianceChecker>()?;
    m.add_class::<AuditLogger>()?;
    m.add_class::<ReportGenerator>()?;
    Ok(())
}
