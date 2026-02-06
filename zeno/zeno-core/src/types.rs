use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesMetadata {
    pub n_rows: usize,
    pub n_series: usize,
    pub time_col: String,
    pub value_cols: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum TransformState {
    Fitted { params: Vec<f64> },
    NotFitted,
}
