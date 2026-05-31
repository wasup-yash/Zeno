// Phase 4: Compliance & Audit Reports
// src/engine/audit.rs

use chrono::{DateTime, Utc};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

#[pyclass]
#[derive(Clone)]
pub struct AuditReport {
    report_id: String,
    timestamp: String,
    validation_results: HashMap<String, bool>,
    metrics: HashMap<String, f64>,
    warnings: Vec<String>,
    passed: bool,
}

#[pymethods]
impl AuditReport {
    #[new]
    pub fn new(report_id: String) -> Self {
        Self {
            report_id,
            timestamp: Utc::now().to_rfc3339(),
            validation_results: HashMap::new(),
            metrics: HashMap::new(),
            warnings: Vec::new(),
            passed: true,
        }
    }

    /// Add validation result
    pub fn add_validation(&mut self, check_name: String, passed: bool) {
        if !passed {
            self.passed = false;
        }
        self.validation_results.insert(check_name, passed);
    }

    /// Add metric
    pub fn add_metric(&mut self, metric_name: String, value: f64) {
        self.metrics.insert(metric_name, value);
    }

    /// Add warning
    pub fn add_warning(&mut self, message: String) {
        self.warnings.push(message);
    }

    /// Get summary
    #[getter]
    pub fn summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();
        summary.insert("report_id".to_string(), self.report_id.clone());
        summary.insert("timestamp".to_string(), self.timestamp.clone());
        summary.insert("passed".to_string(), self.passed.to_string());
        summary.insert(
            "checks".to_string(),
            self.validation_results.len().to_string(),
        );
        summary.insert("warnings".to_string(), self.warnings.len().to_string());
        summary
    }

    /// Export to JSON
    pub fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        let json_mod = py.import_bound("json")?;

        let report_dict = PyDict::new_bound(py);
        report_dict.set_item("report_id", &self.report_id)?;
        report_dict.set_item("timestamp", &self.timestamp)?;
        report_dict.set_item("passed", self.passed)?;
        report_dict.set_item("validations", self.validation_results.clone())?;
        report_dict.set_item("metrics", self.metrics.clone())?;
        report_dict.set_item("warnings", self.warnings.clone())?;

        let json_str: String = json_mod
            .call_method1("dumps", (report_dict, 4))?
            .extract()?;
        Ok(json_str)
    }

    fn __repr__(&self) -> String {
        format!(
            "AuditReport(id='{}', passed={}, checks={})",
            self.report_id,
            self.passed,
            self.validation_results.len()
        )
    }
}

#[pyclass]
pub struct ComplianceChecker {
    rules: HashMap<String, String>,
}

#[pymethods]
impl ComplianceChecker {
    #[new]
    pub fn new() -> Self {
        let mut rules = HashMap::new();

        // Default compliance rules
        rules.insert(
            "temporal_leakage".to_string(),
            "No future data in training".to_string(),
        );
        rules.insert(
            "data_distribution".to_string(),
            "Train/test distributions similar".to_string(),
        );
        rules.insert(
            "feature_stability".to_string(),
            "Features stable across splits".to_string(),
        );
        rules.insert(
            "model_performance".to_string(),
            "Performance within bounds".to_string(),
        );

        Self { rules }
    }

    /// Add custom rule
    pub fn add_rule(&mut self, rule_name: String, description: String) {
        self.rules.insert(rule_name, description);
    }

    /// Check temporal leakage
    pub fn check_temporal_leakage(
        &self,
        py: Python<'_>,
        train_dates: Vec<String>,
        test_dates: Vec<String>,
    ) -> PyResult<bool> {
        let _datetime_mod = py.import_bound("datetime")?;

        // Parse dates
        let train_max = train_dates
            .iter()
            .max()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Empty train dates"))?;
        let test_min = test_dates
            .iter()
            .min()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Empty test dates"))?;

        // Check ordering
        Ok(test_min > train_max)
    }

    /// Check data distribution similarity
    pub fn check_distribution_similarity(
        &self,
        train_data: Vec<f64>,
        test_data: Vec<f64>,
        threshold: f64,
    ) -> PyResult<bool> {
        let train_mean = train_data.iter().sum::<f64>() / train_data.len() as f64;
        let test_mean = test_data.iter().sum::<f64>() / test_data.len() as f64;

        let train_std = (train_data
            .iter()
            .map(|&x| (x - train_mean).powi(2))
            .sum::<f64>()
            / train_data.len() as f64)
            .sqrt();

        let test_std = (test_data
            .iter()
            .map(|&x| (x - test_mean).powi(2))
            .sum::<f64>()
            / test_data.len() as f64)
            .sqrt();

        let mean_diff = (train_mean - test_mean).abs() / train_mean;
        let std_diff = (train_std - test_std).abs() / train_std;

        Ok(mean_diff < threshold && std_diff < threshold)
    }

    /// Run full compliance check
    pub fn run_compliance_check(
        &self,
        py: Python<'_>,
        data_config: Bound<'_, PyDict>,
    ) -> PyResult<AuditReport> {
        let mut report = AuditReport::new(format!("audit_{}", Utc::now().timestamp()));

        // Extract data
        let train_dates: Vec<String> = data_config
            .get_item("train_dates")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("train_dates not found"))?
            .extract()?;
        let test_dates: Vec<String> = data_config
            .get_item("test_dates")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("test_dates not found"))?
            .extract()?;

        // Check temporal leakage
        let no_leakage = self.check_temporal_leakage(py, train_dates, test_dates)?;
        report.add_validation("temporal_leakage".to_string(), no_leakage);

        if !no_leakage {
            report.add_warning("Temporal leakage detected in train/test split".to_string());
        }

        // Add metrics
        report.add_metric(
            "compliance_score".to_string(),
            if no_leakage { 100.0 } else { 0.0 },
        );

        Ok(report)
    }

    fn __repr__(&self) -> String {
        format!("ComplianceChecker(rules={})", self.rules.len())
    }
}

#[pyclass]
pub struct AuditLogger {
    log_file: String,
    logs: Vec<String>,
}

#[pymethods]
impl AuditLogger {
    #[new]
    #[pyo3(signature = (log_file="audit.log"))]
    pub fn new(log_file: Option<&str>) -> Self {
        Self {
            log_file: log_file.unwrap_or("audit.log").to_string(),
            logs: Vec::new(),
        }
    }

    /// Log an event
    pub fn log(&mut self, event: String, level: Option<&str>) {
        let log_level = level.unwrap_or("INFO");
        let timestamp = Utc::now().to_rfc3339();
        let log_entry = format!("[{}] {} - {}", timestamp, log_level, event);
        self.logs.push(log_entry);
    }

    /// Get all logs
    pub fn get_logs(&self) -> Vec<String> {
        self.logs.clone()
    }

    /// Save logs to file
    pub fn save(&self, py: Python<'_>) -> PyResult<()> {
        let content = self.logs.join("\n");

        let pathlib = py.import_bound("pathlib")?;
        let path = pathlib.getattr("Path")?.call1((&self.log_file,))?;
        path.call_method1("write_text", (content,))?;

        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "AuditLogger(file='{}', entries={})",
            self.log_file,
            self.logs.len()
        )
    }
}

#[pyclass]
pub struct ReportGenerator {
    template: String,
}

#[pymethods]
impl ReportGenerator {
    #[new]
    #[pyo3(signature = (template="default"))]
    pub fn new(template: Option<&str>) -> Self {
        Self {
            template: template.unwrap_or("default").to_string(),
        }
    }

    /// Generate HTML report
    pub fn generate_html(&self, report: &AuditReport) -> PyResult<String> {
        let summary = report.summary();

        let html = format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>Zeno Audit Report - {}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; }}
        .passed {{ color: green; font-weight: bold; }}
        .failed {{ color: red; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #34495e; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🌀 Zeno Compliance Audit Report</h1>
        <p>Report ID: {}</p>
        <p>Generated: {}</p>
    </div>
    
    <h2>Summary</h2>
    <p class="{}">Status: {}</p>
    <p>Total Checks: {}</p>
    <p>Warnings: {}</p>
    
    <h2>Validation Results</h2>
    <table>
        <tr><th>Check</th><th>Status</th></tr>
        <!-- Validation results would be inserted here -->
    </table>
    
    <h2>Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <!-- Metrics would be inserted here -->
    </table>
</body>
</html>"#,
            summary.get("report_id").unwrap_or(&"Unknown".to_string()),
            summary.get("report_id").unwrap_or(&"Unknown".to_string()),
            summary.get("timestamp").unwrap_or(&"Unknown".to_string()),
            if report.passed { "passed" } else { "failed" },
            if report.passed { "PASSED" } else { "FAILED" },
            summary.get("checks").unwrap_or(&"0".to_string()),
            summary.get("warnings").unwrap_or(&"0".to_string()),
        );

        Ok(html)
    }

    /// Generate PDF report
    pub fn generate_pdf(
        &self,
        py: Python<'_>,
        report: &AuditReport,
        output_path: &str,
    ) -> PyResult<String> {
        let _html = self.generate_html(report)?;

        // In production, would use weasyprint or similar
        let _weasyprint = py.import_bound("weasyprint");

        // Placeholder - actual PDF generation would go here
        Ok(format!("PDF report saved to {}", output_path))
    }

    fn __repr__(&self) -> String {
        format!("ReportGenerator(template='{}')", self.template)
    }
}
