// Phase 2: Advanced Leakage Detection
// src/validator/leakage.rs

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

#[derive(Debug, Clone)]
pub struct FeatureFingerprint {
    pub window_hash: u64,
    pub timestamps: Vec<i64>,
    pub values_hash: u64,
    pub feature_name: String,
}

impl FeatureFingerprint {
    /// Create fingerprint for a feature window
    pub fn new(
        timestamps: Vec<i64>,
        values: Vec<f64>,
        feature_name: String,
    ) -> Self {
        let mut hasher = DefaultHasher::new();
        
        // Hash the time window
        for ts in &timestamps {
            ts.hash(&mut hasher);
        }
        let window_hash = hasher.finish();

        // Hash the values (approximate)
        let mut value_hasher = DefaultHasher::new();
        for val in values {
            // Convert to bits for hashing
            val.to_bits().hash(&mut value_hasher);
        }
        let values_hash = value_hasher.finish();

        Self {
            window_hash,
            timestamps: timestamps.clone(),
            values_hash,
            feature_name,
        }
    }

    /// Check if two fingerprints overlap in time
    pub fn overlaps_with(&self, other: &FeatureFingerprint) -> bool {
        let self_min = self.timestamps.iter().min().unwrap_or(&0);
        let self_max = self.timestamps.iter().max().unwrap_or(&0);
        let other_min = other.timestamps.iter().min().unwrap_or(&0);
        let other_max = other.timestamps.iter().max().unwrap_or(&0);

        // Check for overlap
        !(self_max < other_min || other_max < self_min)
    }

    /// Similarity score between fingerprints (0.0 to 1.0)
    pub fn similarity(&self, other: &FeatureFingerprint) -> f64 {
        if self.window_hash == other.window_hash {
            return 1.0;
        }

        // Jaccard similarity of timestamps
        let self_set: HashSet<_> = self.timestamps.iter().collect();
        let other_set: HashSet<_> = other.timestamps.iter().collect();
        
        let intersection = self_set.intersection(&other_set).count();
        let union = self_set.union(&other_set).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }
}

#[pyclass]
pub struct LeakageDetector {
    train_fingerprints: Vec<FeatureFingerprint>,
    test_fingerprints: Vec<FeatureFingerprint>,
    threshold: f64,
}

#[pymethods]
impl LeakageDetector {
    #[new]
    pub fn new(threshold: Option<f64>) -> Self {
        Self {
            train_fingerprints: Vec::new(),
            test_fingerprints: Vec::new(),
            threshold: threshold.unwrap_or(0.1),
        }
    }

    /// Register training data fingerprint
    pub fn register_train_window(
        &mut self,
        timestamps: Vec<i64>,
        values: Vec<f64>,
        feature_name: String,
    ) -> PyResult<()> {
        let fingerprint = FeatureFingerprint::new(timestamps, values, feature_name);
        self.train_fingerprints.push(fingerprint);
        Ok(())
    }

    /// Check if test window leaks training data
    pub fn check_test_window(
        &mut self,
        timestamps: Vec<i64>,
        values: Vec<f64>,
        feature_name: String,
    ) -> PyResult<HashMap<String, f64>> {
        let test_fp = FeatureFingerprint::new(timestamps, values, feature_name.clone());
        
        let mut leakage_scores: HashMap<String, f64> = HashMap::new();

        for train_fp in &self.train_fingerprints {
            if test_fp.overlaps_with(train_fp) {
                let similarity = test_fp.similarity(train_fp);
                
                if similarity > self.threshold {
                    leakage_scores.insert(
                        train_fp.feature_name.clone(),
                        similarity,
                    );
                }
            }
        }

        self.test_fingerprints.push(test_fp);

        if !leakage_scores.is_empty() {
            return Err(PyValueError::new_err(
                format!(
                    "LEAKAGE DETECTED in feature '{}': Overlaps with {} training features (max similarity: {:.2})",
                    feature_name,
                    leakage_scores.len(),
                    leakage_scores.values().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0)
                )
            ));
        }

        Ok(leakage_scores)
    }

    /// Get detailed leakage report
    pub fn get_leakage_report(&self) -> PyResult<HashMap<String, Vec<(String, f64)>>> {
        let mut report: HashMap<String, Vec<(String, f64)>> = HashMap::new();

        for test_fp in &self.test_fingerprints {
            let mut overlaps = Vec::new();
            
            for train_fp in &self.train_fingerprints {
                if test_fp.overlaps_with(train_fp) {
                    let similarity = test_fp.similarity(train_fp);
                    if similarity > self.threshold {
                        overlaps.push((train_fp.feature_name.clone(), similarity));
                    }
                }
            }

            if !overlaps.is_empty() {
                report.insert(test_fp.feature_name.clone(), overlaps);
            }
        }

        Ok(report)
    }

    /// Clear all fingerprints
    pub fn reset(&mut self) {
        self.train_fingerprints.clear();
        self.test_fingerprints.clear();
    }

    fn __repr__(&self) -> String {
        format!(
            "LeakageDetector(train_features={}, test_features={}, threshold={})",
            self.train_fingerprints.len(),
            self.test_fingerprints.len(),
            self.threshold
        )
    }
}

/// Rolling hash for time series data (for efficient comparison)
#[pyclass]
pub struct RollingHashValidator {
    hash_size: usize,
    known_hashes: HashSet<u64>,
}

#[pymethods]
impl RollingHashValidator {
    #[new]
    pub fn new(hash_size: Option<usize>) -> Self {
        Self {
            hash_size: hash_size.unwrap_or(100),
            known_hashes: HashSet::new(),
        }
    }

    /// Add a window to known hashes (training data)
    pub fn add_window(&mut self, values: Vec<f64>) -> PyResult<u64> {
        let hash = self.compute_hash(&values);
        self.known_hashes.insert(hash);
        Ok(hash)
    }

    /// Check if window exists in training data
    pub fn check_window(&self, values: Vec<f64>) -> PyResult<bool> {
        let hash = self.compute_hash(&values);
        Ok(self.known_hashes.contains(&hash))
    }

    /// Compute rolling hash for a window
    fn compute_hash(&self, values: &[f64]) -> u64 {
        let mut hasher = DefaultHasher::new();
        
        for val in values.iter().take(self.hash_size.min(values.len())) {
            val.to_bits().hash(&mut hasher);
        }
        
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!("RollingHashValidator(known_windows={})", self.known_hashes.len())
    }
}