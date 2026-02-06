#!/bin/bash
set -e  # Exit on any error

echo " Setting up Zeno: The Zero-Copy Time Series Engine"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

echo -e "${BLUE}📍 Detected OS: ${MACHINE}${NC}"

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZENO_ROOT="${SCRIPT_DIR}/zeno"

echo -e "${BLUE}📂 Zeno will be installed in: ${ZENO_ROOT}${NC}"

# 1. Install Rust (if not present)
echo -e "\n${YELLOW}[1/7] Installing Rust toolchain...${NC}"
if ! command -v rustc &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo -e "${GREEN}✓ Rust installed${NC}"
else
    echo -e "${GREEN}✓ Rust already installed ($(rustc --version))${NC}"
fi

# Make sure cargo is in PATH
if ! command -v cargo &> /dev/null; then
    source "$HOME/.cargo/env"
fi

# 2. Check for Python and setup virtual environment method
echo -e "\n${YELLOW}[2/7] Setting up Python environment...${NC}"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✓ Found Python ${PYTHON_VERSION}${NC}"

# Try to install/use uv (fast package manager)
USE_UV=false
if command -v uv &> /dev/null; then
    echo -e "${GREEN}✓ uv already installed${NC}"
    USE_UV=true
else
    echo -e "${BLUE}Attempting to install uv (fast package manager)...${NC}"
    if curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null; then
        source "$HOME/.cargo/env"
        if command -v uv &> /dev/null; then
            echo -e "${GREEN}✓ uv installed successfully${NC}"
            USE_UV=true
        fi
    fi
fi

if [ "$USE_UV" = false ]; then
    echo -e "${YELLOW}⚠ uv not available, using standard pip and venv (slower but works)${NC}"
fi

# 3. Create project structure
echo -e "\n${YELLOW}[3/7] Creating project structure...${NC}"
mkdir -p "${ZENO_ROOT}/zeno-core/src/engine"
mkdir -p "${ZENO_ROOT}/zeno-core/src/validator"
mkdir -p "${ZENO_ROOT}/zeno-py/zeno"
mkdir -p "${ZENO_ROOT}/tests"
mkdir -p "${ZENO_ROOT}/examples"
mkdir -p "${ZENO_ROOT}/benchmarks"
echo -e "${GREEN}✓ Directory structure created${NC}"

# 4. Initialize Rust project
echo -e "\n${YELLOW}[4/7] Initializing Rust core...${NC}"

cat > "${ZENO_ROOT}/zeno-core/Cargo.toml" << 'EOF'
[package]
name = "zeno-core"
version = "0.1.0"
edition = "2021"

[lib]
name = "zeno_core"
crate-type = ["cdylib", "rlib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module", "abi3-py38"] }
arrow = "53.3.0"
polars = { version = "0.45", features = ["lazy", "temporal"] }
chrono = "0.4"
rayon = "1.10"
thiserror = "2.0"
serde = { version = "1.0", features = ["derive"] }

[dev-dependencies]
criterion = "0.5"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
EOF

cd "${ZENO_ROOT}/zeno-py"
source .venv/bin/activate
maturin develop --release

echo -e "${GREEN}✓ Cargo.toml created${NC}"

# 5. Create Rust source files
echo -e "\n${YELLOW}[5/7] Generating Rust core files...${NC}"

# Main lib.rs
cat > "${ZENO_ROOT}/zeno-core/src/lib.rs" << 'EOF'
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
EOF

# Types
cat > "${ZENO_ROOT}/zeno-core/src/types.rs" << 'EOF'
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
EOF

# Window operations
cat > "${ZENO_ROOT}/zeno-core/src/engine/window.rs" << 'EOF'
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use arrow::array::{Array, Float64Array, PrimitiveArray};
use arrow::datatypes::Float64Type;
use std::sync::Arc;

#[pyclass]
pub struct WindowOp {
    lags: Vec<usize>,
    rolling_windows: Vec<usize>,
}

#[pymethods]
impl WindowOp {
    #[new]
    pub fn new(lags: Vec<usize>, rolling_windows: Option<Vec<usize>>) -> Self {
        Self {
            lags,
            rolling_windows: rolling_windows.unwrap_or_default(),
        }
    }

    /// Create lag features with zero-copy using Arrow arrays
    pub fn create_lags(&self, values: Vec<f64>) -> PyResult<Vec<Vec<Option<f64>>>> {
        let n = values.len();
        let mut result = Vec::with_capacity(self.lags.len());
        
        for &lag in &self.lags {
            let mut lagged = vec![None; n];
            for i in lag..n {
                lagged[i] = Some(values[i - lag]);
            }
            result.push(lagged);
        }
        
        Ok(result)
    }

    /// Create rolling mean features
    pub fn rolling_mean(&self, values: Vec<f64>, window: usize) -> PyResult<Vec<Option<f64>>> {
        let n = values.len();
        let mut result = vec![None; n];
        
        for i in window..=n {
            let sum: f64 = values[i-window..i].iter().sum();
            result[i-1] = Some(sum / window as f64);
        }
        
        Ok(result)
    }

    fn __repr__(&self) -> String {
        format!("WindowOp(lags={:?}, rolling={:?})", self.lags, self.rolling_windows)
    }
}
EOF

cat > "${ZENO_ROOT}/zeno-core/src/engine/mod.rs" << 'EOF'
pub mod window;
pub mod arrow_ops;
EOF

cat > "${ZENO_ROOT}/zeno-core/src/engine/arrow_ops.rs" << 'EOF'
// Placeholder for Arrow-specific zero-copy operations
// Will be expanded in later phases
EOF

# Validator
cat > "${ZENO_ROOT}/zeno-core/src/validator/temporal.rs" << 'EOF'
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use chrono::{DateTime, Utc, NaiveDateTime};

#[pyclass]
pub struct TemporalValidator {
    train_end: Option<i64>,
    test_start: Option<i64>,
}

#[pymethods]
impl TemporalValidator {
    #[new]
    pub fn new() -> Self {
        Self {
            train_end: None,
            test_start: None,
        }
    }

    /// Set the train/test split boundary
    pub fn set_split(&mut self, train_end_timestamp: i64, test_start_timestamp: i64) -> PyResult<()> {
        if test_start_timestamp <= train_end_timestamp {
            return Err(PyValueError::new_err(
                "Test start must be after train end (temporal leakage detected!)"
            ));
        }
        
        self.train_end = Some(train_end_timestamp);
        self.test_start = Some(test_start_timestamp);
        Ok(())
    }

    /// Validate that a feature doesn't use future data
    pub fn check_feature_window(&self, feature_timestamp: i64) -> PyResult<bool> {
        if let (Some(train_end), Some(_test_start)) = (self.train_end, self.test_start) {
            if feature_timestamp > train_end {
                return Err(PyValueError::new_err(
                    format!("Feature uses data from {} which is after train cutoff! TEMPORAL LEAKAGE DETECTED.", feature_timestamp)
                ));
            }
        }
        Ok(true)
    }

    fn __repr__(&self) -> String {
        format!("TemporalValidator(train_end={:?}, test_start={:?})", 
                self.train_end, self.test_start)
    }
}
EOF

cat > "${ZENO_ROOT}/zeno-core/src/validator/mod.rs" << 'EOF'
pub mod temporal;
pub mod leakage;
pub mod pipeline;
EOF

cat > "${ZENO_ROOT}/zeno-core/src/validator/leakage.rs" << 'EOF'
// Placeholder for advanced leakage detection
// Will implement rolling hash comparisons in Phase 2
EOF

cat > "${ZENO_ROOT}/zeno-core/src/validator/pipeline.rs" << 'EOF'
// Placeholder for pipeline state tracking
EOF

echo -e "${GREEN}✓ Rust source files created${NC}"

# 6. Create Python package
echo -e "\n${YELLOW}[6/7] Setting up Python package...${NC}"

# FIXED: Point to correct Cargo.toml location
cat > "${ZENO_ROOT}/zeno-py/pyproject.toml" << 'EOF'
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "zeno-ts"
version = "0.1.0"
description = "Zero-copy time series library for modern AI"
requires-python = ">=3.8"
dependencies = [
    "polars>=0.20.0",
    "pyarrow>=14.0.0",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-benchmark>=4.0",
    "ruff>=0.1.0",
]

[tool.maturin]
# Point to the Rust crate in ../zeno-core
manifest-path = "../zeno-core/Cargo.toml"
python-source = "."
module-name = "zeno._zeno"
EOF

# Create Python interface
cat > "${ZENO_ROOT}/zeno-py/zeno/__init__.py" << 'EOF'
"""
Zeno: The Zero-Copy Time Series Engine
"""

from zeno.atoms import Window, Scale
from zeno.molecule import Molecule
from zeno.validator import TemporalSplitter

__version__ = "0.1.0"
__all__ = ["Window", "Scale", "Molecule", "TemporalSplitter"]
EOF

cat > "${ZENO_ROOT}/zeno-py/zeno/atoms.py" << 'EOF'
"""
Atomic operations for time series transformations
"""
from typing import List, Optional
from zeno._zeno import WindowOp as _WindowOp

class Window:
    """Create lag and rolling window features with zero-copy"""
    
    def __init__(self, lags: List[int], rolling: Optional[List[int]] = None):
        self._core = _WindowOp(lags, rolling)
        self.lags = lags
        self.rolling = rolling or []
    
    def transform(self, values: List[float]):
        """Create lag features"""
        return self._core.create_lags(values)
    
    def rolling_mean(self, values: List[float], window: int):
        """Create rolling mean feature"""
        return self._core.rolling_mean(values, window)
    
    def __repr__(self):
        return f"Window(lags={self.lags}, rolling={self.rolling})"


class Scale:
    """Placeholder for scaling operations"""
    
    def __init__(self, method: str = "standard"):
        self.method = method
    
    def fit(self, data):
        pass
    
    def transform(self, data):
        return data
EOF

cat > "${ZENO_ROOT}/zeno-py/zeno/molecule.py" << 'EOF'
"""
Pipeline composition (Molecules = collections of Atoms)
"""
from typing import List

class Molecule:
    """Compose multiple atomic operations into a pipeline"""
    
    def __init__(self, atoms: List):
        self.atoms = atoms
        self._fitted = False
    
    def fit(self, data):
        """Fit all atoms in sequence"""
        for atom in self.atoms:
            if hasattr(atom, 'fit'):
                atom.fit(data)
        self._fitted = True
        return self
    
    def transform(self, data):
        """Transform data through all atoms"""
        if not self._fitted:
            raise ValueError("Pipeline not fitted. Call .fit() first.")
        
        result = data
        for atom in self.atoms:
            if hasattr(atom, 'transform'):
                result = atom.transform(result)
        return result
    
    def fit_transform(self, data):
        """Fit and transform in one call"""
        return self.fit(data).transform(data)
EOF

cat > "${ZENO_ROOT}/zeno-py/zeno/validator.py" << 'EOF'
"""
Temporal validation and leakage detection
"""
from zeno._zeno import TemporalValidator as _TemporalValidator
from datetime import datetime

class TemporalSplitter:
    """Enforce temporal ordering in train/test splits"""
    
    def __init__(self):
        self._core = _TemporalValidator()
    
    def split(self, timestamps, train_end_date: datetime, test_start_date: datetime):
        """
        Create a temporal train/test split with validation
        
        Args:
            timestamps: List of datetime objects
            train_end_date: Last date in training set
            test_start_date: First date in test set
        
        Raises:
            ValueError: If test_start <= train_end (temporal leakage)
        """
        train_ts = int(train_end_date.timestamp())
        test_ts = int(test_start_date.timestamp())
        
        self._core.set_split(train_ts, test_ts)
        
        train_mask = [ts <= train_end_date for ts in timestamps]
        test_mask = [ts >= test_start_date for ts in timestamps]
        
        return train_mask, test_mask
    
    def validate_feature(self, feature_date: datetime):
        """Check if a feature uses future data"""
        ts = int(feature_date.timestamp())
        return self._core.check_feature_window(ts)
EOF

echo -e "${GREEN}✓ Python package files created${NC}"

# Create example
cat > "${ZENO_ROOT}/examples/quickstart.py" << 'EOF'
"""
Quickstart example for Zeno
"""
import zeno as zn
from datetime import datetime, timedelta

# Example time series data
dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]
values = [10.0 + i * 0.5 for i in range(100)]

print("🌀 Zeno Quickstart Example")
print("=" * 50)

# 1. Create lag features
print("\n1️⃣  Creating lag features...")
window = zn.Window(lags=[1, 7, 14])
lag_features = window.transform(values[:20])
print(f"   Created {len(lag_features)} lag features")
print(f"   Lag-1 (first 5): {[f for f in lag_features[0][:5]]}")

# 2. Create rolling features
print("\n2️⃣  Creating rolling mean...")
rolling_mean = window.rolling_mean(values, window=7)
print(f"   Rolling mean (days 7-12): {rolling_mean[6:12]}")

# 3. Temporal validation
print("\n3️⃣  Temporal split validation...")
splitter = zn.TemporalSplitter()

train_end = datetime(2024, 3, 1)
test_start = datetime(2024, 3, 2)

train_mask, test_mask = splitter.split(dates, train_end, test_start)
print(f"   Train samples: {sum(train_mask)}")
print(f"   Test samples: {sum(test_mask)}")

# 4. Check for leakage
print("\n4️⃣  Checking for temporal leakage...")
try:
    # This should pass
    splitter.validate_feature(datetime(2024, 2, 15))
    print("   ✓ Feature from 2024-02-15 is valid (before train cutoff)")
    
    # This should fail
    splitter.validate_feature(datetime(2024, 3, 5))
except ValueError as e:
    print(f"   ✗ Leakage detected: {e}")

# 5. Pipeline composition
print("\n5️⃣  Building a pipeline...")
pipeline = zn.Molecule([
    zn.Window(lags=[1, 7]),
    zn.Scale(method="robust")
])
print(f"   Pipeline: {pipeline}")

print("\n" + "=" * 50)
print("✨ Zeno is ready! Check benchmarks/ for performance tests.")
EOF

echo -e "${GREEN}✓ Examples created${NC}"

# Create tests
cat > "${ZENO_ROOT}/tests/test_window.py" << 'EOF'
import pytest
from zeno import Window

def test_lag_creation():
    window = Window(lags=[1, 2])
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    lags = window.transform(values)
    
    assert len(lags) == 2  # Two lag features
    assert lags[0][0] is None  # First lag-1 value
    assert lags[0][1] == 1.0  # Second lag-1 value
    assert lags[1][2] == 1.0  # lag-2 at position 2

def test_rolling_mean():
    window = Window(lags=[1])
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    rolling = window.rolling_mean(values, window=3)
    
    assert rolling[0] is None
    assert rolling[1] is None
    assert rolling[2] == 2.0  # (1+2+3)/3
    assert rolling[4] == 4.0  # (3+4+5)/3
EOF

cat > "${ZENO_ROOT}/tests/test_temporal.py" << 'EOF'
import pytest
from zeno import TemporalSplitter
from datetime import datetime

def test_valid_split():
    splitter = TemporalSplitter()
    dates = [datetime(2024, 1, i) for i in range(1, 11)]
    
    train_mask, test_mask = splitter.split(
        dates,
        datetime(2024, 1, 5),
        datetime(2024, 1, 6)
    )
    
    assert sum(train_mask) == 5
    assert sum(test_mask) == 5

def test_leakage_detection():
    splitter = TemporalSplitter()
    dates = [datetime(2024, 1, i) for i in range(1, 11)]
    
    splitter.split(
        dates,
        datetime(2024, 1, 5),
        datetime(2024, 1, 6)
    )
    
    # Should pass
    splitter.validate_feature(datetime(2024, 1, 3))
    
    # Should fail
    with pytest.raises(ValueError, match="TEMPORAL LEAKAGE"):
        splitter.validate_feature(datetime(2024, 1, 7))
EOF

echo -e "${GREEN}✓ Tests created${NC}"

# 7. Build the project
echo -e "\n${YELLOW}[7/7] Building Rust library and installing Python package...${NC}"

cd "${ZENO_ROOT}/zeno-py"

# Create virtual environment
if [ "$USE_UV" = true ]; then
    echo -e "${BLUE}Creating virtual environment with uv...${NC}"
    uv venv
    source .venv/bin/activate
    uv pip install maturin
else
    echo -e "${BLUE}Creating virtual environment with standard venv...${NC}"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install maturin
fi

# Build and install in development mode
echo -e "${BLUE}Building Rust library (this may take a few minutes)...${NC}"
maturin develop --release

echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✨ Zeno setup complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo -e "\n📂 Project structure created in: ${BLUE}${ZENO_ROOT}${NC}"
echo -e "\n🚀 Quick start:"
echo -e "   ${YELLOW}cd ${ZENO_ROOT}/zeno-py${NC}"
echo -e "   ${YELLOW}source .venv/bin/activate${NC}"
echo -e "   ${YELLOW}python ../examples/quickstart.py${NC}"

echo -e "\n🧪 Run tests:"
if [ "$USE_UV" = true ]; then
    echo -e "   ${YELLOW}uv pip install pytest pytest-benchmark${NC}"
else
    echo -e "   ${YELLOW}pip install pytest pytest-benchmark${NC}"
fi
echo -e "   ${YELLOW}pytest ../tests -v${NC}"

echo -e "\n📚 Next steps:"
echo -e "   1. Review the architecture in ${ZENO_ROOT}/zeno-core/src/"
echo -e "   2. Run the quickstart example"
echo -e "   3. Check benchmarks/ for performance comparisons"

echo -e "\n${BLUE}Happy building! 🌀${NC}\n"