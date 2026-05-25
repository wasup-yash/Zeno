"""
Zeno: The Zero-Copy Time Series Engine
"""

# 1. Raw Rust classes
from ._zeno import (
    ArrowPipeline,
    PolarsWindowOp,
    PolarsValidator,
    LeakageDetector,
    RollingHashValidator,
    WindowOp,
    AuditReport,
    Forecast,
    BatchPredictor,
    ChronosWrapper,
    MoiraiWrapper,
    GPUAccelerator,
    TensorConverter,
    ManagedPipeline,
    ValidationScheduler,
    BacktestResult,
    ServerlessConfig,
)

# 2. Python wrappers (advanced features)
from .advanced import (
    ArrowWindow,
    AdvancedLeakageDetector,
    PolarsWindow,
    PolarsTemporalValidator,
    ExpandingWindowValidator,
    GPUManager,
    AuditManager,
    ManagedExecutor,
)

# 3. Phase 1 atoms & molecule
try:
    from .atoms import Window, Scale, EMA
    from .molecule import Molecule
except ImportError:
    # Fallbacks if atoms.py is not final
    Window = None
    Scale = None
    EMA = None
    Molecule = None

# 4. Temporal validator (mask‑based)
from .validator import TemporalSplitter

__version__ = "0.2.0"

__all__ = [
    # Rust Core
    "ArrowPipeline",
    "PolarsWindowOp",
    "WindowOp",
    "PolarsValidator",
    "AuditReport",
    "Forecast",
    "BatchPredictor",
    "ChronosWrapper",
    "MoiraiWrapper",
    "GPUAccelerator",
    "TensorConverter",
    "ManagedPipeline",
    "ValidationScheduler",
    "BacktestResult",
    "ServerlessConfig",
    "LeakageDetector",
    "RollingHashValidator",
    # Python Wrappers
    "ArrowWindow",
    "PolarsWindow",
    "AdvancedLeakageDetector",
    "PolarsTemporalValidator",
    "ExpandingWindowValidator",
    "GPUManager",
    "AuditManager",
    "ManagedExecutor",
    # Phase 1
    "Window",
    "Scale",
    "EMA",
    "Molecule",
    "TemporalSplitter",
]