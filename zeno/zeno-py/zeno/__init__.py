"""
Zeno: The Zero-Copy Time Series Engine
"""

# 1. Import raw Rust classes from the compiled extension
from ._zeno import (
    ArrowPipeline,
    PolarsWindowOp,
    PolarsValidator,
    LeakageDetector,
    RollingHashValidator,
    WindowOp,
)

# 2. Import Python wrappers and logic
from .advanced import (
    ArrowWindow,
    AdvancedLeakageDetector,
    PolarsWindow,
    PolarsTemporalValidator,
    ExpandingWindowValidator,
)

# 3. Import atoms (ensure these exist in zeno/atoms.py)
try:
    from .atoms import Window, Scale
    from .molecule import Molecule
    from .validator import TemporalSplitter
except ImportError:
    # Fallbacks if atoms.py is not yet finalized
    Window = None
    Scale = None

__version__ = "0.2.0"

__all__ = [
    # Rust Core
    "ArrowPipeline", "PolarsWindowOp", "PolarsValidator", 
    "LeakageDetector", "RollingHashValidator",
    # Python Wrappers
    "ArrowWindow", "PolarsWindow", "AdvancedLeakageDetector",
    "PolarsTemporalValidator", "ExpandingWindowValidator",
    # Phase 1 Atoms
    "Window", "Scale", "Molecule", "TemporalSplitter",
]