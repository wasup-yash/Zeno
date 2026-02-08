"""
Zeno: The Zero-Copy Time Series Engine
"""

from zeno.atoms import Window, Scale
from zeno.molecule import Molecule
from zeno.validator import TemporalSplitter

# Phase 2 imports
from zeno.advanced import (
    ArrowWindow,
    PolarsWindow,
    AdvancedLeakageDetector,
    PolarsTemporalValidator,
    ExpandingWindowValidator,
)

__version__ = "0.2.0"  # Updated version

__all__ = [
    # Phase 1
    "Window", "Scale", "Molecule", "TemporalSplitter",
    # Phase 2
    "ArrowWindow", "PolarsWindow", "AdvancedLeakageDetector",
    "PolarsTemporalValidator", "ExpandingWindowValidator",
]