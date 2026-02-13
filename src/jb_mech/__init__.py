"""
JB_mech: Jailbreak Mechanism Analysis with Assistant Axis and GLP

This package provides tools for analyzing jailbreak mechanisms in LLMs
by combining the Assistant Axis representation with GLP-based interpretation.
"""

from .config import (
    MODEL_NAME,
    TOTAL_LAYERS,
    TARGET_LAYER,
    HIDDEN_SIZE,
    GLP_MODEL,
    PROJECT_ROOT,
    THIRD_PARTY_DIR,
    OUTPUT_DIR,
)

__version__ = "0.1.0"
__all__ = [
    "MODEL_NAME",
    "TOTAL_LAYERS",
    "TARGET_LAYER",
    "HIDDEN_SIZE",
    "GLP_MODEL",
    "PROJECT_ROOT",
    "THIRD_PARTY_DIR",
    "OUTPUT_DIR",
]
