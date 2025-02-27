"""
LLM-Opt: NumPy-to-C Optimizer

This package provides functionality to translate NumPy functions to optimized C implementations
using the DeepSeek API for iterative refinement, similar to Numba's @jit but with feedback.
"""

# Version information
__version__ = "0.1.0"

# Core components
from llm_opt.core.c_function import CFunction
from llm_opt.core.signature import Signature
from llm_opt.core.optimizer import Optimizer

# API clients
from llm_opt.api.clients import (
    BaseAPIClient,
    DeepSeekAPIClient,
    MockAPIClient,
    GroqAPIClient,
)

__all__ = [
    # Core components
    "CFunction",
    "Signature",
    "Optimizer",
    # API clients
    "BaseAPIClient",
    "DeepSeekAPIClient",
    "MockAPIClient",
    "GroqAPIClient",
]
