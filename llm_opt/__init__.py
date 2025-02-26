"""
LLM-Opt: NumPy-to-C Optimizer

This package provides functionality to translate NumPy functions to optimized C implementations
using the DeepSeek API for iterative refinement, similar to Numba's @jit but with feedback.
"""

# Version information
__version__ = "0.1.0"

# Main API
from llm_opt.api import optimize

# Core components
from llm_opt.core.c_function import CFunction
from llm_opt.core.signature import Signature
from llm_opt.core.optimizer import Optimizer, LLMOptimizer

# API clients
from llm_opt.api.clients import BaseAPIClient, DeepSeekAPIClient, MockAPIClient

__all__ = [
    # Main API
    "optimize",
    # Core components
    "CFunction",
    "Signature",
    "Optimizer",
    "LLMOptimizer",
    # API clients
    "BaseAPIClient",
    "DeepSeekAPIClient",
    "MockAPIClient",
]
