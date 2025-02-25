"""
LLM-Opt: NumPy-to-C Optimizer

This package provides functionality to translate NumPy functions to optimized C implementations
using the DeepSeek API for iterative refinement, similar to Numba's @jit but with feedback.
"""

from llm_opt.main import optimize
from llm_opt.core.c_function import CFunction, Signature

__all__ = ["optimize", "CFunction", "Signature"]
