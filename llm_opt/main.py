#!/usr/bin/env python3
"""
Main module for the NumPy-to-C optimizer.
"""

import os
import hashlib
import inspect
import time
import ctypes
from typing import Callable, Dict, Optional

import numpy as np

from llm_opt.utils.logging_config import logger
from llm_opt.utils.constants import CACHE_DIR, DTYPE_TO_CTYPES
from llm_opt.core.optimizer import DeepSeekOptimizer
from llm_opt.core.c_function import CFunction, Signature
from llm_opt.core.signature import FunctionSignatureGenerator


def optimize(
    func: Callable,
    input_types: Dict[str, np.dtype],
    test_input_generator: Callable,
    max_iterations: int = 5,
    output_dir: str = "results",
) -> Callable:
    """
    Optimize a NumPy function by translating it to C and iteratively improving it.

    Creates a structured output directory with:
    - logs
    - C code for each iteration
    - JSON results for each iteration
    - The prompts used

    Args:
        func: The NumPy function to optimize
        input_types: Dictionary mapping argument names to NumPy dtypes
        test_input_generator: Function that returns a tuple of test inputs
        max_iterations: Maximum number of optimization iterations
        output_dir: Directory to save optimization artifacts

    Returns:
        A wrapped function that calls the optimized C implementation
    """
    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate a unique hash for the function and input types
    func_source = inspect.getsource(func)
    func_hash = hashlib.md5(func_source.encode()).hexdigest()

    # Create and run the optimizer
    optimizer = DeepSeekOptimizer(
        func, input_types, test_input_generator, max_iterations, output_dir
    )

    optimized_c_implementation = optimizer.optimize()

    if optimized_c_implementation and optimizer.c_function:
        # Use the c_function from the optimizer
        return optimizer.c_function

    # If optimization failed or c_function is not available
    logger.error(f"Failed to optimize {func.__name__}")
    raise Exception(f"Failed to optimize {func.__name__}")
