"""
Main API for the NumPy-to-C optimizer.
"""

import os
from typing import Callable, Dict, Optional, Any

import numpy as np

from llm_opt.utils.logging_config import logger
from llm_opt.utils.helpers import ensure_directory_exists
from llm_opt.core.optimizer import LLMOptimizer
from llm_opt.api.clients import BaseAPIClient
from llm_opt.core.signature import Signature


def optimize(
    func: Callable,
    signature: Signature,
    test_input_generator: Callable,
    max_iterations: int = 5,
    benchmark_runs: int = 100,
    output_dir: str = "results",
    api_client: Optional[BaseAPIClient] = None,
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
        api_client: Optional API client to use (defaults to DeepSeekAPIClient)

    Returns:
        A wrapped function that calls the optimized C implementation
    """
    ensure_directory_exists(output_dir)
    optimizer = LLMOptimizer(
        func,
        signature,
        test_input_generator,
        max_iterations,
        benchmark_runs,
        output_dir,
        api_client,
    )

    optimized_c_implementation = optimizer.optimize()

    if optimized_c_implementation and optimizer.c_function:
        # Use the c_function from the optimizer
        return optimizer.c_function

    # If optimization failed or c_function is not available
    logger.error(f"Failed to optimize {func.__name__}")
    raise Exception(f"Failed to optimize {func.__name__}")
