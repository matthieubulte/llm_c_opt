#!/usr/bin/env python3
"""
Example usage of the NumPy-to-C optimizer with structured output.
"""

import numpy as np
import time
import logging
import os
import sys
import json
import datetime
from typing import Callable, Dict, Any, Tuple

from llm_opt import optimize

# Get the logger for this module
logger = logging.getLogger(__name__)


# Configure logging to show detailed logs
def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # Create formatters
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
    )

    # Create console handler for debug+ messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Create file handler for detailed logging
    log_file = os.path.join("logs", f"example_{time.strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    print(f"Logging to {log_file}")
    logging.info("Logging initialized")

    return log_file


def vec_add(a, b):
    return a + b


def vec_add_inputs():
    n = 10_000_000
    a = np.random.rand(n).astype(np.float64)
    b = np.random.rand(n).astype(np.float64)
    return (a, b)


def sum_of_squares(x):
    return np.sum(x * x)


def sum_of_squares_inputs():
    n = 10_000_000
    x = np.random.rand(n).astype(np.float64)
    return (x,)


def run_example(
    name: str,
    func: Callable,
    args: Tuple,
    input_types: Dict[str, np.dtype],
    test_input_generator: Callable,
    max_iterations: int = 3,
    output_dir: str = "results",
):
    """Run an example function and its optimized version with structured output."""
    print(f"\nExample: {name}")
    logging.info(f"Running example: {name}")

    # Original function
    n_runs = 100
    start = time.time()
    for _ in range(n_runs):
        result_original = func(*args)
    time_original = time.time() - start
    logging.info(f"Original function average time: {time_original/n_runs:.6f} seconds")

    # Use our optimize function with structured output
    try:
        optimized_func = optimize(
            func,
            input_types=input_types,
            test_input_generator=test_input_generator,
            max_iterations=max_iterations,
            output_dir=output_dir,
        )

        # Benchmark the optimized function
        start = time.time()
        for _ in range(n_runs):
            result_optimized = optimized_func(*args)
        time_optimized = time.time() - start
        logging.info(
            f"Optimized function average time: {time_optimized/n_runs:.6f} seconds"
        )

        # Calculate speedup
        speedup = time_original / time_optimized if time_optimized > 0 else 0
        logging.info(f"Speedup: {speedup:.2f}x")

        # Verify results
        if isinstance(result_original, np.ndarray) and isinstance(
            result_optimized, np.ndarray
        ):
            max_diff = (
                np.max(np.abs(result_original - result_optimized))
                if result_original.size > 0
                else 0
            )
            logging.info(f"Maximum difference: {max_diff:.10e}")
            logging.info(f"Results match: {max_diff < 1e-10}")
        else:
            # Convert both to numpy arrays for consistent comparison
            result_original_arr = np.array([result_original])
            result_optimized_arr = np.array([result_optimized])

            # Compare using numpy's isclose
            match = np.allclose(
                result_original_arr, result_optimized_arr, rtol=1e-10, atol=1e-10
            )
            logging.info(f"Results match: {match}")
            logging.debug(
                f"Original result: {result_original}, Optimized result: {result_optimized}"
            )
    except Exception as e:
        logging.error(f"Optimization failed: {e}")
        return


def main():
    # Set up logging first
    setup_logging()

    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    print("NumPy-to-C Optimizer Examples")
    print("============================")
    logging.info("Starting NumPy-to-C Optimizer Examples")

    # Example 1: Vector addition
    a, b = vec_add_inputs()
    run_example(
        "vec_add",
        vec_add,
        (a, b),
        {
            "a": np.dtype(np.float64),
            "b": np.dtype(np.float64),
        },
        test_input_generator=vec_add_inputs,
        max_iterations=3,
        output_dir=results_dir,
    )

    # Example 2: Sum of squares
    (x,) = sum_of_squares_inputs()
    run_example(
        "sum_of_squares",
        sum_of_squares,
        (x,),
        {
            "x": np.dtype(np.float64),
        },
        test_input_generator=sum_of_squares_inputs,
        max_iterations=3,
        output_dir=results_dir,
    )


if __name__ == "__main__":
    main()
