import numpy as np
import time
import logging
import argparse
from typing import Callable, Dict, Any, Tuple

from llm_opt import optimize, DeepSeekAPIClient, BaseAPIClient
from llm_opt.utils.logging_config import setup_logging
from llm_opt.utils.helpers import ensure_directory_exists
from llm_opt.core.signature import Signature
from llm_opt.core.type_interface import DOUBLE

# Get the logger for this module
logger = logging.getLogger(__name__)


def vec_add(a, b, out):
    out[0] = np.linalg.norm(a + b)


def vec_add_inputs():
    """Generate test inputs for vec_add."""
    n = 10_000_000
    a = np.random.rand(n).astype(np.float64)
    b = np.random.rand(n).astype(np.float64)
    out = np.zeros(1).astype(np.float64)
    return (a, b, out)


def run_example(
    name: str,
    func: Callable,
    args: Tuple,
    signature: Signature,
    test_input_generator: Callable,
    api_client: BaseAPIClient,
    max_iterations: int = 3,
    benchmark_runs: int = 100,
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
            signature,
            test_input_generator,
            max_iterations,
            benchmark_runs,
            output_dir,
            api_client,
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run NumPy-to-C optimizer examples")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = setup_logging(log_level)
    print(f"Logging to {log_file}")

    # Create results directory if it doesn't exist
    results_dir = "results"
    ensure_directory_exists(results_dir)

    print("NumPy-to-C Optimizer Examples")
    print("============================")
    logging.info("Starting NumPy-to-C Optimizer Examples")

    api_client = DeepSeekAPIClient()

    # Example 1: Vector addition
    a, b, out = vec_add_inputs()
    run_example(
        "vec_add",
        vec_add,
        (a, b, out),
        Signature(
            [
                ("a", DOUBLE.array_of()),
                ("b", DOUBLE.array_of()),
                ("out", DOUBLE.array_of()),
            ]
        ),
        test_input_generator=vec_add_inputs,
        api_client=api_client,
        max_iterations=5,
        benchmark_runs=100,
        output_dir=results_dir,
    )


if __name__ == "__main__":
    main()
