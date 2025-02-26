#!/usr/bin/env python3
"""
Optimizer for the NumPy-to-C optimizer.
"""

import os
import time
import inspect
import datetime
import numpy as np
from typing import Callable, Dict, Optional, Any

from llm_opt.utils.logging_config import logger
from llm_opt.utils.helpers import compare_outputs, ensure_directory_exists
from llm_opt.core.signature import Signature
from llm_opt.core.c_function import CFunction
from llm_opt.utils.prompts import gen_initial_prompt, gen_update_prompt
from llm_opt.api.clients import DeepSeekAPIClient, extract_code_from_response
from llm_opt.utils.llm_artifact_logger import LLMOptimizerArtifactLogger


class Optimizer:
    """
    Base class for optimizers that translate NumPy functions to optimized code.
    """

    def __init__(
        self,
        func: Callable,
        signature: Signature,
        test_input_generator: Callable,
        max_iterations: int = 5,
        benchmark_runs: int = 100,
        output_dir: str = "results",
    ):
        self.func = func
        self.signature = signature
        self.func_name = func.__name__
        self.signature_str = signature.generate_c_function_signature(self.func_name)
        self.max_iterations = max_iterations
        self.benchmark_runs = benchmark_runs
        self.best_implementation = None
        self.best_performance = float("inf")
        self.numpy_source = inspect.getsource(func)
        self.test_input_generator = test_input_generator
        self.c_function = None

        # Create a unique run folder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{self.func_name}_{timestamp}"
        self.run_dir = os.path.join(output_dir, self.run_id)

        # Create directory structure
        ensure_directory_exists(self.run_dir)
        ensure_directory_exists(os.path.join(self.run_dir, "iterations"))
        ensure_directory_exists(os.path.join(self.run_dir, "logs"))

        # Set up a file logger for this run
        self.log_file = os.path.join(self.run_dir, "logs", "optimization.log")
        self._setup_file_logger()

        logger.info(f"Created run directory: {self.run_dir}")

    def _setup_file_logger(self):
        """Set up a file handler for logging to the run directory."""
        import logging

        # Create a file handler for this run
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)

        # Create a formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(file_handler)

        logger.info(f"Logging to {self.log_file}")

    def test_function(self, c_function: CFunction) -> Dict[str, Any]:
        """
        Test the C implementation against the original NumPy function.
        """
        try:
            c_function.compile_and_load()
        except Exception as e:
            logger.error(f"Error compiling and loading C function: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

        verification_passed = self.verify_against_numpy(c_function)
        if not verification_passed:
            return {
                "success": False,
                "error": "Verification failed",
            }

        return self.eval_performance(c_function)

    def eval_performance(self, c_function: CFunction) -> Dict[str, Any]:
        try:
            test_inputs_tuple = self.test_input_generator()

            # Time the NumPy implementation
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                self.func(*test_inputs_tuple)
            numpy_time = (time.time() - start_time) / self.benchmark_runs

            # Time the C implementation
            c_args = self.signature.python_args_to_c_args(test_inputs_tuple)

            start_time = time.time()
            for _ in range(self.benchmark_runs):
                c_function(*c_args)
            c_time = (time.time() - start_time) / self.benchmark_runs

            speedup = numpy_time / c_time if c_time > 0 else 0

            return {
                "success": True,
                "numpy_time": numpy_time,
                "speedup": speedup,
                "performance": c_time,
            }

        except Exception as e:
            logger.error(f"Error during testing: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "performance": float("inf"),
            }

    def verify_against_numpy(self, c_function: CFunction) -> bool:
        """
        Verify the C implementation against the original NumPy function.
        Generate test data, run both implementations, and compare results.
        """
        try:
            test_inputs_tuple = self.test_input_generator()

            np_args = [np.copy(arg) for arg in test_inputs_tuple]
            np_output = np_args[-1]

            c_args_np = [np.copy(arg) for arg in test_inputs_tuple]
            c_args = self.signature.python_args_to_c_args(c_args_np)
            c_output = c_args_np[-1]

            self.func(*np_args)
            c_function(*c_args)

            match, _ = compare_outputs(np_output, c_output)
            return match

        except Exception as e:
            logger.error(f"Error during verification: {e}", exc_info=True)
            return False

    def optimize(self) -> Optional[str]:
        """
        Run the optimization loop to translate the NumPy function to optimized code.
        """
        raise NotImplementedError("Subclasses must implement optimize()")


class LLMOptimizer(Optimizer):
    """
    Optimizes NumPy functions by translating them to C using the DeepSeek API.

    For each run, creates a folder with:
    - logs
    - C code for each iteration
    - JSON results for each iteration
    - The prompts used
    """

    def __init__(
        self,
        func: Callable,
        signature: Signature,
        test_input_generator: Callable,
        max_iterations: int = 5,
        benchmark_runs: int = 100,
        output_dir: str = "results",
        api_client=None,
    ):
        super().__init__(
            func,
            signature,
            test_input_generator,
            max_iterations,
            benchmark_runs,
            output_dir,
        )
        self.artifact_logger = LLMOptimizerArtifactLogger(self.run_dir, self.func_name)
        self.api_client = api_client or DeepSeekAPIClient()

    # def _get_next_fn(self, )

    def optimize(self) -> Optional[str]:
        """
        Run the optimization loop to translate the NumPy function to optimized C.
        """

        initial_prompt = gen_initial_prompt(self.numpy_source, self.signature_str)

        logger.info(f"Starting optimization loop for function {self.func_name}")
        logger.debug(f"Function source:\n{self.numpy_source}")
        logger.debug(f"Function signature:\n{self.signature}")

        # Save the initial function information
        func_info = {
            "function_name": self.func_name,
            "numpy_source": self.numpy_source,
            "function_signature": self.signature_str,
        }

        self.artifact_logger.log_func_info(func_info)
        self.artifact_logger.log_initial_prompt(initial_prompt)

        current_prompt = initial_prompt

        for iteration in range(self.max_iterations):
            logger.info(f"\nStarting iteration {iteration + 1}/{self.max_iterations}")

            response = self.api_client.call_api(current_prompt)
            if not response:
                logger.error("Failed to get response from API")
                break

            # Extract the C implementation
            c_implementation = extract_code_from_response(response, self.func_name)
            c_function = CFunction(self.func_name, self.signature, c_implementation)

            # Test it
            test_results = self.test_function(c_function)

            # Save iteration artifacts
            self.artifact_logger.save_iteration_artifacts(
                iteration + 1,
                c_implementation,
                current_prompt,
                response,
                test_results,
            )

            if (
                test_results["success"]
                and test_results["performance"] < self.best_performance
            ):
                self.best_performance = test_results["performance"]
                self.best_implementation = c_implementation
                self.c_function = c_function
                logger.info(
                    f"New best implementation with performance: {self.best_performance:.6f} seconds"
                )
            else:
                logger.info(
                    f"No improvement in iteration {iteration + 1}. Best performance: {self.best_performance:.6f} seconds"
                )

            # Generate the next prompt with feedback
            if iteration < self.max_iterations - 1:
                feedback = self._generate_feedback(test_results)
                current_prompt = gen_update_prompt(
                    self.numpy_source,
                    self.signature.generate_c_function_signature(self.func_name),
                    c_implementation,
                    feedback,
                )

        # Return the best implementation
        return self.best_implementation

    def _generate_feedback(self, test_results: Dict[str, Any]) -> str:
        """
        Generate feedback for the next iteration based on test results.
        """
        if not test_results["success"]:
            return "The implementation does not produce the same results as the NumPy function. Please fix the numerical accuracy."

        if not test_results["success"]:
            return f"The implementation failed with error: {test_results.get('error', 'Unknown error')}"

        if test_results["speedup"] < 1:
            return "The implementation is slower than the NumPy function. Please optimize for better performance."

        return f"The implementation is {test_results['speedup']:.2f}x faster than the NumPy function. Please optimize further for even better performance."


#     # Always provide optimization suggestions
# -        feedback.append("\nConsider exploring these optimization techniques:")
# -
# -        # Vectorization suggestions
# -        feedback.append("\nVectorization:")
# -        feedback.append(
# -            "- Use SIMD instructions (SSE, AVX, NEON) for parallel operations"
# -        )
# -        feedback.append("- Align data to vector boundaries for efficient SIMD access")
# -        feedback.append("- Vectorize inner loops where possible")
# -
# -        # Memory access suggestions
# -        feedback.append("\nMemory access optimizations:")
# -        feedback.append(
# -            "- Implement cache blocking/tiling to improve cache utilization"
# -        )
# -        feedback.append("- Optimize data layout for better spatial locality")
# -        feedback.append("- Use prefetching to reduce cache misses")
# -
# -        # Loop optimization suggestions
# -        feedback.append("\nLoop optimizations:")
# -        feedback.append("- Unroll loops to reduce branch prediction misses")
# -        feedback.append("- Fuse loops to reduce loop overhead")
# -        feedback.append("- Interchange loops to improve memory access patterns")
# -
# -        # Algorithm suggestions
# -        feedback.append("\nAlgorithm improvements:")
# -        feedback.append("- Look for mathematical simplifications")
# -        feedback.append("- Consider specialized algorithms for common patterns")
# -        feedback.append("- Reduce redundant computations")
