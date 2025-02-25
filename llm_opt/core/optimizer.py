#!/usr/bin/env python3
"""
Optimizer for the NumPy-to-C optimizer.
"""

import os
import time
import inspect
import ctypes
import json
import datetime
import numpy as np
from typing import Callable, Dict, Optional, List, Any, Tuple, Type, cast

from llm_opt.utils.logging_config import logger
from llm_opt.utils.constants import CACHE_DIR, DTYPE_TO_CTYPES
from llm_opt.core.analyzer import NumPyFunctionAnalyzer
from llm_opt.core.signature import FunctionSignatureGenerator
from llm_opt.core.c_function import CFunction, Signature
from llm_opt.utils.prompts import gen_initial_prompt, gen_update_prompt
from llm_opt.api.deepseek import call_deepseek_api, extract_code_from_response


class DeepSeekOptimizer:
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
        input_types: Dict[str, np.dtype],
        test_input_generator: Callable,
        max_iterations: int = 5,
        output_dir: str = "results",
    ):
        self.func = func
        self.func_name = func.__name__
        self.input_types = input_types
        self.max_iterations = max_iterations
        self.best_implementation = None
        self.best_performance = float("inf")
        self.analyzer = NumPyFunctionAnalyzer(func)
        self.signature_generator = FunctionSignatureGenerator(func, input_types)
        self.function_signature = (
            self.signature_generator.generate_c_function_signature()
        )
        self.numpy_source = inspect.getsource(func)
        self.test_input_generator = test_input_generator
        self.c_function = None

        # Create a unique run folder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{self.func_name}_{timestamp}"
        self.run_dir = os.path.join(output_dir, self.run_id)

        # Create directory structure
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "iterations"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "logs"), exist_ok=True)

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

    def create_c_file_with_implementation(self, c_implementation: str) -> str:
        """
        Create a C file with just the function implementation.
        """
        c_code = f"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

// Implementation of {self.func_name} function
{c_implementation}
"""
        return c_code

    def save_iteration_artifacts(
        self,
        iteration: int,
        c_implementation: str,
        prompt: str,
        response: str,
        test_results: Dict[str, Any],
        verification_passed: bool,
    ):
        """Save all artifacts for a single iteration."""
        iteration_dir = os.path.join(
            self.run_dir, "iterations", f"iteration_{iteration}"
        )
        os.makedirs(iteration_dir, exist_ok=True)

        # Save C implementation
        c_file_path = os.path.join(iteration_dir, f"{self.func_name}.c")
        with open(c_file_path, "w") as f:
            f.write(self.create_c_file_with_implementation(c_implementation))

        # Save prompt
        prompt_file_path = os.path.join(iteration_dir, "prompt.txt")
        with open(prompt_file_path, "w") as f:
            f.write(prompt)

        # Save response
        response_file_path = os.path.join(iteration_dir, "response.txt")
        with open(response_file_path, "w") as f:
            f.write(response)

        # Save test results as JSON
        results = {
            "iteration": iteration,
            "verification_passed": verification_passed,
            "test_results": test_results | {"result": None},
            "timestamp": datetime.datetime.now().isoformat(),
        }

        results_file_path = os.path.join(iteration_dir, "results.json")
        with open(results_file_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved artifacts for iteration {iteration} to {iteration_dir}")

    def save_final_results(self):
        """Save the final results of the optimization process."""
        if self.best_implementation:
            # Save the best implementation
            best_file_path = os.path.join(self.run_dir, f"{self.func_name}_best.c")
            with open(best_file_path, "w") as f:
                f.write(
                    self.create_c_file_with_implementation(self.best_implementation)
                )

            # Create the best c_function if not already set
            if self.c_function is None:
                signature = self.signature_generator.generate_ctypes_signature()
                self.c_function = CFunction(
                    self.func_name, signature, self.best_implementation
                )
                if not (self.c_function.compile() and self.c_function.load()):
                    logger.error("Failed to compile and load the best implementation")
                    self.c_function = None

            # Save a summary of the optimization process
            summary = {
                "function_name": self.func_name,
                "run_id": self.run_id,
                "max_iterations": self.max_iterations,
                "best_performance": self.best_performance,
                "timestamp": datetime.datetime.now().isoformat(),
                "numpy_source": self.numpy_source,
                "function_signature": self.function_signature,
            }

            summary_file_path = os.path.join(self.run_dir, "summary.json")
            with open(summary_file_path, "w") as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Saved final results to {self.run_dir}")
        else:
            logger.warning(
                "No successful implementation found, no final results saved."
            )

    def compile_and_test(self, c_function: CFunction) -> Dict[str, Any]:
        """
        Compile the C implementation and test its performance.

        Args:
            c_implementation: The C implementation as a string

        Returns:
            A dictionary with the test results
        """
        try:
            # Generate test data
            test_inputs_tuple = self.test_input_generator()
            c_args, c_output = self.signature_generator.python_args_to_c_args(
                test_inputs_tuple
            )

            # Measure NumPy performance first
            numpy_start_time = time.time()
            for _ in range(100):
                self.func(*test_inputs_tuple)
            numpy_end_time = time.time()
            numpy_performance = numpy_end_time - numpy_start_time

            # Measure C implementation performance
            start_time = time.time()
            for _ in range(100):
                c_function(*c_args)
            end_time = time.time()

            c_performance = end_time - start_time

            # Calculate performance multiplier (NumPy time / C time)
            # Higher is better - means C is faster than NumPy
            perf_multiplier = (
                numpy_performance / c_performance if c_performance > 0 else 0
            )

            logger.info(f"NumPy performance: {numpy_performance:.6f} seconds")
            logger.info(f"C implementation performance: {c_performance:.6f} seconds")
            logger.info(
                f"Performance multiplier: {perf_multiplier:.2f}x (higher is better)"
            )

            return {
                "success": True,
                "performance": c_performance,
                "numpy_performance": numpy_performance,
                "perf_multiplier": perf_multiplier,
                "result": c_output,
            }
        except Exception as e:
            logger.error(f"Error testing C implementation: {e}", exc_info=True)
            raise Exception(f"Error testing C implementation: {e}")

    def verify_against_numpy(self, c_function: CFunction) -> bool:
        """
        Verify the C implementation against the original NumPy function.
        Generate test data, run both implementations, and compare results.
        """
        try:
            test_inputs_tuple = self.test_input_generator()
            c_args, c_output = self.signature_generator.python_args_to_c_args(
                test_inputs_tuple
            )

            numpy_output = self.func(*test_inputs_tuple)
            c_function(*c_args)

            return self._compare_outputs(numpy_output, c_output)

        except Exception as e:
            logger.error(f"Error during verification: {e}", exc_info=True)
            return False

    def optimize(self) -> Optional[str]:
        """
        Run the optimization loop to translate the NumPy function to optimized C.
        """

        initial_prompt = gen_initial_prompt(self.numpy_source, self.function_signature)

        logger.info(f"Starting optimization loop for function {self.func_name}")
        logger.debug(f"Function source:\n{self.numpy_source}")
        logger.debug(f"Function signature:\n{self.function_signature}")

        # Save the initial function information
        func_info = {
            "function_name": self.func_name,
            "numpy_source": self.numpy_source,
            "function_signature": self.function_signature,
        }

        func_info_path = os.path.join(self.run_dir, "function_info.json")
        with open(func_info_path, "w") as f:
            json.dump(func_info, f, indent=2)

        # Save the initial prompt
        initial_prompt_path = os.path.join(self.run_dir, "initial_prompt.txt")
        with open(initial_prompt_path, "w") as f:
            f.write(initial_prompt)

        current_prompt = initial_prompt

        for iteration in range(self.max_iterations):
            logger.info(f"\nStarting iteration {iteration + 1}/{self.max_iterations}")

            response = call_deepseek_api(current_prompt)
            if not response:
                logger.error("Failed to get response from DeepSeek API")
                break

            # Extract the C implementation
            c_implementation = extract_code_from_response(response, self.func_name)
            signature = self.signature_generator.generate_ctypes_signature()
            c_function = CFunction(self.func_name, signature, c_implementation)
            compiles = c_function.compile() and c_function.load()
            if compiles:
                verification_passed = self.verify_against_numpy(c_function)
                test_results = self.compile_and_test(c_function)

                # Store the c_function if it's valid
                if verification_passed and test_results["success"]:
                    self.c_function = c_function
            else:
                verification_passed = False
                test_results = {
                    "success": False,
                    "error": "Failed to compile or load C implementation",
                    "performance": float("inf"),
                }

            # Save iteration artifacts
            self.save_iteration_artifacts(
                iteration + 1,  # 1-indexed for user-friendliness
                c_implementation,
                current_prompt,
                response,
                test_results,
                verification_passed,
            )

            if (
                compiles
                and verification_passed
                and test_results["success"]
                and test_results["performance"] < self.best_performance
            ):
                self.best_performance = test_results["performance"]
                self.best_implementation = c_implementation
                # Update c_function to use the best implementation
                self.c_function = c_function
                logger.info(
                    f"New best implementation with performance: {self.best_performance:.6f} seconds"
                )
            else:
                logger.info(
                    f"Not updating best implementation. Current best: {self.best_performance:.6f} seconds"
                )

            # Generate feedback
            feedback = self.generate_feedback(test_results, verification_passed)

            # Update the prompt with feedback
            current_prompt = gen_update_prompt(
                self.numpy_source,
                self.function_signature,
                self.best_implementation,
                feedback,
            )

        # Save final results
        self.save_final_results()

        # Return the best implementation or None if none were successful
        if self.best_implementation:
            logger.info(
                f"Optimization completed successfully with best performance: {self.best_performance:.6f} seconds"
            )
            logger.debug(f"Best implementation:\n{self.best_implementation}")
        else:
            logger.warning("Optimization failed to produce a valid implementation")

        return self.best_implementation

    def generate_feedback(self, test_results: Dict, verification_passed: bool) -> str:
        """
        Generate feedback for the DeepSeek API based on test results and verification.
        """
        feedback = []

        if not verification_passed:
            message = (
                "The C implementation does not match the NumPy function's behavior. "
                "Please ensure numerical correctness."
            )
            feedback.append(message)

        if not test_results["success"]:
            if (
                "error" in test_results
                and "Compilation failed" in test_results["error"]
            ):
                message = (
                    "The code failed to compile. Please fix the compilation errors."
                )
                feedback.append(message)
            else:
                message = "The implementation failed to run correctly. Please fix any runtime issues."
                feedback.append(message)
        else:
            message = f"Your implementation took {test_results['performance']:.6f} seconds to run."
            feedback.append(message)

        # Always provide optimization suggestions
        feedback.append("\nConsider exploring these optimization techniques:")

        # Vectorization suggestions
        feedback.append("\nVectorization:")
        feedback.append(
            "- Use SIMD instructions (SSE, AVX, NEON) for parallel operations"
        )
        feedback.append("- Align data to vector boundaries for efficient SIMD access")
        feedback.append("- Vectorize inner loops where possible")

        # Memory access suggestions
        feedback.append("\nMemory access optimizations:")
        feedback.append(
            "- Implement cache blocking/tiling to improve cache utilization"
        )
        feedback.append("- Optimize data layout for better spatial locality")
        feedback.append("- Use prefetching to reduce cache misses")

        # Loop optimization suggestions
        feedback.append("\nLoop optimizations:")
        feedback.append("- Unroll loops to reduce branch prediction misses")
        feedback.append("- Fuse loops to reduce loop overhead")
        feedback.append("- Interchange loops to improve memory access patterns")

        # Algorithm suggestions
        feedback.append("\nAlgorithm improvements:")
        feedback.append("- Look for mathematical simplifications")
        feedback.append("- Consider specialized algorithms for common patterns")
        feedback.append("- Reduce redundant computations")

        full_feedback = "\n".join(feedback)
        return full_feedback

    def _compare_outputs(self, numpy_output: np.ndarray, c_output: np.ndarray) -> bool:
        """
        Compare NumPy and C function outputs, handling both scalar and array outputs.
        Uses the function signature to determine the expected output type.

        Args:
            numpy_output: Output from the NumPy implementation
            c_output: Output from the C implementation

        Returns:
            bool: True if outputs match within tolerance, False otherwise
        """
        match = np.allclose(numpy_output, c_output, rtol=1e-10, atol=1e-10)
        if not match:
            logger.warning(
                f"Array verification failed: NumPy shape={numpy_output.shape}, "
                f"C shape={c_output.shape if isinstance(c_output, np.ndarray) else None}\n"
                f"First few elements - NumPy: {numpy_output.flat[:3]}, C: {c_output.flat[:3] if isinstance(c_output, np.ndarray) else c_output}"
            )

        return bool(match)
