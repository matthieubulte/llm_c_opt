import os
import time
import inspect
import datetime
import numpy as np
import json
from typing import Callable, Dict, Optional, Any, List

from llm_opt.utils.logging_config import logger
from llm_opt.utils.helpers import compare_outputs, ensure_directory_exists
from llm_opt.core.signature import Signature
from llm_opt.core.c_function import CFunction
from llm_opt.utils.prompts import gen_initial_prompt, gen_update_prompt
from llm_opt.api.clients import BaseAPIClient, extract_code_from_response
from llm_opt.utils.artifact_collection import ArtifactCollection
from llm_opt.core.iteration_artifact import IterationArtifact
from llm_opt.core.performance_report import PerformanceReport


class Optimizer:
    """
    Base class for optimizers that translate NumPy functions to optimized code.
    """

    def __init__(
        self,
        func: Callable,
        signature: Signature,
        test_input_generator: Callable,
        api_client: BaseAPIClient,
        max_iterations: int = 5,
        benchmark_runs: int = 100,
        output_dir: str = "results",
    ):
        self.func = func
        self.signature = signature
        self.max_iterations = max_iterations
        self.benchmark_runs = benchmark_runs
        self.numpy_source = inspect.getsource(func)
        self.test_input_generator = test_input_generator
        self.best_artifact = None

        # Create a unique run folder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{func.__name__}_{timestamp}"
        self.run_dir = os.path.join(output_dir, self.run_id)

        # Create directory structure
        ensure_directory_exists(self.run_dir)
        ensure_directory_exists(os.path.join(self.run_dir, "iterations"))
        ensure_directory_exists(os.path.join(self.run_dir, "logs"))

        # Set up a file logger for this run
        self.log_file = os.path.join(self.run_dir, "logs", "optimization.log")
        self._setup_file_logger()

        logger.info(f"Created run directory: {self.run_dir}")

        self.artifacts = ArtifactCollection(self.run_dir, func.__name__)
        self.api_client = api_client

    def optimize(self) -> Optional[str]:
        signature_str = self.signature.generate_c_function_signature(self.func.__name__)
        initial_prompt = gen_initial_prompt(self.numpy_source, signature_str)

        logger.info(f"Starting optimization loop for function {self.func.__name__}")
        logger.debug(f"Function source:\n{self.numpy_source}")
        logger.debug(f"Function signature:\n{self.signature}")

        # Save the initial function information
        func_info = {
            "function_name": self.func.__name__,
            "numpy_source": self.numpy_source,
            "function_signature": signature_str,
        }

        self.artifacts.log_func_info(func_info)
        self.artifacts.log_initial_prompt(initial_prompt)

        current_prompt = initial_prompt

        for iteration in range(self.max_iterations):
            logger.info(f"\nStarting iteration {iteration + 1}/{self.max_iterations}")

            response = self.api_client.call_api(current_prompt)
            if not response:
                logger.error("Failed to get response from API")
                break

            # Extract the C implementation
            c_implementation = extract_code_from_response(response)
            c_function = CFunction(self.func.__name__, self.signature, c_implementation)

            artifact = IterationArtifact(
                iteration + 1,
                c_implementation,
                current_prompt,
                response,
            )

            # Test it
            self.test_function(c_function, artifact)

            # Save iteration artifacts
            self.artifacts.add_artifact(artifact)

            if artifact.success and artifact.performance_report.avg_speedup() > (
                self.best_artifact.performance_report.avg_speedup()
                if self.best_artifact
                else 0
            ):
                self.best_artifact = artifact
                logger.info(
                    f"New best implementation: {self.best_artifact.short_desc()}"
                )
            else:
                logger.info(f"No improvement in iteration {iteration + 1}.")

            if iteration < self.max_iterations - 1:
                current_prompt = gen_update_prompt(
                    self.numpy_source,
                    self.signature.generate_c_function_signature(self.func.__name__),
                    self.artifacts.to_str(),
                )

        return self.best_artifact

    def test_function(self, c_function: CFunction, artifact: IterationArtifact):
        try:
            c_function.compile_and_load()
        except Exception as e:
            logger.error(f"Error compiling and loading C function: {e}", exc_info=True)
            artifact.success = False
            artifact.error = str(e)
            return

        verification_passed = self.verify_against_numpy(c_function)
        if not verification_passed:
            artifact.success = False
            artifact.error = "Verification failed"
            return

        artifact.performance_report = self.eval_performance(c_function)

    def eval_performance(self, c_function: CFunction) -> PerformanceReport:
        test_inputs_tuple = self.test_input_generator()
        perf_report = PerformanceReport()

        # Time the NumPy implementation
        for _ in range(self.benchmark_runs):
            start_time = time.time()
            self.func(*test_inputs_tuple)
            perf_report.add_numpy_runtime(time.time() - start_time)

        # Time the C implementation
        c_args = self.signature.python_args_to_c_args(test_inputs_tuple)

        for _ in range(self.benchmark_runs):
            start_time = time.time()
            c_function(*c_args)
            perf_report.add_c_runtime(time.time() - start_time)

        return perf_report

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
