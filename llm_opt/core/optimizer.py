import os
import time
import inspect
import datetime
import numpy as np
from typing import Callable, Optional
import hashlib

from llm_opt.utils.logging_config import logger
from llm_opt.utils.helpers import assert_outputs_equal, ensure_directory_exists
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
        err_tol: float = 1e-8,
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
        self.err_tol = err_tol
        self.best_artifact = None
        self.seen_implementations_hashes = set()

        # Create a unique run folder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{func.__name__}_{timestamp}"
        self.run_dir = os.path.join(output_dir, self.run_id)

        # Create directory structure
        ensure_directory_exists(self.run_dir)
        ensure_directory_exists(os.path.join(self.run_dir, "iterations"))

        # Set up a file logger for this run
        self.log_file = os.path.join(self.run_dir, "optimization.log")
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

        # Just to be sure, warm up the numpy implementation
        for _ in range(self.benchmark_runs):
            self.func(*self.test_input_generator())

        self.artifacts.log_func_info(func_info)
        self.artifacts.log_initial_prompt(initial_prompt)

        current_prompt = initial_prompt

        for iteration in range(self.max_iterations):
            logger.info(f"Starting iteration {iteration + 1}/{self.max_iterations}")
            logger.info(f"Querying LLM API")
            response = self.api_client.call_api(current_prompt)
            if not response:
                logger.error("Failed to get response from API")
                break

            # Extract the C implementation
            c_implementation = extract_code_from_response(response)
            artifact = IterationArtifact(
                iteration + 1,
                c_implementation,
                current_prompt,
                response,
            )
            self.artifacts.add_artifact(artifact)

            logger.info(f"Testing implementation.")
            self.test_implementation(c_implementation, artifact)

            if artifact.success and artifact.performance_report.speedup_medians() > (
                self.best_artifact.performance_report.speedup_medians()
                if self.best_artifact
                else 0
            ):
                self.best_artifact = artifact
                logger.info(
                    f"Iteration {iteration + 1} is the new best implementation!"
                )
            else:
                logger.info(f"No improvement in iteration {iteration + 1}.")

            self.artifacts._save_iteration_artifact(artifact)  # save after run
            if self.best_artifact:
                self.artifacts.checkpoint(self.best_artifact)

            if iteration < self.max_iterations - 1:
                current_prompt = gen_update_prompt(
                    self.numpy_source,
                    self.signature.generate_c_function_signature(self.func.__name__),
                    self.artifacts.to_str(),
                )

        logger.info("=" * 80)
        logger.info(f"Best implementation\n {self.best_artifact.short_desc()}")
        logger.info("=" * 80)
        return self.best_artifact

    def test_implementation(self, c_implementation: str, artifact: IterationArtifact):
        implementation_hash = hashlib.sha256(c_implementation.encode()).hexdigest()
        if implementation_hash in self.seen_implementations_hashes:
            err_str = f"Implementation proposed already seen, skipping"
            logger.error(err_str, exc_info=True)
            artifact.success = False
            artifact.error = err_str
            artifact.c_code = "[skipped]"
            return
        self.seen_implementations_hashes.add(implementation_hash)

        c_function = CFunction(self.func.__name__, self.signature, c_implementation)

        try:
            c_function.compile_and_load()
        except Exception as e:
            err_str = f"Error compiling and loading C function: {e}"
            logger.error(err_str, exc_info=True)
            artifact.success = False
            artifact.error = err_str
            return

        verification_passed, err_str = self.verify_against_numpy(c_function)
        if not verification_passed:
            artifact.success = False
            artifact.error = err_str
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

    def verify_against_numpy(self, c_function: CFunction) -> tuple[bool, Optional[str]]:
        """
        Verify the C implementation against the original NumPy function.
        Generate test data, run both implementations, and compare results.
        """
        try:
            for _ in range(self.benchmark_runs):
                test_inputs_tuple = self.test_input_generator()

                np_args = [np.copy(arg) for arg in test_inputs_tuple]
                np_output = np_args[-1]

                c_args_np = [np.copy(arg) for arg in test_inputs_tuple]
                c_args = self.signature.python_args_to_c_args(c_args_np)
                c_output = c_args_np[-1]

                self.func(*np_args)
                c_function(*c_args)

                assert_outputs_equal(np_output, c_output, self.err_tol)

            return True, None

        except Exception as e:
            err_str = f"Error during verification:\n {e}"
            logger.error(err_str, exc_info=True)
            return False, err_str

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
