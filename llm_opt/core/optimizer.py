import os
import time
import inspect
import datetime
import numpy as np
import logging
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
        self.test_input_generator = test_input_generator
        self.api_client = api_client
        self.err_tol = err_tol
        self.max_iterations = max_iterations
        self.benchmark_runs = benchmark_runs

        self.seen_implementations_hashes = set()

        self.output_dir = output_dir
        self._setup_run_dir()

        self.log_file = os.path.join(self.run_dir, "optimization.log")
        self._setup_file_logger()
        logger.info(f"Created run directory: {self.run_dir}")

        self.artifacts = ArtifactCollection(self.run_dir, func.__name__)

    def optimize(self) -> Optional[str]:
        signature_str = self.signature.generate_c_function_signature(self.func.__name__)
        numpy_source = inspect.getsource(self.func)
        initial_prompt = gen_initial_prompt(numpy_source, signature_str)

        logger.info(f"Starting optimization loop for function {self.func.__name__}")
        logger.info(f"Function source:\n{numpy_source}")
        logger.info(f"Function signature:\n{signature_str}")

        for _ in range(self.benchmark_runs):
            self.func(*self.test_input_generator())

        current_prompt = initial_prompt

        for iteration in range(self.max_iterations):
            logger.info(f"Starting iteration {iteration + 1}/{self.max_iterations}")
            logger.info(f"Querying LLM API")
            response = self.api_client.call_api(current_prompt)
            if not response:
                logger.error("Failed to get response from API")
                break

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
            self.artifacts.checkpoint()

            if iteration < self.max_iterations - 1:
                current_prompt = gen_update_prompt(
                    numpy_source,
                    self.signature.generate_c_function_signature(self.func.__name__),
                    self.artifacts.to_str(),
                )

        logger.info("=" * 80)
        logger.info(
            f"Best implementation\n {self.artifacts.get_best_artifact().short_desc()}"
        )
        logger.info("=" * 80)

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

        try:
            self.verify_against_numpy(c_function)
        except Exception as e:
            err_str = f"Error during verification:\n {e}"
            logger.error(err_str, exc_info=True)
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

    def verify_against_numpy(self, c_function: CFunction):
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

    def _setup_file_logger(self):
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to {self.log_file}")

    def _setup_run_dir(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{self.func.__name__}_{timestamp}"
        self.run_dir = os.path.join(self.output_dir, self.run_id)
        ensure_directory_exists(self.run_dir)
        ensure_directory_exists(os.path.join(self.run_dir, "iterations"))
