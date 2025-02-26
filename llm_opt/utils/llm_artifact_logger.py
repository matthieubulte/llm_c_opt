import os
import json
from typing import Dict, Any

from llm_opt.utils.logging_config import logger
from llm_opt.utils.helpers import ensure_directory_exists


class LLMOptimizerArtifactLogger:
    """
    Logs artifacts from the optimization process.
    """

    def __init__(self, run_dir: str, func_name: str):
        self.run_dir = run_dir
        self.func_name = func_name

    def log_func_info(self, func_info: Dict[str, Any]):
        func_info_path = os.path.join(self.run_dir, "function_info.json")
        with open(func_info_path, "w") as f:
            json.dump(func_info, f, indent=2)

    def log_initial_prompt(self, initial_prompt: str):
        initial_prompt_path = os.path.join(self.run_dir, "initial_prompt.txt")
        with open(initial_prompt_path, "w") as f:
            f.write(initial_prompt)

    def save_iteration_artifacts(
        self,
        iteration: int,
        c_implementation: str,
        prompt: str,
        response: str,
        test_results: Dict[str, Any],
    ):
        """
        Save artifacts from an optimization iteration.
        """
        iteration_dir = os.path.join(
            self.run_dir, "iterations", f"iteration_{iteration}"
        )
        ensure_directory_exists(iteration_dir)

        # Save the C implementation
        c_file_path = os.path.join(iteration_dir, f"{self.func_name}.c")
        with open(c_file_path, "w") as f:
            f.write(c_implementation)

        # Save the prompt
        prompt_path = os.path.join(iteration_dir, "prompt.txt")
        with open(prompt_path, "w") as f:
            f.write(prompt)

        # Save the response
        response_path = os.path.join(iteration_dir, "response.txt")
        with open(response_path, "w") as f:
            f.write(response)

        # Save the test results
        test_results_path = os.path.join(iteration_dir, "test_results.json")
        with open(test_results_path, "w") as f:
            json.dump(test_results, f, indent=2)

        # Log the results
        logger.info(f"Iteration {iteration} results:")
        logger.info(f"  Test results: {test_results}")
