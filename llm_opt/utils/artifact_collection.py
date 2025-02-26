import os
import json
from typing import Dict, Any, List
from datetime import datetime

from llm_opt.utils.logging_config import logger
from llm_opt.utils.helpers import ensure_directory_exists
from llm_opt.core.iteration_artifact import IterationArtifact
from llm_opt.utils.artifacts_to_html import to_html


class ArtifactCollection:
    """
    Logs artifacts from the optimization process.
    """

    def __init__(self, run_dir: str, func_name: str):
        self.run_dir = run_dir
        self.func_name = func_name
        self.artifacts = []

    def log_func_info(self, func_info: Dict[str, Any]):
        func_info_path = os.path.join(self.run_dir, "function_info.json")
        with open(func_info_path, "w") as f:
            json.dump(func_info, f, indent=2)

    def log_initial_prompt(self, initial_prompt: str):
        initial_prompt_path = os.path.join(self.run_dir, "initial_prompt.txt")
        with open(initial_prompt_path, "w") as f:
            f.write(initial_prompt)

    def add_artifact(self, artifact: IterationArtifact):
        self.artifacts.append(artifact)
        self._save_iteration_artifact(artifact)

    def to_str(self):
        return "\n".join([artifact.short_desc() for artifact in self.artifacts])

    def _save_iteration_artifact(
        self,
        artifact: IterationArtifact,
    ):
        """
        Save artifacts from an optimization iteration.
        """
        iteration_dir = os.path.join(
            self.run_dir, "iterations", f"iteration_{artifact.idx}"
        )
        ensure_directory_exists(iteration_dir)

        # Save the C implementation
        c_file_path = os.path.join(iteration_dir, f"{self.func_name}.c")
        with open(c_file_path, "w") as f:
            f.write(artifact.c_code)

        # Save the prompt
        prompt_path = os.path.join(iteration_dir, "prompt.txt")
        with open(prompt_path, "w") as f:
            f.write(artifact.prompt)

        # Save the response
        response_path = os.path.join(iteration_dir, "response.txt")
        with open(response_path, "w") as f:
            f.write(artifact.response)

        # Save the test results
        test_results_path = os.path.join(iteration_dir, "artifact.txt")
        with open(test_results_path, "w") as f:
            f.write(artifact.short_desc())

        # Log the results
        logger.info(f"Iteration {artifact.idx} results:")
        logger.info(f"\tShort desc results: {artifact.short_desc()}")

    def checkpoint(self, best_artifact: IterationArtifact):
        self._save_iteration_artifact(self.artifacts[-1])
        html_path = os.path.join(self.run_dir, "artifacts.html")
        logger.info(f"Saving artifacts to {html_path}")
        to_html(self.artifacts, html_path)
        with open(os.path.join(self.run_dir, "best_artifact.txt"), "w") as f:
            f.write(best_artifact.short_desc())
