"""
Base API client for the NumPy-to-C optimizer.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseAPIClient(ABC):
    """
    Abstract base class for API clients.
    """

    @abstractmethod
    def call_api(self, prompt: str) -> Optional[str]:
        """
        Call the API with the given prompt.

        Args:
            prompt: The prompt to send to the API

        Returns:
            The response from the API or None if an error occurred
        """
        pass


def extract_code_from_response(response: Optional[str], func_name: str) -> str:
    """
    Extract the C code from the API response.

    Args:
        response: The response from the API
        func_name: The name of the function to extract

    Returns:
        The extracted C code or an empty string if extraction failed
    """
    if response is None:
        return ""

    # Look for code blocks in markdown format
    if "```c" in response:
        code_blocks = response.split("```c")
        if len(code_blocks) > 1:
            return code_blocks[1].split("```")[0].strip()

    # Look for the function definition directly
    function_signature_start = f"void {func_name}"
    if function_signature_start in response:
        lines = response.split("\n")
        start_idx = None
        end_idx = None
        brace_count = 0

        for i, line in enumerate(lines):
            if function_signature_start in line and start_idx is None:
                start_idx = i

            if start_idx is not None:
                if "{" in line:
                    brace_count += line.count("{")
                if "}" in line:
                    brace_count -= line.count("}")

                if brace_count == 0 and "}" in line:
                    end_idx = i
                    break

        if start_idx is not None and end_idx is not None:
            return "\n".join(lines[start_idx : end_idx + 1])

    return response
