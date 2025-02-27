from abc import ABC, abstractmethod
from typing import Optional


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


def extract_code_from_response(response: Optional[str]) -> str:
    if response is None:
        return ""

    # Look for code blocks in markdown format
    if "```c" in response:
        code_blocks = response.split("```c")
        if len(code_blocks) > 1:
            return code_blocks[1].split("```")[0].strip()

    raise ValueError(f"No ```c ... ``` code block found in the response:\n{response}")
