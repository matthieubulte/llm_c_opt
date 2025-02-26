from typing import Optional, List

from llm_opt.api.clients.base import BaseAPIClient


class MockAPIClient(BaseAPIClient):
    """
    A mock API client for testing that returns predefined responses.
    """

    def __init__(self, responses=None):
        """
        Initialize the mock client with predefined responses.

        Args:
            responses: A dictionary mapping prompts to responses
        """
        self.responses = responses or {}
        self.calls: List[str] = []

    def call_api(self, prompt: str) -> Optional[str]:
        """
        Mock API call that returns a predefined response.

        Args:
            prompt: The prompt to send to the API

        Returns:
            The predefined response or a default response
        """
        self.calls.append(prompt)

        # If we have a predefined response for this prompt, return it
        if prompt in self.responses:
            return self.responses[prompt]

        # Otherwise, return a default response
        return """
        ```c
        void test_func(double* a, int a_size, double* b, int b_size, double* output, int output_size) {
            for (int i = 0; i < a_size && i < output_size; i++) {
                output[i] = a[i] + b[i];
            }
        }
        ```
        """

    def add_response(self, prompt: str, response: str):
        """
        Add a predefined response for a prompt.

        Args:
            prompt: The prompt to respond to
            response: The response to return
        """
        self.responses[prompt] = response

    def get_calls(self) -> List[str]:
        """
        Get the list of prompts that have been sent to the API.

        Returns:
            The list of prompts
        """
        return self.calls
