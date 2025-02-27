from typing import Optional
from groq import Groq

from llm_opt.utils.logging_config import logger
from llm_opt.utils.constants import GROQ_API_KEY, GROQ_MODEL
from llm_opt.api.clients.base import BaseAPIClient


class GroqAPIClient(BaseAPIClient):
    """
    Groq API client for the NumPy-to-C optimizer.
    """

    def __init__(self, api_key=None, model_name=None):
        """
        Initialize the Groq API client.

        Args:
            api_key: The API key to use (defaults to GROQ_API_KEY)
            model_name: The model name to use (defaults to GROQ_MODEL)
        """
        self.api_key = api_key or GROQ_API_KEY
        self.model_name = model_name or GROQ_MODEL
        self.client = Groq(api_key=self.api_key)

        if not self.api_key:
            logger.error("GROQ_API_KEY environment variable not set")
            raise ValueError("GROQ_API_KEY environment variable not set")
        if not self.model_name:
            logger.error("GROQ_MODEL environment variable not set")
            raise ValueError("GROQ_MODEL environment variable not set")

    def call_api(self, prompt: str) -> Optional[str]:
        """
        Call the Groq API with the given prompt.

        Args:
            prompt: The prompt to send to the Groq API

        Returns:
            The response from the Groq API or None if an error occurred
        """

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_completion_tokens=8000,
            top_p=0.95,
            stream=False,
            stop=None,
        )

        return completion.choices[0].message.content
