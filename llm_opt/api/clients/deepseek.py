import requests
from typing import Optional

from llm_opt.utils.logging_config import logger
from llm_opt.utils.constants import DEEPSEEK_API_KEY, DEEPSEEK_API_URL
from llm_opt.api.clients.base import BaseAPIClient


class DeepSeekAPIClient(BaseAPIClient):
    """
    DeepSeek API client for the NumPy-to-C optimizer.
    """

    def __init__(self, api_key=None, api_url=None):
        """
        Initialize the DeepSeek API client.

        Args:
            api_key: The API key to use (defaults to DEEPSEEK_API_KEY)
            api_url: The API URL to use (defaults to DEEPSEEK_API_URL)
        """
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.api_url = api_url or DEEPSEEK_API_URL

        if not self.api_key:
            logger.error("DEEPSEEK_API_KEY environment variable not set")
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")

    def call_api(self, prompt: str) -> Optional[str]:
        """
        Call the DeepSeek API with the given prompt.

        Args:
            prompt: The prompt to send to the DeepSeek API

        Returns:
            The response from the DeepSeek API or None if an error occurred
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert C programmer specializing in translating NumPy functions to optimized C code. Your task is to create efficient C implementations of numerical algorithms.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.5,
            "max_tokens": 7000,
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling DeepSeek API: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response status code: {e.response.status_code}")
                logger.error(f"Response headers: {dict(e.response.headers)}")
                logger.error(f"Response content: {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling DeepSeek API: {e}", exc_info=True)
            return None
