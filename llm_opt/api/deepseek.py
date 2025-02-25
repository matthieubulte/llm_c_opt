#!/usr/bin/env python3
"""
DeepSeek API client for the NumPy-to-C optimizer.
"""

import requests
import json
import time
from typing import Optional, Dict, Any

from llm_opt.utils.logging_config import logger
from llm_opt.utils.constants import DEEPSEEK_API_KEY, DEEPSEEK_API_URL


def call_deepseek_api(prompt: str) -> Optional[str]:
    """
    Call the DeepSeek API with the given prompt.

    Args:
        prompt: The prompt to send to the DeepSeek API

    Returns:
        The response from the DeepSeek API or None if an error occurred
    """
    if not DEEPSEEK_API_KEY:
        logger.error("DEEPSEEK_API_KEY environment variable not set")
        raise ValueError("DEEPSEEK_API_KEY environment variable not set")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
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
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
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


def extract_code_from_response(response: Optional[str], func_name: str) -> str:
    """
    Extract the C code from the API response.

    Args:
        response: The response from the DeepSeek API
        func_name: The name of the function to extract

    Returns:
        The extracted C code or an empty string if extraction failed
    """
    if response is None:
        logger.warning("Response is None, returning empty string")
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

    logger.warning("Could not extract code properly, returning full response")
    return response
