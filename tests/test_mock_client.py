"""
Tests for the mock API client.
"""

import unittest
from typing import Optional
from llm_opt.api.clients import MockAPIClient


class TestMockAPIClient(unittest.TestCase):
    """Test the MockAPIClient class."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = MockAPIClient()

    def test_default_response(self):
        """Test that the mock client returns a default response."""
        response = self.client.call_api("test prompt")
        self.assertIsNotNone(response)
        if response is not None:  # Type check for linter
            self.assertIn("void test_func", response)

    def test_custom_response(self):
        """Test that the mock client returns a custom response."""
        custom_response = "custom response"
        self.client.add_response("test prompt", custom_response)
        response = self.client.call_api("test prompt")
        self.assertEqual(response, custom_response)

    def test_call_tracking(self):
        """Test that the mock client tracks calls."""
        self.client.call_api("test prompt 1")
        self.client.call_api("test prompt 2")
        calls = self.client.get_calls()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], "test prompt 1")
        self.assertEqual(calls[1], "test prompt 2")


if __name__ == "__main__":
    unittest.main()
