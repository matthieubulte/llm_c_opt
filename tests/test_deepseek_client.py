import unittest
from unittest.mock import patch, MagicMock
from llm_opt.api.clients import DeepSeekAPIClient, extract_code_from_response
from llm_opt.api.clients.deepseek import call_deepseek_api


class TestDeepSeekAPIClient(unittest.TestCase):
    """Test the DeepSeekAPIClient class."""

    @patch("llm_opt.api.clients.deepseek.requests.post")
    def test_call_api_success(self, mock_post):
        """Test successful API call."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "test response"}}]
        }
        mock_post.return_value = mock_response

        # Create a client with a mock API key
        client = DeepSeekAPIClient(api_key="test_key", api_url="test_url")
        response = client.call_api("test prompt")

        # Check that the response is correct
        self.assertEqual(response, "test response")

        # Check that the API was called with the correct arguments
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "test_url")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test_key")
        self.assertEqual(kwargs["json"]["messages"][1]["content"], "test prompt")

    @patch("llm_opt.api.clients.deepseek.requests.post")
    def test_call_api_error(self, mock_post):
        """Test API call with error."""
        # Mock the response to raise an exception
        mock_post.side_effect = Exception("test error")

        # Create a client with a mock API key
        client = DeepSeekAPIClient(api_key="test_key", api_url="test_url")
        response = client.call_api("test prompt")

        # Check that the response is None
        self.assertIsNone(response)

    @patch("llm_opt.api.clients.deepseek.DeepSeekAPIClient")
    def test_call_deepseek_api(self, mock_client_class):
        """Test the call_deepseek_api function."""
        # Mock the client
        mock_client = MagicMock()
        mock_client.call_api.return_value = "test response"
        mock_client_class.return_value = mock_client

        # Call the function
        response = call_deepseek_api("test prompt")

        # Check that the response is correct
        self.assertEqual(response, "test response")

        # Check that the client was created and called
        mock_client_class.assert_called_once()
        mock_client.call_api.assert_called_once_with("test prompt")

    def test_extract_code_from_response_markdown(self):
        """Test extracting code from a markdown response."""
        response = """
        Here's the implementation:

        ```c
        void test_func(double* a, int a_size, double* b, int b_size, double* output, int output_size) {
            for (int i = 0; i < a_size && i < output_size; i++) {
                output[i] = a[i] + b[i];
            }
        }
        ```

        This implementation adds the two arrays element-wise.
        """

        code = extract_code_from_response(response)
        self.assertIn("void test_func", code)
        self.assertIn("output[i] = a[i] + b[i]", code)

    def test_extract_code_from_response_none(self):
        """Test extracting code from a None response."""
        code = extract_code_from_response(None)
        self.assertEqual(code, "")


if __name__ == "__main__":
    unittest.main()
