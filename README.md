# LLM-Opt: NumPy-to-C Optimizer

This project implements a feedback loop for optimizing NumPy functions by translating them to C code using the DeepSeek API, inspired by NVIDIA's inference-time scaling approach.

## Overview

The NumPy-to-C Optimizer provides a generic approach for translating NumPy functions to optimized C implementations, effectively creating an alternative to Numba's `@jit` decorator with iterative refinement. It follows these steps:

1. Take a NumPy function as input
2. Send the NumPy function directly to DeepSeek API for C translation
3. Verify the C implementation against the original NumPy function
4. Iteratively optimize the C implementation using feedback
5. Provide a Python wrapper to call the optimized C function

This approach is generic and can be applied to any NumPy function, making it widely applicable across scientific computing, data science, and machine learning workflows.

## Requirements

- Python 3.12+
- NumPy
- GCC compiler
- DeepSeek API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-opt.git
cd llm-opt
```

2. Set up your DeepSeek API key:
```bash
export DEEPSEEK_API_KEY=your_api_key_here
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Basic Usage

```python
import numpy as np
from llm_opt import optimize

# Define a NumPy function
def sum_of_squares(x):
    return np.sum(x * x)

# Define input types
input_types = {"x": np.dtype(np.float64)}

# Define a function to generate test inputs
def test_input_generator():
    n = 1000
    x = np.random.rand(n).astype(np.float64)
    return (x,)

# Optimize the function
optimized_func = optimize(
    sum_of_squares,
    input_types=input_types,
    test_input_generator=test_input_generator,
    max_iterations=3,
    output_dir="results",
)

# Use the optimized function
x = np.random.rand(1000).astype(np.float64)
result = optimized_func(x)
```

### Using a Mock API Client for Testing

```python
from llm_opt import optimize, MockAPIClient

# Create a mock API client
mock_client = MockAPIClient()

# Add a predefined response
mock_client.add_response(
    "test prompt",
    """
    ```c
    void sum_of_squares(double* x, int x_size, double* output, int output_size) {
        double sum = 0.0;
        for (int i = 0; i < x_size; i++) {
            sum += x[i] * x[i];
        }
        output[0] = sum;
    }
    ```
    """
)

# Optimize the function using the mock client
optimized_func = optimize(
    sum_of_squares,
    input_types=input_types,
    test_input_generator=test_input_generator,
    max_iterations=3,
    output_dir="results",
    api_client=mock_client,
)
```

### Running the Example

The repository includes an example script that demonstrates the optimizer with two functions:

```bash
python example.py
```

To use a mock API client instead of the real DeepSeek API:

```bash
python example.py --mock
```

## Project Structure

- `llm_opt/`: Main package directory
  - `api/`: API client implementations
  - `core/`: Core optimizer functionality
  - `utils/`: Utility functions and constants
- `tests/`: Test suite
- `example.py`: Example usage

## Testing

Run the test suite with:

```bash
python -m unittest discover tests
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project was inspired by NVIDIA's inference-time scaling approach and the Numba project.
