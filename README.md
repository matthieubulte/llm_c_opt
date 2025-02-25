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
def matrix_power(x, n):
    """Compute the nth power of a square matrix."""
    result = np.eye(x.shape[0], dtype=x.dtype)
    temp = x.copy()
    
    while n > 0:
        if n % 2 == 1:
            result = np.dot(result, temp)
        temp = np.dot(temp, temp)
        n //= 2
    
    return result

# Define a test input generator
def matrix_power_inputs():
    """Generate test inputs for matrix_power function."""
    return (np.random.rand(10, 10), 5)

# Optimize the function
optimized_matrix_power = optimize(
    matrix_power,
    input_types={"x": np.dtype(np.float64), "n": np.dtype(np.int32)},
    test_input_generator=matrix_power_inputs
)

# Use the optimized function
A = np.random.rand(1000, 1000)
result = optimized_matrix_power(A, 10)
```

### Command-line Usage

Run the example optimization:

```bash
./run.sh
```

## Project Structure

The project is organized into the following modules:

- `llm_opt/`: Main package directory
  - `__init__.py`: Package initialization and exports
  - `main.py`: Main module with the `optimize` function
  - `core/`: Core functionality
    - `analyzer.py`: Function analysis
    - `signature.py`: C function signature generation
    - `compiler.py`: C compilation utilities
    - `optimizer.py`: DeepSeek optimization loop
    - `loader.py`: Function loading utilities
  - `api/`: API clients
    - `deepseek.py`: DeepSeek API client
  - `utils/`: Utility modules
    - `constants.py`: Constants and type mappings
    - `logging_config.py`: Logging configuration
- `example.py`: Example usage of the optimizer
- `run.sh`: Script to run the example optimization

## Optimization Techniques

The system applies various optimization techniques:

1. **Vectorization**
   - Use SIMD instructions (SSE, AVX, NEON)
   - Optimize for specific CPU architectures

2. **Memory Access Optimization**
   - Cache blocking/tiling
   - Data alignment
   - Prefetching

3. **Loop Optimization**
   - Loop unrolling
   - Loop fusion
   - Loop interchange

4. **Algorithm Improvements**
   - Specialized algorithms for common patterns
   - Mathematical simplifications
   - Numerical approximations when appropriate

5. **Parallelization**
   - Multi-threading for large computations
   - Task parallelism for independent operations

## Advantages Over Numba

1. **Iterative Refinement**: Unlike Numba's one-pass compilation, our approach iteratively improves the implementation.
2. **Hardware-Specific Optimization**: Tailors optimizations to the specific hardware.
3. **Feedback-Driven**: Uses performance feedback to guide optimization decisions.
4. **Explainable**: Provides insights into the optimizations applied.
5. **Extensible**: Can incorporate new optimization techniques as they are developed.

## Inspiration

This project is inspired by NVIDIA's experiment where they used DeepSeek-R1 with a feedback mechanism during inference to generate optimized GPU attention kernels. The approach demonstrated that by allocating additional computational resources during inference and implementing a feedback loop, AI models can produce increasingly optimized code.

## License

MIT
