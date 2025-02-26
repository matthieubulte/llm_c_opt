"""
Helper functions for the NumPy-to-C optimizer.
"""

import os
import numpy as np
from typing import Any, Tuple


def ensure_directory_exists(directory: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory: The directory path to ensure exists
    """
    os.makedirs(directory, exist_ok=True)


def compare_outputs(numpy_output: Any, c_output: Any) -> Tuple[bool, float]:
    """
    Compare the outputs of the NumPy and C implementations.

    Args:
        numpy_output: The output from the NumPy implementation
        c_output: The output from the C implementation

    Returns:
        A tuple of (match, max_difference)
    """
    if isinstance(numpy_output, np.ndarray):
        max_diff = np.max(np.abs(numpy_output - c_output))
        return max_diff < 1e-10, float(max_diff)
    else:
        # Convert to numpy arrays for consistent comparison
        numpy_output_arr = np.array([numpy_output])
        c_output_arr = np.array([c_output[0]])
        match = np.allclose(numpy_output_arr, c_output_arr, rtol=1e-10, atol=1e-10)
        return match, 0.0 if match else 1.0
