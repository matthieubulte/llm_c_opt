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


def assert_outputs_equal(numpy_output: Any, c_output: Any, err_tol: float):
    """
    Compare the outputs of the NumPy and C implementations.

    Args:
        numpy_output: The output from the NumPy implementation
        c_output: The output from the C implementation

    Returns:
        A tuple of (match, max_difference)
    """
    max_diff = np.max(np.abs(numpy_output - c_output))
    if max_diff > err_tol:
        raise ValueError(
            f"""
        The outputs of the NumPy and C implementations are not equal.
        Max difference: {max_diff}
        NumPy output: {numpy_output}
        C output: {c_output}
        """
        )
