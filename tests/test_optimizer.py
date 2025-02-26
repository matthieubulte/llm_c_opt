"""
Tests for the optimizer module.
"""

import unittest
import numpy as np
import tempfile
import shutil
import ctypes

from llm_opt.core.c_function import CFunction
from llm_opt.core.signature import Signature
from llm_opt.core.type_interface import DOUBLE


def vec_add(a, b):
    """Simple vector addition function for testing."""
    return a + b


def vec_add_inputs():
    """Generate test inputs for vec_add."""
    n = 1000  # Smaller size for tests
    a = np.random.rand(n).astype(np.float64)
    b = np.random.rand(n).astype(np.float64)
    return (a, b)


def sum_of_squares(x):
    """Simple sum of squares function for testing."""
    return np.sum(x * x)


def sum_of_squares_inputs():
    """Generate test inputs for sum_of_squares."""
    n = 1000  # Smaller size for tests
    x = np.random.rand(n).astype(np.float64)
    return (x,)


class TestCFunction(unittest.TestCase):
    """Test the CFunction class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Tear down test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_compile_and_load(self):
        """Test compiling and loading a C function."""
        func_name = "test_func"
        c_code = """
        void test_func(double* a, int a_size, double* b, int b_size, double* output, int output_size) {
            for (int i = 0; i < a_size && i < output_size; i++) {
                output[i] = a[i] + b[i];
            }
        }
        """

        # Create a signature
        arg_types = [
            ("a", DOUBLE.array_of()),  # a
            ("b", DOUBLE.array_of()),  # b
            ("output", DOUBLE.array_of()),  # output
        ]
        signature = Signature(arg_types)

        # Create a CFunction
        c_function = CFunction(func_name, signature, c_code)

        # Test compilation and loading
        self.assertTrue(c_function.compile_and_load())

        # Test calling
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        output = np.zeros(3, dtype=np.float64)

        a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        c_function(a_ptr, 3, b_ptr, 3, output_ptr, 3)

        np.testing.assert_array_equal(output, np.array([5.0, 7.0, 9.0]))


if __name__ == "__main__":
    unittest.main()
