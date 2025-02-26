"""
Tests for the signature module.
"""

import unittest
import numpy as np
import ctypes
from typing import Dict

from llm_opt.core.signature import Signature
from llm_opt.core.type_interface import DOUBLE


def sample_func(a, b, c):
    """Test function with multiple arguments."""
    return a + b + c


class TestSignature(unittest.TestCase):
    """Test the Signature class."""

    def setUp(self):
        """Set up test fixtures."""
        self.signature = Signature(
            [
                ("a", DOUBLE.array_of()),
                ("b", DOUBLE.array_of()),
                ("c", DOUBLE.array_of()),
            ]
        )

    def test_generate_c_function_signature(self):
        """Test generating a C function signature."""
        signature = self.signature.generate_c_function_signature("sample_func")
        expected = "void sample_func(double* a, int a_size, double* b, int b_size, double* c, int c_size)"
        self.assertEqual(signature, expected)

    def test_generate_ctypes_signature(self):
        """Test generating a ctypes signature."""
        signature = self.signature.generate_ctypes_signature()

        # Check that the signature has the correct number of argument types
        self.assertEqual(len(signature), 6)  # 3 args * 2 (ptr + size)

        # Check that the argument types are correct
        self.assertEqual(signature[0]._type_.__name__, "c_double")  # a pointer
        self.assertEqual(signature[1].__name__, "c_int")  # a_size
        self.assertEqual(signature[2]._type_.__name__, "c_double")  # b pointer
        self.assertEqual(signature[3].__name__, "c_int")  # b_size
        self.assertEqual(signature[4]._type_.__name__, "c_double")  # c pointer
        self.assertEqual(signature[5].__name__, "c_int")  # c_size

    def test_python_args_to_c_args(self):
        """Test converting Python arguments to C arguments."""
        # Create test inputs
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        c = np.array([7.0, 8.0, 9.0], dtype=np.float64)

        # Convert to C arguments
        c_args = self.signature.python_args_to_c_args(list([a, b, c]))

        # Check that we have the correct number of arguments
        self.assertEqual(len(c_args), 6)  # 3 args * 2 (ptr + size)

        # Check that the sizes are correct
        self.assertEqual(c_args[1], 3)  # a_size
        self.assertEqual(c_args[3], 3)  # b_size
        self.assertEqual(c_args[5], 3)  # c_size


if __name__ == "__main__":
    unittest.main()
