#!/usr/bin/env python3
"""
Function signature generator for the NumPy-to-C optimizer.
"""

import ctypes
import numpy as np
from typing import Callable, Dict, List, Tuple, Any

from llm_opt.core.analyzer import NumPyFunctionAnalyzer
from llm_opt.utils.constants import DTYPE_TO_CTYPE, DTYPE_TO_CTYPES
from llm_opt.core.c_function import Signature


class FunctionSignatureGenerator:
    """
    Generates C function signatures from NumPy functions.
    """

    def __init__(self, func: Callable, input_types: Dict[str, np.dtype]):
        self.func = func
        self.func_name = func.__name__
        self.input_types = input_types
        self.analyzer = NumPyFunctionAnalyzer(func)

    def python_args_to_c_args(self, args: List[Any]) -> Tuple[List[Any], np.ndarray]:
        """
        Convert Python arguments to C arguments by adding array sizes and output buffer.

        Args:
            args: List of Python arguments (numpy arrays and scalars)

        Returns:
            List of arguments suitable for the C function, including array sizes and output buffer
        """
        c_args = []
        for arg, arg_name in zip(args, self.analyzer.get_input_args()):
            if arg_name in self.input_types:
                dtype = self.input_types[arg_name]
                if isinstance(dtype, np.dtype):
                    # For array arguments, add the array and its size
                    # Convert numpy array to ctypes pointer
                    c_args.append(
                        arg.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                    )  # The array itself
                    c_args.append(len(arg))  # The array size
                else:
                    # For scalar arguments, just add the value
                    c_args.append(arg)
            else:
                # Default to array handling for unknown types
                if isinstance(arg, np.ndarray):
                    c_args.append(arg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
                    c_args.append(len(arg))
                else:
                    # If it's not an array, just pass it as is
                    c_args.append(arg)
                    c_args.append(1)  # Default size

        # Add output array and size
        # Use the first array argument's size as output size
        output_size = next(
            (len(arg) for arg in args if isinstance(arg, np.ndarray)),
            1,  # Default to 1 if no array arguments
        )
        output = np.zeros(output_size, dtype=np.float64)

        # Convert numpy array to ctypes pointer for the C function
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        c_args.extend([output_ptr, output_size])

        return c_args, output

    def generate_c_function_signature(self) -> str:
        """Generate the C function signature as a string."""
        args = []
        for arg_name in self.analyzer.get_input_args():
            if arg_name in self.input_types:
                dtype = self.input_types[arg_name]
                if isinstance(dtype, np.dtype):
                    # Handle array arguments
                    c_type = DTYPE_TO_CTYPE.get(dtype.type, "double")
                    args.append(f"{c_type}* {arg_name}")
                    args.append(f"int {arg_name}_size")
                else:
                    # Handle scalar arguments
                    c_type = DTYPE_TO_CTYPE.get(dtype, "double")
                    args.append(f"{c_type} {arg_name}")
            else:
                # Default to double for unknown types
                args.append(f"double* {arg_name}")
                args.append(f"int {arg_name}_size")

        # Add output argument
        args.append("double* output")
        args.append("int output_size")

        return f"void {self.func_name}({', '.join(args)})"

    def generate_ctypes_signature(self) -> Signature:
        """
        Generate a Signature object for use with the CFunction class.

        Returns:
            A Signature object containing arg_types and return_type
        """
        from numpy.ctypeslib import ndpointer

        arg_types = []
        for arg_name in self.analyzer.get_input_args():
            if arg_name in self.input_types:
                dtype = self.input_types[arg_name]
                if isinstance(dtype, np.dtype):
                    # Array argument - use pointer to match what we're passing
                    arg_types.append(ctypes.POINTER(ctypes.c_double))
                    arg_types.append(ctypes.c_int)  # size
                else:
                    # Scalar argument
                    c_type = DTYPE_TO_CTYPES.get(dtype, ctypes.c_double)
                    arg_types.append(c_type)
            else:
                # Default to double pointer for unknown types
                arg_types.append(ctypes.POINTER(ctypes.c_double))
                arg_types.append(ctypes.c_int)  # size

        # Add output argument
        arg_types.append(ctypes.POINTER(ctypes.c_double))  # output
        arg_types.append(ctypes.c_int)  # output_size

        # Create and return the Signature object
        return Signature(arg_types, None)  # Assuming void return type

    def generate_test_harness(self) -> str:
        """
        Generate a C test harness to verify the implementation.
        """
        return f"""
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

{self.generate_c_function_signature()};

int main() {{
    // Initialize random seed
    srand(time(NULL));
    
    // TODO: Set up test data based on function signature
    
    // Measure performance
    clock_t start, end;
    start = clock();
    
    // Call the function
    // TODO: Call the function with test data
    
    end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("Time taken: %f seconds\\n", time_taken);
    
    return 0;
}}
"""
