#!/usr/bin/env python3
"""
Function loader for the NumPy-to-C optimizer.
"""

import ctypes
import inspect
import numpy as np
import logging
import time
import os
from typing import Callable, Dict, Optional, Any, Type, cast

from llm_opt.utils.logging_config import logger
from llm_opt.utils.constants import DTYPE_TO_CTYPES


def load_cached_function(
    shared_lib_path: str,
    func_name: str,
    input_types: Dict[str, np.dtype],
    original_func: Callable,
) -> Callable:
    """
    Load a function from a compiled shared library.

    Args:
        shared_lib_path: Path to the shared library
        func_name: Name of the function
        input_types: Dictionary mapping argument names to NumPy dtypes
        original_func: The original Python function to use as fallback

    Returns:
        A wrapped function that calls the compiled C function
    """
    try:
        # Load the shared library
        lib = ctypes.CDLL(shared_lib_path)
        c_func = getattr(lib, func_name)

        # Set up the function signature
        arg_types = []
        for arg_name, dtype in input_types.items():
            if isinstance(dtype, np.dtype):
                # Array argument
                c_type = DTYPE_TO_CTYPES.get(dtype.type, ctypes.c_double)
                # Cast to the correct type for POINTER
                ptr_type = ctypes.POINTER(cast(Type[ctypes._SimpleCData], c_type))
                arg_types.append(ptr_type)
                arg_types.append(ctypes.c_int)  # size
            else:
                # Scalar argument
                c_type = DTYPE_TO_CTYPES.get(dtype, ctypes.c_double)
                arg_types.append(c_type)

        # Add output argument
        arg_types.append(ctypes.POINTER(ctypes.c_double))  # output
        arg_types.append(ctypes.c_int)  # output_size

        # Set the argument types
        c_func.argtypes = arg_types
        c_func.restype = None

        # Create a Python wrapper function
        def optimized_wrapper(*args, **kwargs):
            """
            Wrapper function that calls the compiled C function.
            """
            try:
                # Convert positional args to keyword args based on function signature
                sig = inspect.signature(original_func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                kwargs = bound_args.arguments

                # Prepare arguments for the C function
                c_args = []
                for arg_name, dtype in input_types.items():
                    value = kwargs.get(arg_name)
                    if value is None:
                        logger.warning(
                            f"Missing argument {arg_name}, falling back to original function"
                        )
                        result = original_func(*args, **kwargs)
                        return result

                    if isinstance(dtype, np.dtype):
                        # Array argument
                        if not isinstance(value, np.ndarray):
                            logger.debug(
                                f"Converting non-array value to numpy array for '{arg_name}'"
                            )
                            value = np.array(value, dtype=dtype)
                        else:
                            if value.dtype != dtype:
                                logger.debug(
                                    f"Converting array from {value.dtype} to {dtype} for '{arg_name}'"
                                )
                            value = value.astype(dtype)

                        c_type = DTYPE_TO_CTYPES.get(dtype.type, ctypes.c_double)
                        ptr_type = cast(Type[ctypes._SimpleCData], c_type)
                        c_args.append(value.ctypes.data_as(ctypes.POINTER(ptr_type)))
                        c_args.append(ctypes.c_int(len(value)))
                    else:
                        c_type = DTYPE_TO_CTYPES.get(dtype, ctypes.c_double)
                        c_args.append(c_type(value))

                # Determine output size and type
                result = original_func(*args, **kwargs)

                if isinstance(result, np.ndarray):
                    output_size = result.size
                    output = np.zeros(output_size, dtype=np.float64)
                else:
                    # Scalar output
                    output_size = 1
                    output = np.zeros(1, dtype=np.float64)

                # Add output arguments
                c_args.append(output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
                c_args.append(ctypes.c_int(output_size))

                # Call the C function
                c_func(*c_args)

                # Return the result
                if output_size == 1:
                    result_value = output[0]  # Return scalar
                else:
                    # Try to reshape the output to match the original result shape
                    if isinstance(result, np.ndarray) and result.shape != output.shape:
                        result_value = output.reshape(result.shape)
                    else:
                        result_value = output

                return result_value

            except Exception as e:
                logger.warning(
                    f"Error calling optimized function: {e}, falling back to original",
                    exc_info=True,
                )
                return original_func(*args, **kwargs)

        return optimized_wrapper

    except Exception as e:
        logger.error(f"Error loading compiled function: {e}", exc_info=True)
        # Fall back to the original function
        logger.warning(f"Falling back to original function due to loading error")
        return original_func
