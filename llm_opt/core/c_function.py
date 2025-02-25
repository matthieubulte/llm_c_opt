#!/usr/bin/env python3
"""
C function representation for the NumPy-to-C optimizer using the improved CFunction implementation.
"""

import os
import tempfile
import subprocess
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import uuid
import logging
from typing import Callable, Dict, Optional, Any, List, Tuple

from llm_opt.utils.logging_config import logger
from llm_opt.utils.constants import CACHE_DIR, DTYPE_TO_CTYPES


class Signature:
    """
    Represents the signature of a C function.
    """

    def __init__(self, arg_types, return_type):
        self.arg_types = arg_types
        self.return_type = return_type


class CFunction:
    """
    A class representing a C function that can be compiled, loaded, and called.
    This implementation provides improved performance and better handling of
    vectorized operations.
    """

    def __init__(self, func_name, signature, c_code, compiler="gcc"):
        """
        Initialize a C function.

        Args:
            func_name: Name of the function
            signature: Signature object containing arg_types and return_type
            c_code: C implementation of the function
            compiler: Compiler to use (default: "gcc")
        """
        self.c_code = c_code
        self.func_name = func_name
        self.signature = signature
        self.compiler = compiler
        self.lib_path = None
        self.lib = None
        self.func = None

    def compile(self):
        """
        Compile the C code to a shared library.

        Returns:
            True if compilation was successful, False otherwise
        """
        try:
            temp_dir = tempfile.mkdtemp()
            c_file = os.path.join(temp_dir, f"{self.func_name}_{uuid.uuid4().hex}.c")
            lib_file = os.path.join(temp_dir, f"{self.func_name}.so")

            # Create the final library path in the cache directory
            os.makedirs(CACHE_DIR, exist_ok=True)
            final_lib_path = os.path.join(
                CACHE_DIR, f"{self.func_name}_{uuid.uuid4().hex}.so"
            )

            compile_cmd = [
                self.compiler,
                c_file,
                "-shared",
                "-fPIC",
                "-o",
                lib_file,
                "-O3",
                "-march=native",
                "-ftree-vectorize",
                "-ffast-math",
            ]

            with open(c_file, "w") as f:
                f.write(self.c_code)

            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Compilation failed:\n{result.stderr}")
                return False

            # Copy the library to the cache directory
            import shutil

            shutil.copy(lib_file, final_lib_path)

            self.lib_path = final_lib_path
            return True

        except Exception as e:
            logger.error(f"Unexpected error during compilation: {e}", exc_info=True)
            return False

    def load(self):
        """
        Load the compiled function from the shared library.

        Returns:
            True if loading was successful, False otherwise
        """
        try:
            if not self.lib_path or not os.path.exists(self.lib_path):
                logger.error(f"Shared library not found at {self.lib_path}")
                return False

            self.lib = ctypes.CDLL(self.lib_path)
            self.func = getattr(self.lib, self.func_name)
            self.func.argtypes = self.signature.arg_types
            self.func.restype = self.signature.return_type
            return True

        except Exception as e:
            logger.error(f"Error loading compiled function: {e}", exc_info=True)
            return False

    def __del__(self):
        """
        Clean up resources when the object is deleted.
        """
        if hasattr(self, "lib") and self.lib:
            del self.lib
            self.lib = None

    def __call__(self, *args, **kwargs):
        """
        Call the C function with the given arguments.

        Returns:
            The result of the C function call
        """
        if self.func is None:
            raise ValueError("Function not loaded. Call compile() and load() first.")
        return self.func(*args)

    @staticmethod
    def nparray(dtype):
        """
        Helper method to create a numpy array pointer for use in function signatures.

        Args:
            dtype: NumPy data type

        Returns:
            A numpy array pointer type for use in ctypes
        """
        return ndpointer(dtype=dtype, flags="C_CONTIGUOUS")

    @classmethod
    def from_shared_library(cls, lib_path, func_name, input_types, original_func=None):
        """
        Create a CFunction instance from an existing shared library.

        Args:
            lib_path: Path to the shared library
            func_name: Name of the function
            input_types: Dictionary mapping argument names to NumPy dtypes
            original_func: The original Python function (optional)

        Returns:
            A CFunction instance
        """
        try:
            # Import here to avoid circular imports
            from llm_opt.core.signature import FunctionSignatureGenerator

            # Create a signature generator if we have the original function
            if original_func:
                signature_generator = FunctionSignatureGenerator(
                    original_func, input_types
                )
                signature = signature_generator.generate_ctypes_signature()
            else:
                # Create a signature based on input_types
                arg_types = []
                for arg_name, dtype in input_types.items():
                    if isinstance(dtype, np.dtype):
                        # Array argument
                        arg_types.append(cls.nparray(dtype))
                        arg_types.append(ctypes.c_int)  # size
                    else:
                        # Scalar argument
                        c_type = DTYPE_TO_CTYPES.get(dtype, ctypes.c_double)
                        arg_types.append(c_type)

                # Create a signature object
                signature = Signature(arg_types, None)  # Assuming void return type

            # Create a CFunction instance
            instance = cls(func_name, signature, "", compiler="gcc")
            instance.lib_path = lib_path

            # Load the function
            if not instance.load():
                if original_func:
                    return original_func
                else:
                    raise ValueError(
                        f"Failed to load function {func_name} from {lib_path}"
                    )

            return instance
        except Exception as e:
            logger.error(f"Error in from_shared_library: {e}", exc_info=True)
            if original_func:
                return original_func
            else:
                raise
