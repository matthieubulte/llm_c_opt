"""
C function representation for the NumPy-to-C optimizer using the improved CFunction implementation.
"""

import os
import tempfile
import subprocess
import ctypes
import uuid


class CFunction:
    """
    A class representing a C function that can be compiled, loaded, and called.
    This implementation provides improved performance and better handling of
    vectorized operations.
    """

    def __init__(self, func_name, signature, c_code):
        """
        Initialize a C function.

        Args:
            func_name: Name of the function
            signature: Signature object containing arg_types and return_type
            c_code: C implementation of the function
        """
        self.c_code = c_code
        self.func_name = func_name
        self.signature = signature
        self.lib = None
        self.func = None

    def compile_and_load(self, compiler: str = "gcc"):
        temp_dir = tempfile.mkdtemp()
        c_file = os.path.join(temp_dir, f"{self.func_name}_{uuid.uuid4().hex}.c")
        lib_file = os.path.join(temp_dir, f"{self.func_name}.so")

        compile_cmd = [
            compiler,
            c_file,
            "-shared",
            "-fPIC",
            "-o",
            lib_file,
            "-O3",
            "-march=native",
            "-ftree-vectorize",
            "-ffast-math",
            "-funroll-loops",
        ]

        with open(c_file, "w") as f:
            f.write(self.c_code)

        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Compilation failed:\n{result.stderr}")

        self.lib = ctypes.CDLL(lib_file)
        self.func = getattr(self.lib, self.func_name)
        self.func.argtypes = self.signature.generate_ctypes_signature()
        self.func.restype = ctypes.c_void_p

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
