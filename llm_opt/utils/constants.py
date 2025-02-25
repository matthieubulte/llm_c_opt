#!/usr/bin/env python3
"""
Constants and type definitions for the NumPy-to-C optimizer.
"""

import os
import numpy as np
import ctypes
from typing import Dict

# DeepSeek API configuration
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# Cache directory for storing compiled functions
CACHE_DIR = os.environ.get(
    "CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
)
os.makedirs(CACHE_DIR, exist_ok=True)

# Mapping of NumPy dtypes to C types
DTYPE_TO_CTYPE: Dict[type, str] = {
    np.float32: "float",
    np.float64: "double",
    np.int32: "int",
    np.int64: "long long",
    np.uint32: "unsigned int",
    np.uint64: "unsigned long long",
    np.bool_: "bool",
}

# Mapping of NumPy dtypes to ctypes
DTYPE_TO_CTYPES: Dict[type, type] = {
    np.float32: ctypes.c_float,
    np.float64: ctypes.c_double,
    np.int32: ctypes.c_int,
    np.int64: ctypes.c_longlong,
    np.uint32: ctypes.c_uint,
    np.uint64: ctypes.c_ulonglong,
    np.bool_: ctypes.c_bool,
}
