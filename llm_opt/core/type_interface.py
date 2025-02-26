import numpy as np
import ctypes

from llm_opt.utils.constants import DTYPE_TO_CTYPE_STR, DTYPE_TO_CTYPES
from typing import List, Any


class TypeInterface:

    def __init__(self, dtype: np.dtype):
        self.dtype = dtype

    def append_to_c_signature_str(self, name: str, signature: List[str]):
        if isinstance(self.dtype, np.dtype):
            c_type = DTYPE_TO_CTYPE_STR.get(self.dtype.type, "double")
            signature.append(f"{c_type}* {name}")
            signature.append(f"int {name}_size")
        else:
            c_type = DTYPE_TO_CTYPE_STR.get(self.dtype, "double")
            signature.append(f"{c_type} {name}")

    def append_to_c_args_list(self, args: List[type]):
        if isinstance(self.dtype, np.dtype):
            args.append(ctypes.POINTER(ctypes.c_double))
            args.append(ctypes.c_int)
        else:
            args.append(DTYPE_TO_CTYPES.get(self.dtype, ctypes.c_double))

    def convert_to_c_val(self, arg, c_args: List[Any]):
        if isinstance(self.dtype, np.dtype):
            c_args.append(arg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
            c_args.append(len(arg))
        else:
            c_args.append(arg)

    def array_of(self) -> "TypeInterface":
        return TypeInterface(np.dtype(self.dtype))


FLOAT = TypeInterface(np.float32)
DOUBLE = TypeInterface(np.float64)
INT = TypeInterface(np.int32)
LONG = TypeInterface(np.int64)
BOOL = TypeInterface(np.bool_)
