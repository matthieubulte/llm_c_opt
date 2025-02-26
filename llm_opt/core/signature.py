"""
Function signature generator for the NumPy-to-C optimizer.
"""

from typing import List, Tuple
from llm_opt.core.type_interface import TypeInterface


class Signature:
    def __init__(self, args: List[Tuple[str, TypeInterface]]):
        self.args = args

    def generate_c_function_signature(self, func_name: str) -> str:
        args = []
        for arg_name, type in self.args:
            type.append_to_c_signature_str(arg_name, args)

        return f"void {func_name}({', '.join(args)})"

    def python_args_to_c_args(self, python_args: Tuple | List) -> Tuple:
        assert len(python_args) == len(self.args)
        c_args = []
        for i, (_, type) in enumerate(self.args):
            type.convert_to_c_val(python_args[i], c_args)
        return tuple(c_args)

    def generate_ctypes_signature(self) -> List[type]:
        args = []
        for _, type in self.args:
            type.append_to_c_args_list(args)

        return args
