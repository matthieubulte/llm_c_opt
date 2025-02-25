#!/usr/bin/env python3
"""
Function analyzer for the NumPy-to-C optimizer.
"""

import ast
import inspect
from typing import Callable, Dict, List, Any


class NumPyFunctionAnalyzer(ast.NodeVisitor):
    """
    Analyzes a NumPy function to understand its structure and operations.
    """

    def __init__(self, func: Callable):
        self.func = func
        self.source = inspect.getsource(func)
        self.tree = ast.parse(self.source)
        self.numpy_calls = []
        self.input_args = []
        self.visit(self.tree)

    def visit_Call(self, node):
        """Visit function calls to identify NumPy operations."""
        if isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Name
        ):
            if node.func.value.id == "np":
                self.numpy_calls.append({"name": node.func.attr, "lineno": node.lineno})
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Visit function definition to identify input arguments."""
        self.input_args = [arg.arg for arg in node.args.args]
        self.generic_visit(node)

    def get_numpy_operations(self) -> List[Dict[str, Any]]:
        """Get the list of NumPy operations used in the function."""
        return self.numpy_calls

    def get_input_args(self) -> List[str]:
        """Get the list of input arguments to the function."""
        return self.input_args
