#!/usr/bin/env python3
"""
Command-line interface for the NumPy-to-C optimizer.
"""

import argparse
import importlib
import inspect
import os
import sys
from typing import Dict, Any, Callable

import numpy as np

from llm_opt import optimize
from llm_opt.api.clients import MockAPIClient
from llm_opt.utils.logging_config import setup_logging


def import_function(function_path: str) -> Callable:
    """
    Import a function from a module path.

    Args:
        function_path: Path to the function in the format 'module.submodule:function_name'

    Returns:
        The imported function
    """
    module_path, function_name = function_path.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, function_name)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="NumPy-to-C Optimizer")
    parser.add_argument(
        "function",
        help="Function to optimize in the format 'module.submodule:function_name'",
    )
    parser.add_argument(
        "input_generator",
        help="Function that generates test inputs in the format 'module.submodule:function_name'",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save optimization artifacts",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum number of optimization iterations",
    )
    parser.add_argument("--mock", action="store_true", help="Use mock API client")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = setup_logging(log_level)
    print(f"Logging to {log_file}")

    try:
        # Import the function and input generator
        func = import_function(args.function)
        input_generator = import_function(args.input_generator)

        # Get the input types from the function signature
        sig = inspect.signature(func)
        input_types = {}
        for param_name in sig.parameters:
            # Default to float64 for all parameters
            input_types[param_name] = np.dtype(np.float64)

        # Create API client
        api_client = MockAPIClient() if args.mock else None

        # Run the optimization
        optimized_func = optimize(
            func,
            input_types=input_types,
            test_input_generator=input_generator,
            max_iterations=args.max_iterations,
            output_dir=args.output_dir,
            api_client=api_client,
        )

        print(f"Optimization successful. Results saved to {args.output_dir}")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
