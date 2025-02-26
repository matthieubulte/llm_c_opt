#!/usr/bin/env python3
"""
Logging configuration for the NumPy-to-C optimizer.
"""

import os
import logging
import sys
from datetime import datetime

from llm_opt.utils.helpers import ensure_directory_exists

# Create logs directory if it doesn't exist
LOGS_DIR = "logs"
ensure_directory_exists(LOGS_DIR)

# Generate a timestamp for the log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOGS_DIR, f"llm_opt_{timestamp}.log")

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Create formatters
detailed_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
)
console_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create file handler for detailed logging
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(detailed_formatter)
root_logger.addHandler(file_handler)

# Create console handler for info+ messages
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

# Create a logger for the package
logger = logging.getLogger("llm_opt")
logger.info(f"Logging initialized. Detailed logs will be saved to: {LOG_FILE}")


def get_logger(name):
    """Get a logger with the given name."""
    return logging.getLogger(name)


def setup_logging(log_level="INFO"):
    """
    Set up logging with the specified log level.

    Args:
        log_level: The log level to use (e.g., "DEBUG", "INFO")

    Returns:
        The path to the log file
    """
    # Create logs directory if it doesn't exist
    ensure_directory_exists(LOGS_DIR)

    # Generate a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"llm_opt_{timestamp}.log")

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Clear existing handlers
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
    )
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create file handler for detailed logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)

    # Create console handler for the specified log level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Log initialization
    logging.info(
        f"Logging initialized with level {log_level}. Detailed logs will be saved to: {log_file}"
    )

    return log_file
