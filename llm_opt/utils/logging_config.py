#!/usr/bin/env python3
"""
Logging configuration for the NumPy-to-C optimizer.
"""

import logging
import sys


# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Create console handler for info+ messages
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
    )
)
root_logger.addHandler(handler)

# Create a logger for the package
logger = logging.getLogger("llm_opt")
