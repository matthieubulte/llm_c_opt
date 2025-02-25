#!/usr/bin/env python3
"""
Setup script for the llm-opt package.
"""

from setuptools import setup, find_packages

setup(
    name="llm-opt",
    version="0.1.0",
    description="NumPy-to-C Optimizer using DeepSeek API",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/llm-opt",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.2.0",
        "requests>=2.31.0",
    ],
    python_requires=">=3.12",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
