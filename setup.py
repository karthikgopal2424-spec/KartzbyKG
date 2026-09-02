"""Compatibility shim so `pip install -e .` works on older pip/setuptools too.

All real metadata lives in ``pyproject.toml``.
"""

from setuptools import setup

setup()
