"""
Backward-compatibility shim.

All functionality has been consolidated into run.py.
This module re-exports symbols for any code that still imports from utils.
"""
from .run import run_attack, _set_seed as set_seed
