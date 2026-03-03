"""
NumPy compatibility utilities.

Provides compatibility wrappers for functions that were renamed or removed
in newer NumPy versions.
"""

import numpy as np

# numpy.trapz was renamed to numpy.trapezoid in NumPy 2.0
try:
    trapz = np.trapezoid  # NumPy 2.0+
except AttributeError:
    trapz = np.trapz  # NumPy 1.x
