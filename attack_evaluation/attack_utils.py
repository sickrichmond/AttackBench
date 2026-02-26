"""
Attack utility functions.

This module provides utility functions for adversarial attacks,
replacing dependencies from adv_lib.utils.attack_utils.
"""

from collections import OrderedDict
from .distances import l0_distances, l1_distances, l2_distances, linf_distances


# Default metrics dictionary for tracking distance metrics
# This matches the adv_lib._default_metrics structure
_default_metrics = OrderedDict([
    ('linf', linf_distances),
    ('l2', l2_distances),
    ('l1', l1_distances),
    ('l0', l0_distances),
])
