"""Compatibility helpers for importing the PyPI release of RobustBench."""

import importlib
import sys
from typing import Callable


def get_robustbench_loader() -> Callable:
    """Return ``robustbench.load_model`` with its PyPI namespace mismatch repaired.

    RobustBench 1.1.1 depends on ``pyautoattack`` but imports it as ``autoattack``.
    The two names refer to the same PyPI distribution; registering aliases before
    importing RobustBench makes its declared dependency usable without a Git install.
    """
    try:
        importlib.import_module("autoattack")
    except ModuleNotFoundError as exc:
        if exc.name != "autoattack":
            raise
        try:
            pyautoattack = importlib.import_module("pyautoattack")
            pyautoattack_state = importlib.import_module("pyautoattack.state")
        except ModuleNotFoundError as dependency_exc:
            raise ImportError(
                "RobustBench requires pyautoattack. Install the models extra with: "
                "pip install 'attackbenchlib[models]'"
            ) from dependency_exc
        sys.modules.setdefault("autoattack", pyautoattack)
        sys.modules.setdefault("autoattack.state", pyautoattack_state)

    return importlib.import_module("robustbench").load_model
