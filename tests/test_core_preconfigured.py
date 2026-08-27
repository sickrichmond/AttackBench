"""Ensure preconfigured attacks do not make optional libraries core dependencies."""

import os
import subprocess
import sys


def test_preconfigured_attacks_import_without_foolbox_or_eagerpy():
    script = r'''
import builtins

original_import = builtins.__import__

def without_optional_attacks(name, *args, **kwargs):
    if name.split(".", 1)[0] in {"eagerpy", "foolbox"}:
        raise ModuleNotFoundError(f"No module named '{name}'", name=name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = without_optional_attacks

from attackbench.attacks import (
    pgd, fgsm, apgd, fab, fmn, deepfool, superdeepfool, trust_region,
)

assert all(callable(attack) for attack in (
    pgd, fgsm, apgd, fab, fmn, deepfool, superdeepfool, trust_region,
))

try:
    fmn(None, None, None)
except ImportError as exc:
    assert "pip install 'attackbenchlib[attacks]'" in str(exc)
else:
    raise AssertionError("FMN should require the attacks extra when invoked")
'''

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    subprocess.run([sys.executable, "-c", script], check=True, env=env)
