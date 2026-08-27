"""Source-checkout wrapper for the installed AttackBench acceptance command."""

import sys
from pathlib import Path

# Prefer the checkout over any older wheel already installed in the environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attackbench.acceptance import main

if __name__ == "__main__":
    sys.exit(main())
