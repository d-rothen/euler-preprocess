"""Thin wrapper so ``python main.py`` keeps working.

The actual logic lives in :mod:`src.cli` (which is also the installed
``euler-fog`` console-script entry point).
"""

import sys

from euler_fog.cli import main

if __name__ == "__main__":
    sys.exit(main())
