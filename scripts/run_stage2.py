#!/usr/bin/env python3
"""Entrypoint for Stage 2 clean master preparation. Run from repo root: python scripts/run_stage2.py"""

import os
import sys

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from stage2.cli import main

if __name__ == "__main__":
    sys.exit(main())
