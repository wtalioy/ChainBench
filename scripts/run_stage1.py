#!/usr/bin/env python3
"""Entrypoint for Stage 1 source curation. Run from repo root: python scripts/run_stage1.py"""

import sys

# Ensure scripts dir is on path when run from repo root
import os
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from stage1.cli import main

if __name__ == "__main__":
    sys.exit(main())
