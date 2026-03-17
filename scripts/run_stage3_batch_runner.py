#!/usr/bin/env python3
"""Entrypoint for Stage-3 generator batch runner (conda-run script).

Invoked by stage3/runners.run_generator_batch() inside a generator-specific conda env.
Adds scripts dir to path and runs stage3.runner.main().
"""

import os
import sys

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from stage3.runner import main

if __name__ == "__main__":
    sys.exit(main())
