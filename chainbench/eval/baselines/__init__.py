"""Baseline registry."""

from __future__ import annotations

from .native.aasist import AASISTLRunner, AASISTRunner
from .native.safeear import SafeEarRunner
from .native.sls_df import SlsDfRunner
from .native.nes2net import Nes2NetRunner

BASELINE_MAP = {
    "aasist": AASISTRunner,
    "aasist-l": AASISTLRunner,
    "sls_df": SlsDfRunner,
    "safeear": SafeEarRunner,
    "nes2net": Nes2NetRunner,
}

__all__ = ["BASELINE_MAP"]
