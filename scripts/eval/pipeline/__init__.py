"""Public API for the evaluation pipeline package."""

from __future__ import annotations

from .orchestrator import run_all_baselines

__all__ = ["run_all_baselines"]
