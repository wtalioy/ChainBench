"""Base adapter and shared helpers for stage3 generator batch runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chainbench.lib.logging import get_logger

LOGGER = get_logger("stage3-runner")


def map_qwen_language(language: str) -> str:
    return {"zh": "Chinese", "en": "English"}.get(language, "Auto")


def resolve_local_or_hf_model_dir(
    repo_path: Path,
    local_path_str: str,
    hf_repo_id: str | None,
) -> Path:
    local_path = Path(local_path_str)
    if not local_path.is_absolute():
        local_path = repo_path / local_path
    if local_path.exists():
        if not local_path.is_dir():
            raise NotADirectoryError(f"Model path exists but is not a directory: {local_path}")
        if not hf_repo_id:
            return local_path
        LOGGER.info("verify model snapshot %s -> %s", hf_repo_id, local_path)
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=hf_repo_id,
                local_dir=str(local_path),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to verify/download HF model snapshot {hf_repo_id} into existing directory {local_path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return local_path
    if not hf_repo_id:
        raise FileNotFoundError(f"Local model directory not found: {local_path}")
    LOGGER.info("download model %s -> %s", hf_repo_id, local_path)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=hf_repo_id,
        local_dir=str(local_path),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return local_path


class AdapterRunner:
    """Base class for a single generator adapter (load once, run many jobs)."""

    def __init__(self, repo_path: Path, config: dict[str, Any]) -> None:
        self.repo_path = repo_path
        self.config = config

    def setup(self) -> None:
        raise NotImplementedError

    def run_job(self, job: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
