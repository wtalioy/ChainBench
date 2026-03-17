"""Checkpoint and shared-training helpers for the evaluation pipeline."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from lib.io import write_json
from lib.logging import get_logger

from ..tasks import TaskPack

LOGGER = get_logger("eval.pipeline")


def write_checkpoint_skip_note(run_dir: Path, checkpoint: Path, epochs: int) -> None:
    (run_dir / "train.log").write_text(
        f"Training skipped: using existing checkpoint {checkpoint}\n"
        f"(Delete checkpoint to retrain with config epochs={epochs})\n",
        encoding="utf-8",
    )


def write_shared_checkpoint_note(run_dir: Path, checkpoint: Path) -> None:
    (run_dir / "train.log").write_text(
        f"Training skipped: reusing shared checkpoint {checkpoint}\n",
        encoding="utf-8",
    )


def row_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha1()
    normalized_rows = sorted(
        json.dumps(row, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
        for row in rows
    )
    for normalized_row in normalized_rows:
        digest.update(normalized_row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def shared_training_key(pack: TaskPack, baseline_name: str, baseline_cfg: dict[str, Any]) -> str | None:
    if not pack.train_rows:
        return None
    payload = {
        "baseline": baseline_name,
        "repo_path": baseline_cfg["repo_path"],
        "conda_prefix": baseline_cfg["conda_prefix"],
        "env": baseline_cfg.get("env", {}),
        "assets": baseline_cfg.get("assets", {}),
        "adapter": baseline_cfg.get("adapter", {}),
        "train": baseline_cfg["train"],
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
    digest.update(row_digest(pack.train_rows).encode("utf-8"))
    digest.update(row_digest(pack.dev_rows).encode("utf-8"))
    return digest.hexdigest()


def safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def materialize_shared_checkpoint(run_dir: Path, checkpoint: Path) -> Path:
    local_checkpoint = run_dir / "shared_checkpoint" / checkpoint.name
    safe_link_or_copy(checkpoint, local_checkpoint)
    return local_checkpoint


def checkpoint_manifest_path(run_dir: Path) -> Path:
    return run_dir / "checkpoint_manifest.json"


def write_checkpoint_manifest(run_dir: Path, shared_training_key_value: str) -> None:
    write_json(checkpoint_manifest_path(run_dir), {"shared_training_key": shared_training_key_value})


def checkpoint_manifest_status(run_dir: Path, shared_training_key_value: str) -> str:
    manifest_path = checkpoint_manifest_path(run_dir)
    if not manifest_path.exists():
        return "missing"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "invalid"
    return "match" if payload.get("shared_training_key") == shared_training_key_value else "mismatch"


def clear_stale_run_artifacts(run_dir: Path, runner: Any) -> None:
    for pattern in runner.checkpoint_patterns:
        for path in run_dir.rglob(pattern):
            path.unlink()
    shared_checkpoint_dir = run_dir / "shared_checkpoint"
    if shared_checkpoint_dir.exists():
        shutil.rmtree(shared_checkpoint_dir)
    manifest_path = checkpoint_manifest_path(run_dir)
    if manifest_path.exists():
        manifest_path.unlink()


def checkpoint_for_run(run_dir: Path, runner: Any, shared_training_key_value: str | None) -> Path | None:
    checkpoint = runner.find_checkpoint(run_dir)
    if checkpoint is None or shared_training_key_value is None:
        return checkpoint
    manifest_status = checkpoint_manifest_status(run_dir, shared_training_key_value)
    if manifest_status == "match":
        return checkpoint
    if manifest_status == "missing":
        LOGGER.info("ignoring unmanaged checkpoint for %s because no checkpoint manifest was found", run_dir)
    elif manifest_status == "invalid":
        LOGGER.info("ignoring checkpoint for %s because checkpoint manifest is invalid", run_dir)
    else:
        LOGGER.info("ignoring stale checkpoint for %s because training inputs changed", run_dir)
    clear_stale_run_artifacts(run_dir, runner)
    return None
