"""Dataset materialization helpers for stage5."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chainbench.lib.config import resolve_path


@dataclass
class ExportResult:
    ok: bool
    status: str
    input_row: dict[str, Any]
    output_row: dict[str, Any] | None = None
    error: str | None = None


def build_dataset_audio_relpath(row: dict[str, Any]) -> str:
    suffix = Path(str(row.get("audio_path", ""))).suffix or ".wav"
    return (
        Path(str(row["split"]))
        / "audio"
        / row["language"]
        / row["label"]
        / f"{row['sample_id']}{suffix}"
    ).as_posix()


def export_single_audio(
    row: dict[str, Any],
    workspace_root: Path,
    dataset_root: Path,
    overwrite: bool,
) -> ExportResult:
    source_path = resolve_path(str(row["audio_path"]), workspace_root)
    if not source_path.exists():
        return ExportResult(
            ok=False,
            status="missing_source_audio",
            input_row=row,
            error=f"missing source audio: {source_path}",
        )

    audio_relpath = build_dataset_audio_relpath(row)
    target_path = dataset_root / audio_relpath
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if target_path.exists() and not overwrite:
            if target_path.stat().st_size > 0:
                exported = dict(row)
                exported["audio_path"] = audio_relpath
                return ExportResult(
                    ok=True,
                    status="skipped_existing_audio",
                    input_row=row,
                    output_row=exported,
                )
        shutil.copy2(source_path, target_path)
    except Exception as exc:
        return ExportResult(
            ok=False,
            status="audio_export_failed",
            input_row=row,
            error=f"{type(exc).__name__}: {exc}",
        )

    exported = dict(row)
    exported["audio_path"] = audio_relpath
    return ExportResult(
        ok=True,
        status="exported_audio",
        input_row=row,
        output_row=exported,
    )
