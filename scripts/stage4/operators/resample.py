"""O1. Resample operator: 16k→8k, 16k→24k, 16k→32k→16k (paper §8.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lib.proc import run_command

from .base import DeliveryOperator


class ResampleOperator(DeliveryOperator):
    @property
    def op_name(self) -> str:
        return "resample"

    def _apply_impl(
        self,
        input_path: Path,
        output_path: Path,
        params: dict[str, Any],
        config: dict[str, Any],
        seed: int,
        op_index: int,
        metadata: dict[str, Any],
    ) -> None:
        mode = params["mode"]
        metadata["mode"] = mode
        if mode == "16k_to_8k":
            self._resample_roundtrip(input_path, output_path, 8000)
        elif mode == "16k_to_24k":
            self._resample_roundtrip(input_path, output_path, 24000)
        elif mode == "16k_to_32k_to_16k":
            self._resample_roundtrip(input_path, output_path, 32000)
        else:
            raise ValueError(f"Unsupported resample mode: {mode}")

    @staticmethod
    def _resample_roundtrip(input_path: Path, output_path: Path, mid_sample_rate: int) -> None:
        mid_path = output_path.with_name(f"{output_path.stem}__mid_{mid_sample_rate}.wav")
        try:
            cmd1 = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(input_path),
                "-ac", "1", "-ar", str(mid_sample_rate), "-c:a", "pcm_s16le",
                str(mid_path),
            ]
            r1 = run_command(cmd1)
            if r1.returncode != 0:
                raise RuntimeError(r1.stderr.strip() or "ffmpeg resample step failed")
            cmd2 = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(mid_path),
                "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                str(output_path),
            ]
            r2 = run_command(cmd2)
            if r2.returncode != 0:
                raise RuntimeError(r2.stderr.strip() or "ffmpeg resample return step failed")
        finally:
            try:
                mid_path.unlink()
            except FileNotFoundError:
                pass
