"""O4. PacketLoss operator (paper §8.1, §13.3)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from .base import DeliveryOperator, load_audio, write_audio


class PacketLossOperator(DeliveryOperator):
    @property
    def op_name(self) -> str:
        return "packet_loss"

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
        loss_rate_pct = float(params["loss_rate_pct"])
        frame_ms = int(config["packet_loss"]["frame_ms"])
        audio, sr = load_audio(input_path)
        dropped = self._apply_packet_loss_numpy(audio, sr, loss_rate_pct, frame_ms)
        write_audio(output_path, dropped, sr)
        metadata["loss_rate_pct"] = loss_rate_pct
        metadata["frame_ms"] = frame_ms

    @staticmethod
    def _apply_packet_loss_numpy(
        audio: np.ndarray,
        sample_rate: int,
        loss_rate_pct: float,
        frame_ms: int,
    ) -> np.ndarray:
        frame_len = max(1, int(sample_rate * frame_ms / 1000.0))
        dropped = loss_rate_pct / 100.0
        frames = []
        previous = np.zeros(frame_len, dtype=np.float32)
        for start in range(0, len(audio), frame_len):
            frame = audio[start : start + frame_len]
            if len(frame) < frame_len:
                frame = np.pad(frame, (0, frame_len - len(frame)))
            if random.random() < dropped:
                frames.append(previous.copy())
            else:
                frames.append(frame.copy())
                previous = frame.copy()
        merged = np.concatenate(frames)[: len(audio)]
        return merged.astype(np.float32)
