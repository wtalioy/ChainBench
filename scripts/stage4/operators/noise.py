"""O6. Noise operator: additive noise at given SNR (paper §8.1, §13.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import DeliveryOperator, load_audio, peak_normalize, write_audio


class NoiseOperator(DeliveryOperator):
    @property
    def op_name(self) -> str:
        return "noise"

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
        snr_db = float(params["snr_db"])
        noise_type = params["noise_type"]
        metadata["snr_db"] = snr_db
        metadata["noise_type"] = noise_type
        audio, sr = load_audio(input_path)
        noisy = self._apply_noise_numpy(audio, snr_db, noise_type, seed + op_index)
        write_audio(output_path, noisy, sr)

    @staticmethod
    def _generate_colored_noise(
        num_samples: int,
        noise_type: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        white = rng.standard_normal(num_samples).astype(np.float32)
        if noise_type == "white":
            return white
        if noise_type == "brown":
            brown = np.cumsum(white)
            brown = brown / (np.std(brown) + 1e-8)
            return brown.astype(np.float32)
        if noise_type == "pink":
            spectrum = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(num_samples, d=1.0)
            freqs[0] = 1.0
            spectrum = spectrum / np.sqrt(freqs)
            pink = np.fft.irfft(spectrum, n=num_samples)
            pink = pink / (np.std(pink) + 1e-8)
            return pink.astype(np.float32)
        raise ValueError(f"Unsupported noise type: {noise_type}")

    @staticmethod
    def _apply_noise_numpy(
        audio: np.ndarray,
        snr_db: float,
        noise_type: str,
        seed: int,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        noise = NoiseOperator._generate_colored_noise(len(audio), noise_type, rng)
        signal_rms = float(np.sqrt(np.mean(np.square(audio))) + 1e-8)
        noise_rms = float(np.sqrt(np.mean(np.square(noise))) + 1e-8)
        desired_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
        scaled_noise = noise * (desired_noise_rms / noise_rms)
        mixed = audio + scaled_noise
        return peak_normalize(mixed)
