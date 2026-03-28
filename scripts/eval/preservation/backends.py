"""Backend protocols and loaders for preservation analysis."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

try:
    from typing import Protocol
except ImportError:  # pragma: no cover - Python < 3.8 fallback.
    from typing_extensions import Protocol


class AsrBackend(Protocol):
    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str = "",
        row: Mapping[str, str] | None = None,
    ) -> str:
        """Return a transcript hypothesis for the audio."""


class SpeakerEmbeddingBackend(Protocol):
    def embed(
        self,
        audio_path: Path,
        *,
        language: str = "",
        row: Mapping[str, str] | None = None,
    ) -> Sequence[float]:
        """Return a speaker embedding vector for the audio."""


def _read_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(np.asarray(samples, dtype=np.float32), axis=1)
    return mono, int(sample_rate)


class IdentityAsrBackend:
    """Testing and smoke-test backend that reuses the metadata transcript."""

    batch_size = 1024

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str = "",
        row: Mapping[str, str] | None = None,
    ) -> str:
        del audio_path, language
        if row is None:
            return ""
        return str(row.get("transcript", "") or row.get("raw_transcript", "")).strip()

    def transcribe_many(
        self,
        audio_paths: Sequence[Path],
        *,
        language: str = "",
        rows: Sequence[Mapping[str, str] | None] | None = None,
    ) -> list[str]:
        del language
        if rows is None:
            return ["" for _ in audio_paths]
        return [self.transcribe(path, row=row) for path, row in zip(audio_paths, rows)]


class TransformersAsrBackend:
    """Optional Hugging Face ASR backend loaded lazily at runtime."""

    def __init__(
        self,
        model_id: str,
        *,
        device: int | str | None = None,
        chunk_length_s: float = 30.0,
        batch_size: int = 1,
        trust_remote_code: bool = False,
        generate_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        pipeline_kwargs: dict[str, Any] | None = None,
    ) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - exercised only when optional deps missing.
            raise RuntimeError("transformers is required for the Transformers ASR backend") from exc

        extra_pipeline_kwargs = dict(pipeline_kwargs or {})
        normalized_model_kwargs = _normalize_transformers_model_kwargs(model_kwargs or {})
        self.batch_size = max(int(batch_size), 1)
        self._pipeline = pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            device=device,
            chunk_length_s=chunk_length_s,
            batch_size=self.batch_size,
            trust_remote_code=trust_remote_code,
            model_kwargs=normalized_model_kwargs,
            **extra_pipeline_kwargs,
        )
        self._generate_kwargs = dict(generate_kwargs or {})

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str = "",
        row: Mapping[str, str] | None = None,
    ) -> str:
        del row
        kwargs = dict(self._generate_kwargs)
        if language and "language" not in kwargs:
            kwargs["language"] = language
        result = self._pipeline(str(audio_path), generate_kwargs=kwargs or None)
        if isinstance(result, dict):
            return str(result.get("text", "")).strip()
        return str(result).strip()

    def transcribe_many(
        self,
        audio_paths: Sequence[Path],
        *,
        language: str = "",
        rows: Sequence[Mapping[str, str] | None] | None = None,
    ) -> list[str]:
        del rows
        if not audio_paths:
            return []
        kwargs = dict(self._generate_kwargs)
        if language and "language" not in kwargs:
            kwargs["language"] = language
        result = self._pipeline([str(path) for path in audio_paths], generate_kwargs=kwargs or None)
        if isinstance(result, dict):
            result = [result]
        return [
            str(item.get("text", "")).strip() if isinstance(item, dict) else str(item).strip()
            for item in result
        ]


class SpeechBrainSpeakerBackend:
    """Optional SpeechBrain ECAPA backend loaded lazily at runtime."""

    def __init__(
        self,
        source: str = "speechbrain/spkrec-ecapa-voxceleb",
        *,
        savedir: str | None = None,
        run_opts: dict[str, Any] | None = None,
    ) -> None:
        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:  # pragma: no cover - exercised only when optional deps missing.
            raise RuntimeError("speechbrain and torch are required for the SpeechBrain speaker backend") from exc

        self._torch = torch
        self._classifier = EncoderClassifier.from_hparams(
            source=source,
            savedir=savedir,
            run_opts=run_opts or {},
        )

    def embed(
        self,
        audio_path: Path,
        *,
        language: str = "",
        row: Mapping[str, str] | None = None,
    ) -> Sequence[float]:
        del language, row
        samples, _sample_rate = _read_audio_mono(audio_path)
        tensor = self._torch.from_numpy(samples).float().unsqueeze(0)
        embedding = self._classifier.encode_batch(tensor).detach().cpu().numpy().reshape(-1)
        return embedding.tolist()


class TransformersSpeakerBackend:
    """Optional transformers-based speaker embedding backend."""

    def __init__(
        self,
        model_id: str = "microsoft/wavlm-base-plus-sv",
        *,
        device: int | str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        batch_size: int = 16,
    ) -> None:
        try:
            import torch
            import torchaudio
            from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
        except ImportError as exc:  # pragma: no cover - exercised only when optional deps missing.
            raise RuntimeError(
                "transformers, torch, and torchaudio are required for the transformers speaker backend"
            ) from exc

        self._torch = torch
        self._torchaudio = torchaudio
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        normalized_model_kwargs = _normalize_transformers_model_kwargs(model_kwargs or {})
        self._model = AutoModelForAudioXVector.from_pretrained(model_id, **normalized_model_kwargs)
        self._device = self._normalize_device(device)
        self._model.to(self._device)
        self._model.eval()
        self._sampling_rate = int(getattr(self._feature_extractor, "sampling_rate", 16000))
        self.batch_size = max(int(batch_size), 1)

    def embed(
        self,
        audio_path: Path,
        *,
        language: str = "",
        row: Mapping[str, str] | None = None,
    ) -> Sequence[float]:
        del language, row
        return self.embed_many([audio_path])[0]

    def embed_many(
        self,
        audio_paths: Sequence[Path],
        *,
        language: str = "",
        rows: Sequence[Mapping[str, str] | None] | None = None,
    ) -> list[list[float]]:
        del language, rows
        if not audio_paths:
            return []
        waveforms: list[np.ndarray] = []
        for audio_path in audio_paths:
            samples, sample_rate = _read_audio_mono(audio_path)
            waveform = self._torch.from_numpy(samples).float()
            if sample_rate != self._sampling_rate:
                waveform = self._torchaudio.functional.resample(waveform, sample_rate, self._sampling_rate)
            waveforms.append(waveform.cpu().numpy())
        inputs = self._feature_extractor(
            waveforms,
            sampling_rate=self._sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        embeddings = self._forward_embeddings(inputs)
        return [embeddings[index].detach().cpu().numpy().reshape(-1).tolist() for index in range(embeddings.shape[0])]

    def _forward_embeddings(self, inputs: Mapping[str, Any]) -> Any:
        model_dtype = next(self._model.parameters()).dtype
        normalized_inputs: dict[str, Any] = {}
        for key, value in inputs.items():
            tensor = value.to(self._device)
            if self._torch.is_floating_point(tensor):
                tensor = tensor.to(dtype=model_dtype)
            normalized_inputs[key] = tensor
        with self._torch.inference_mode():
            outputs = self._model(**normalized_inputs)
        embeddings = getattr(outputs, "embeddings", None)
        if embeddings is None:
            embeddings = getattr(outputs, "xvector", None)
        if embeddings is None:
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                raise RuntimeError("speaker model did not return embeddings or hidden states")
            embeddings = hidden.mean(dim=1)
        return self._torch.nn.functional.normalize(embeddings, dim=-1)

    def _normalize_device(self, device: int | str | None) -> Any:
        if device is None:
            return self._torch.device("cuda:0" if self._torch.cuda.is_available() else "cpu")
        if isinstance(device, int):
            return self._torch.device(f"cuda:{device}" if self._torch.cuda.is_available() else "cpu")
        return self._torch.device(device)


def build_identity_asr_backend() -> IdentityAsrBackend:
    return IdentityAsrBackend()


def build_transformers_asr_backend(
    model_id: str,
    *,
    device: int | str | None = None,
    chunk_length_s: float = 30.0,
    batch_size: int = 1,
    trust_remote_code: bool = False,
    generate_kwargs: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    pipeline_kwargs: dict[str, Any] | None = None,
) -> TransformersAsrBackend:
    return TransformersAsrBackend(
        model_id=model_id,
        device=device,
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
        trust_remote_code=trust_remote_code,
        generate_kwargs=generate_kwargs,
        model_kwargs=model_kwargs,
        pipeline_kwargs=pipeline_kwargs,
    )


def build_speechbrain_speaker_backend(
    source: str = "speechbrain/spkrec-ecapa-voxceleb",
    *,
    savedir: str | None = None,
    device: str | None = None,
) -> SpeechBrainSpeakerBackend:
    run_opts = {"device": device} if device else None
    return SpeechBrainSpeakerBackend(source=source, savedir=savedir, run_opts=run_opts)


def build_transformers_speaker_backend(
    model_id: str = "microsoft/wavlm-base-plus-sv",
    *,
    device: int | str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    batch_size: int = 16,
) -> TransformersSpeakerBackend:
    return TransformersSpeakerBackend(
        model_id=model_id,
        device=device,
        model_kwargs=model_kwargs,
        batch_size=batch_size,
    )


BACKEND_SHORTCUTS = {
    "identity_asr": "eval.preservation:build_identity_asr_backend",
    "transformers_asr": "eval.preservation:build_transformers_asr_backend",
    "speechbrain_speaker": "eval.preservation:build_speechbrain_speaker_backend",
    "transformers_speaker": "eval.preservation:build_transformers_speaker_backend",
}


def _parse_backend_kwargs(raw_value: str) -> dict[str, Any]:
    if not raw_value:
        return {}
    parsed = json.loads(raw_value)
    if not isinstance(parsed, dict):
        raise ValueError("backend kwargs must decode to a JSON object")
    return parsed


def _normalize_transformers_model_kwargs(model_kwargs: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(model_kwargs)
    for key in ("torch_dtype", "dtype"):
        dtype_value = normalized.get(key)
        if not isinstance(dtype_value, str):
            continue
        try:
            import torch
        except ImportError:
            return normalized
        dtype = getattr(torch, dtype_value, None)
        if dtype is not None:
            normalized[key] = dtype
    return normalized


def load_backend_from_spec(spec: str, kwargs: dict[str, Any] | None = None) -> Any:
    if not spec:
        return None
    resolved_spec = BACKEND_SHORTCUTS.get(spec, spec)
    module_name, separator, attribute_name = resolved_spec.partition(":")
    if not module_name or not separator or not attribute_name:
        raise ValueError(f"backend spec must look like module:path, got {spec!r}")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if factory is None or not callable(factory):
        raise ValueError(f"backend factory not found: {resolved_spec}")
    return factory(**(kwargs or {}))


def load_backend_from_cli(spec: str, kwargs_json: str) -> Any:
    return load_backend_from_spec(spec, _parse_backend_kwargs(kwargs_json))


def _call_backend(
    backend: Any,
    method_name: str,
    audio_path: Path,
    *,
    language: str,
    row: Mapping[str, str],
) -> Any:
    candidate = getattr(backend, method_name, None)
    if candidate is None:
        if callable(backend):
            candidate = backend
        else:
            raise TypeError(f"backend for {method_name} is not callable")
    try:
        return candidate(audio_path, language=language, row=row)
    except TypeError:
        try:
            return candidate(audio_path, language=language)
        except TypeError:
            return candidate(audio_path)


__all__ = [
    "AsrBackend",
    "BACKEND_SHORTCUTS",
    "IdentityAsrBackend",
    "SpeakerEmbeddingBackend",
    "SpeechBrainSpeakerBackend",
    "TransformersAsrBackend",
    "TransformersSpeakerBackend",
    "build_identity_asr_backend",
    "build_speechbrain_speaker_backend",
    "build_transformers_asr_backend",
    "build_transformers_speaker_backend",
    "load_backend_from_cli",
    "load_backend_from_spec",
]
