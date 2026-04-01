"""Shared training and evaluation helpers for native baseline runtimes."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .metrics import compute_eer_from_labels
from .progress import create_progress


@dataclass
class ProtocolRow:
    sample_id: str
    label: int


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot interpret {value!r} as a boolean")


def resolve_device(device_name: str) -> torch.device:
    primary_name = split_device_names(device_name)[0]
    if primary_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(primary_name)


def split_device_names(device_name: str) -> list[str]:
    parts = [part.strip() for part in str(device_name).split(",") if part.strip()]
    if not parts:
        return ["cpu"]
    if len(parts) == 1:
        return parts
    if parts[0].startswith("cuda:"):
        return [parts[0], *[part if part.startswith("cuda:") else f"cuda:{part}" for part in parts[1:]]]
    if all(part.isdigit() for part in parts):
        return [f"cuda:{part}" for part in parts]
    return parts


def resolve_device_ids(device_name: str) -> list[int]:
    device_ids: list[int] = []
    for name in split_device_names(device_name):
        if name.startswith("cuda:"):
            _, gpu_idx = name.split(":", 1)
            device_ids.append(int(gpu_idx))
    return device_ids


def prepare_model_for_devices(model: torch.nn.Module, device_name: str) -> torch.nn.Module:
    device = resolve_device(device_name)
    model = model.to(device)
    device_ids = resolve_device_ids(device_name)
    if len(device_ids) > 1 and torch.cuda.is_available():
        return nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    return model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def save_model_state(model: torch.nn.Module, checkpoint_path: Path) -> None:
    torch.save(unwrap_model(model).state_dict(), checkpoint_path)


def load_model_state(model: torch.nn.Module, checkpoint_path: Path, map_location: torch.device) -> None:
    state_dict = torch.load(checkpoint_path, map_location=map_location)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    unwrap_model(model).load_state_dict(state_dict)


def compute_eer(target_scores: np.ndarray, nontarget_scores: np.ndarray) -> float:
    labels = np.concatenate((np.ones(target_scores.size, dtype=np.int32), np.zeros(nontarget_scores.size, dtype=np.int32)))
    predictions = np.concatenate((target_scores, nontarget_scores)).astype(np.float64, copy=False)
    eer, _ = compute_eer_from_labels(labels, predictions)
    return eer


def pad_audio(x: np.ndarray, max_len: int = 64600, random_crop: bool = False) -> np.ndarray:
    if x.shape[0] >= max_len:
        if random_crop:
            start = np.random.randint(0, x.shape[0] - max_len + 1)
            return x[start : start + max_len]
        return x[:max_len]
    return np.tile(x, int(max_len / x.shape[0]) + 1)[:max_len]


def read_protocol(protocol_path: Path) -> list[ProtocolRow]:
    rows: list[ProtocolRow] = []
    for line in protocol_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            _, sample_id, _, _, label = line.strip().split()
            rows.append(ProtocolRow(sample_id=sample_id, label=1 if label == "bonafide" else 0))
    return rows


class AudioProtocolDataset(Dataset):
    def __init__(
        self,
        protocol_rows: list[ProtocolRow],
        audio_root: Path,
        *,
        extension: str = "flac",
        max_len: int = 64600,
        random_crop: bool,
    ) -> None:
        self.protocol_rows = protocol_rows
        self.audio_root = audio_root
        self.extension = extension
        self.max_len = max_len
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.protocol_rows)

    def __getitem__(self, index: int):
        row = self.protocol_rows[index]
        audio, _ = sf.read(str(self.audio_root / f"{row.sample_id}.{self.extension}"))
        audio = pad_audio(audio.astype(np.float32), self.max_len, self.random_crop)
        return torch.from_numpy(audio), row.label, row.sample_id


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(
    protocol_path: str,
    audio_root: str,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    extension: str = "flac",
    max_len: int = 64600,
    random_crop: bool,
    shuffle: bool,
    drop_last: bool,
) -> tuple[list[ProtocolRow], DataLoader]:
    rows = read_protocol(Path(protocol_path))
    worker_count = max(0, min(int(num_workers), os.cpu_count() or 1))
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": worker_count,
        "pin_memory": bool(pin_memory and torch.cuda.is_available()),
    }
    if worker_count > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None and int(prefetch_factor) > 0:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    loader = DataLoader(
        AudioProtocolDataset(
            rows,
            Path(audio_root),
            extension=extension,
            max_len=max_len,
            random_crop=random_crop,
        ),
        **loader_kwargs,
    )
    return rows, loader


def evaluate_scores(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    forward: Callable[[Any], Any] | None = None,
    progress_desc: str | None = None,
) -> tuple[float, list[tuple[str, float, int]]]:
    predict = forward or model
    model.eval()
    scores: list[tuple[str, float, int]] = []
    progress = create_progress(total=len(data_loader), desc=progress_desc, unit="batch", leave=False) if progress_desc else None
    with torch.no_grad():
        for batch_x, batch_y, batch_ids in data_loader:
            batch_x = batch_x.to(device)
            batch_out = predict(batch_x)
            batch_scores = batch_out[:, 1].detach().cpu().numpy().ravel().tolist()
            for sample_id, score, label in zip(batch_ids, batch_scores, batch_y.tolist()):
                scores.append((sample_id, float(score), int(label)))
            if progress is not None:
                progress.update(1)
    if progress is not None:
        progress.close()
    bona = np.array([score for _, score, label in scores if label == 1], dtype=np.float64)
    spoof = np.array([score for _, score, label in scores if label == 0], dtype=np.float64)
    eer = compute_eer(bona, spoof) if len(bona) and len(spoof) else 1.0
    return eer, scores


def fit_classifier(
    model: torch.nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    scheduler: Any = None,
    forward: Callable[[Any], Any] | None = None,
    logger: Any | None = None,
) -> None:
    predict = forward or model
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]).to(device))
    best_eer = 1.0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    progress = create_progress(total=epochs, desc="train", unit="epoch")
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            loss = criterion(predict(batch_x), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        dev_eer, _ = evaluate_scores(model, dev_loader, device, forward=forward)
        if dev_eer <= best_eer:
            best_eer = dev_eer
            save_model_state(model, checkpoint_path)
        progress.set_postfix(
            {"epoch": f"{epoch + 1}/{epochs}", "dev_eer": f"{dev_eer:.4f}", "best": f"{best_eer:.4f}"},
            refresh=False,
        )
        progress.update(1)
    progress.close()
    if logger is not None:
        logger.info("training finished: best_dev_eer=%.4f epochs=%d", best_eer, epochs)


def write_scores(path: Path, rows: list[tuple[str, float, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample_id, score, _ in rows:
            handle.write(f"{sample_id} {score}\n")
