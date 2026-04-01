"""ChainBench-native runtime for SLS-DF."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from ....runtime_support import (
    build_loader,
    evaluate_scores,
    fit_classifier,
    load_model_state,
    prepare_model_for_devices,
    read_protocol,
    resolve_device,
    seed_everything,
    str_to_bool,
    unwrap_model,
    write_scores,
)

SLS_RAWBOOST_DEFAULTS = {
    "nBands": 5,
    "minF": 20,
    "maxF": 8000,
    "minBW": 100,
    "maxBW": 1000,
    "minCoeff": 10,
    "maxCoeff": 100,
    "minG": 0,
    "maxG": 0,
    "minBiasLinNonLin": 5,
    "maxBiasLinNonLin": 20,
    "N_f": 5,
    "P": 10,
    "g_sd": 2,
    "SNRmin": 10,
    "SNRmax": 40,
    "cudnn_deterministic_toggle": True,
    "cudnn_benchmark_toggle": False,
}


def _extract_feat_parallel_safe(self, input_data):
    model_tensor = next(self.model.parameters(), None)
    if model_tensor is None:
        model_tensor = next(self.model.buffers(), None)
    if model_tensor is None or model_tensor.device != input_data.device or model_tensor.dtype != input_data.dtype:
        self.model.to(input_data.device, dtype=input_data.dtype)
        self.model.train()

    if input_data.ndim == 3:
        input_tmp = input_data[:, :, 0]
    else:
        input_tmp = input_data
    output = self.model(input_tmp, mask=False, features_only=True)
    return output["x"], output["layer_results"]


class _SlsTrainDatasetWithIds(Dataset):
    def __init__(self, protocol_path: str, audio_root: str, augmentation_args: SimpleNamespace, algo: int) -> None:
        from data_utils_SSL import Dataset_ASVspoof2019_train

        self.rows = read_protocol(Path(protocol_path))
        self.sample_ids = [row.sample_id for row in self.rows]
        labels = {row.sample_id: row.label for row in self.rows}
        split_root = Path(audio_root).resolve().parent
        self.dataset = Dataset_ASVspoof2019_train(
            augmentation_args,
            list_IDs=self.sample_ids,
            labels=labels,
            base_dir=f"{split_root}/",
            algo=algo,
        )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int):
        audio, label = self.dataset[index]
        return audio, label, self.sample_ids[index]


def _build_sls_loader(
    protocol_path: str,
    audio_root: str,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
    shuffle: bool,
    drop_last: bool,
    augmentation_args: SimpleNamespace,
    algo: int,
) -> tuple[list[Any], DataLoader]:
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
    dataset = _SlsTrainDatasetWithIds(protocol_path, audio_root, augmentation_args, algo)
    return dataset.rows, DataLoader(dataset, **loader_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChainBench-native SLS runtime")
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--mode", choices=("train", "eval"), required=True)
    parser.add_argument("--train-protocol", required=True)
    parser.add_argument("--dev-protocol", required=True)
    parser.add_argument("--eval-protocol", required=True)
    parser.add_argument("--train-audio-root", required=True)
    parser.add_argument("--dev-audio-root", required=True)
    parser.add_argument("--eval-audio-root", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--scores-path", required=True)
    parser.add_argument("--train-device", required=True)
    parser.add_argument("--eval-device", required=True)
    parser.add_argument("--train-batch-size", type=int, required=True)
    parser.add_argument("--eval-batch-size", type=int, required=True)
    parser.add_argument("--train-num-workers", type=int, required=True)
    parser.add_argument("--eval-num-workers", type=int, required=True)
    parser.add_argument("--train-pin-memory", type=str_to_bool, required=True)
    parser.add_argument("--eval-pin-memory", type=str_to_bool, required=True)
    parser.add_argument("--train-persistent-workers", type=str_to_bool, required=True)
    parser.add_argument("--eval-persistent-workers", type=str_to_bool, required=True)
    parser.add_argument("--train-prefetch-factor", type=int, required=True)
    parser.add_argument("--eval-prefetch-factor", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--algo", type=int, default=5)
    parser.add_argument("--xlsr-path", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_path = Path(args.repo_path).resolve()
    sys.path.insert(0, str(repo_path))
    os.chdir(repo_path)

    if args.xlsr_path:
        src = Path(args.xlsr_path).resolve()
        dst = repo_path / "xlsr2_300m.pt"
        if src.exists() and not dst.exists():
            dst.symlink_to(src)

    from core_scripts.startup_config import set_random_seed
    from model import Model, SSLModel

    SSLModel.extract_feat = _extract_feat_parallel_safe

    augmentation_args = SimpleNamespace(**SLS_RAWBOOST_DEFAULTS)
    set_random_seed(args.seed, augmentation_args)
    seed_everything(args.seed)
    train_device = resolve_device(args.train_device)
    eval_device = resolve_device(args.eval_device)
    model_args = SimpleNamespace(algo=args.algo)
    model = prepare_model_for_devices(
        Model(model_args, str(train_device if args.mode == "train" else eval_device)),
        args.train_device if args.mode == "train" else args.eval_device,
    )
    checkpoint_path = Path(args.checkpoint_path)

    if args.mode == "train":
        _, train_loader = _build_sls_loader(
            args.train_protocol,
            args.train_audio_root,
            batch_size=args.train_batch_size,
            num_workers=args.train_num_workers,
            pin_memory=args.train_pin_memory,
            persistent_workers=args.train_persistent_workers,
            prefetch_factor=args.train_prefetch_factor,
            shuffle=True,
            drop_last=True,
            augmentation_args=augmentation_args,
            algo=args.algo,
        )
        _, dev_loader = _build_sls_loader(
            args.dev_protocol,
            args.dev_audio_root,
            batch_size=args.train_batch_size,
            num_workers=args.train_num_workers,
            pin_memory=args.train_pin_memory,
            persistent_workers=args.train_persistent_workers,
            prefetch_factor=args.train_prefetch_factor,
            shuffle=False,
            drop_last=False,
            augmentation_args=augmentation_args,
            algo=args.algo,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        fit_classifier(
            model,
            train_loader,
            dev_loader,
            train_device,
            epochs=args.epochs,
            optimizer=optimizer,
            checkpoint_path=checkpoint_path,
        )
        if checkpoint_path.exists():
            model = prepare_model_for_devices(Model(model_args, str(eval_device)), args.eval_device)
            load_model_state(model, checkpoint_path, eval_device)
        else:
            model = prepare_model_for_devices(unwrap_model(model), args.eval_device)
    else:
        model = prepare_model_for_devices(unwrap_model(model), args.eval_device)
        load_model_state(model, checkpoint_path, eval_device)

    _, eval_loader = build_loader(
        args.eval_protocol,
        args.eval_audio_root,
        batch_size=args.eval_batch_size,
        num_workers=args.eval_num_workers,
        pin_memory=args.eval_pin_memory,
        persistent_workers=args.eval_persistent_workers,
        prefetch_factor=args.eval_prefetch_factor,
        extension="flac",
        max_len=64600,
        random_crop=False,
        shuffle=False,
        drop_last=False,
    )
    _, score_rows = evaluate_scores(model, eval_loader, eval_device, progress_desc="eval")
    write_scores(Path(args.scores_path), score_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
