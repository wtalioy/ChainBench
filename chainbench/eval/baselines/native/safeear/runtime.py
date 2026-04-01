"""ChainBench-native runtime for SafeEar."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import MethodType
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChainBench-native SafeEar runtime")
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--eval-config-path", default="")
    parser.add_argument("--mode", choices=("train", "eval"), required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def _instantiate_system(cfg: Any) -> tuple[Any, Any]:
    import hydra
    import torch

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    decouple_model = hydra.utils.instantiate(cfg.decouple_model)
    decouple_model.load_state_dict(torch.load(cfg.speechtokenizer_path))
    detect_model = hydra.utils.instantiate(cfg.detect_model)
    system = hydra.utils.instantiate(
        cfg.system,
        decouple_model=decouple_model,
        detect_model=detect_model,
    )
    return datamodule, system


def _patch_test_writer(system: Any, output_path: Path) -> None:
    import numpy as np
    import torch

    def patched_on_test_epoch_end(self):  # type: ignore[no-redef]
        string_list = [list(item) for item in self.eval_filename_loader]
        all_filename = np.array(string_list).reshape(-1)
        all_index = torch.cat(self.eval_index_loader, dim=0).view(-1).cpu().numpy()
        all_score = torch.cat(self.eval_score_loader, dim=0).view(-1).cpu().numpy()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["audio_path", "score", "label"])
            writer.writeheader()
            for audio_path, score, label_idx in zip(all_filename, all_score, all_index):
                writer.writerow(
                    {
                        "audio_path": str(audio_path),
                        "score": float(score),
                        "label": "bonafide" if int(label_idx) == 0 else "spoof",
                    }
                )
        self.eval_index_loader.clear()
        self.eval_score_loader.clear()
        self.eval_filename_loader.clear()

    system.on_test_epoch_end = MethodType(patched_on_test_epoch_end, system)


def _pad_sequence(batch: list[Any]) -> Any:
    import torch

    batch = [item.permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    return batch.permute(0, 2, 1)


def _safeear_test_collate(batch: list[tuple[Any, ...]]) -> tuple[Any, ...]:
    import torch

    if not batch:
        raise ValueError("SafeEar test batch is empty.")

    wavs = []
    feats = []
    targets = []
    audio_paths = []
    include_paths = len(batch[0]) == 4
    for sample in batch:
        if len(sample) not in (3, 4):
            raise ValueError(f"Unexpected SafeEar test sample size: {len(sample)}")
        wav, feat, target = sample[:3]
        wavs.append(wav)
        feats.append(feat)
        targets.append(target)
        if include_paths:
            audio_paths.append(sample[3])

    padded_wavs = _pad_sequence(wavs)
    padded_feats = _pad_sequence(feats).permute(0, 2, 1)
    target_tensor = torch.tensor(targets).long()
    if include_paths:
        return padded_wavs, padded_feats, target_tensor, audio_paths
    return padded_wavs, padded_feats, target_tensor


def _build_test_dataloader(datamodule: Any) -> Any:
    from torch.utils.data import DataLoader

    datamodule.setup(stage="test")
    num_workers = int(getattr(datamodule.hparams, "num_workers", 0))
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(datamodule.hparams.batch_size),
        "num_workers": num_workers,
        "pin_memory": bool(getattr(datamodule.hparams, "pin_memory", False)),
        "shuffle": False,
        "collate_fn": _safeear_test_collate,
    }
    persistent_workers = bool(getattr(datamodule.hparams, "persistent_workers", False))
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        prefetch_factor = getattr(datamodule.hparams, "prefetch_factor", None)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(datamodule.data_test, **loader_kwargs)


def _device_count(cfg: Any) -> int:
    devices = cfg.trainer.devices
    if isinstance(devices, int):
        return devices
    if isinstance(devices, (list, tuple)):
        return len(devices)
    if isinstance(devices, str):
        return len([item for item in devices.split(",") if item.strip()])
    return 1


def _trainer_overrides(cfg: Any) -> dict[str, Any]:
    if _device_count(cfg) <= 1:
        return {}
    from pytorch_lightning.strategies.ddp import DDPStrategy

    return {"strategy": DDPStrategy(find_unused_parameters=True)}


def _run_train(cfg: Any) -> tuple[Path | None, Any, Any, Any]:
    import hydra
    from pytorch_lightning import Callback, Trainer

    datamodule, system = _instantiate_system(cfg)
    datamodule.setup()

    callbacks: list[Callback] = []
    checkpoint_callback = None
    if cfg.get("early_stopping"):
        callbacks.append(hydra.utils.instantiate(cfg.early_stopping))
    if cfg.get("checkpoint"):
        checkpoint_callback = hydra.utils.instantiate(cfg.checkpoint)
        callbacks.append(checkpoint_callback)

    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name, "logs"), exist_ok=True)
    logger = hydra.utils.instantiate(cfg.logger)
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        **_trainer_overrides(cfg),
    )
    trainer.fit(system, datamodule=datamodule)

    if checkpoint_callback is not None:
        best_k = {k: v.item() for k, v in checkpoint_callback.best_k_models.items()}
        with open(os.path.join(cfg.exp.dir, cfg.exp.name, "best_k_models.json"), "w", encoding="utf-8") as handle:
            json.dump(best_k, handle, indent=2)
        if checkpoint_callback.best_model_path:
            return Path(checkpoint_callback.best_model_path), datamodule, system, trainer
    return None, datamodule, system, trainer


def _run_eval(cfg: Any, checkpoint_path: str, output_path: Path) -> None:
    import hydra
    from pytorch_lightning import Trainer

    datamodule, system = _instantiate_system(cfg)
    _patch_test_writer(system, output_path)
    test_dataloader = _build_test_dataloader(datamodule)
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        **_trainer_overrides(cfg),
    )
    trainer.test(system, dataloaders=test_dataloader, ckpt_path=checkpoint_path or None)


def _run_eval_reusing_training(
    *,
    datamodule: Any,
    system: Any,
    trainer: Any,
    checkpoint_path: str,
    output_path: Path,
) -> None:
    _patch_test_writer(system, output_path)
    test_dataloader = _build_test_dataloader(datamodule)
    trainer.test(system, dataloaders=test_dataloader, ckpt_path=checkpoint_path or None)


def _can_reuse_training_runtime(train_cfg: Any, eval_cfg: Any) -> bool:
    return (
        train_cfg.trainer.devices == eval_cfg.trainer.devices
        and train_cfg.datamodule.batch_size == eval_cfg.datamodule.batch_size
    )


def main() -> int:
    args = parse_args()
    repo_path = Path(args.repo_path).resolve()
    config_path = Path(args.config_path).resolve()
    output_path = Path(args.output_path).resolve()

    sys.path.insert(0, str(repo_path))

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    eval_cfg = OmegaConf.load(args.eval_config_path) if args.eval_config_path else cfg
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.exp.dir, cfg.exp.name, "config.yaml"))
    if args.eval_config_path:
        os.makedirs(os.path.join(eval_cfg.exp.dir, eval_cfg.exp.name), exist_ok=True)
        OmegaConf.save(eval_cfg, os.path.join(eval_cfg.exp.dir, eval_cfg.exp.name, "config_eval.yaml"))

    if args.mode == "train":
        best_model_path, datamodule, system, trainer = _run_train(cfg)
        checkpoint_arg = str(best_model_path or args.checkpoint_path)
        if _can_reuse_training_runtime(cfg, eval_cfg):
            _run_eval_reusing_training(
                datamodule=datamodule,
                system=system,
                trainer=trainer,
                checkpoint_path=checkpoint_arg,
                output_path=output_path,
            )
        else:
            _run_eval(eval_cfg, checkpoint_arg, output_path)
    else:
        _run_eval(cfg, args.checkpoint_path, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
