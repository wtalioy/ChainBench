"""ChainBench-native runtime for AASIST."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from chainbench.lib.logging import get_logger, setup_logging
from ....runtime_support import (
    build_loader,
    evaluate_scores,
    fit_classifier,
    load_model_state,
    prepare_model_for_devices,
    resolve_device,
    seed_everything,
    str_to_bool,
    unwrap_model,
    write_scores,
)


LOGGER = get_logger("eval.aasist.runtime")


def _load_model_args(repo_path: Path, template_name: str) -> dict[str, Any]:
    template_path = repo_path / "config" / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"AASIST config template not found: {template_path}")
    payload = json.loads(template_path.read_text(encoding="utf-8"))
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict) or not model_config:
        raise ValueError(f"AASIST config template missing model_config: {template_path}")
    return dict(model_config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChainBench-native AASIST runtime")
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--config-template", default="AASIST.conf")
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
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    LOGGER.info("starting")
    repo_path = Path(args.repo_path).resolve()
    sys.path.insert(0, str(repo_path))

    LOGGER.info("importing model")
    from models.AASIST import Model
    from utils import create_optimizer

    seed_everything(args.seed)
    train_device = resolve_device(args.train_device)
    eval_device = resolve_device(args.eval_device)
    LOGGER.info("using train device %s and eval device %s", train_device, eval_device)
    model_args = _load_model_args(repo_path, args.config_template)
    LOGGER.info("loaded model config from %s", repo_path / "config" / args.config_template)
    model = prepare_model_for_devices(
        Model(model_args),
        args.train_device if args.mode == "train" else args.eval_device,
    )
    checkpoint_path = Path(args.checkpoint_path)
    forward = lambda batch_x: model(batch_x)[1]

    if args.mode == "train":
        LOGGER.info("loading protocols and building dataloaders")
        train_rows, train_loader = build_loader(
            args.train_protocol,
            args.train_audio_root,
            batch_size=args.train_batch_size,
            num_workers=args.train_num_workers,
            pin_memory=args.train_pin_memory,
            persistent_workers=args.train_persistent_workers,
            prefetch_factor=args.train_prefetch_factor,
            extension="flac",
            max_len=64600,
            random_crop=True,
            shuffle=True,
            drop_last=True,
        )
        dev_rows, dev_loader = build_loader(
            args.dev_protocol,
            args.dev_audio_root,
            batch_size=args.train_batch_size,
            num_workers=args.train_num_workers,
            pin_memory=args.train_pin_memory,
            persistent_workers=args.train_persistent_workers,
            prefetch_factor=args.train_prefetch_factor,
            extension="flac",
            max_len=64600,
            random_crop=False,
            shuffle=False,
            drop_last=False,
        )
        LOGGER.info("train=%d, dev=%d", len(train_rows), len(dev_rows))
        optim_config = {
            "optimizer": "adam",
            "amsgrad": "false",
            "base_lr": args.learning_rate,
            "lr_min": 5e-6,
            "betas": [0.9, 0.999],
            "weight_decay": args.weight_decay,
            "scheduler": "cosine",
            "steps_per_epoch": len(train_loader),
            "epochs": args.epochs,
        }
        optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
        LOGGER.info("starting training for %d epochs", args.epochs)
        fit_classifier(
            model,
            train_loader,
            dev_loader,
            train_device,
            epochs=args.epochs,
            optimizer=optimizer,
            checkpoint_path=checkpoint_path,
            scheduler=scheduler,
            forward=forward,
            logger=LOGGER,
        )
        if checkpoint_path.exists():
            model = prepare_model_for_devices(Model(model_args), args.eval_device)
            forward = lambda batch_x: model(batch_x)[1]
            load_model_state(model, checkpoint_path, eval_device)
        else:
            model = prepare_model_for_devices(unwrap_model(model), args.eval_device)
            forward = lambda batch_x: model(batch_x)[1]
    else:
        LOGGER.info("loading checkpoint for evaluation")
        model = prepare_model_for_devices(unwrap_model(model), args.eval_device)
        forward = lambda batch_x: model(batch_x)[1]
        load_model_state(model, checkpoint_path, eval_device)

    LOGGER.info("running evaluation")
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
    _, score_rows = evaluate_scores(model, eval_loader, eval_device, forward=forward, progress_desc="eval")
    write_scores(Path(args.scores_path), score_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
