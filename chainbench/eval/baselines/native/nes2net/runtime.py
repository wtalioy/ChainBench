"""ChainBench-native runtime for Nes2Net."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChainBench-native Nes2Net runtime")
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
    parser.add_argument("--model-name", default="wav2vec2_Nes2Net_X")
    parser.add_argument("--pool-func", default="mean")
    parser.add_argument("--dilation", type=int, default=2)
    parser.add_argument("--nes-ratio", nargs="+", type=int, default=[8, 8])
    parser.add_argument("--se-ratio", nargs="+", type=int, default=[1])
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

    from model_scripts.wav2vec2_Nes2Net_X import wav2vec2_Nes2Net_no_Res_w_allT

    seed_everything(args.seed)
    train_device = resolve_device(args.train_device)
    eval_device = resolve_device(args.eval_device)
    model_args = SimpleNamespace(
        n_output_logits=2,
        dilation=args.dilation,
        pool_func=args.pool_func,
        Nes_ratio=args.nes_ratio,
        SE_ratio=args.se_ratio,
    )
    model = prepare_model_for_devices(
        wav2vec2_Nes2Net_no_Res_w_allT(
            model_args,
            str(train_device if args.mode == "train" else eval_device),
        ),
        args.train_device if args.mode == "train" else args.eval_device,
    )
    checkpoint_path = Path(args.checkpoint_path)

    if args.mode == "train":
        _, train_loader = build_loader(
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
        _, dev_loader = build_loader(
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
            model = prepare_model_for_devices(
                wav2vec2_Nes2Net_no_Res_w_allT(model_args, str(eval_device)),
                args.eval_device,
            )
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
