"""Common helpers for ASVspoof-like baseline wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaselineRunner
from ..tasks import TaskPack
from ..views import (
    build_asvspoof_view,
    extract_score_rows,
    write_normalized_scores,
)


class ASVspoofBaselineRunner(BaselineRunner):
    runtime_module = ""
    audio_extension = "flac"
    train_track = "LA"
    eval_track = "DF"
    eval_protocol_key = "eval_protocol_2021"

    def _shared_view_root(self, run_dir: Path) -> Path:
        adapter = self.config.get("adapter", {})
        train_track = adapter.get("train_track", self.train_track)
        eval_track = adapter.get("eval_track", self.eval_track)
        audio_extension = adapter.get("audio_extension", self.audio_extension)
        return run_dir.parent / "_views" / f"asvspoof_{train_track}_{eval_track}_{audio_extension}"

    def prepare_view(self, data_view: TaskPack, run_dir: Path, dataset_root: Path) -> dict[str, Any]:
        asvspoof = build_asvspoof_view(
            data_view,
            run_dir,
            dataset_root,
            root=self._shared_view_root(run_dir),
            train_track=self.config.get("adapter", {}).get("train_track", self.train_track),
            eval_track=self.config.get("adapter", {}).get("eval_track", self.eval_track),
            audio_extension=self.config.get("adapter", {}).get("audio_extension", self.audio_extension),
        )
        prepared = {
            "asvspoof_root": str(asvspoof.root),
            "database_root": str(asvspoof.database_root),
            "protocols_root": str(asvspoof.protocols_root),
            "train_protocol": str(asvspoof.train_protocol),
            "dev_protocol": str(asvspoof.dev_protocol),
            "eval_protocol_2019": str(asvspoof.eval_protocol_2019),
            "eval_protocol_2021": str(asvspoof.eval_protocol_2021),
            "eval_protocol_2021_la": str(asvspoof.eval_protocol_2021_la),
            "eval_protocol_itw": str(asvspoof.eval_protocol_itw),
            "asv_score_path": str(asvspoof.asv_score_path),
        }
        return prepared

    def normalize_scores(self, prepared_view: dict[str, Any], run_dir: Path, raw_output_path: Path) -> Path:
        scores = extract_score_rows(raw_output_path)
        return write_normalized_scores(run_dir / "scores.csv", scores)

    def _checkpoint_path(self, run_dir: Path) -> Path:
        return run_dir / "checkpoints" / "best.pth"

    def _scores_path(self, run_dir: Path) -> Path:
        return run_dir / "raw_scores.txt"

    def _audio_roots(self, prepared_view: dict[str, Any]) -> tuple[Path, Path, Path]:
        database_root = Path(prepared_view["database_root"])
        adapter = self.config.get("adapter", {})
        train_track = adapter.get("train_track", self.train_track)
        eval_track = adapter.get("eval_track", self.eval_track)
        audio_extension = adapter.get("audio_extension", self.audio_extension)
        return (
            database_root / f"ASVspoof2019_{train_track}_train" / audio_extension,
            database_root / f"ASVspoof2019_{train_track}_dev" / audio_extension,
            database_root / f"ASVspoof2021_{eval_track}_eval" / audio_extension,
        )

    def _runtime_extra_args(self) -> list[str]:
        return []

    def _runtime_command(
        self,
        prepared_view: dict[str, Any],
        run_dir: Path,
        *,
        mode: str,
        checkpoint: Path,
    ) -> list[str]:
        train_cfg = self.config["train"]
        eval_cfg = self.config["eval"]
        train_audio_root, dev_audio_root, eval_audio_root = self._audio_roots(prepared_view)
        return self._command_prefix() + [
            "-m",
            self.runtime_module,
            "--repo-path",
            str(self.repo_path),
            "--mode",
            mode,
            "--train-protocol",
            prepared_view["train_protocol"],
            "--dev-protocol",
            prepared_view["dev_protocol"],
            "--eval-protocol",
            prepared_view[self.eval_protocol_key],
            "--train-audio-root",
            str(train_audio_root),
            "--dev-audio-root",
            str(dev_audio_root),
            "--eval-audio-root",
            str(eval_audio_root),
            "--checkpoint-path",
            str(checkpoint),
            "--scores-path",
            str(self._scores_path(run_dir)),
            "--train-device",
            str(train_cfg["device"]),
            "--eval-device",
            str(eval_cfg["device"]),
            "--train-batch-size",
            str(train_cfg["batch_size"]),
            "--eval-batch-size",
            str(eval_cfg["batch_size"]),
            "--train-num-workers",
            str(train_cfg["num_workers"]),
            "--eval-num-workers",
            str(eval_cfg["num_workers"]),
            "--train-pin-memory",
            str(train_cfg["pin_memory"]).lower(),
            "--eval-pin-memory",
            str(eval_cfg["pin_memory"]).lower(),
            "--train-persistent-workers",
            str(train_cfg["persistent_workers"]).lower(),
            "--eval-persistent-workers",
            str(eval_cfg["persistent_workers"]).lower(),
            "--train-prefetch-factor",
            str(train_cfg["prefetch_factor"]),
            "--eval-prefetch-factor",
            str(eval_cfg["prefetch_factor"]),
            "--epochs",
            str(train_cfg["epochs"]),
            "--learning-rate",
            str(train_cfg["learning_rate"]),
            "--weight-decay",
            str(train_cfg["weight_decay"]),
            "--seed",
            str(train_cfg["seed"]),
            *self._runtime_extra_args(),
        ]
