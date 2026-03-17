"""ChainBench-native AASIST pipeline."""

from __future__ import annotations

from pathlib import Path

from ...asvspoof import ASVspoofBaselineRunner
from ...base import BaselineRunResult


class AasistRunner(ASVspoofBaselineRunner):
    """Own the ChainBench -> AASIST train/eval translation."""

    name = "aasist"
    runtime_module = "eval.baselines.native.aasist.runtime"
    checkpoint_patterns = ("best.pth", "epoch_*.pth")
    train_track = "LA"
    eval_track = "LA"
    eval_protocol_key = "eval_protocol_2021_la"

    def train(self, prepared_view: dict[str, str], run_dir: Path) -> BaselineRunResult:
        checkpoint_path = self._checkpoint_path(run_dir)
        command = self._runtime_command(
            prepared_view,
            run_dir,
            mode="train",
            checkpoint=checkpoint_path,
        )
        result = self._run_command(
            command,
            cwd=run_dir,
            log_path=run_dir / "train.log",
        )
        raw_output = self._scores_path(run_dir)
        normalized = self.normalize_scores(prepared_view, run_dir, raw_output) if raw_output.exists() else None
        return BaselineRunResult(
            ok=result.ok,
            returncode=result.returncode,
            command=command,
            model_path=checkpoint_path if checkpoint_path.exists() else None,
            raw_output_path=raw_output if raw_output.exists() else None,
            scores_path=normalized,
        )

    def evaluate(self, prepared_view: dict[str, str], run_dir: Path, checkpoint: Path | None) -> BaselineRunResult:
        raw_output = self._scores_path(run_dir)
        checkpoint_path = checkpoint or self._checkpoint_path(run_dir)
        command = self._runtime_command(
            prepared_view,
            run_dir,
            mode="eval",
            checkpoint=checkpoint_path,
        )
        result = self._run_command(
            command,
            cwd=run_dir,
            log_path=run_dir / "eval.log",
        )
        normalized = self.normalize_scores(prepared_view, run_dir, raw_output) if raw_output.exists() else None
        return BaselineRunResult(
            ok=result.ok,
            returncode=result.returncode,
            command=command,
            model_path=checkpoint_path,
            raw_output_path=raw_output if raw_output.exists() else None,
            scores_path=normalized,
        )
