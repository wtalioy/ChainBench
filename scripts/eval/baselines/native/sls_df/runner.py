"""ChainBench-native SLS pipeline."""

from __future__ import annotations

from pathlib import Path

from ...asvspoof import ASVspoofBaselineRunner
from ...base import BaselineRunResult


class SlsDfRunner(ASVspoofBaselineRunner):
    """Own the ChainBench -> SLS train/eval translation."""

    name = "sls_df"
    runtime_module = "eval.baselines.native.sls_df.runtime"
    checkpoint_patterns = ("epoch_*.pth", "best.pth")
    train_track = "LA"
    eval_track = "DF"

    def _runtime_extra_args(self) -> list[str]:
        args = ["--algo", str(self.config.get("adapter", {}).get("algo", 5))]
        xlsr_path = self.config.get("assets", {}).get("xlsr_model_path", "")
        if xlsr_path:
            args.extend(["--xlsr-path", xlsr_path])
        return args

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
        if raw_output.exists():
            raw_output.unlink()
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
            model_path=checkpoint_path if checkpoint_path.exists() else None,
            raw_output_path=raw_output if raw_output.exists() else None,
            scores_path=normalized,
        )
