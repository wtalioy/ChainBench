"""ChainBench-native Nes2Net pipeline."""

from __future__ import annotations

from pathlib import Path

from ...asvspoof import ASVspoofBaselineRunner
from ...base import BaselineRunResult


class Nes2NetRunner(ASVspoofBaselineRunner):
    """Own the ChainBench -> Nes2Net train/eval translation."""

    name = "nes2net"
    runtime_module = "eval.baselines.native.nes2net.runtime"
    checkpoint_patterns = ("*.pth",)
    train_track = "LA"
    eval_track = "DF"

    def _runtime_extra_args(self) -> list[str]:
        adapter = self.config.get("adapter", {})
        nes_ratio = [str(item) for item in adapter.get("Nes_ratio", [8, 8])]
        se_ratio = [str(item) for item in adapter.get("SE_ratio", [1])]
        args = [
            "--model-name",
            adapter.get("model_name", "wav2vec2_Nes2Net_X"),
            "--pool-func",
            adapter.get("pool_func", "mean"),
            "--dilation",
            str(adapter.get("dilation", 2)),
        ]
        if nes_ratio:
            args.extend(["--nes-ratio", *nes_ratio])
        if se_ratio:
            args.extend(["--se-ratio", *se_ratio])
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
