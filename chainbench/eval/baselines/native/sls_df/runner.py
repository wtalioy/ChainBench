"""ChainBench-native SLS pipeline."""

from __future__ import annotations

from ...asvspoof import ASVspoofBaselineRunner


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
