"""ChainBench-native Nes2Net pipeline."""

from __future__ import annotations

from ...asvspoof import ASVspoofBaselineRunner


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
