"""ChainBench-native AASIST pipeline."""

from __future__ import annotations

from ...asvspoof import ASVspoofBaselineRunner


class AASISTRunner(ASVspoofBaselineRunner):
    """Own the ChainBench -> AASIST train/eval translation."""

    name = "aasist"
    runtime_module = "eval.baselines.native.aasist.runtime"
    checkpoint_patterns = ("best.pth", "epoch_*.pth")
    train_track = "LA"
    eval_track = "LA"
    eval_protocol_key = "eval_protocol_2021_la"

    def _runtime_extra_args(self) -> list[str]:
        template_name = self.config.get("adapter", {}).get("template", "AASIST.conf")
        return ["--config-template", str(template_name)]


class AASISTLRunner(AASISTRunner):
    """Thin alias for the large AASIST config variant."""

    name = "aasist-l"
