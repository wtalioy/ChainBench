"""ChainBench-native SafeEar pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...base import BaselineRunResult, BaselineRunner
from ....tasks import TaskPack
from ....views import build_safeear_view, extract_csv_score_rows, write_normalized_scores
from lib.logging import get_logger

WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
LOGGER = get_logger("eval.safeear")


def _device_to_lightning_devices(device_name: str) -> list[int]:
    device_parts = [part.strip() for part in str(device_name).split(",") if part.strip()]
    if not device_parts:
        return [0]
    if len(device_parts) == 1 and ":" not in device_parts[0]:
        return [0]
    devices: list[int] = []
    for part in device_parts:
        normalized = part if ":" in part else f"cuda:{part}"
        _, gpu_idx = normalized.split(":", 1)
        devices.append(int(gpu_idx))
    return devices or [0]


class SafeEarRunner(BaselineRunner):
    """Own the ChainBench -> SafeEar train/eval translation."""

    name = "safeear"
    checkpoint_patterns = ("*.ckpt",)

    def _resolve_config_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (WORKSPACE_ROOT / path).resolve()

    def _resolve_assets(self, run_dir: Path) -> dict[str, Path]:
        assets = self.config.get("assets", {})
        default_hubert_root = (run_dir.parent / "_views" / "safeear_hubert").resolve()
        hubert_2019_root = (
            self._resolve_config_path(assets["hubert_features_2019_root"])
            if assets.get("hubert_features_2019_root")
            else default_hubert_root
        )
        hubert_2021_root = (
            self._resolve_config_path(assets["hubert_features_2021_root"])
            if assets.get("hubert_features_2021_root")
            else hubert_2019_root
        )
        return {
            "speechtokenizer_path": self._resolve_config_path(assets["speechtokenizer_path"]),
            "hubert_checkpoint_path": self._resolve_config_path(
                assets.get("hubert_checkpoint_path", "baselines/SafeEar/model_zoos/hubert_base_ls960.pt")
            ),
            "hubert_features_2019_root": hubert_2019_root,
            "hubert_features_2021_root": hubert_2021_root,
        }

    def _first_missing_feature(self, audio_dir: Path, feature_dir: Path) -> Path | None:
        if not audio_dir.exists():
            raise FileNotFoundError(f"SafeEar audio directory does not exist: {audio_dir}")
        for audio_path in audio_dir.rglob("*.flac"):
            expected = (feature_dir / audio_path.relative_to(audio_dir)).with_suffix(".npy")
            if not expected.exists():
                return expected
        return None

    def _dump_hubert_split(
        self,
        *,
        split_name: str,
        audio_dir: Path,
        feature_dir: Path,
        checkpoint_path: Path,
        run_dir: Path,
    ) -> None:
        missing = self._first_missing_feature(audio_dir, feature_dir)
        if missing is None:
            return
        LOGGER.info("dumping missing SafeEar HuBERT features for %s into %s", split_name, feature_dir)
        feature_dir.mkdir(parents=True, exist_ok=True)
        command = self._command_prefix() + [
            str(self.repo_path / "datas" / "dump_hubert_avg_feature.py"),
            str(audio_dir),
            str(feature_dir),
            str(checkpoint_path),
            "9",
        ]
        result = self._run_command(
            command,
            cwd=self.repo_path / "datas",
            log_path=run_dir / f"prepare_{split_name}_hubert.log",
        )
        if not result.ok:
            raise RuntimeError(f"SafeEar HuBERT feature dump failed for {split_name}; see prepare_{split_name}_hubert.log")
        missing = self._first_missing_feature(audio_dir, feature_dir)
        if missing is not None:
            raise RuntimeError(f"SafeEar HuBERT feature dump for {split_name} is incomplete; missing {missing}")

    def _ensure_hubert_features(self, prepared_view: dict[str, Any], run_dir: Path, assets: dict[str, Path]) -> None:
        checkpoint_path = assets["hubert_checkpoint_path"]
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"SafeEar HuBERT checkpoint not found at {checkpoint_path}. "
                "Set assets.hubert_checkpoint_path or place hubert_base_ls960.pt in baselines/SafeEar/model_zoos/."
            )
        database_root = Path(prepared_view["database_root"])
        self._dump_hubert_split(
            split_name="train",
            audio_dir=database_root / "ASVspoof2019_LA_train" / "flac",
            feature_dir=assets["hubert_features_2019_root"] / "ASVspoof2019_LA_train" / "flac",
            checkpoint_path=checkpoint_path,
            run_dir=run_dir,
        )
        self._dump_hubert_split(
            split_name="dev",
            audio_dir=database_root / "ASVspoof2019_LA_dev" / "flac",
            feature_dir=assets["hubert_features_2019_root"] / "ASVspoof2019_LA_dev" / "flac",
            checkpoint_path=checkpoint_path,
            run_dir=run_dir,
        )
        self._dump_hubert_split(
            split_name="eval_2021",
            audio_dir=database_root / "ASVspoof2021_LA_eval" / "flac",
            feature_dir=assets["hubert_features_2021_root"] / "ASVspoof2021_LA_eval" / "flac",
            checkpoint_path=checkpoint_path,
            run_dir=run_dir,
        )

    def _template_path(self) -> Path:
        template_name = self.config.get("adapter", {}).get("template", "train21.yaml")
        return self.repo_path / "config" / template_name

    def _generated_config_path(self, run_dir: Path, mode: str) -> Path:
        return run_dir / "generated" / f"safeear_{mode}.yaml"

    def _output_root(self, run_dir: Path) -> Path:
        return run_dir / "outputs"

    def _normalize_scores(self, run_dir: Path, raw_output: Path) -> Path | None:
        return write_normalized_scores(run_dir / "scores.csv", extract_csv_score_rows(raw_output)) if raw_output.exists() else None

    def prepare_view(self, data_view: TaskPack, run_dir: Path, dataset_root: Path) -> dict[str, Any]:
        safeear_view = build_safeear_view(
            data_view,
            run_dir,
            dataset_root,
            root=run_dir.parent / "_views" / "safeear",
        )
        return {
            "safeear_root": str(safeear_view.root.resolve()),
            "database_root": str(safeear_view.asvspoof_view.database_root.resolve()),
            "train_tsv": str(safeear_view.train_tsv.resolve()),
            "dev_tsv": str(safeear_view.dev_tsv.resolve()),
            "test_tsv_2019": str(safeear_view.test_tsv_2019.resolve()),
            "test_tsv_2021": str(safeear_view.test_tsv_2021.resolve()),
            "train_protocol": str(safeear_view.asvspoof_view.train_protocol.resolve()),
            "dev_protocol": str(safeear_view.asvspoof_view.dev_protocol.resolve()),
            "eval_protocol_2019": str(safeear_view.asvspoof_view.eval_protocol_2019.resolve()),
            "eval_protocol_2021": str(safeear_view.asvspoof_view.eval_protocol_2021_la.resolve()),
        }

    def _build_config(self, prepared_view: dict[str, Any], run_dir: Path, *, mode: str, assets: dict[str, Path]) -> Path:
        import yaml

        cfg = yaml.safe_load(self._template_path().read_text(encoding="utf-8"))
        output_root = self._output_root(run_dir)
        exp_name = f"{self.name}_{run_dir.parent.parent.name}_{run_dir.parent.name}"
        phase_cfg = self.config[mode]
        trainer_device = _device_to_lightning_devices(phase_cfg["device"])

        cfg["datamodule"]["batch_size"] = int(phase_cfg["batch_size"])
        cfg["datamodule"]["DataClass_dict"]["train_path"] = [
            prepared_view["train_tsv"],
            prepared_view["train_protocol"],
            str(assets["hubert_features_2019_root"]),
        ]
        cfg["datamodule"]["DataClass_dict"]["val_path"] = [
            prepared_view["dev_tsv"],
            prepared_view["dev_protocol"],
            str(assets["hubert_features_2019_root"]),
        ]
        cfg["datamodule"]["DataClass_dict"]["test_path"] = [
            prepared_view["test_tsv_2021"],
            prepared_view["eval_protocol_2021"],
            str(assets["hubert_features_2021_root"]),
        ]
        cfg["speechtokenizer_path"] = str(assets["speechtokenizer_path"])
        cfg["exp"]["dir"] = str(output_root.resolve())
        cfg["exp"]["name"] = exp_name
        cfg["trainer"]["devices"] = trainer_device
        cfg["trainer"]["max_epochs"] = int(self.config["train"]["epochs"])
        cfg["trainer"]["enable_progress_bar"] = False
        cfg["trainer"]["progress_bar_refresh_rate"] = 0
        cfg["system"]["save_score_path"] = str((output_root / exp_name).resolve())

        config_path = self._generated_config_path(run_dir, mode)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        return config_path

    def train(self, prepared_view: dict[str, Any], run_dir: Path) -> BaselineRunResult:
        assets = self._resolve_assets(run_dir)
        self._ensure_hubert_features(prepared_view, run_dir, assets)
        train_config_path = self._build_config(prepared_view, run_dir, mode="train", assets=assets)
        eval_config_path = self._build_config(prepared_view, run_dir, mode="eval", assets=assets)
        raw_output = run_dir / "score.csv"
        command = self._command_prefix() + [
            "-m",
            "eval.baselines.native.safeear.runtime",
            "--repo-path",
            str(self.repo_path),
            "--config-path",
            str(train_config_path),
            "--eval-config-path",
            str(eval_config_path),
            "--mode",
            "train",
            "--checkpoint-path",
            str(run_dir / "checkpoints" / "best.ckpt"),
            "--output-path",
            str(raw_output),
        ]
        result = self._run_command(
            command,
            cwd=run_dir,
            log_path=run_dir / "train.log",
        )
        normalized = self._normalize_scores(run_dir, raw_output)
        return BaselineRunResult(
            ok=result.ok,
            returncode=result.returncode,
            command=command,
            model_path=self.find_checkpoint(self._output_root(run_dir)),
            raw_output_path=raw_output if raw_output.exists() else None,
            scores_path=normalized,
        )

    def evaluate(self, prepared_view: dict[str, Any], run_dir: Path, checkpoint: Path | None) -> BaselineRunResult:
        assets = self._resolve_assets(run_dir)
        self._ensure_hubert_features(prepared_view, run_dir, assets)
        config_path = self._build_config(prepared_view, run_dir, mode="eval", assets=assets)
        raw_output = run_dir / "score.csv"
        command = self._command_prefix() + [
            "-m",
            "eval.baselines.native.safeear.runtime",
            "--repo-path",
            str(self.repo_path),
            "--config-path",
            str(config_path),
            "--mode",
            "eval",
            "--checkpoint-path",
            str(checkpoint) if checkpoint else "",
            "--output-path",
            str(raw_output),
        ]
        result = self._run_command(
            command,
            cwd=run_dir,
            log_path=run_dir / "eval.log",
        )
        normalized = self._normalize_scores(run_dir, raw_output)
        return BaselineRunResult(
            ok=result.ok,
            returncode=result.returncode,
            command=command,
            model_path=checkpoint,
            raw_output_path=raw_output if raw_output.exists() else None,
            scores_path=normalized,
        )
