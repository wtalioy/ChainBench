"""Generator adapters: one class per TTS clone (Qwen3, CosyVoice3, SparkTTS, F5-TTS, VoxCPM, IndexTTS2)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .base import AdapterRunner, map_qwen_language, resolve_local_or_hf_model_dir


class Qwen3CloneRunner(AdapterRunner):
    def setup(self) -> None:
        os.chdir(self.repo_path)
        sys.path.insert(0, str(self.repo_path))
        import torch
        from qwen_tts import Qwen3TTSModel
        self.torch = torch
        dtype_name = self.config.get("dtype", "bfloat16")
        dtype = getattr(torch, dtype_name)
        self.model = Qwen3TTSModel.from_pretrained(
            self.config["model_path"],
            device_map=self.config.get("device", "cuda:0"),
            dtype=dtype,
            attn_implementation=self.config.get("attn_implementation", "flash_attention_2"),
        )

    def run_job(self, job: dict[str, Any]) -> dict[str, Any]:
        import soundfile as sf
        output_path = Path(job["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wavs, sr = self.model.generate_voice_clone(
            text=job["text"],
            language=map_qwen_language(job["language"]),
            ref_audio=job["prompt_audio_path"],
            ref_text=job["prompt_text"],
            **self.config.get("generation_kwargs", {}),
        )
        sf.write(output_path, wavs[0], sr)
        return {"sample_rate": sr}


class CosyVoice3CloneRunner(AdapterRunner):
    def setup(self) -> None:
        os.chdir(self.repo_path)
        sys.path.insert(0, str(self.repo_path))
        sys.path.insert(0, str(self.repo_path / "third_party" / "Matcha-TTS"))
        import torch
        from cosyvoice.cli.cosyvoice import AutoModel
        self.torch = torch
        model_dir = resolve_local_or_hf_model_dir(
            self.repo_path,
            self.config["model_path"],
            self.config.get("hf_repo_id"),
        )
        self.model = AutoModel(model_dir=str(model_dir))

    def run_job(self, job: dict[str, Any]) -> dict[str, Any]:
        import torchaudio
        output_path = Path(job["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_text = job["prompt_text"]
        prefix = self.config.get("prepend_prompt_prefix", "")
        if prefix:
            prompt_text = f"{prefix}{prompt_text}"
        outputs = list(
            self.model.inference_zero_shot(
                job["text"],
                prompt_text,
                job["prompt_audio_path"],
                stream=bool(self.config.get("stream", False)),
                speed=float(self.config.get("speed", 1.0)),
                text_frontend=bool(self.config.get("text_frontend", True)),
            )
        )
        speech = self.torch.cat([item["tts_speech"] for item in outputs], dim=1).cpu()
        torchaudio.save(str(output_path), speech, self.model.sample_rate)
        return {"sample_rate": self.model.sample_rate}


class SparkTTSCloneRunner(AdapterRunner):
    def setup(self) -> None:
        os.chdir(self.repo_path)
        sys.path.insert(0, str(self.repo_path))
        import torch
        from cli.SparkTTS import SparkTTS
        device_name = self.config.get("device", "cuda:0")
        self.torch = torch
        self.device = torch.device(device_name)
        model_dir = resolve_local_or_hf_model_dir(
            self.repo_path,
            self.config["model_dir"],
            self.config.get("hf_repo_id"),
        )
        self.model = SparkTTS(model_dir, self.device)

    def run_job(self, job: dict[str, Any]) -> dict[str, Any]:
        import soundfile as sf
        output_path = Path(job["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wav = self.model.inference(
            job["text"],
            prompt_speech_path=Path(job["prompt_audio_path"]),
            prompt_text=job["prompt_text"],
            temperature=float(self.config.get("temperature", 0.8)),
            top_k=int(self.config.get("top_k", 50)),
            top_p=float(self.config.get("top_p", 0.95)),
        )
        sf.write(output_path, wav, samplerate=self.model.sample_rate)
        return {"sample_rate": self.model.sample_rate}


class F5TTSCloneRunner(AdapterRunner):
    def setup(self) -> None:
        os.chdir(self.repo_path)
        sys.path.insert(0, str(self.repo_path / "src"))
        from f5_tts.api import F5TTS
        self.model = F5TTS(
            model=self.config.get("model", "F5TTS_v1_Base"),
            ckpt_file=self.config.get("ckpt_file", ""),
            vocab_file=self.config.get("vocab_file", ""),
            device=self.config.get("device"),
            hf_cache_dir=self.config.get("hf_cache_dir"),
            vocoder_local_path=self.config.get("vocoder_local_path"),
        )

    def run_job(self, job: dict[str, Any]) -> dict[str, Any]:
        output_path = Path(job["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _, sr, _ = self.model.infer(
            ref_file=job["prompt_audio_path"],
            ref_text=job["prompt_text"],
            gen_text=job["text"],
            target_rms=float(self.config.get("target_rms", 0.1)),
            cross_fade_duration=float(self.config.get("cross_fade_duration", 0.15)),
            sway_sampling_coef=float(self.config.get("sway_sampling_coef", -1)),
            cfg_strength=float(self.config.get("cfg_strength", 2.0)),
            nfe_step=int(self.config.get("nfe_step", 32)),
            speed=float(self.config.get("speed", 1.0)),
            fix_duration=self.config.get("fix_duration"),
            remove_silence=bool(self.config.get("remove_silence", False)),
            file_wave=str(output_path),
            seed=self.config.get("seed"),
        )
        return {"sample_rate": sr}


class VoxCPMCloneRunner(AdapterRunner):
    def setup(self) -> None:
        os.chdir(self.repo_path)
        sys.path.insert(0, str(self.repo_path / "src"))
        import soundfile as sf
        from voxcpm import VoxCPM
        self.sf = sf
        if self.config.get("model_path"):
            self.model = VoxCPM(
                voxcpm_model_path=self.config["model_path"],
                zipenhancer_model_path=self.config.get("zipenhancer_model_path"),
                enable_denoiser=bool(self.config.get("denoise", False)),
            )
        else:
            self.model = VoxCPM.from_pretrained(
                hf_model_id=self.config.get("hf_model_id", "openbmb/VoxCPM1.5"),
                cache_dir=self.config.get("cache_dir"),
                local_files_only=bool(self.config.get("local_files_only", False)),
                load_denoiser=bool(self.config.get("denoise", False)),
                zipenhancer_model_id=self.config.get("zipenhancer_model_path"),
            )

    def run_job(self, job: dict[str, Any]) -> dict[str, Any]:
        output_path = Path(job["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wav = self.model.generate(
            text=job["text"],
            prompt_wav_path=job["prompt_audio_path"],
            prompt_text=job["prompt_text"],
            cfg_value=float(self.config.get("cfg_value", 2.0)),
            inference_timesteps=int(self.config.get("inference_timesteps", 10)),
            normalize=bool(self.config.get("normalize", False)),
            denoise=bool(self.config.get("denoise", False)),
        )
        sr = self.model.tts_model.sample_rate
        self.sf.write(output_path, wav, sr)
        return {"sample_rate": sr}


class IndexTTS2CloneRunner(AdapterRunner):
    def setup(self) -> None:
        os.chdir(self.repo_path)
        sys.path.insert(0, str(self.repo_path))
        from indextts.infer_v2 import IndexTTS2
        cfg_path = self.repo_path / self.config.get("cfg_path", "checkpoints/config.yaml")
        model_dir = resolve_local_or_hf_model_dir(
            self.repo_path,
            self.config["model_dir"],
            self.config.get("hf_repo_id"),
        )
        self.model = IndexTTS2(
            cfg_path=str(cfg_path),
            model_dir=str(model_dir),
            use_fp16=bool(self.config.get("use_fp16", False)),
            device=self.config.get("device"),
            use_cuda_kernel=self.config.get("use_cuda_kernel"),
            use_deepspeed=bool(self.config.get("use_deepspeed", False)),
            use_accel=bool(self.config.get("use_accel", False)),
            use_torch_compile=bool(self.config.get("use_torch_compile", False)),
        )

    def run_job(self, job: dict[str, Any]) -> dict[str, Any]:
        output_path = Path(job["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.infer(
            spk_audio_prompt=job["prompt_audio_path"],
            text=job["text"],
            output_path=str(output_path),
            verbose=bool(self.config.get("verbose", False)),
        )
        return {"sample_rate": None}


RUNNER_REGISTRY: dict[str, type[AdapterRunner]] = {
    "qwen3_clone": Qwen3CloneRunner,
    "cosyvoice3_clone": CosyVoice3CloneRunner,
    "sparktts_clone": SparkTTSCloneRunner,
    "f5tts_clone": F5TTSCloneRunner,
    "voxcpm_clone": VoxCPMCloneRunner,
    "indextts2_clone": IndexTTS2CloneRunner,
}
