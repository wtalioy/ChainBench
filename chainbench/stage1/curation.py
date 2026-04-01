"""Per-speaker curation: probe, quality check, accept/reject."""

from __future__ import annotations

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from chainbench.lib.audio import ffprobe_audio
from chainbench.lib.proc import run_command

from .candidates import AcceptedSample, Candidate, ProbedCandidate

MEAN_VOL_RE = re.compile(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")
MAX_VOL_RE = re.compile(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")
SILENCE_DUR_RE = re.compile(r"silence_duration:\s*(\d+(?:\.\d+)?)")


def counter_summary(counter: Counter, top_k: int = 5) -> str:
    if not counter:
        return "none"
    parts = [f"{k}={v}" for k, v in counter.most_common(top_k)]
    return ", ".join(parts)


def analyze_audio_quality(
    path: str,
    silence_noise_db: int,
    silence_min_duration: float,
) -> dict[str, float] | None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        path,
        "-af",
        f"silencedetect=noise={silence_noise_db}dB:d={silence_min_duration},volumedetect",
        "-f",
        "null",
        "-",
    ]
    result = run_command(command)
    if result.returncode != 0:
        return None
    stderr = result.stderr
    mean_match = MEAN_VOL_RE.search(stderr)
    max_match = MAX_VOL_RE.search(stderr)
    if mean_match is None or max_match is None:
        return None
    silence_total = sum(float(m.group(1)) for m in SILENCE_DUR_RE.finditer(stderr))
    return {
        "mean_volume_db": float(mean_match.group(1)),
        "max_volume_db": float(max_match.group(1)),
        "silence_duration_sec": silence_total,
    }


def parallel_map(
    items: list[Any],
    worker: Any,
    max_workers: int,
) -> list[tuple[Any, Any]]:
    if not items:
        return []
    results: list[tuple[Any, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(worker, item): item for item in items}
        for future in as_completed(future_map):
            item = future_map[future]
            try:
                results.append((item, future.result()))
            except Exception:
                results.append((item, None))
    return results


def duration_rank(
    duration: float,
    preferred_min: float,
    preferred_max: float,
) -> tuple[int, float]:
    in_preferred = preferred_min <= duration <= preferred_max
    if in_preferred:
        return (0, abs(duration - (preferred_min + preferred_max) / 2.0))
    if duration < preferred_min:
        return (1, preferred_min - duration)
    return (1, duration - preferred_max)


def resolve_max_audio_checks(num_candidates: int, lang_cfg: dict[str, Any]) -> int:
    configured = int(lang_cfg.get("max_audio_checks_per_speaker", 0) or 0)
    if configured <= 0:
        return num_candidates
    return min(num_candidates, configured)


def curate_single_speaker(
    candidates: list[Candidate],
    lang_cfg: dict[str, Any],
    filters: dict[str, Any],
    workers: int,
    rng: Any,
    language: str,
    source_speaker_id: str,
    logger: Any,
) -> tuple[list[AcceptedSample] | None, dict[str, int]]:
    stats = Counter()
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    target_utts = lang_cfg["target_utterances_per_speaker"]
    min_utts = lang_cfg["min_utterances_per_speaker"]
    max_checks = resolve_max_audio_checks(len(shuffled), lang_cfg)
    batch_size = min(12, max(4, workers * 2))

    accepted: list[AcceptedSample] = []
    checked = 0
    cursor = 0

    while cursor < len(shuffled) and checked < max_checks and len(accepted) < target_utts:
        remaining_budget = max_checks - checked
        batch = shuffled[cursor : cursor + min(batch_size, remaining_budget)]
        cursor += len(batch)

        probed_batch: list[ProbedCandidate] = []
        for candidate, probe in parallel_map(
            batch,
            lambda item: ffprobe_audio(item.source_audio_path),
            workers,
        ):
            checked += 1
            if probe is None:
                stats["audio_probe_failed"] += 1
                continue
            duration = float(probe["duration"])
            if duration < filters["min_duration_sec"] or duration > filters["max_duration_sec"]:
                stats["duration_rejected"] += 1
                continue
            probed_batch.append(
                ProbedCandidate(
                    candidate=candidate,
                    duration=duration,
                    sample_rate=int(probe["sample_rate"]),
                    channels=int(probe["channels"]),
                    codec_name=str(probe["codec_name"]),
                )
            )

        analyzed = parallel_map(
            probed_batch,
            lambda item: analyze_audio_quality(
                item.candidate.source_audio_path,
                filters["silence_noise_threshold_db"],
                filters["silence_min_duration_sec"],
            ),
            workers,
        )

        for probed, quality in analyzed:
            if quality is None:
                stats["audio_quality_failed"] += 1
                continue
            mean_volume_db = float(quality["mean_volume_db"])
            max_volume_db = float(quality["max_volume_db"])
            silence_duration_sec = float(quality["silence_duration_sec"])
            speech_ratio = max(0.0, 1.0 - (silence_duration_sec / probed.duration))

            if mean_volume_db < filters["min_mean_volume_db"]:
                stats["low_volume_rejected"] += 1
                continue
            if max_volume_db > filters["max_peak_volume_db"]:
                stats["clipping_rejected"] += 1
                continue
            if speech_ratio < filters["min_speech_ratio"]:
                stats["low_speech_ratio_rejected"] += 1
                continue

            accepted.append(
                AcceptedSample(
                    candidate=probed.candidate,
                    duration=probed.duration,
                    sample_rate=probed.sample_rate,
                    channels=probed.channels,
                    codec_name=probed.codec_name,
                    mean_volume_db=mean_volume_db,
                    max_volume_db=max_volume_db,
                    silence_duration_sec=silence_duration_sec,
                    speech_ratio=speech_ratio,
                )
            )

        logger.info(
            "[%s][speaker=%s] checked %d/%d candidates in latest batch, accepted=%d, rejected=%s",
            language,
            source_speaker_id,
            checked,
            max_checks,
            len(accepted),
            counter_summary(stats),
        )

    if len(accepted) < min_utts:
        stats["speaker_rejected_insufficient_valid_utterances"] += 1
        logger.info(
            "[%s][speaker=%s] rejected after checking %d/%d candidates: accepted=%d (< %d), stats=%s",
            language,
            source_speaker_id,
            checked,
            max_checks,
            len(accepted),
            min_utts,
            counter_summary(stats, top_k=8),
        )
        return None, dict(stats)

    accepted.sort(
        key=lambda item: (
            duration_rank(
                item.duration,
                filters["preferred_min_duration_sec"],
                filters["preferred_max_duration_sec"],
            ),
            -item.speech_ratio,
            abs(item.mean_volume_db + 20.0),
            item.candidate.utterance_id,
        )
    )
    logger.info(
        "[%s][speaker=%s] accepted with %d valid utterances after %d checks",
        language,
        source_speaker_id,
        len(accepted),
        checked,
    )
    return accepted[:target_utts], dict(stats)
