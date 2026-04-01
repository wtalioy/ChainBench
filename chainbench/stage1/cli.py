"""Stage 1 CLI: source curation."""

from __future__ import annotations

import argparse
import random
from collections import Counter
import sys
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from chainbench.lib.cli import add_log_level_argument, resolve_config_argument
from chainbench.lib.logging import get_logger, setup_logging
from chainbench.lib.config import default_workspace_root, load_json, relative_to_workspace
from chainbench.lib.io import write_csv, write_json
from chainbench.lib.summary import print_json, utc_now_iso

from .candidates import load_aishell_candidates, load_common_voice_candidates
from .curation import curate_single_speaker, counter_summary
from .manifest import ensure_symlink, sample_to_manifest_row
from .splits import assign_splits


LOGGER = get_logger("stage1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/stage1.json", help="Path to the JSON config file.")
    add_log_level_argument(parser)
    parser.add_argument(
        "--log-every-speakers",
        type=int,
        default=10,
        help="Emit a progress log every N speakers during curation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    workspace_root = default_workspace_root()
    config_path = resolve_config_argument(args.config, workspace_root)
    config = load_json(config_path)
    output_root = resolve_config_argument(config["output_root"], workspace_root)
    manifest_root = output_root / "manifests"
    raw_root = output_root / "raw"
    manifest_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    workers = int(config["workers"])
    filters = config["audio_filters"]

    all_manifest_rows: list[dict[str, Any]] = []
    overall_summary: dict[str, Any] = {
        "generated_at_utc": utc_now_iso(),
        "config_path": relative_to_workspace(config_path, workspace_root),
        "output_root": relative_to_workspace(output_root, workspace_root),
        "seed": seed,
        "workers": workers,
        "languages": {},
    }

    for language in ("zh", "en"):
        lang_cfg = config["languages"][language]
        counters: dict[str, Counter] = {"text": Counter(), "selection": Counter()}

        LOGGER.info("[%s] loading metadata from %s", language, lang_cfg["dataset_root"])
        if language == "zh":
            speaker_to_candidates = load_aishell_candidates(lang_cfg, counters)
        else:
            speaker_to_candidates = load_common_voice_candidates(lang_cfg, counters)

        min_text_candidates = lang_cfg["min_utterances_per_speaker"]
        candidate_speakers = [
            sid for sid, items in speaker_to_candidates.items() if len(items) >= min_text_candidates
        ]
        counters["selection"]["candidate_speakers_after_text_filter"] = len(candidate_speakers)
        LOGGER.info(
            "[%s] metadata scan complete: %d candidate speakers, text counters=%s",
            language,
            len(candidate_speakers),
            counter_summary(counters["text"], top_k=8),
        )
        if len(candidate_speakers) < lang_cfg["target_speakers"]:
            raise RuntimeError(
                f"{language}: only {len(candidate_speakers)} candidate speakers available; "
                f"target is {lang_cfg['target_speakers']}"
            )

        rng = random.Random(seed + (11 if language == "zh" else 17))
        rng.shuffle(candidate_speakers)

        speaker_bundles: list[dict[str, Any]] = []
        speaker_logs: list[dict[str, Any]] = []
        LOGGER.info(
            "[%s] curating speakers: target=%d, min_utts=%d, target_utts=%d",
            language,
            lang_cfg["target_speakers"],
            lang_cfg["min_utterances_per_speaker"],
            lang_cfg["target_utterances_per_speaker"],
        )

        with tqdm(
            total=len(candidate_speakers),
            desc=f"stage1 {language} speakers",
            unit="spk",
            dynamic_ncols=True,
        ) as progress:
            for source_speaker_id in candidate_speakers:
                if len(speaker_bundles) >= lang_cfg["target_speakers"]:
                    break

                accepted_samples, speaker_stats = curate_single_speaker(
                    speaker_to_candidates[source_speaker_id],
                    lang_cfg,
                    filters,
                    workers,
                    random.Random(f"{seed}:{language}:{source_speaker_id}"),
                    language,
                    source_speaker_id,
                    LOGGER,
                )
                log_entry = {
                    "source_speaker_id": source_speaker_id,
                    "candidate_utterances_after_text_filter": len(speaker_to_candidates[source_speaker_id]),
                    "status": "accepted" if accepted_samples is not None else "rejected",
                    "stats": speaker_stats,
                }
                speaker_logs.append(log_entry)
                progress.update(1)

                if accepted_samples is None:
                    counters["selection"]["rejected_speakers_audio_filters"] += 1
                else:
                    speaker_bundles.append(
                        {
                            "source_speaker_id": source_speaker_id,
                            "samples": accepted_samples,
                            "speaker_meta": accepted_samples[0].candidate.speaker_meta,
                        }
                    )
                    counters["selection"]["accepted_speakers"] += 1
                    counters["selection"]["accepted_samples"] += len(accepted_samples)

                if progress.n <= 3 or progress.n % args.log_every_speakers == 0:
                    progress.set_postfix(
                        accepted=len(speaker_bundles),
                        target=lang_cfg["target_speakers"],
                        rejected=counters["selection"]["rejected_speakers_audio_filters"],
                        samples=counters["selection"]["accepted_samples"],
                    )

        target_shortfall = lang_cfg["target_speakers"] - len(speaker_bundles)
        if target_shortfall > 0:
            if config.get("allow_partial_target", False):
                LOGGER.warning(
                    "[%s] target shortfall: accepted %d speakers out of requested %d; proceeding with partial dataset",
                    language,
                    len(speaker_bundles),
                    lang_cfg["target_speakers"],
                )
            else:
                raise RuntimeError(
                    f"{language}: accepted {len(speaker_bundles)} speakers, "
                    f"but target is {lang_cfg['target_speakers']}"
                )

        speaker_bundles.sort(key=lambda b: b["source_speaker_id"])
        for idx, bundle in enumerate(speaker_bundles, start=1):
            bundle["speaker_id"] = f"{lang_cfg['speaker_id_prefix']}{idx:04d}"

        split_map = assign_splits(
            speaker_bundles,
            config["splits"],
            random.Random(seed + (101 if language == "zh" else 103)),
        )

        selected_speakers_payload: list[dict[str, Any]] = []
        manifest_rows: list[dict[str, Any]] = []
        split_speaker_counts = Counter(split_map.values())
        split_sample_counts = Counter()

        LOGGER.info("[%s] materializing curated layout and manifests", language)
        for bundle in speaker_bundles:
            speaker_id = bundle["speaker_id"]
            split = split_map[speaker_id]
            selected_speakers_payload.append(
                {
                    "speaker_id": speaker_id,
                    "source_speaker_id": bundle["source_speaker_id"],
                    "split": split,
                    "language": language,
                    "source_corpus": lang_cfg["source_corpus"],
                    "speaker_meta": bundle["speaker_meta"],
                    "selected_utterances": [s.candidate.utterance_id for s in bundle["samples"]],
                }
            )
            for sample in bundle["samples"]:
                src = Path(sample.candidate.source_audio_path)
                dst = raw_root / language / split / speaker_id / f"{sample.candidate.utterance_id}{src.suffix}"
                ensure_symlink(src, dst)
                row = sample_to_manifest_row(sample, speaker_id, split, seed, workspace_root, raw_root)
                manifest_rows.append(row)
                split_sample_counts[split] += 1

        manifest_rows.sort(key=lambda r: (r["language"], r["split"], r["speaker_id"], r["utterance_id"]))
        all_manifest_rows.extend(manifest_rows)

        language_summary = {
            "source_corpus": lang_cfg["source_corpus"],
            "target_speakers": lang_cfg["target_speakers"],
            "selected_speakers": len(speaker_bundles),
            "target_shortfall": max(0, lang_cfg["target_speakers"] - len(speaker_bundles)),
            "target_utterances_per_speaker": lang_cfg["target_utterances_per_speaker"],
            "min_utterances_per_speaker": lang_cfg["min_utterances_per_speaker"],
            "selected_samples": len(manifest_rows),
            "candidate_speakers_after_text_filter": len(candidate_speakers),
            "split_speaker_counts": dict(split_speaker_counts),
            "split_sample_counts": dict(split_sample_counts),
            "text_filter_counters": dict(counters["text"]),
            "selection_counters": dict(counters["selection"]),
            "speaker_logs_path": f"manifests/{language}_speaker_logs.json",
            "speaker_map_path": f"manifests/{language}_selected_speakers.json",
        }
        overall_summary["languages"][language] = language_summary

        write_json(manifest_root / f"{language}_speaker_logs.json", speaker_logs)
        write_json(manifest_root / f"{language}_selected_speakers.json", selected_speakers_payload)
        write_csv(manifest_root / f"clean_real_manifest_{language}.csv", manifest_rows)
        LOGGER.info(
            "[%s] finished: speakers=%d, samples=%d, split_speakers=%s, split_samples=%s",
            language,
            len(speaker_bundles),
            len(manifest_rows),
            dict(split_speaker_counts),
            dict(split_sample_counts),
        )

    all_manifest_rows.sort(key=lambda r: (r["language"], r["split"], r["speaker_id"], r["utterance_id"]))
    write_csv(manifest_root / "clean_real_manifest.csv", all_manifest_rows)
    write_json(manifest_root / "stage1_summary.json", overall_summary)

    LOGGER.info("all languages finished: total samples=%d", len(all_manifest_rows))
    print_json(overall_summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
