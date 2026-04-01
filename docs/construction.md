# Construction Guide

This document explains how to build ChainBench-ADD end to end using the repository’s **five-stage pipeline**.

## 1. Pipeline overview

The construction pipeline is intentionally staged so that **speaker/text control**, **spoof generation**, and **post-generation delivery** stay separated:

| Stage | Goal | Main input | Main output |
|---|---|---|---|
| Stage 1 | Curate source speech and assign speaker-disjoint splits | Raw corpora | `data/stage1_source_curation/manifests/clean_real_manifest.csv` |
| Stage 2 | Build standardized clean masters | Stage 1 manifest | `data/stage2_clean_masters/manifests/clean_parent_manifest.csv` |
| Stage 3 | Generate spoof clean parents | Stage 2 manifest | `data/stage3_spoof_generation/manifests/clean_parent_manifest_all.csv` |
| Stage 4 | Render delivery chains | Stage 3 combined manifest | `data/stage4_delivery_chain/manifests/delivered_manifest.csv` |
| Stage 5 | Validate, annotate, and package the release | Stage 4 manifest | `data/ChainBench/metadata.csv` and exported audio |

The design mirrors the paper: first fix transcript/speaker conditions, then synthesize spoof parents, then apply post-generation delivery so detector changes can be attributed to delivery rather than uncontrolled upstream variation.

## 2. Prerequisites

### Source data

The default config expects:

- English from Common Voice 24.0
- Mandarin Chinese from AISHELL-3

Adjust the `dataset_root` fields in `config/stage1.json` if your local layout differs.

### Software

The orchestration environment needs:

- Python 3.11+
- `ffmpeg` / `ffprobe`
- `conda`
- the package itself installed with `pip install -e .`

Stage 3 additionally expects generator repos and conda environments named in `config/stage3.json`. Stage 4 uses `pyroomacoustics` for RIR simulation by default, with a synthetic fallback.

### Workspace convention

All default configs are repo-relative. If you want to keep data elsewhere, export:

```bash
export CHAINBENCH_ROOT=/abs/path/to/ChainBench-ADD
```

## 3. Stage 1 — source curation

### Purpose

Stage 1 scans the raw corpora, normalizes transcripts, probes candidate audio, filters low-quality examples, selects speakers, and assigns **speaker-disjoint train/dev/test splits**.

### Implementation

`chainbench/stage1` performs four main jobs:

1. **Load corpus metadata**
   - `load_aishell_candidates(...)`
   - `load_common_voice_candidates(...)`

2. **Normalize transcripts**
   - English keeps tokenized text with at least 4 tokens and rejects numeric-only strings.
   - AISHELL-3 collapses tokenized Chinese text and rejects too-short/numeric-only strings.

3. **Screen audio quality**
   - duration bounds
   - mean/max volume checks
   - speech ratio estimated via `silencedetect`
   - optional cap on the number of audio checks per speaker

4. **Select speakers and split them**
   - assign benchmark speaker IDs
   - speaker-level split assignment via `assign_splits(...)`
   - create a curated raw layout using symlinks

### Default config highlights

`config/stage1.json` controls:

- speaker targets per language
- utterances per speaker
- minimum utterances required to keep a speaker
- audio filtering thresholds
- train/dev/test ratios
- random seed

### Run

```bash
chainbench stage1 --config config/stage1.json
```

Useful flags:

```bash
chainbench stage1 --config config/stage1.json --language en
chainbench stage1 --config config/stage1.json --log-level DEBUG
```

### Outputs

Main outputs under `data/stage1_source_curation/`:

- `manifests/clean_real_manifest.csv`
- `manifests/clean_real_manifest_en.csv`
- `manifests/clean_real_manifest_zh.csv`
- `manifests/en_selected_speakers.json`
- `manifests/zh_selected_speakers.json`
- `manifests/en_speaker_logs.json`
- `manifests/zh_speaker_logs.json`
- `manifests/stage1_summary.json`
- `raw/...` symlinked source audio in curated split/speaker layout

### Important contract

Every row emitted here is still **bona fide** and still linked to a single curated speaker and transcript. No spoofing or delivery edits have happened yet.

---

## 4. Stage 2 — clean master preparation

### Purpose

Stage 2 turns the curated Stage 1 audio into standardized **clean parents**.

### Implementation

`chainbench/stage2` reads the Stage 1 manifest and applies an FFmpeg pipeline built by `build_filter_chain(...)`:

- optional silence trimming
- downmix to mono
- resample to 16 kHz
- loudness normalization
- PCM 16-bit WAV encoding

It validates every rendered file against the configured output spec and duration bounds.

### Default config highlights

`config/stage2.json` controls:

- Stage 1 manifest path
- output root
- worker count
- FFmpeg/FFprobe timeouts
- audio output format
- trim settings
- loudness normalization settings
- post-render validation bounds

### Run

```bash
chainbench stage2 --config config/stage2.json
```

Useful flags:

```bash
chainbench stage2 --config config/stage2.json --language zh
chainbench stage2 --config config/stage2.json --limit 1000
```

### Outputs

Main outputs under `data/stage2_clean_masters/`:

- `audio/...` clean master WAVs
- `manifests/clean_parent_manifest.csv`
- `manifests/clean_parent_manifest_en.csv`
- `manifests/clean_parent_manifest_zh.csv`
- `manifests/stage2_failures.json`
- `manifests/stage2_summary.json`

### Important contract

Rows remain one-to-one with curated bona fide parents, but now have a canonical clean-master audio path in `clean_parent_path`.

---

## 5. Stage 3 — spoof clean-parent generation

### Purpose

Stage 3 creates **matched spoof clean parents** from the Stage 2 bona fide clean parents.

### Implementation

Stage 3 deliberately separates generator choice from later delivery rendering.

For each clean parent, it:

1. selects a fixed number of generators (`generators_per_parent`, default 2)
2. chooses a **same-speaker but different-utterance** prompt/reference audio when possible
3. materializes generator-specific job JSONL files
4. launches one internal worker per generator batch inside the configured conda env
5. optionally postprocesses raw generator outputs into standardized WAV
6. validates spoof outputs
7. merges bona fide clean parents and valid spoof clean parents into a combined manifest for Stage 4

### Generators wired by default

`config/stage3.json` includes six generator adapters:

- `qwen3_tts_base`
- `cosyvoice3`
- `spark_tts`
- `f5_tts`
- `index_tts2`
- `voxcpm`

Each entry declares:

- repo path
- conda env
- supported languages
- adapter name
- adapter-specific model/runtime args

### Run

```bash
chainbench stage3 --config config/stage3.json
```

Useful variants:

```bash
chainbench stage3 --config config/stage3.json --plan-only
chainbench stage3 --config config/stage3.json --only-generator qwen3_tts_base
chainbench stage3 --config config/stage3.json --language en --generators-per-parent 1
```

### Outputs

Main outputs under `data/stage3_spoof_generation/`:

- `jobs/*.jsonl` per-generator job files
- `jobs/*.adapter_config.json`
- `results/*.jsonl` per-generator result logs
- `logs/*.log`
- `audio_raw/...` raw generator outputs
- `audio/...` postprocessed spoof clean parents
- `manifests/spoof_clean_manifest.csv`
- `manifests/clean_parent_manifest_all.csv` (**Stage 4 input**)
- `manifests/stage3_failures.json`
- `manifests/stage3_summary.json`

### Important contract

After Stage 3 you have:

- original bona fide clean parents from Stage 2
- spoof clean parents generated from them

Both still live in the **clean-parent** regime. Delivery has not started yet.

---

## 6. Stage 4 — delivery-chain rendering

### Purpose

Stage 4 is where ChainBench-ADD becomes a **delivery-aware** benchmark. It samples delivery templates, instantiates operator parameters, renders realized chains, and writes delivery metadata.

### Implementation

Stage 4 reads `clean_parent_manifest_all.csv`, so it renders delivery variants for **both bona fide and spoof clean parents**.

For each parent and selected family, it:

1. samples up to `variants_per_parent` realized variants
2. concretizes template specs using parameter pools
3. applies operators sequentially in waveform space
4. optionally writes trace JSON
5. standardizes final output when needed
6. emits a delivered-manifest row with operator sequence and sampled params

### Delivery families and default template counts

The default `config/stage4.json` defines:

- `direct` — 1 template
- `platform_like` — 6 templates
- `telephony` — 10 templates
- `simreplay` — 7 templates
- `hybrid` — 9 templates

That is **33 templates total**, consistent with the paper.

### Operators in the implementation

Stage 4 implements reusable operators under `chainbench/stage4/operators/`:

- `resample`
- `bandlimit`
- `codec`
- `reencode`
- `packet_loss`
- `noise`
- `rir`
- `call_path`

Notable implementation details:

- codec/re-encode use actual encode/decode round trips
- packet loss supports rate, burst length, and concealment mode
- RIR defaults to `pyroomacoustics` with a synthetic fallback
- final outputs are standardized to 16 kHz mono PCM WAV

### Paired template groups

Some families include paired template groups that deliberately preserve the same operator multiset while changing order. These are what make order-swap style analyses possible downstream.

### Run

```bash
chainbench stage4 --config config/stage4.json
```

Useful variants:

```bash
chainbench stage4 --config config/stage4.json --plan-only
chainbench stage4 --config config/stage4.json --families telephony hybrid
chainbench stage4 --config config/stage4.json --language en --limit 500
```

### Outputs

Main outputs under `data/stage4_delivery_chain/`:

- `audio/<family>/...` delivered WAVs
- `traces/<family>/...` optional render traces
- `jobs/stage4_job_plan.json`
- `manifests/delivered_manifest.csv`
- `manifests/delivered_manifest_<family>.csv`
- `manifests/delivered_manifest_<language>.csv`
- `manifests/stage4_failures.json`
- `manifests/stage4_summary.json`

### Important contract

Each row now represents a **realized delivery child** linked back to its clean parent through `parent_id`, `clean_parent_path`, `operator_seq`, and `operator_params`.

---

## 7. Stage 5 — validation, annotation, and packaging

### Purpose

Stage 5 validates the Stage 4 outputs, exports the final dataset layout, and adds the structural metadata used by the benchmark tasks.

### Implementation

Stage 5 performs three steps:

1. **Validate audio**
   - duration
   - sample rate / channels / codec
   - NaN/Inf checks
   - peak/clipping checks
   - duration ratio relative to the clean parent

2. **Export packaged audio**
   - copies files into the final release layout:
     - `train/audio/<language>/<label>/...`
     - `dev/audio/<language>/<label>/...`
     - `test/audio/<language>/<label>/...`

3. **Annotate structural metadata**
   - operator substitution groups
   - parameter perturbation groups
   - order swap groups
   - path groups / path step indices
   - operator multiset keys
   - path endpoint keys

These annotations are generated by `annotate_structural_group_fields(...)` and are what later drive the evaluation tasks.

### Run

```bash
chainbench stage5 --config config/stage5.json
```

Useful variants:

```bash
chainbench stage5 --config config/stage5.json --skip-validation
chainbench stage5 --config config/stage5.json --language zh
```

### Outputs

Main outputs under `data/ChainBench/`:

- `audio/...` split audio trees
- `metadata.csv`
- `train/metadata.csv`
- `dev/metadata.csv`
- `test/metadata.csv`
- `manifest/stage5_failures.json`
- `manifest/dataset_summary.json`
- `manifest/stats_label_language_split.csv`
- `manifest/stats_chain_family_label.csv`
- `manifest/stats_generator_family_label.csv`

### Final release metadata

The root `metadata.csv` includes release-critical fields such as:

- identifiers: `sample_id`, `parent_id`, `speaker_id`, `utterance_id`
- benchmark labels: `label`, `split_standard`, `language`
- provenance: `source_corpus`, `generator_family`, `generator_name`
- delivery structure: `chain_family`, `chain_template_id`, `chain_variant_index`
- operator fields: `operator_seq`, `operator_params`, `operator_multiset_key`
- task annotations:
  - `operator_substitution_group_id`
  - `parameter_perturbation_group_id`
  - `order_swap_group_id`
  - `path_group_id`
  - `path_step_index`

---

## 8. Recommended end-to-end command sequence

```bash
chainbench stage1 --config config/stage1.json
chainbench stage2 --config config/stage2.json
chainbench stage3 --config config/stage3.json
chainbench stage4 --config config/stage4.json
chainbench stage5 --config config/stage5.json
```

If you want to inspect plans before heavy stages:

```bash
chainbench stage3 --config config/stage3.json --plan-only
chainbench stage4 --config config/stage4.json --plan-only
```

---

## 9. Common knobs you will actually change

### Stage 1
- source corpus roots
- target speaker counts
- utterances per speaker
- audio thresholds

### Stage 3
- enabled generators
- conda env names
- generators per parent
- model/checkpoint paths

### Stage 4
- selected families
- `variants_per_parent`
- parameter pools
- RIR backend and fallback
- worker count

### Stage 5
- output dataset root
- validation thresholds
- required family coverage list

---

## 10. Sanity checks after construction

A healthy run should leave you with:

1. `data/ChainBench/metadata.csv`
2. split-level metadata files under `train/`, `dev/`, and `test/`
3. exported audio under split/language/label folders
4. `manifest/dataset_summary.json` showing non-zero dataset rows
5. speaker-disjoint splits and structural group fields populated for downstream evaluation

If those are present, the benchmark is ready for `chainbench eval`.
