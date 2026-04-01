# Evaluation Guide

This document explains how to evaluate models on ChainBench-ADD using the unified pipeline in `chainbench/eval/`.

## 1. What evaluation consumes

The evaluation pipeline is **metadata-driven**. Its main required inputs are:

- final packaged metadata, usually `data/ChainBench/metadata.csv`
- final dataset root, usually `data/ChainBench`
- baseline repos and baseline-specific conda environments
- `config/eval.json`

The orchestrator uses the metadata to derive task packs and then builds baseline-specific views from those packs.

## 2. Benchmark tasks supported

The pipeline exposes five task IDs:

| Task ID | What it measures | Test construction |
|---|---|---|
| `in_chain_detection` | bona fide vs spoof discrimination within a delivery family | family-specific train/dev/test rows |
| `operator_substitution` | robustness to swapping operator identity under matched context | matched same-label pairs |
| `parameter_perturbation` | robustness to parameter changes within one operator slot | matched same-label pairs |
| `order_swap` | robustness to adjacent minimal order swaps | matched same-label pairs |
| `delivery_robustness` | robustness along observed lineage graphs as edits accumulate | test rows grouped by `(parent_id, chain_family, label)` |

These tasks are built from Stage 5 metadata annotations, not from hand-written split files.

## 3. Baselines currently wired

`config/eval.json` registers five baselines:

- `aasist`
- `aasist-l`
- `sls_df`
- `safeear`
- `nes2net`

Each baseline entry defines:

- `repo_path`
- `conda_env`
- `train` config
- `eval` config
- optional `assets`
- adapter/runtime config

The orchestrator is unified, but the actual model/runtime code still comes from the external baseline repos.

## 4. Baseline prerequisites

Before you run evaluation, make sure:

1. the baseline submodules are initialized
2. all baseline conda envs exist
3. model assets referenced in `config/eval.json` are present
4. the final dataset exists at `dataset_root`
5. `metadata_path` points to the packaged Stage 5 metadata

Example submodule/bootstrap step:

```bash
git submodule update --init --recursive
```

## 5. Main command

```bash
chainbench eval --config config/eval.json
```

This command:

1. loads the metadata
2. builds task packs
3. schedules baseline runs across configured devices
4. prepares baseline-specific views
5. trains and/or evaluates each baseline
6. normalizes scores to `scores.csv`
7. computes task metrics
8. writes per-run metrics and an overall eval summary

## 6. Useful execution modes

### Dry run

Build task packs and report counts without training/evaluating anything.

```bash
chainbench eval --config config/eval.json --dry-run
```

### Eval only

Reuse existing checkpoints and only score/evaluate.

```bash
chainbench eval --config config/eval.json --eval-only
```

### Train only

Train runs without final evaluation.

```bash
chainbench eval --config config/eval.json --train-only
```

### Restrict tasks

```bash
chainbench eval --config config/eval.json --tasks in_chain_detection
chainbench eval --config config/eval.json --tasks operator_substitution parameter_perturbation
```

### Restrict baselines

```bash
chainbench eval --config config/eval.json --baselines nes2net sls_df
```

### Subsample packs

```bash
chainbench eval --config config/eval.json --sample-ratio 0.1
```

### Force retraining

Ignore cached checkpoints.

```bash
chainbench eval --config config/eval.json --force-retrain
```

## 7. Sampling

You can specify the sampling ratio for each task and split by `sample_ratio` in `config/eval.json`. It can be provided in several ways.

### Single float

Apply the same ratio to every task and split.

```json
"sample_ratio": 0.1
```

### Per-task list aligned to task order

```json
"sample_ratio": [
  [1.0, 1.0, 1.0],
  [0.1, 0.1, 1.0],
  [0.1, 0.1, 1.0],
  [0.1, 0.1, 1.0],
  [1.0, 1.0, 1.0]
]
```

## 8. Output layout

By default the pipeline writes to `outputs/eval/`.

A typical run produces:

```text
outputs/eval/
  <task_id>/
    <variant>/
      <baseline>/
        train.log
        eval.log
        scores.csv
        metrics.json
        checkpoints/...
  eval_summary_<timestamp>.json
```

The run directory pattern is:

```text
<output_root>/<task_id>/<variant>/<baseline>/
```

## 9. Practical evaluation workflow

### Standard benchmark run

```bash
chainbench eval --config config/eval.json --dry-run
chainbench eval --config config/eval.json
```

### Intervention-only study

```bash
chainbench eval --config config/eval.json \
  --tasks operator_substitution parameter_perturbation order_swap
```

## 10. Adding or swapping a baseline

At a high level you need to:

1. add the repo under `baselines/`
2. create its runtime/adapter wrapper under `chainbench/eval/`
3. register the baseline in `config/eval.json`
4. make sure it outputs scores that the normalization path can ingest

The current pipeline is already structured around baseline runners, prepared views, and normalized score files, so new baselines plug in most cleanly by following the existing runner pattern.
