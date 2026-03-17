# ChainBench Eval Pipeline

This directory now contains the unified ChainBench baseline evaluation pipeline.

## Main Entry

Run evaluation from the repo root with:

```bash
conda run --no-capture-output -n chainbench python scripts/run_eval.py --config config/eval_baselines.json
```

Useful modes:

```bash
conda run --no-capture-output -n chainbench python scripts/run_eval.py --config config/eval_baselines.json --eval-only
conda run --no-capture-output -n chainbench python scripts/run_eval.py --config config/eval_baselines.json --train-only
conda run --no-capture-output -n chainbench python scripts/run_eval.py --config config/eval_baselines.json --sample-ratio 0.1
conda run --no-capture-output -n chainbench python scripts/run_eval.py --config config/eval_baselines.json --force-retrain
```

## Config Shape

The shared config lives in `config/eval_baselines.json`.

Top-level fields:

- `metadata_path`
- `dataset_root`
- `output_root` for generated eval artifacts, logs, summaries, and prepared views. The default config writes these to `outputs/eval`, not inside `data/`.
- `sample_ratio` to deterministically subsample every generated task pack split. The same sampled train/dev/test rows are reused for every baseline. For counterfactual evaluation, test rows are sampled by parent group to preserve matched pairs.
- `tasks`
- `baselines`

Each baseline uses the same high-level structure:

```json
{
  "repo_path": "baselines/aasist",
  "conda_prefix": "/remote-home/wangruiming/conda-env/aasist",
  "train": {
    "enabled": true,
    "devices": ["cuda:0", "cuda:1"],
    "seed": 1234,
    "epochs": 100,
    "batch_size": 24,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001
  },
  "eval": {
    "enabled": true,
    "devices": ["cuda:0", "cuda:1"],
    "batch_size": 24
  },
  "assets": {},
  "adapter": {}
}
```

`devices` is required for both `train` and `eval`. Each entry is one execution slot, so it can be a single GPU like `"cuda:1"` or a multi-GPU runtime string like `"cuda:0,cuda:1"`.

When training is enabled, the scheduler round-robins end-to-end runs across the configured training `devices` list so multiple task/baseline runs can proceed concurrently. In `--eval-only` mode it uses the evaluation `devices` list instead.

## Package Layout

Shared pipeline code:

- `config.py`
- `cli.py`
- `pipeline.py`
- `tasks.py`
- `views.py`
- `metrics.py`

Shared baseline helpers:

- `baselines/base.py`
- `baselines/asvspoof.py`
- `baselines/__init__.py`

Per-baseline native packages:

- `baselines/native/aasist/`
- `baselines/native/sls_df/`
- `baselines/native/safeear/`
- `baselines/native/nes2net/`

Each native baseline package owns:

- a package-local `runner.py`
- a package-local `runtime.py` when train/eval execution is ChainBench-owned

## Runtime Model

The pipeline works in these steps:

1. load ChainBench metadata
2. derive task packs
3. choose baselines from the registry
4. build a baseline-specific view from each task pack
5. run training/evaluation inside the baseline conda prefix
6. normalize scores into `scores.csv`
7. compute ChainBench metrics

## Task Scope

The current baseline pipeline supports these task ids:

- `in_chain`
- `cross_chain`
- `unseen_composition`
- `unseen_order`
- `counterfactual_consistency`

These names are used directly in config, CLI selection, summaries, and output directories.

## Important Caveat

The orchestration, config shape, view building, and runtime entry surfaces are now ChainBench-owned.

The baseline packages still import model and helper code from the cloned upstream repos under `baselines/`. So the system is unified at the framework level, but not yet a full reimplementation of each baseline architecture inside ChainBench itself.
