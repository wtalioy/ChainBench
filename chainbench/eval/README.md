# ChainBench Eval Pipeline

This directory now contains the unified ChainBench baseline evaluation pipeline.

## Main Entry

After activating the main `ChainBench` environment, run evaluation with:

```bash
chainbench eval --config config/eval.json
```

Useful modes:

```bash
chainbench eval --config config/eval.json --eval-only
chainbench eval --config config/eval.json --train-only
chainbench eval --config config/eval.json --dry-run
chainbench eval --config config/eval.json --sample-ratio 0.1
chainbench eval --config config/eval.json --force-retrain
```

Template-holdout evaluation now uses the same config plus CLI overrides:

```bash
chainbench eval --config config/eval.json --tasks in_chain_detection --template-holdout --output-root outputs/eval_template_holdout
```

`--dry-run` loads metadata, derives task packs, and reports per-pack plus total `train`/`dev`/`test` sample counts without running any baseline training or evaluation.

## Config Shape

The shared config lives in `config/eval.json`.

Top-level fields:

- `metadata_path`
- `dataset_root`
- `output_root` for generated eval artifacts, logs, summaries, and prepared views. The default config writes these to `outputs/eval`, not inside `data/`.
- `sample_ratio` to deterministically subsample generated task pack splits. This can be a single float applied to every task, a per-task list aligned with the `tasks` order such as `[0.04, 0.03, 0.03, 0.03]`, or a per-task-per-split list such as `[[0.04, 0.04, 0.04], [0.04, 0.04, 1.0], [0.04, 0.04, 1.0], [0.04, 0.04, 1.0]]` where each inner list is `[train, dev, test]`. For the redesigned paired tasks, test rows are sampled by matched pair so each comparison stays intact.
- `generalization` for optional eval-time holdout protocols. The current supported block is:

```json
{
  "generalization": {
    "protocol": "leave_one_template_out",
    "scope": "per_family"
  }
}
```

This protocol is currently supported only for `in_chain_detection`. It keeps training and validation on non-held-out templates within a family, and evaluates on the held-out template's original `test` rows.
- `tasks`
- `baselines`

Each baseline uses the same high-level structure:

```json
{
  "repo_path": "baselines/aasist",
  "conda_env": "aasist",
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

`conda_env` can be either a conda environment name like `"aasist"` or an absolute conda env prefix path like `"/remote-home/wangruiming/conda-env/aasist"`.

`devices` is required for both `train` and `eval`. Each entry is one execution slot, so it can be a single GPU like `"cuda:1"` or a multi-GPU runtime string like `"cuda:0,cuda:1"`.

When training is enabled, the scheduler round-robins end-to-end runs across the configured training `devices` list so multiple task/baseline runs can proceed concurrently. In `--eval-only` mode it uses the evaluation `devices` list instead.

## Package Layout

Shared pipeline code inside `chainbench/eval/`:

- `app.py`
- `config.py`
- `cli.py`
- `pipeline/`
- `tasks/`
- `views.py`
- `metrics/`

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
5. run training/evaluation inside the baseline conda environment
6. normalize scores into `scores.csv`
7. compute ChainBench metrics

## Task Scope

The current baseline pipeline supports these task ids:

- `in_chain_detection`
- `operator_substitution`
- `parameter_perturbation`
- `order_swap`
- `delivery_robustness`

These names are used directly in config, CLI selection, summaries, and output directories.

Task semantics are Stage 5 metadata-driven:

- `in_chain_detection`: one variant per `chain_family`, using native train/dev/test splits within that family. This is the in-domain anchor task for each delivery chain family. When `generalization.protocol = leave_one_template_out`, this expands to one variant per held-out `(chain_family, chain_template_id)` fold and reports unseen-template generalization instead of the native in-domain split.
- `operator_substitution`: same-label matched pairs where the operator *name* changes at one or more indices (full operator substitutions only; parameter-only edits are excluded). Rows share the same parent, chain family, and label; cross-label pairs are dropped. Variant is `default`; metrics use `operator_substitution_*` keys.
- `parameter_perturbation`: same-label matched pairs that differ at exactly one signature index and only along a `parameter@…` axis (same operator name, different attributes). Mutually exclusive per row with operator-substitution pairing. Variant is `default`; metrics use `parameter_perturbation_*` keys. Metadata columns `parameter_perturbation_group_id` / `parameter_perturbation_axis` identify pairs.
- `order_swap`: a single task (variant `default`) of same-label matched pairs with the same operator multiset and an adjacent minimal order swap. Cross-label pairs are dropped; `metric_profile` is `order_swap` and metrics use `order_swap_*` keys.
- `delivery_robustness`: test rows are grouped by **`(parent_id, chain_family, label)`** and only lineages with at least two rendered variants are kept. The shortest operator-signature row in each lineage is the reference chain. Distances are computed on the observed lineage graph over atomic chain edits: single-index operator substitutions, single-index parameter perturbations, adjacent minimal order swaps, and one-step insertions/deletions. Metrics report **`delivery_robustness_robust_at_k`** (fraction of lineages still correct for all reachable variants within graph-distance budget `k`), **`delivery_robustness_mean_radius`** / **`delivery_robustness_mean_radius_capped`**, **`delivery_robustness_mean_radius_by_label`**, and **`delivery_robustness_aurc_chain`** (area under Robust@k).

The task builders attach richer metadata to each `TaskPack`, including `metric_profile`, `chain_family`, `shared_training_group`, and matched-pair IDs where applicable. `operator_substitution` and `parameter_perturbation` both set `shared_training_group` to `intervention_robustness` so they can reuse one trained checkpoint when train/dev are identical. Metrics use that metadata to report subgroup summaries over `chain_template_id`, operator-sequence length, `language`, and `generator_family`. For intervention-style paired tasks, reported pair metrics are: **`pair_count`**, **`pair_consistency_rate`**, **`pair_joint_accuracy`**, **`pair_symmetric_misclassification_risk`** (mean \(\tfrac12(\ell(s_1,y)+\ell(s_2,y))\) with \(\ell\) the 0–1 loss at \(\tau\)), and **`mean_normalized_score_drift`** (mean \(\lvert s_1-s_2\rvert\) divided by the score **IQR**), plus **`by_label`** and task-specific **`by_*` breakdowns**.

**Threshold \(\tau\):** same as headline accuracy — **EER-derived** on the scored set when both classes are present, else **0.5**. Decisions: \(\hat{y}=\mathbb{1}[s\ge\tau]\).

## Important Caveat

The orchestration, config shape, view building, and runtime entry surfaces are now ChainBench-owned.

The baseline packages still import model and helper code from the cloned upstream repos under `baselines/`. So the system is unified at the framework level, but not yet a full reimplementation of each baseline architecture inside ChainBench itself.
