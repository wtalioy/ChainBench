# Native Baseline Packages

These packages are the ChainBench-owned baseline integrations. They are intentionally separate from the raw upstream repos under `baselines/`.

## Layout

- `aasist/runner.py`
- `sls_df/runner.py`
- `safeear/runner.py`
- `nes2net/runner.py`

The registry in `chainbench/eval/baselines/__init__.py` points to these package-local runners.

## What Is ChainBench-owned

ChainBench owns:

- the shared config schema in `chainbench/eval/config.py`
- the top-level orchestration in `chainbench/eval/cli.py`
- the shared task/view pipeline in `chainbench/eval/views.py`
- score normalization in `chainbench/eval/metrics/`
- the baseline package runners in this directory

Each runner translates the shared ChainBench train/eval settings into the underlying baseline-specific execution path.

## Runtime Ownership

### Fully or partially ChainBench-owned runtime paths

- `aasist`:
  - `native/aasist/runner.py`
  - `native/aasist/runtime.py`
- `sls_df`:
  - `native/sls_df/runner.py`
  - `native/sls_df/runtime.py`
- `nes2net`:
  - `native/nes2net/runner.py`
  - `native/nes2net/runtime.py`
- `safeear`:
  - `native/safeear/runner.py`
  - `native/safeear/runtime.py`

### Still upstream-script launched

- none for the baseline packages listed above

The remaining upstream dependency is now mainly at the level of imported model definitions and helper modules inside the raw baseline repos, not their top-level train/eval scripts.

## Current Validation Status

Validated:

- import/registry resolution points to these package-local runners
- `chainbench eval --dry-run` succeeds with the new pipeline
- package-local view preparation succeeds for at least one baseline package (`aasist`)
- Python compilation succeeds for the updated eval package
- Python compilation succeeds for the package-local runtimes in `aasist`, `sls_df`, `nes2net`, and `safeear`

## Remaining Prerequisites

### `aasist`

- no checkpoints are present in the cloned repo
- evaluation requires either a trained checkpoint from the new ChainBench runtime or a configured external checkpoint

### `sls_df`

- no pretrained checkpoints or required SSL assets were found in the cloned repo
- evaluation requires a checkpoint path or successful training run
- the ChainBench-native runtime expects an XLSR checkpoint path if it is not already staged as `xlsr2_300m.pt` in the repo environment

### `safeear`

- `SpeechTokenizer.pt` was not found in the cloned repo
- Hubert feature roots for the generated ChainBench view are not configured yet
- the ChainBench-native runtime now owns both training and evaluation orchestration, but it still depends on upstream SafeEar model/datamodule code plus the required feature assets

### `nes2net`

- no pretrained checkpoints were found in the cloned repo
- evaluation requires a checkpoint path or successful training run
- the ChainBench-native runtime expects an XLSR checkpoint path if it is not already staged as `xlsr2_300m.pt` in the repo environment

## Design Note

These packages are currently hybrid integrations:

- ChainBench owns the data view, config shape, orchestration, and normalized outputs
- all four baseline packages now also have ChainBench-owned runtime modules or package-local orchestration for train/eval

The next reduction in upstream dependency would be to move more train/eval runtime logic from those upstream scripts into ChainBench-owned runtime modules, package by package.
