# BrownBear_ReID — `testing` branch

Experiment infrastructure for retraining the BrownBear ReID model.
See `brown-bear-server/docs/superpowers/specs/2026-04-16-reid-retraining-experiments-design.md`
in the parent repo for the full design spec.

## Quick start

### Run a 1-epoch smoke test (CPU or MPS)

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1 python tools/train.py \
    --config configs/experiments/_smoke_baseline.yaml \
    --output runs/_smoke_baseline/ \
    --device mps
```

### Run a real experiment on the 40-ID subsample

Edit your experiment config to set `DATA.SUBSAMPLE_PATH: configs/data/subsample_40ids.json`,
or create a local override config:

```yaml
# configs/experiments/_local_fixes_subsample.yaml
_base_: 01_fixes_only.yaml
DATA:
  SUBSAMPLE_PATH: configs/data/subsample_40ids.json
```

Then run:

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1 python tools/train.py \
    --config configs/experiments/_local_fixes_subsample.yaml \
    --output runs/01_fixes_subsample/ \
    --device mps
```

### Run on full dataset

Use the experiment config as-is (no SUBSAMPLE_PATH override):

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1 python tools/train.py \
    --config configs/experiments/01_fixes_only.yaml \
    --output runs/01_fixes_only/ \
    --device mps
```

## Adding a new experiment

1. Copy an existing config:
   ```bash
   cp configs/experiments/01_fixes_only.yaml configs/experiments/02_my_experiment.yaml
   ```
2. Edit the override fields. The file inherits from `../base.yaml` via `_base_:`.
3. Run with `--config configs/experiments/02_my_experiment.yaml --output runs/02_my_experiment/`.

## Output structure

Each run writes to `runs/<name>/`:

```
runs/<name>/
├── config.yaml         frozen post-merge config
├── git.txt             submodule + parent-repo SHAs
├── log.txt             training log
├── tensorboard/        TensorBoard events
├── checkpoints/
│   ├── best.pth        best by EVAL.CHECKPOINT_METRIC
│   └── final.pth       last epoch
└── metrics.json        final summary
```

`runs/` is gitignored. Monitor training with: `tensorboard --logdir runs/`

## Subsample mode

Use `configs/data/subsample_40ids.json` for fast hyperparameter iteration (~30 min/run on M4).
Built via `tools/build_subsample.py`. Reuse the same subsample across all sweeps.

## Tests

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -v
```

## Migrating to GPU

Same CLI works on CUDA:

```bash
python tools/train.py --config configs/experiments/01_fixes_only.yaml --output runs/01_fixes_only/ --device cuda
```

No code changes needed.

## What's in this branch (Plan 1)

- Config-driven CLI (`tools/train.py`)
- AdamW + cosine-warmup scheduler
- Val-Rank-1 checkpoint selection
- Strong augmentation profile (RandomResizedCrop + ColorJitter)
- Subsample mode for fast iteration
- Per-run output directory structure
- Two experiments: `00_baseline_current` (faithful repro) and `01_fixes_only` (new defaults)

Plans 2-4 add: loss ablations, eval harness, comparison layer, backbone registry (DINOv2), pose precomputation.
