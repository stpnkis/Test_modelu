# PatchCore — Anomaly Detection Experiments

Automated experiments evaluating how many normal (OK) training images PatchCore needs
to reliably detect defects (NOK images) on a casting dataset.

## Quick start

```bash
# From this directory (models/patchcore/)
docker compose build
docker compose run --rm patchcore python src/experiment_runner.py
```

Results are saved to `../../experiments/patchcore/results.csv`.

## What the experiment does

Trains and evaluates PatchCore on training sets of different sizes:
`n_train ∈ [10, 20, 50, 100, 200]`

For each size it:
1. Creates a dataset split (symlinks into `../../dataset/`)
2. Trains PatchCore on `n_train` OK images
3. Evaluates on a fixed test set (200 OK + all NOK images)
4. Records AUROC, Accuracy, Precision, Recall to a shared CSV

## Run a single size

```bash
docker compose run --rm patchcore python src/experiment_runner.py --sizes 10
```

## Dataset layout expected

```
../../dataset/
    ok/     # Normal (defect-free) images
    nok/    # Anomalous (defective) images
```

## Requirements

- Docker with NVIDIA Container Toolkit
- GPU with CUDA support
- `docker context use default` (not Docker Desktop context)

## Source files

| File | Purpose |
|------|---------|
| `src/dataset_splitter.py` | Creates train/test splits via symlinks |
| `src/train.py` | Trains PatchCore using anomalib |
| `src/evaluate.py` | Runs inference, computes sklearn metrics |
| `src/experiment_runner.py` | Orchestrates the full experiment loop |
