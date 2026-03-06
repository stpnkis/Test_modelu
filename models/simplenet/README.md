# SimpleNet — Anomaly Detection Experiments

Automated experiments evaluating how many normal (OK) training images SimpleNet
needs to reliably detect defects (NOK images) on a casting dataset.

Based on: **"SimpleNet: A Simple Network for Image Anomaly Detection and
Localization"** (Liu et al., CVPR 2023).

## Quick start

```bash
# From this directory (models/simplenet/)
docker compose build
docker compose run --rm simplenet python src/experiment_runner.py
```

Results are saved to `../../experiments/simplenet/results.csv`.

## What the experiment does

Trains and evaluates SimpleNet on training sets of different sizes:
`n_train ∈ [50, 100, 150, 200]`

> Minimum 50 training images required (enforced by `dataset_splitter.py`).

For each size it:
1. Creates a dataset split (symlinks into `../../dataset/`)
2. Trains SimpleNet on `n_train` OK images
3. Evaluates on a fixed test set (200 OK + all NOK images)
4. Records AUROC, Accuracy, Precision, Recall to a shared CSV

## How SimpleNet works

| Step | Description |
|------|-------------|
| 1. Feature extraction | Pretrained WideResNet-50-2 (frozen) extracts patch features from `layer2` + `layer3` |
| 2. Feature adaptor | Linear → BN → LeakyReLU projects features to a 512-d embedding space |
| 3. Anomaly generation | Gaussian noise (σ = 0.015) is added to L2-normalised adapted features |
| 4. Discriminator training | MLP learns to separate normal features from synthetic anomalies (BCE loss) |
| 5. Inference | Max-pooled sigmoid score over the spatial patch map → image-level anomaly score |

### Hyper-parameters (defaults)

| Parameter | Value | CLI flag |
|-----------|-------|----------|
| Backbone | `wide_resnet50_2` | `--backbone` |
| Image size | 256 × 256 | `--image-size` |
| Feature layers | `layer2`, `layer3` | — |
| Adaptor output dim | 512 | — |
| Discriminator hidden | 256 | — |
| Noise std (σ) | 0.015 | `--noise-std` |
| Training epochs | 200 | `--epochs` |
| Early stopping patience | 30 | `--patience` |
| Learning rate (adaptor) | 2 × 10⁻⁴ | `--lr` |
| Learning rate (discriminator) | 1 × 10⁻⁴ | — |
| LR schedule | CosineAnnealingLR | — |
| Weight decay | 1 × 10⁻⁵ | `--weight-decay` |
| Gradient clip max norm | 1.0 | — |
| Batch size (images) | 32 | `--batch-size` |
| Patch batch size | 256 | — |
| Seed | 42 | (in experiment_runner) |

### Score aggregation

```
anomaly_score(image) = 0.7 × max(sigmoid(D(A(f)))) + 0.3 × mean(sigmoid(D(A(f))))
```

Where `D` is the discriminator, `A` is the adaptor, and `f` are the backbone features.
The weighted combination of max and mean provides a balance between sensitivity
to local defects and global image score stability.

## Run a single size

```bash
docker compose run --rm simplenet python src/experiment_runner.py --sizes 50
```

## Run unit tests

```bash
docker compose run --rm simplenet python -m pytest tests/ -v
# or without pytest:
docker compose run --rm simplenet python -m unittest discover -s tests -v
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
| `src/dataset_splitter.py` | Creates train/test splits via symlinks (identical logic to PatchCore) |
| `src/train.py` | Implements SimpleNet (feature extractor + adaptor + discriminator), trains on feature bank |
| `src/evaluate.py` | Runs inference, computes sklearn metrics (AUROC, Acc, Prec, Rec via Youden J) |
| `src/experiment_runner.py` | Orchestrates the full experiment loop |
| `tests/test_splitter.py` | Unit tests for deterministic splitting + symlink correctness |
| `tests/test_metrics.py` | Unit tests for metric computation stability |

## Comparison with PatchCore

Both models share:
- **Identical dataset splits** (same seed, same test set of 200 OK + all NOK)
- **Identical metric computation** (AUROC + Youden J threshold + Accuracy/Precision/Recall)
- **Identical CSV format** (`experiments/<model>/results.csv`)
- **Same CLI interface** (`--split-dir`, `--output-dir`, `--image-size`, `--backbone`, `--batch-size`)
- **Same Docker conventions** (GPU mount, shared dataset, volume layout)

Key differences:
- PatchCore uses a **memory bank + KNN** (no gradient training) → 1 epoch  
- SimpleNet uses a **discriminator + synthetic anomalies** (gradient training) → up to 200 epochs (early stopping)  
- PatchCore depends on **anomalib**; SimpleNet is implemented in **pure PyTorch**

---

## Experiment results on casting dataset

> **Status: training collapse — AUROC below 0.5 on all runs**

| n_train | AUROC | Accuracy | Precision | Recall | F1 |
|---------|-------|----------|-----------|--------|----|
| 50 | 0.606 | 0.422 | 0.912 | 0.304 | 0.455 |
| 100 | 0.390 | 0.219 | 1.000 | 0.019 | 0.038 |
| 150 | 0.444 | 0.220 | 0.944 | 0.022 | 0.043 |
| 200 | 0.395 | 0.803 | 0.803 | 0.997 | 0.890 |

### Root cause analysis

AUROC < 0.5 indicates **inverted scores** — the discriminator assigns higher
anomaly scores to OK images than to defective ones.

Training loss converged normally (1.39 → 0.04), so the model did learn
something — but it learned a **flipped decision boundary**.

**Suspected cause:** `noise_std = 0.015` (calibrated for MVTec dataset features)
is too small relative to the natural variance of WideResNet features on this
casting dataset. The synthetic anomalies are indistinguishable from real normal
features, so the discriminator effectively learns the wrong direction.

### Recommended fixes (not yet implemented)

1. **Adaptive noise std**: Scale σ to the empirical standard deviation of features
   in each channel: `noise = torch.randn_like(feat) * feat.std(dim=0) * scale_factor`
2. **Noise std sweep**: Try `σ ∈ [0.05, 0.1, 0.2, 0.5]` as a hyperparameter search
3. **Score inversion guard**: If final training AUROC < 0.5, flip scores automatically
4. **Noise in raw (pre-normalisation) feature space** rather than L2-normalised space
