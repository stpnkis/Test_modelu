# AnomalyDINO — Anomaly Detection Experiments

Evaluates how many normal (OK) reference images **AnomalyDINO** needs to
reliably detect defects on a casting dataset.

> **Paper:** "AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection
> with DINOv2" — Damm et al., WACV 2025.

---

## Quick Start

```bash
# From this directory: models/anomalydino/
docker compose build          # one-time image build
docker compose run --rm anomalydino python src/experiment_runner.py
```

Results → `../../experiments/anomalydino/results.csv`

---

## What the Experiment Does

For each `n_train ∈ [10, 50, 100, 150, 200]`:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `dataset_splitter.py` | Select `n_train` OK images for reference; fixed 200 OK + all NOK for test. |
| 2 | `train.py` | Extract DINOv2 patch features → build memory bank → save `.pth`. |
| 3 | `evaluate.py` | Score every test image via cosine NN → compute AUROC, Precision, Recall, F1. |
| 4 | `evaluate.py` | Append metrics to shared CSV. |

---

## How AnomalyDINO Works

```
Reference images ──► DINOv2 ViT (frozen) ──► patch features ──► Memory Bank
                                                                     │
Test image ────────► DINOv2 ViT (frozen) ──► patch features ─────► cosine NN
                                                                     │
                                                          max(distance) = score
```

1. **Feature extraction** — Frozen DINOv2 ViT-B/14 produces 1 024 patch
   features (768-dim) per 448 × 448 image.
2. **Memory bank** — All patch features from the reference images are stored.
3. **Scoring** — Each test patch is compared to the nearest reference patch
   via cosine distance.  `max(distances)` = image-level anomaly score.
4. **Threshold** — Youden's J statistic (identical to PatchCore / SimpleNet).

No gradient training — "training" is a single forward pass.

---

## Hyper-parameters

| Parameter | Default | CLI flag | Notes |
|-----------|---------|----------|-------|
| Backbone | `dinov2_vitb14` | `--backbone` | ViT-B/14, 768-dim features |
| Image size | 448 × 448 | `--image-size` | Must be divisible by 14 |
| Patch size | 14 × 14 | — | Fixed by backbone |
| Patches / image | 32 × 32 = 1 024 | — | |
| Scoring | cosine NN | — | |
| Aggregation | `max(anomaly_map)` | — | Per AnomalyDINO paper §3.3 |
| Seed | 42 | — | Edit in `experiment_runner.py` |

### Backbone variants

| Variant | Hub name | Feat dim | Speed (GTX 1080) | Accuracy |
|---------|----------|----------|-------------------|----------|
| ViT-S/14 | `dinov2_vits14` | 384 | ~30–60 ms/img | Good |
| **ViT-B/14** | `dinov2_vitb14` | 768 | ~90–150 ms/img | **Best** |

---

## CLI Examples

```bash
# All sizes, default ViT-B backbone
docker compose run --rm anomalydino python src/experiment_runner.py

# Single size
docker compose run --rm anomalydino python src/experiment_runner.py --sizes 50

# Multiple sizes
docker compose run --rm anomalydino python src/experiment_runner.py --sizes 10 50 100

# ViT-S backbone (lighter, faster)
docker compose run --rm anomalydino python src/experiment_runner.py --backbone dinov2_vits14

# Custom results file
docker compose run --rm anomalydino python src/experiment_runner.py --results-csv experiments/custom.csv

# Standalone train + evaluate
docker compose run --rm anomalydino python src/train.py \
    --split-dir splits/n50 --output-dir experiments/n50
docker compose run --rm anomalydino python src/evaluate.py \
    --split-dir splits/n50 --checkpoint experiments/n50/model_best.pth --n-train 50
```

---

## Unit Tests

```bash
docker compose run --rm anomalydino python -m pytest tests/ -v
# or without pytest:
docker compose run --rm anomalydino python -m unittest discover -s tests -v
```

---

## File Overview

```
models/anomalydino/
├── Dockerfile              # Docker image (PyTorch + CUDA 11.8 base)
├── docker-compose.yml      # GPU mount, shared volumes, hub cache
├── requirements.txt        # Pinned Python dependencies
├── README.md               # ← you are here
├── src/
│   ├── dataset_splitter.py # Train/test split via symlinks (same logic as PatchCore/SimpleNet)
│   ├── train.py            # DINOv2 feature extraction → memory bank → .pth checkpoint
│   ├── evaluate.py         # Inference + metrics (AUROC, Acc, Prec, Rec, F1, timing)
│   └── experiment_runner.py# Orchestrates split → train → eval for all n_train sizes
└── tests/
    ├── __init__.py
    ├── test_splitter.py    # Deterministic splits, symlink correctness
    └── test_metrics.py     # Metric computation on synthetic data
```

---

## Dataset Layout

```
../../dataset/
    ok/         # Normal (defect-free) images
    nok/        # Anomalous (defective) images
```

---

## CSV Output Format

```csv
n_train,backbone,auroc,accuracy,precision,recall,f1,threshold,n_test_total,n_test_ok,n_test_nok,avg_inference_ms,build_time_s
50,dinov2_vitb14,0.9629,0.8767,0.9882,0.8553,0.9170,0.291458,981,200,781,101.2,4.32
```

---

## Comparison with Other Models

All three models in this repository share:

- **Same dataset splits** — identical seed, always 200 OK + all NOK for test.
- **Same metric pipeline** — AUROC + Youden J + Accuracy / Precision / Recall / F1.
- **Same CSV saved to** `experiments/<model>/results.csv`.
- **Same Docker conventions** — GPU mount, shared dataset, volume layout.

| Model | Backbone | Training | Key difference |
|-------|----------|----------|----------------|
| PatchCore | WideResNet-50-2 (CNN) | Memory bank + KNN | anomalib, no grads |
| SimpleNet | ResNet-18 + MLP | Discriminator + synth. anomalies | Gradient training |
| **AnomalyDINO** | **DINOv2 ViT-B/14** | **Memory bank + cosine NN** | **No grads, ViT features** |

---

## Jetson Orin Nano Deployment

The code is ready for Jetson with minimal changes:

1. **Dockerfile** — swap the `FROM` line to the L4T PyTorch base image:
   ```dockerfile
   FROM nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.5-py3
   ```
2. **docker-compose.yml** — remove `--platform=linux/amd64` if present.
3. **No code changes needed** — `train.py` and `evaluate.py` auto-detect
   `cuda` / `cpu` and work on both amd64 and aarch64.

Optimisation tips for edge deployment:
- Use ViT-S backbone (`--backbone dinov2_vits14`) for ~2× faster inference.
- Reduce `--image-size` to 224 for ~4× fewer patches (but lower accuracy).
- Pre-download DINOv2 weights into the Docker image to avoid runtime downloads.

---

## Requirements

- Docker with **NVIDIA Container Toolkit** (`nvidia-docker2`)
- GPU with CUDA support (also runs on CPU, much slower)
- `docker context use default` (not Docker Desktop context)
- Internet on first run (DINOv2 weights are downloaded from torch.hub)
