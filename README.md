# Test_modelu — Anomaly Detection Model Comparison

Educational repository for benchmarking anomaly detection models on
industrial visual inspection data.

> **Research question:** How many "OK" training samples are needed for
> reliable unsupervised anomaly detection?

---

## Quick Start (after cloning)

```bash
bash setup.sh              # create experiments/ and splits/ output dirs
cd models/anomalydino
docker compose build
docker compose run --rm anomalydino python src/experiment_runner.py
```

---

## Repository Structure

```
Test_modelu/
├── setup.sh              ← run once after cloning (creates output dirs)
├── dataset/              ← shared dataset (gitignored — add manually)
│   ├── ok/               ← normal (defect-free) images
│   └── nok/              ← defective images
├── experiments/          ← model outputs: .pth, logs, results.csv (gitignored)
│   ├── anomalydino/
│   ├── patchcore/
│   └── simplenet/
├── splits/               ← auto-generated train/test splits (gitignored)
├── models/
│   ├── anomalydino/      ← AnomalyDINO (WACV 2025) — DINOv2 ViT-B
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   ├── requirements.txt
│   │   ├── README.md
│   │   ├── src/
│   │   │   ├── dataset_splitter.py
│   │   │   ├── train.py
│   │   │   ├── evaluate.py
│   │   │   └── experiment_runner.py
│   │   └── tests/
│   │       ├── test_splitter.py
│   │       └── test_metrics.py
│   ├── patchcore/        ← PatchCore — WideResNet + coreset
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   ├── requirements.txt
│   │   ├── README.md
│   │   └── src/
│   └── simplenet/        ← SimpleNet (CVPR 2023) — discriminator
│       ├── Dockerfile
│       ├── docker-compose.yml
│       ├── requirements.txt
│       ├── README.md
│       ├── src/
│       └── tests/
```

Each model has its own `Dockerfile`, `docker-compose.yml`, `requirements.txt`,
and source code. The `dataset/` directory is shared across all models.

---

## Dataset

**Casting Product Image Data for Quality Inspection**
([Kaggle](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product))

| Property | Value |
|---|---|
| Subject | Submersible pump impeller |
| View | Top view |
| OK images | ~519 |
| Defective images | ~781 |

Download and place images into:

```
dataset/ok/    ← normal images
dataset/nok/   ← defective images
```

---

## Models

| Model | Backbone | Status | AUROC (n=200) |
|---|---|---|---|
| [AnomalyDINO](models/anomalydino/) | DINOv2 ViT-B/14 | ✅ Working | **0.981** |
| [PatchCore](models/patchcore/) | WideResNet-50-2 | ✅ Working | — |
| [SimpleNet](models/simplenet/) | ResNet-18 + MLP | ⚠️ Low accuracy | < 0.5 on casting data |

---

## Running an Experiment

```bash
cd models/anomalydino      # or patchcore / simplenet
docker compose build
docker compose run --rm anomalydino python src/experiment_runner.py
```

Results are written to `experiments/<model>/results.csv`.

See each model's `README.md` for model-specific CLI options.

---

## Requirements

- Docker with NVIDIA Container Toolkit (`nvidia-docker2`)
- GPU with CUDA support (also runs on CPU, significantly slower)
- System Docker context: `docker context use default`
