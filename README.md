# Test_modelu — Anomaly Detection Model Comparison

Educational repository for benchmarking anomaly detection models on
industrial visual inspection data.

> **Research question:** How many "OK" training samples are needed for
> reliable unsupervised anomaly detection?

---

## Repository structure

```
Test_modelu/
├── dataset/              ← shared dataset (gitignored, add manually)
│   ├── ok/               ← normal (defect-free) images
│   └── nok/              ← defective images
├── experiments/          ← results per model (gitignored)
│   └── patchcore/
│       └── results.csv
├── models/
│   ├── patchcore/        ← PatchCore experiments
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   ├── requirements.txt
│   │   ├── README.md
│   │   └── src/
│   │       ├── dataset_splitter.py
│   │       ├── train.py
│   │       ├── evaluate.py
│   │       └── experiment_runner.py
│   └── <next_model>/     ← add further models here
└── splits/               ← auto-generated train/test splits (gitignored)
```

Each model lives in its own subdirectory under `models/` with its own
`Dockerfile`, `docker-compose.yml`, `requirements.txt`, and source code.
The `dataset/` directory is shared across all models.

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

| Model | Status | Description |
|---|---|---|
| [PatchCore](models/patchcore/) | ✅ Working | Nearest-neighbour coreset memory bank |

---

## Running an experiment

```bash
cd models/patchcore
docker compose build
docker compose run --rm patchcore python src/experiment_runner.py
```

See each model's `README.md` for model-specific instructions.

---

## Requirements

- Docker with NVIDIA Container Toolkit
- GPU with CUDA support
- System Docker context: `docker context use default`
