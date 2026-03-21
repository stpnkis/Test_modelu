# Test_modelu — Anomaly Detection Benchmark

Reproducible benchmark for comparing unsupervised anomaly detection models
on industrial visual inspection data.

> **Research question:** How many "OK" training samples are needed for
> reliable unsupervised anomaly detection?

## Key design principles

- **Fair comparison** — all models share the same deterministic split, preprocessing, and evaluation
- **Val-only thresholding** — anomaly threshold is derived from val/ok scores (quantile p=0.99), never from test data
- **3-seed reproducibility** — every experiment runs three seeds; results are mean ± std
- **Few-shot invariant** — val and test sets remain constant across different `n_train` values
- **Docker isolation** — one container per model, shared code mounted read-only

---

## Quick Start

```bash
# 1. Clone and set up directory structure
git clone <repo-url> && cd Test_modelu
bash setup.sh          # creates dirs + installs Python deps

# 1b. (alternative) manual install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Place your dataset
#    datasets/<dataset_id>/ok/   ← normal (defect-free) images
#    datasets/<dataset_id>/nok/  ← defective images
#    datasets/<dataset_id>/masks/ ← optional pixel-level masks

# 3. Run a single model
python benchmark/run.py --dataset casting --model patchcore

# 4. Run all models
python benchmark/run.py --dataset casting --all-models

# 5. Aggregate only (skip training)
python benchmark/run.py --dataset casting --aggregate-only
```

---

## Dataset layout

Place images in `datasets/<dataset_id>/`:

```
datasets/
└── casting/
    ├── ok/          ← normal images (.png, .jpg, .bmp, .tiff)
    ├── nok/         ← defective images
    └── masks/       ← optional: pixel-level ground truth for AU-PRO
```

The dataset validator runs before any experiment and will report missing
directories, empty folders, unsupported extensions, or orphan masks.

---

## Repository Structure

```
Test_modelu/
├── README.md
├── setup.sh                  ← run once after cloning
├── benchmark/
│   ├── config.yaml           ← central configuration (seeds, n_train, etc.)
│   ├── run.py                ← CLI entry point
│   └── orchestrator.py       ← dataset × model × seed loop
├── shared/                   ← shared library (mounted read-only into containers)
│   ├── dataset_schema.py     ← fail-fast dataset validation
│   ├── split_manager.py      ← deterministic train/val/test split + manifest
│   ├── preprocessing.py      ← unified transforms (baseline/high_accuracy/edge_safe)
│   ├── thresholding.py       ← val-only anomaly threshold
│   ├── metrics.py            ← AUROC, F1, AU-PRO
│   ├── runtime_profiler.py   ← latency + GPU peak memory
│   ├── results_io.py         ← per-seed JSON results
│   ├── aggregate.py          ← mean ± std across seeds, LaTeX output
│   └── seed_utils.py         ← full reproducibility (torch, numpy, python)
├── datasets/                 ← your image datasets (gitignored)
├── splits/                   ← auto-generated split directories + manifests
├── experiments/              ← per-seed results JSON files
├── tests/                    ← centralized test suite
│   ├── test_split_manager.py
│   ├── test_thresholding.py
│   ├── test_metrics.py
│   ├── test_dataset_schema.py
│   ├── test_smoke.py
│   ├── test_benchmark_fixes.py
│   └── test_protocol_compliance.py  ← full protocol compliance checks
└── models/
    ├── anomalydino/          ← AnomalyDINO (DINOv2 ViT-B/14 memory bank)
    ├── patchcore/            ← PatchCore (anomalib, WideResNet-50-2 coreset)
    ├── rd_plus_plus/         ← RD++ (knowledge distillation, reverse decoder)
    └── simplenet/            ← SimpleNet (WideResNet-50-2 + discriminator)
```

---

## Models

| Model | Backbone | Method |
|---|---|---|
| [AnomalyDINO](models/anomalydino/) | DINOv2 ViT-B/14 | Frozen features + memory bank |
| [PatchCore](models/patchcore/) | WideResNet-50-2 | Coreset subsampling |
| [RD++](models/rd_plus_plus/) | WideResNet-50-2 | Reverse distillation + multi-projection |
| [SimpleNet](models/simplenet/) | WideResNet-50-2 | Feature adaptation + discriminator |

---

## Benchmark CLI

```bash
# Single model
python benchmark/run.py --dataset casting --model anomalydino

# All models
python benchmark/run.py --dataset casting --all-models

# Few-shot evaluation (sweep over n_train = 10, 25, 50, 100)
python benchmark/run.py --dataset casting --model patchcore --few-shot

# Custom settings
python benchmark/run.py --dataset casting --model patchcore \
    --n-train-ok 50 --seeds 42 1337 2026 --preprocessing-mode high_accuracy

# Validate dataset only (no training)
python benchmark/run.py --dataset casting --validate-only

# Aggregate existing results
python benchmark/run.py --dataset casting --aggregate-only
```

Configuration in [benchmark/config.yaml](benchmark/config.yaml) provides
defaults; CLI flags override them.

---

## Preprocessing modes

| Mode | Resize | Final size | Use case |
|---|---|---|---|
| `baseline` (default) | Shortest edge → 256 + center crop | 224×224 | Official benchmark |
| `high_accuracy` | Square resize | 448×448 | Maximum accuracy |
| `edge_safe` | Keep aspect + pad | 224×224 | No cropping edge artifacts |

All modes apply ImageNet normalization. No augmentations.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Requirements

- Docker with NVIDIA Container Toolkit (`nvidia-docker2`)
- GPU with CUDA support (runs on CPU too, but significantly slower)
- Python 3.10+ (for benchmark CLI on host)

Host-side Python dependencies (installed via `pip install -r requirements.txt`):
torch, torchvision, numpy, scikit-learn, scipy, PyYAML, Pillow, pytest
