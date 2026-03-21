# AnomalyDINO — Anomaly Detection

Implementation of **AnomalyDINO** (Damm et al., WACV 2025) — few-shot
anomaly detection using frozen DINOv2 features and a memory bank.

## Official benchmark

```bash
# From the repository root
python benchmark/run.py --dataset casting --model anomalydino

# Few-shot evaluation
python benchmark/run.py --dataset casting --model anomalydino --few-shot
```

The entrypoint inside the Docker container is `src/run_benchmark.py`,
invoked automatically by `benchmark/orchestrator.py`.

## Source files

| File | Purpose |
|------|---------|
| `src/run_benchmark.py` | Benchmark entrypoint — trains, scores, computes metrics via `shared/` |
| `src/train.py` | Extracts DINOv2 features, builds memory bank (.pth) |

All benchmark logic (splits, thresholding, metrics, results I/O, runtime
profiling, seed handling) lives in `shared/` and is shared across all models.

## How AnomalyDINO works

1. **Feature extraction** — Frozen DINOv2 ViT-B/14 extracts 1024 patch features (768-dim) per 448×448 image
2. **Memory bank** — All reference patch features stored in a `.pth` file
3. **Scoring** — Cosine nearest-neighbour distance to memory bank; max(distances) = image score

No gradient training — the model is fully determined by the reference images.

## Requirements

- Docker with NVIDIA Container Toolkit
- GPU with CUDA support
