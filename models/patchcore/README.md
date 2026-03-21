# PatchCore — Anomaly Detection

PatchCore (Roth et al., CVPR 2022) using anomalib, with a WideResNet-50-2
backbone and coreset subsampling.

## Official benchmark

PatchCore is run **only** through the centralized benchmark pipeline:

```bash
# From the repository root
python benchmark/run.py --dataset casting --model patchcore

# Few-shot evaluation
python benchmark/run.py --dataset casting --model patchcore --few-shot
```

The entrypoint inside the Docker container is `src/run_benchmark.py`,
invoked automatically by `benchmark/orchestrator.py`.

## Source files

| File | Purpose |
|------|---------|
| `src/run_benchmark.py` | Benchmark entrypoint — trains, scores, computes metrics via `shared/` |
| `src/train.py` | PatchCore training using anomalib |

All benchmark logic (splits, thresholding, metrics, results I/O, runtime
profiling, seed handling) lives in `shared/` and is shared across all models.

## How PatchCore works

1. **Feature extraction** — Frozen WideResNet-50-2 extracts patch-level features
2. **Coreset subsampling** — Greedy coreset reduces the memory bank to a representative subset
3. **Inference** — k-NN distance to the nearest coreset patch = anomaly score

No gradient training — the model is fully determined by the training images.

## Requirements

- Docker with NVIDIA Container Toolkit
- GPU with CUDA support
