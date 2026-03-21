# RD++ — Revisiting Reverse Distillation for Anomaly Detection

Implementation of **RD++** (Tran Dinh Tien et al., CVPR 2023) for
anomaly detection benchmarking.

## Official benchmark

```bash
# From the repository root
python benchmark/run.py --dataset casting --model rd_plus_plus

# Few-shot evaluation
python benchmark/run.py --dataset casting --model rd_plus_plus --few-shot
```

The entrypoint inside the Docker container is `src/run_benchmark.py`,
invoked automatically by `benchmark/orchestrator.py`.

## Source files

| File | Purpose |
|------|---------|
| `src/run_benchmark.py` | Benchmark entrypoint — trains, scores, computes metrics via `shared/` |
| `src/train.py` | RD++ training (encoder–decoder distillation + multi-projection) |

All benchmark logic (splits, thresholding, metrics, results I/O, runtime
profiling, seed handling) lives in `shared/` and is shared across all models.

## Algorithm overview

1. **Encoder** — Frozen pretrained WideResNet-50-2 extracts multi-scale features
2. **Bottleneck (OCE-BN)** — Aggregates and compresses multi-scale features
3. **Decoder** — Reverse ResNet with transposed convolutions reconstructs encoder features
4. **Multi-Projection Layer** — Projects encoder features for SSOT regularisation
5. **Pseudo-Anomaly Generation** — Structured noise injected during training

**Anomaly scoring**: sum of `1 − cosine_similarity(encoder, decoder)` at each
scale, smoothed with Gaussian (σ=4). Image score = max(anomaly_map).

RD++ is the only model in this benchmark that produces pixel-level anomaly maps,
so AU-PRO is reported only for RD++ (when ground-truth masks are available).

## Hyperparameters

| Parameter           | Default | Description                           |
|---------------------|---------|---------------------------------------|
| `image_size`        | 256     | Input image size (square)             |
| `batch_size`        | 16      | Training batch size                   |
| `epochs`            | 200     | Max training epochs (early stopping)  |
| `proj_lr`           | 1e-3    | Projection layer learning rate (Adam) |
| `distill_lr`        | 5e-3    | Decoder + BN learning rate (Adam)     |
| `weight_proj`       | 0.2     | Weight for projection loss            |
| `patience`          | 30      | Early stopping patience               |

## Requirements

- Docker with NVIDIA Container Toolkit
- GPU with CUDA support
