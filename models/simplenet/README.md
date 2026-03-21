# SimpleNet — Anomaly Detection

Implementation of **SimpleNet** (Liu et al., CVPR 2023) — feature
adaptation with a discriminator network for anomaly detection.

## Official benchmark

```bash
# From the repository root
python benchmark/run.py --dataset casting --model simplenet

# Few-shot evaluation
python benchmark/run.py --dataset casting --model simplenet --few-shot
```

The entrypoint inside the Docker container is `src/run_benchmark.py`,
invoked automatically by `benchmark/orchestrator.py`.

## Source files

| File | Purpose |
|------|---------|
| `src/run_benchmark.py` | Benchmark entrypoint — trains, scores, computes metrics via `shared/` |
| `src/train.py` | SimpleNet training (feature adaptor + discriminator) |

All benchmark logic (splits, thresholding, metrics, results I/O, runtime
profiling, seed handling) lives in `shared/` and is shared across all models.

## How SimpleNet works

| Step | Description |
|------|-------------|
| 1. Feature extraction | Pretrained WideResNet-50-2 (frozen) extracts patch features from `layer2` + `layer3` |
| 2. Feature adaptor | Linear → BN → LeakyReLU projects features to a 512-d embedding space |
| 3. Anomaly generation | Gaussian noise (σ = 0.015) added to L2-normalised adapted features |
| 4. Discriminator training | MLP learns to separate normal features from synthetic anomalies (BCE loss) |
| 5. Inference | Max-pooled sigmoid score over spatial patch map → image-level anomaly score |

### Score aggregation

```
anomaly_score(image) = 0.7 × max(sigmoid(D(A(f)))) + 0.3 × mean(sigmoid(D(A(f))))
```

### Key hyperparameters

| Parameter | Value |
|-----------|-------|
| Backbone | `wide_resnet50_2` |
| Image size | 256 × 256 |
| Noise std (σ) | 0.015 |
| Training epochs | 200 (early stopping, patience 30) |

## Known issue: training collapse on casting dataset

AUROC < 0.5 on some configurations indicates inverted scores — the
discriminator assigns higher scores to OK images. Suspected cause:
`noise_std = 0.015` may be too small for these specific features.

## Requirements

- Docker with NVIDIA Container Toolkit
- GPU with CUDA support
