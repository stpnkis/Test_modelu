# RD++ — Revisiting Reverse Distillation for Anomaly Detection

Implementation of **RD++** (CVPR 2023, Tran Dinh Tien et al.) for the casting
defect detection experiment.

> **Paper:** [Revisiting Reverse Distillation for Anomaly Detection](https://openaccess.thecvf.com/content/CVPR2023/papers/Tien_Revisiting_Reverse_Distillation_for_Anomaly_Detection_CVPR_2023_paper.pdf)
>
> **Original repo:** [tientrandinh/Revisiting-Reverse-Distillation](https://github.com/tientrandinh/Revisiting-Reverse-Distillation)

## Algorithm Overview

RD++ is a knowledge-distillation-based anomaly detection method:

1. **Encoder** — A frozen, pretrained WideResNet-50-2 extracts multi-scale
   features (layer1, layer2, layer3) from input images.
2. **Bottleneck Network (OCE-BN)** — Aggregates and compresses features from
   all three scales into a compact one-class embedding.
3. **Decoder** — A reverse ResNet with transposed convolutions reconstructs
   the encoder features from the bottleneck output. The decoder is trained to
   perfectly reconstruct *normal* features — anomalous regions cause high
   reconstruction error.
4. **Multi-Projection Layer** — Projects encoder features for self-supervised
   optimal transport (SSOT) regularisation, improving feature compactness and
   suppressing anomalous signals.
5. **Pseudo-Anomaly Generation** — Structured noise (approximating simplex noise)
   is injected into training images to create pseudo-anomalies for the
   projection loss.

**Anomaly scoring** (inference): For each test image, the anomaly map is
computed as the sum of `1 − cosine_similarity(encoder, decoder)` at each scale,
smoothed with a Gaussian filter (σ=4). The image-level score is the maximum
value of the smoothed anomaly map.

## Quick Start

```bash
# Build the Docker image
docker compose build

# Run the full experiment suite (all training sizes)
docker compose run --rm rd_plus_plus python src/experiment_runner.py

# Run a single training size
docker compose run --rm rd_plus_plus python src/experiment_runner.py --sizes 100

# Run unit tests
docker compose run --rm rd_plus_plus python -m pytest tests/ -v
```

## Hyperparameters

| Parameter           | Default | Description                                      |
|---------------------|---------|--------------------------------------------------|
| `image_size`        | 256     | Input image size (square)                        |
| `batch_size`        | 16      | Training batch size                              |
| `epochs`            | 200     | Max training epochs (early stopping)             |
| `proj_lr`           | 1e-3    | Projection layer learning rate (Adam)            |
| `distill_lr`        | 5e-3    | Decoder + BN learning rate (Adam)                |
| `weight_proj`       | 0.2     | Weight for projection loss in total loss         |
| `accumulation_steps`| 2       | Gradient accumulation steps                      |
| `patience`          | 30      | Early stopping patience (epochs)                 |

## CLI Examples

```bash
# Train on a specific split
python src/train.py --split-dir splits/n100 --output-dir experiments/n100

# Evaluate a trained model
python src/evaluate.py --split-dir splits/n100 --checkpoint experiments/n100/model_best.pth

# Create a dataset split
python src/dataset_splitter.py --n-train 100 --output-dir splits/n100
```

## Architecture Details

```
Input Image (256×256)
    │
    ▼
┌─────────────────────────────────┐
│  Encoder (WRN50-2, frozen)      │
│  layer1 → [B, 256, 64, 64]     │
│  layer2 → [B, 512, 32, 32]     │
│  layer3 → [B, 1024, 16, 16]    │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Multi-Projection Layer         │
│  ProjLayer per scale (conv×4)   │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Bottleneck Network (OCE)       │
│  Aggregate 3 scales → 2048ch    │
│  [B, 2048, 8, 8]               │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Decoder (reverse WRN50-2)      │
│  layer1: 2048→1024, 8→16       │
│  layer2: 1024→512, 16→32       │
│  layer3: 512→256, 32→64        │
└─────────────────────────────────┘
    │
    ▼
  Anomaly Map = Σ(1 - cos_sim)
  + Gaussian smoothing (σ=4)
  Image score = max(anomaly_map)
```

## Training Losses

- **L_distill**: Cosine similarity between encoder and decoder features
- **L_proj**: (L_SSOT + 0.01 × L_reconstruct + 0.1 × L_contrast) / 1.11
- **L_total**: L_distill + 0.2 × L_proj

## Citation

```bibtex
@InProceedings{Tien_2023_CVPR,
    author    = {Tien, Tran Dinh and Nguyen, Anh Tuan and Tran, Nguyen Hoang
                 and Huy, Ta Duc and Duong, Soan T.M. and Nguyen, Chanh D. Tr.
                 and Truong, Steven Q. H.},
    title     = {Revisiting Reverse Distillation for Anomaly Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and
                 Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {24511-24520}
}
```
