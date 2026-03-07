"""
AnomalyDINO — Evaluation Script
=================================

Evaluates a trained AnomalyDINO model (memory bank checkpoint) on a
held-out test set and computes image-level metrics.

Metrics
-------
- **AUROC** — threshold-independent, area under the ROC curve.
- **Accuracy** — fraction of correct predictions at the optimal threshold.
- **Precision** — of predicted anomalies, how many are truly anomalous?
- **Recall** — of all true anomalies, how many were detected?
- **F1** — harmonic mean of precision and recall.
- **Inference time** — average wall-clock ms per image.

Pipeline
--------
1. Load the memory bank checkpoint (``.pth``) produced by ``train.py``.
2. Load the same DINOv2 backbone that was used during training.
3. **Once** — L2-normalise the entire memory bank (``ref_norm``).
4. For each test image:

   a. Extract dense patch features via DINOv2.
   b. Compute cosine distance to the **nearest** reference patch.
   c. Image-level score = ``max(distances)`` — follows the AnomalyDINO
      paper (Damm et al., WACV 2025, Sec. 3.3).

5. Compute metrics using scikit-learn.

The binary threshold is chosen via **Youden's J statistic**
(maximises ``TPR − FPR``), identical to PatchCore and SimpleNet so
that results are directly comparable.

Usage (standalone)::

    python src/evaluate.py \\
        --split-dir splits/n50 \\
        --checkpoint experiments/n50/model_best.pth \\
        --n-train 50

Typically called automatically by ``experiment_runner.py``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torchvision import transforms

from train import (
    BACKBONE_CONFIG,
    DEFAULT_BACKBONE,
    DEFAULT_IMAGE_SIZE,
    IMAGE_EXTENSIONS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DINOv2FeatureExtractor,
)

__all__ = [
    "evaluate_anomalydino",
    "save_results",
]

logger = logging.getLogger(__name__)


# ── Labelled test dataset ───────────────────────────────────────────────────

class LabeledImageDataset:
    """Load test images from ``good/`` and ``defective/`` with binary labels.

    Convention: **0 = normal (good), 1 = anomalous (defective)**.
    """

    def __init__(self, good_dir: str | Path, defective_dir: str | Path) -> None:
        self.items: List[Tuple[Path, int]] = []
        for p in sorted(Path(good_dir).iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                self.items.append((p, 0))
        for p in sorted(Path(defective_dir).iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                self.items.append((p, 1))

    def __len__(self) -> int:
        return len(self.items)


# ── Inference helper ─────────────────────────────────────────────────────────

@torch.no_grad()
def _compute_anomaly_score(
    image_path: Path,
    extractor: DINOv2FeatureExtractor,
    ref_norm: torch.Tensor,
    transform: transforms.Compose,
    device: torch.device,
) -> float:
    """Compute the image-level anomaly score for one test image.

    Steps:
      1. Extract DINOv2 patch features.
      2. For each test patch find the **maximum cosine similarity** to any
         reference patch → convert to distance ``1 − similarity``.
      3. Image score = ``max(distances)`` over all patches.

    Args:
        image_path: Path to test image.
        extractor:  Feature extractor (on *device*).
        ref_norm:   L2-normalised memory bank ``[N_ref, D]`` on *device*.
                    Pre-computed once by :func:`evaluate_anomalydino`.
        transform:  Image preprocessing pipeline.
        device:     Torch device.

    Returns:
        Scalar anomaly score (higher → more anomalous).
    """
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]

    # Patch features → flat [N_test, D]
    patch_feats = extractor(tensor)
    test_patches = patch_feats.reshape(-1, patch_feats.shape[-1])

    # Cosine similarity via normalised dot product
    test_norm = F.normalize(test_patches, dim=1)          # [N_test, D]
    similarity = torch.mm(test_norm, ref_norm.t())        # [N_test, N_ref]
    max_sim, _ = similarity.max(dim=1)                    # [N_test]

    distances = 1.0 - max_sim                             # [N_test]
    return float(distances.max().cpu())


# ── Main evaluation ──────────────────────────────────────────────────────────

def evaluate_anomalydino(
    split_dir: str,
    checkpoint_path: str,
    image_size: int = DEFAULT_IMAGE_SIZE,
    backbone: str = DEFAULT_BACKBONE,
) -> Dict[str, object]:
    """Evaluate an AnomalyDINO checkpoint on the test split.

    Args:
        split_dir:        Split root (must contain ``test/good/`` and
                          ``test/defective/``).
        checkpoint_path:  ``.pth`` file produced by ``train.py``.
        image_size:       Must match the value used during training (read
                          from the checkpoint automatically).
        backbone:         Fallback if the checkpoint has no ``backbone`` key.

    Returns:
        Dictionary with all computed metrics.
    """
    split_dir = Path(split_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Evaluating checkpoint: %s", checkpoint_path)
    logger.info("Test data:  %s", split_dir)
    logger.info("Device:     %s", device)

    # 1. Load checkpoint ──────────────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    memory_bank = ckpt["memory_bank"].to(device)               # [N_ref, D]

    used_backbone = cfg.get("backbone", backbone)
    used_image_size = cfg.get("image_size", image_size)

    logger.info(
        "Memory bank: %d patches × %d-dim  |  backbone=%s  image_size=%d",
        memory_bank.shape[0], memory_bank.shape[1],
        used_backbone, used_image_size,
    )

    # 2. Load DINOv2 backbone ─────────────────────────────────────────────
    extractor = DINOv2FeatureExtractor(used_backbone).to(device)

    # 3. Prepare test data ────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((used_image_size, used_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    test_dataset = LabeledImageDataset(
        good_dir=split_dir / "test" / "good",
        defective_dir=split_dir / "test" / "defective",
    )
    logger.info("Test set: %d images", len(test_dataset))

    # 4. Pre-normalise memory bank (done **once**, not per image) ─────────
    ref_norm = F.normalize(memory_bank, dim=1)                  # [N_ref, D]

    # 5. Inference loop with timing ───────────────────────────────────────
    all_scores: List[float] = []
    all_labels: List[int] = []
    inference_times: List[float] = []

    # Warmup pass — first CUDA call is slow (context initialisation).
    if device.type == "cuda" and test_dataset.items:
        _compute_anomaly_score(
            test_dataset.items[0][0], extractor, ref_norm, transform, device,
        )
        torch.cuda.synchronize()

    for img_path, label in test_dataset.items:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        score = _compute_anomaly_score(
            img_path, extractor, ref_norm, transform, device,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        all_scores.append(score)
        all_labels.append(label)
        inference_times.append(elapsed)

    scores_np = np.array(all_scores)
    labels_np = np.array(all_labels, dtype=int)
    avg_inference_ms = float(np.mean(inference_times) * 1000)

    logger.info(
        "Inference done — %d images  (%d OK, %d NOK)  avg %.2f ms/img",
        len(labels_np),
        int((labels_np == 0).sum()),
        int((labels_np == 1).sum()),
        avg_inference_ms,
    )

    # 6. Compute metrics ──────────────────────────────────────────────────
    auroc = float(roc_auc_score(labels_np, scores_np))

    # Optimal binary threshold via Youden's J = TPR − FPR
    fpr, tpr, thresholds = roc_curve(labels_np, scores_np)
    j_scores = tpr - fpr
    optimal_idx = int(np.argmax(j_scores))
    threshold = float(thresholds[optimal_idx])

    preds = (scores_np >= threshold).astype(int)

    metrics: Dict[str, object] = {
        "auroc":            round(auroc, 4),
        "accuracy":         round(float(accuracy_score(labels_np, preds)), 4),
        "precision":        round(float(precision_score(labels_np, preds, zero_division=0)), 4),
        "recall":           round(float(recall_score(labels_np, preds, zero_division=0)), 4),
        "f1":               round(float(f1_score(labels_np, preds, zero_division=0)), 4),
        "threshold":        round(threshold, 6),
        "n_test_total":     len(labels_np),
        "n_test_ok":        int((labels_np == 0).sum()),
        "n_test_nok":       int((labels_np == 1).sum()),
        "avg_inference_ms": round(avg_inference_ms, 2),
        "build_time_s":     cfg.get("build_time_s"),
    }

    logger.info("Metrics: %s", metrics)
    return metrics


# ── Results persistence ──────────────────────────────────────────────────────

def save_results(
    metrics: Dict[str, object],
    results_path: str,
    n_train: int,
    backbone: str = DEFAULT_BACKBONE,
) -> None:
    """Append one experiment row to the results CSV.

    The CSV is created on first call and appended to on subsequent calls.

    Args:
        metrics:      Dict returned by :func:`evaluate_anomalydino`.
        results_path: Path to the CSV file.
        n_train:      Number of reference images.
        backbone:     Backbone used (recorded in the ``backbone`` column).
    """
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    row = {"n_train": n_train, "backbone": backbone, **metrics}
    df = pd.DataFrame([row])

    write_header = not results_path.exists()
    df.to_csv(results_path, mode="a", header=write_header, index=False)
    logger.info("Results appended → %s", results_path)


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate a trained AnomalyDINO model on the test split.",
    )
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--n-train", type=int, required=True,
        help="Number of reference images (recorded in the CSV)",
    )
    parser.add_argument("--results-csv", default="experiments/results.csv")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--backbone", default=DEFAULT_BACKBONE,
        choices=sorted(BACKBONE_CONFIG),
    )
    args = parser.parse_args()

    m = evaluate_anomalydino(
        split_dir=args.split_dir,
        checkpoint_path=args.checkpoint,
        image_size=args.image_size,
        backbone=args.backbone,
    )
    save_results(m, args.results_csv, args.n_train, backbone=args.backbone)
    print(f"\nMetrics: {m}")
