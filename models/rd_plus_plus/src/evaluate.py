"""
RD++ Evaluation Script
=======================

Evaluates a trained RD++ model on the test set and computes:

  - AUROC       (threshold-independent ranking quality)
  - Accuracy    (fraction of correct predictions at optimal threshold)
  - Precision   (of predicted anomalies, how many are truly anomalous?)
  - Recall      (of all true anomalies, how many were detected?)
  - F1          (harmonic mean of precision and recall)

Evaluation strategy:
  1. Load the trained model checkpoint (.pth).
  2. Run inference on every test image (good + defective).
  3. For each image:
       a. Extract multi-scale encoder features.
       b. Project features → BN → decoder → reconstructed features.
       c. Compute per-scale anomaly maps via 1 − cosine_similarity.
       d. Sum maps, apply Gaussian smoothing (σ=4).
       e. Image-level anomaly_score = max(smoothed_anomaly_map).
  4. Compute sklearn metrics with optimal threshold chosen via
     Youden's J statistic (maximises TPR − FPR on the ROC curve).

Metric computation is **identical** to SimpleNet's and PatchCore's evaluate.py
so that results in experiments/results.csv are directly comparable.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Import model architecture + helpers from train module
from train import (
    IMAGE_EXTENSIONS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    BottleneckNetwork,
    Decoder,
    Encoder,
    MultiProjectionLayer,
    compute_anomaly_map,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Labelled test dataset
# ---------------------------------------------------------------------------

class LabeledImageDataset(Dataset):
    """Load test images from good/ and defective/ directories with labels.

    Label convention: 0 = normal (good), 1 = anomalous (defective).
    """

    def __init__(
        self,
        good_dir: str | Path,
        defective_dir: str | Path,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.items: list[tuple[Path, int]] = []
        for p in sorted(Path(good_dir).iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                self.items.append((p, 0))
        for p in sorted(Path(defective_dir).iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                self.items.append((p, 1))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_rd_plus_plus(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[BottleneckNetwork, Decoder, MultiProjectionLayer, dict]:
    """Reconstruct RD++ from a .pth checkpoint.

    Returns:
        (bottleneck_network, decoder, projection_layer, config_dict)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    bn = BottleneckNetwork(width_per_group=cfg["width_per_group"]).to(device)
    decoder = Decoder(
        layers=cfg["decoder_layers"],
        width_per_group=cfg["width_per_group"],
    ).to(device)
    proj_layer = MultiProjectionLayer(base=cfg["proj_base"]).to(device)

    bn.load_state_dict(ckpt["bn_state_dict"])
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    proj_layer.load_state_dict(ckpt["proj_state_dict"])

    bn.eval()
    decoder.eval()
    proj_layer.eval()

    return bn, decoder, proj_layer, cfg


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _predict_scores(
    encoder: Encoder,
    bn: BottleneckNetwork,
    decoder: Decoder,
    proj_layer: MultiProjectionLayer,
    dataloader: DataLoader,
    device: torch.device,
    image_size: int = 256,
) -> tuple[list[float], list[int]]:
    """Run inference and return (image_scores, labels).

    Anomaly score per image:
      1. Encoder → multi-scale features.
      2. Projection → BN → Decoder → reconstructed features.
      3. Anomaly map = sum of (1 − cos_sim) at each scale.
      4. Gaussian smoothing (σ=4).
      5. Image score = max of smoothed anomaly map.

    Higher score = more anomalous.
    """
    all_scores: list[float] = []
    all_labels: list[int] = []

    for images, labels in dataloader:
        images = images.to(device)

        # Encoder (frozen)
        inputs = encoder(images)

        # Forward: projection → BN → decoder
        features = proj_layer(inputs)
        outputs = decoder(bn(features))

        # Anomaly map: per-scale cosine distance, summed
        anomaly_map = compute_anomaly_map(inputs, outputs, image_size)
        anomaly_map = anomaly_map.cpu().numpy()  # [B, H, W]

        # Gaussian smoothing + image-level score
        for j in range(anomaly_map.shape[0]):
            smoothed = scipy_gaussian_filter(anomaly_map[j], sigma=4)
            score = float(smoothed.max())
            all_scores.append(score)

        all_labels.extend(labels.tolist())

    return all_scores, all_labels


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_rd_plus_plus(
    split_dir: str,
    checkpoint_path: str,
    image_size: int = 256,
    batch_size: int = 32,
) -> dict:
    """
    Evaluate a trained RD++ model on the test set.

    Args:
        split_dir:        Path to the dataset split (same layout used for training).
        checkpoint_path:  Path to the .pth file produced by train.py.
        image_size:       Must match the value used during training.
        batch_size:       Batch size for inference.

    Returns:
        Dictionary of metric name -> value.
    """
    split_dir = Path(split_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Evaluating checkpoint: {checkpoint_path}")
    logger.info(f"Test data: {split_dir}")

    # ------------------------------------------------------------------
    # Step 1: Load models
    # ------------------------------------------------------------------
    encoder = Encoder().to(device)
    encoder.eval()

    bn, decoder, proj_layer, cfg = _load_rd_plus_plus(checkpoint_path, device)

    # ------------------------------------------------------------------
    # Step 2: Test data
    # ------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    test_dataset = LabeledImageDataset(
        good_dir=split_dir / "test" / "good",
        defective_dir=split_dir / "test" / "defective",
        transform=transform,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Step 3: Run inference
    # ------------------------------------------------------------------
    all_scores, all_labels = _predict_scores(
        encoder, bn, decoder, proj_layer, test_loader, device, image_size,
    )

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels, dtype=int)

    logger.info(
        f"Test set: {len(all_labels)} images  "
        f"({(all_labels == 0).sum()} OK, {(all_labels == 1).sum()} NOK)"
    )

    # ------------------------------------------------------------------
    # Step 4: Compute metrics (identical to SimpleNet / PatchCore)
    # ------------------------------------------------------------------
    # AUROC (threshold-independent)
    auroc = roc_auc_score(all_labels, all_scores)

    # Find optimal threshold via Youden's J statistic
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    j_scores = tpr - fpr
    optimal_idx = int(np.argmax(j_scores))
    threshold = float(thresholds[optimal_idx])

    # Binary predictions at optimal threshold
    pred_labels = (all_scores >= threshold).astype(int)

    accuracy = accuracy_score(all_labels, pred_labels)
    precision = precision_score(all_labels, pred_labels, zero_division=0)
    recall = recall_score(all_labels, pred_labels, zero_division=0)
    f1 = f1_score(all_labels, pred_labels, zero_division=0)

    metrics = {
        "auroc": round(float(auroc), 4),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "threshold": round(threshold, 6),
        "n_test_total": len(all_labels),
        "n_test_ok": int((all_labels == 0).sum()),
        "n_test_nok": int((all_labels == 1).sum()),
    }

    logger.info(f"Metrics: {metrics}")
    return metrics


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def save_results(metrics: dict, results_path: str, n_train: int) -> None:
    """Append one experiment's metrics to the results CSV.

    Args:
        metrics:      Dictionary returned by evaluate_rd_plus_plus().
        results_path: Path to the CSV file (created if missing).
        n_train:      Number of training images (recorded in the CSV).
    """
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    row = {"n_train": n_train, **metrics}
    df = pd.DataFrame([row])

    write_header = not results_path.exists()
    df.to_csv(results_path, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate RD++ model")
    parser.add_argument("--split-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    metrics = evaluate_rd_plus_plus(
        split_dir=args.split_dir,
        checkpoint_path=args.checkpoint,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )
    for k, v in metrics.items():
        print(f"  {k}: {v}")
