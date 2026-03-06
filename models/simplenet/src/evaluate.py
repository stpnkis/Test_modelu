"""
SimpleNet Evaluation Script
============================

Evaluates a trained SimpleNet model on the test set and computes:

  - AUROC       (threshold-independent ranking quality)
  - Accuracy    (fraction of correct predictions at optimal threshold)
  - Precision   (of predicted anomalies, how many are truly anomalous?)
  - Recall      (of all true anomalies, how many were detected?)
  - F1          (harmonic mean of precision and recall)

Evaluation strategy:
  1. Load the trained model checkpoint (.pth).
  2. Run inference on every test image (good + defective).
  3. Aggregate per-patch discriminator scores to a single image-level
     anomaly_score via a weighted blend:

       score = 0.7 × max(patch_probs) + 0.3 × mean(patch_probs)

     The max term detects localised defects while the mean term
     provides context sensitivity for diffuse anomalies.
  4. Compute sklearn metrics with optimal threshold chosen via
     Youden's J statistic (maximises TPR − FPR on the ROC curve).

Metric computation is **identical** to PatchCore's evaluate.py so that
results in experiments/results.csv are directly comparable.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
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
    Adaptor,
    Discriminator,
    FeatureExtractor,
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

def _load_simplenet(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[FeatureExtractor, Adaptor, Discriminator, dict]:
    """Reconstruct SimpleNet from a .pth checkpoint.

    Returns:
        (feature_extractor, adaptor, discriminator, config_dict)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    extractor = FeatureExtractor(cfg["backbone"], cfg["layers"]).to(device)
    adaptor = Adaptor(cfg["in_features"], cfg["adaptor_dim"]).to(device)
    discriminator = Discriminator(cfg["adaptor_dim"]).to(device)

    adaptor.load_state_dict(ckpt["adaptor_state_dict"])
    discriminator.load_state_dict(ckpt["discriminator_state_dict"])

    extractor.eval()
    adaptor.eval()
    discriminator.eval()

    return extractor, adaptor, discriminator, cfg


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _predict_scores(
    extractor: FeatureExtractor,
    adaptor: Adaptor,
    discriminator: Discriminator,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[float], list[int]]:
    """Run inference and return (image_scores, labels).

    Anomaly score per image = weighted blend of max and mean patch
    probabilities, aggregated over the spatial patch map.
    Higher score = more anomalous.
    """
    all_scores: list[float] = []
    all_labels: list[int] = []

    for images, labels in dataloader:
        images = images.to(device)
        feats = extractor(images)  # [B, C, H, W]
        B, C, H, W = feats.shape

        patches = feats.permute(0, 2, 3, 1).reshape(-1, C)

        # Forward through adaptor → discriminator.  No extra normalisations
        # here — the adaptor's internal BatchNorm layers handle feature
        # scaling, exactly mirroring the train.py forward pass.
        adapted = adaptor(patches)
        logits = discriminator(adapted)  # [B*H*W, 1]

        probs = torch.sigmoid(logits).reshape(B, H, W)
        # Blend max and mean over the spatial patch map.
        # max  → detects a single highly anomalous patch (localised defects)
        # mean → captures diffuse/spread anomalies that individually look mild
        image_scores = 0.7 * probs.amax(dim=(1, 2)) + 0.3 * probs.mean(dim=(1, 2))

        all_scores.extend(image_scores.cpu().tolist())
        all_labels.extend(labels.tolist())

    return all_scores, all_labels


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_simplenet(
    split_dir: str,
    checkpoint_path: str,
    image_size: int = 256,
    batch_size: int = 32,
    backbone: str = "wide_resnet50_2",
) -> dict:
    """
    Evaluate a trained SimpleNet model on the test set.

    Args:
        split_dir:        Path to the dataset split (same layout used for training).
        checkpoint_path:  Path to the .pth file produced by train.py.
        image_size:       Must match the value used during training.
        batch_size:       Batch size for inference.
        backbone:         Backbone name (used only for logging; loaded from ckpt).

    Returns:
        Dictionary of metric name -> value.
    """
    split_dir = Path(split_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Evaluating checkpoint: {checkpoint_path}")
    logger.info(f"Test data: {split_dir}")

    # ------------------------------------------------------------------
    # Step 1: Load model
    # ------------------------------------------------------------------
    extractor, adaptor, discriminator, cfg = _load_simplenet(
        checkpoint_path, device,
    )

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
        extractor, adaptor, discriminator, test_loader, device,
    )

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels, dtype=int)

    logger.info(
        f"Test set: {len(all_labels)} images  "
        f"({(all_labels == 0).sum()} OK, {(all_labels == 1).sum()} NOK)"
    )

    # ------------------------------------------------------------------
    # Step 4: Compute metrics (identical to PatchCore evaluate.py)
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
        metrics:      Dictionary returned by evaluate_simplenet().
        results_path: Path to the CSV file (created if missing).
        n_train:      Number of training images (recorded in the CSV).
    """
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    row = {"n_train": n_train, **metrics}
    df = pd.DataFrame([row])

    write_header = not results_path.exists()
    df.to_csv(results_path, mode="a", header=write_header, index=False)

    logger.info(f"Results appended to {results_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate SimpleNet model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split-dir", type=str, required=True,
                        help="Path to dataset split directory")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model_best.pth checkpoint")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    metrics = evaluate_simplenet(
        split_dir=args.split_dir,
        checkpoint_path=args.checkpoint,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )
    print(f"\nMetrics: {metrics}")
