"""
PatchCore Evaluation Script
============================

Evaluates a trained PatchCore model on the test set and computes:

  - AUROC      (threshold-independent ranking quality)
  - Accuracy   (fraction of correct predictions at optimal threshold)
  - Precision  (of predicted anomalies, how many are truly anomalous?)
  - Recall     (of all true anomalies, how many were detected?)

Evaluation strategy:
  1. Load the trained model checkpoint.
  2. Run anomalib's engine.test()  -> built-in image-level AUROC.
  3. Run engine.predict()          -> per-image anomaly scores.
  4. Compute sklearn metrics from the per-image scores.

The optimal binary threshold is chosen via Youden's J statistic
(maximises  TPR − FPR  on the ROC curve).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_field(obj, *names):
    """Try to extract a named field from an object or dict."""
    for name in names:
        if hasattr(obj, name):
            val = getattr(obj, name)
            if val is not None:
                return val
        if isinstance(obj, dict) and name in obj:
            val = obj[name]
            if val is not None:
                return val
    return None


def _to_numpy(tensor_or_array) -> np.ndarray:
    """Convert a torch Tensor or array-like to a flat numpy array."""
    if hasattr(tensor_or_array, "cpu"):
        return tensor_or_array.cpu().numpy().flatten()
    return np.asarray(tensor_or_array).flatten()


def _label_from_path(path: str) -> int:
    """Infer ground truth label from file path (fallback when labels are missing).

    Returns 0 for normal (path contains 'good'), 1 for anomalous.
    """
    return 0 if "good" in str(path).lower() else 1


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate_patchcore(
    split_dir: str,
    checkpoint_path: str,
    image_size: int = 256,
    batch_size: int = 32,
    backbone: str = "wide_resnet50_2",
) -> dict:
    """
    Evaluate a trained PatchCore model on the test set.

    Args:
        split_dir:        Path to the dataset split (same layout used for training).
        checkpoint_path:  Path to the .ckpt file produced by train.py.
        image_size:       Must match the value used during training.
        batch_size:       Batch size for inference.

    Returns:
        Dictionary of metric name -> value.
    """
    split_dir = Path(split_dir)

    logger.info(f"Evaluating checkpoint: {checkpoint_path}")
    logger.info(f"Test data: {split_dir}")

    # ------------------------------------------------------------------
    # Step 1: Data module  (same configuration as training)
    # ------------------------------------------------------------------
    datamodule = Folder(
        name="casting",
        root=str(split_dir),
        normal_dir="train/good",
        abnormal_dir="test/defective",
        normal_test_dir="test/good",
        task="classification",
        image_size=(
            (image_size, image_size) if isinstance(image_size, int) else image_size
        ),
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
    )

    # ------------------------------------------------------------------
    # Step 2: Load the trained model from checkpoint
    # ------------------------------------------------------------------
    # The checkpoint contains backbone weights AND the memory bank
    # (coreset), so the model is ready for inference immediately.
    # Instantiate the class standardly so signature changes don't brick the loader
    model = Patchcore(
        backbone=backbone,
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )

    # ------------------------------------------------------------------
    # Step 3: Run anomalib Engine
    # ------------------------------------------------------------------
    engine = Engine(
        task="classification",
        default_root_dir=str(split_dir.parent),
        devices=1,
        accelerator="auto",
    )

    # 3a. engine.test() -> anomalib's built-in metrics (dict) -> disabled to not iterate twice over dataset unnecessarily
    # test_results = engine.test(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)
    # logger.info(f"anomalib test results: {test_results}")

    # 3b. engine.predict() -> per-image predictions
    predictions = engine.predict(
        model=model, datamodule=datamodule, ckpt_path=checkpoint_path
    )

    # ------------------------------------------------------------------
    # Step 4: Collect per-image scores and ground-truth labels
    # ------------------------------------------------------------------
    all_scores: list[float] = []
    all_labels: list[int] = []

    for batch in predictions:
        # Anomaly score (higher = more anomalous)
        scores = _extract_field(batch, "pred_score", "pred_scores")
        # Ground-truth label (0 = normal, 1 = anomalous)
        labels = _extract_field(batch, "gt_label", "label", "gt_labels")
        # Image paths (used as fallback to infer labels)
        paths = _extract_field(batch, "image_path", "image_paths")

        if scores is not None:
            all_scores.extend(_to_numpy(scores).tolist())

        if labels is not None:
            all_labels.extend(_to_numpy(labels).astype(int).tolist())
        elif paths is not None:
            # Fallback: infer labels from filesystem path
            all_labels.extend(_label_from_path(p) for p in paths)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels, dtype=int)

    logger.info(
        f"Test set: {len(all_labels)} images  "
        f"({(all_labels == 0).sum()} OK, {(all_labels == 1).sum()} NOK)"
    )

    # ------------------------------------------------------------------
    # Step 5: Compute metrics
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

    metrics = {
        "auroc": round(float(auroc), 4),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
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
        metrics:      Dictionary returned by evaluate_patchcore().
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

    parser = argparse.ArgumentParser(description="Evaluate PatchCore model")
    parser.add_argument("--split-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--n-train",
        type=int,
        required=True,
        help="Number of training images (recorded in CSV)",
    )
    parser.add_argument("--results-csv", type=str, default="experiments/results.csv")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    metrics = evaluate_patchcore(
        split_dir=args.split_dir,
        checkpoint_path=args.checkpoint,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    save_results(metrics, args.results_csv, args.n_train)
    print(f"\nMetrics: {metrics}")
