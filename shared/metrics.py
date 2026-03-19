"""
Metrics — image-level and pixel-level evaluation.

Image-level:
    AUROC, Precision, Recall, F1

Pixel-level:
    AU-PRO (integrated up to FPR = 0.3)
    Reported only when the model produces anomaly maps AND the dataset
    has ground-truth masks.  Otherwise ``N/A`` with a reason.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ── Image-level metrics ─────────────────────────────────────────────────────


def compute_image_metrics(
    labels: np.ndarray | List[int],
    scores: np.ndarray | List[float],
    threshold: float,
) -> Dict[str, Any]:
    """Compute image-level classification metrics.

    Args:
        labels:    Ground-truth binary labels (0=normal, 1=anomalous).
        scores:    Anomaly scores (higher = more anomalous).
        threshold: Decision threshold derived from val/ok.

    Returns:
        Dictionary with ``auroc``, ``precision``, ``recall``, ``f1``,
        ``threshold``, and count fields.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    auroc = float(roc_auc_score(labels, scores))
    preds = (scores >= threshold).astype(int)

    return {
        "auroc": round(auroc, 4),
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, preds, zero_division=0)), 4),
        "threshold": round(threshold, 6),
        "n_test_total": len(labels),
        "n_test_ok": int((labels == 0).sum()),
        "n_test_nok": int((labels == 1).sum()),
    }


# ── Pixel-level metrics (AU-PRO) ────────────────────────────────────────────


def _compute_pro_curve(
    anomaly_maps: List[np.ndarray],
    masks: List[np.ndarray],
    num_thresholds: int = 300,
) -> tuple:
    """Compute the per-region overlap (PRO) curve.

    For each threshold, compute the mean TPR per connected component
    (per-region overlap) and the overall FPR.

    Returns:
        (fpr_array, pro_array) — arrays of shape ``(num_thresholds,)``.
    """
    from scipy import ndimage

    all_scores = np.concatenate([am.ravel() for am in anomaly_maps])
    thresholds = np.linspace(all_scores.max(), all_scores.min(), num_thresholds)

    fprs = []
    pros = []

    for t in thresholds:
        fpr_per_image = []
        pro_per_image = []

        for amap, mask in zip(anomaly_maps, masks):
            binary_pred = (amap >= t).astype(int)
            binary_mask = (mask > 0).astype(int)

            # FPR: false positive pixels / total normal pixels
            normal_pixels = (binary_mask == 0).sum()
            if normal_pixels > 0:
                fp = ((binary_pred == 1) & (binary_mask == 0)).sum()
                fpr_per_image.append(fp / normal_pixels)

            # Per-region overlap
            labeled, num_components = ndimage.label(binary_mask)
            for comp_idx in range(1, num_components + 1):
                region = labeled == comp_idx
                region_size = region.sum()
                if region_size > 0:
                    overlap = (binary_pred[region] == 1).sum()
                    pro_per_image.append(overlap / region_size)

        fprs.append(np.mean(fpr_per_image) if fpr_per_image else 0.0)
        pros.append(np.mean(pro_per_image) if pro_per_image else 0.0)

    return np.array(fprs), np.array(pros)


def compute_aupro(
    anomaly_maps: Optional[List[np.ndarray]],
    masks: Optional[List[np.ndarray]],
    fpr_limit: float = 0.3,
) -> Dict[str, Any]:
    """Compute AU-PRO integrated up to *fpr_limit*.

    Returns:
        Dictionary with ``aupro`` (float or ``"N/A"``) and ``aupro_reason``
        if not available.
    """
    if anomaly_maps is None or masks is None:
        return {
            "aupro": "N/A",
            "aupro_reason": "Model does not produce anomaly maps or dataset has no masks.",
        }

    if len(anomaly_maps) == 0 or len(masks) == 0:
        return {
            "aupro": "N/A",
            "aupro_reason": "No anomaly maps or masks available for AU-PRO.",
        }

    if len(anomaly_maps) != len(masks):
        return {
            "aupro": "N/A",
            "aupro_reason": (
                f"Mismatch: {len(anomaly_maps)} anomaly maps vs " f"{len(masks)} masks."
            ),
        }

    fpr, pro = _compute_pro_curve(anomaly_maps, masks)

    # Sort by FPR ascending
    order = np.argsort(fpr)
    fpr = fpr[order]
    pro = pro[order]

    # Clip to fpr_limit
    valid = fpr <= fpr_limit
    if valid.sum() < 2:
        return {
            "aupro": "N/A",
            "aupro_reason": f"Insufficient data points below FPR={fpr_limit}.",
        }

    fpr_clipped = fpr[valid]
    pro_clipped = pro[valid]

    # Normalize so that perfect = 1.0
    aupro_raw = float(auc(fpr_clipped, pro_clipped))
    aupro_normalized = aupro_raw / fpr_limit

    return {
        "aupro": round(aupro_normalized, 4),
        "aupro_fpr_limit": fpr_limit,
    }
