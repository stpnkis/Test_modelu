"""
Thresholding — derive anomaly threshold from validation data only.

The threshold is **never** derived from test data or test labels.
No ``roc_curve``, no Youden J on test data.

Supported strategies:
    ``quantile`` — p-quantile of val/ok scores (default p=0.99)
    ``max``      — maximum of val/ok scores
    ``k_sigma``  — mean + k × std of val/ok scores (default k=3)
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def compute_threshold(
    val_scores: List[float] | np.ndarray,
    strategy: str = "quantile",
    *,
    quantile_p: float = 0.99,
    k_sigma: float = 3.0,
) -> float:
    """Compute anomaly threshold from **validation-only** normal scores.

    Args:
        val_scores: Anomaly scores of **val/ok** images (all normal).
        strategy:   ``"quantile"`` | ``"max"`` | ``"k_sigma"``.
        quantile_p: Percentile for the quantile strategy (0–1).
        k_sigma:    Multiplier for the k-sigma strategy.

    Returns:
        Scalar threshold.  Scores >= threshold are classified as anomalous.

    Raises:
        ValueError: If val_scores is empty or strategy is unknown.
    """
    scores = np.asarray(val_scores, dtype=float)
    if scores.size == 0:
        raise ValueError("val_scores must not be empty.")

    if strategy == "quantile":
        threshold = float(np.quantile(scores, quantile_p))
        logger.info(
            "Threshold (quantile p=%.3f): %.6f  " "[val_ok n=%d, min=%.4f, max=%.4f]",
            quantile_p,
            threshold,
            len(scores),
            float(scores.min()),
            float(scores.max()),
        )
    elif strategy == "max":
        threshold = float(scores.max())
        logger.info(
            "Threshold (max): %.6f  [val_ok n=%d]",
            threshold,
            len(scores),
        )
    elif strategy == "k_sigma":
        mu = float(scores.mean())
        sigma = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
        threshold = mu + k_sigma * sigma
        logger.info(
            "Threshold (k_sigma, k=%.1f): %.6f  " "[mean=%.4f, std=%.4f, val_ok n=%d]",
            k_sigma,
            threshold,
            mu,
            sigma,
            len(scores),
        )
    else:
        raise ValueError(
            f"Unknown threshold strategy '{strategy}'. "
            f"Available: quantile, max, k_sigma."
        )

    return threshold
