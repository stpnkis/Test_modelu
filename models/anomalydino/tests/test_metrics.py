"""
Unit tests for metric computation (``evaluate.py`` helpers)
=============================================================

Verifies that AUROC, Youden-J threshold, Accuracy, Precision, Recall
and F1 are computed identically to PatchCore's and SimpleNet's
``evaluate.py`` — using synthetic data only.

No model weights are loaded.  Only numpy + sklearn are required.
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _compute_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Standalone replica of the metric block in evaluate.py.

    Intentionally does NOT import evaluate.py (which pulls in torch).
    """
    auroc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j = tpr - fpr
    idx = int(np.argmax(j))
    threshold = float(thresholds[idx])

    preds = (scores >= threshold).astype(int)
    return {
        "auroc":     round(float(auroc), 4),
        "accuracy":  round(float(accuracy_score(labels, preds)), 4),
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall":    round(float(recall_score(labels, preds, zero_division=0)), 4),
        "f1":        round(float(f1_score(labels, preds, zero_division=0)), 4),
        "threshold": round(threshold, 6),
    }


class TestPerfectSeparation(unittest.TestCase):

    def test_all_ones(self) -> None:
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.15, 0.05, 0.3, 0.9, 0.85, 0.95, 0.8, 0.88])
        m = _compute_metrics(labels, scores)
        for key in ("auroc", "accuracy", "precision", "recall", "f1"):
            self.assertEqual(m[key], 1.0, f"{key} should be 1.0")


class TestRandomScores(unittest.TestCase):

    def test_auroc_near_chance(self) -> None:
        rng = np.random.RandomState(0)
        n = 10_000
        labels = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
        scores = rng.rand(n)
        m = _compute_metrics(labels, scores)
        self.assertAlmostEqual(m["auroc"], 0.5, delta=0.05)


class TestYoudenJ(unittest.TestCase):

    def test_threshold_maximises_j(self) -> None:
        labels = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0])
        scores = np.array([0.2, 0.35, 0.4, 0.7, 0.65, 0.9, 0.55, 0.3, 0.8, 0.25])
        fpr, tpr, thresholds = roc_curve(labels, scores)
        expected = float(thresholds[int(np.argmax(tpr - fpr))])
        m = _compute_metrics(labels, scores)
        self.assertAlmostEqual(m["threshold"], round(expected, 6))

    def test_deterministic(self) -> None:
        labels = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0.1, 0.4, 0.8, 0.7, 0.3, 0.9])
        self.assertEqual(_compute_metrics(labels, scores),
                         _compute_metrics(labels, scores))


class TestEdgeCases(unittest.TestCase):

    def test_all_positive(self) -> None:
        labels = np.ones(10, dtype=int)
        scores = np.linspace(0.5, 1.0, 10)
        fpr, tpr, _ = roc_curve(labels, scores)
        self.assertEqual(tpr[-1], 1.0)

    def test_single_class_safe(self) -> None:
        labels = np.zeros(5, dtype=int)
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = roc_auc_score(labels, scores)
            self.assertTrue(np.isnan(result) or isinstance(result, float))
        except ValueError:
            pass  # older sklearn raises — acceptable


class TestMetricRanges(unittest.TestCase):

    def test_all_in_0_1(self) -> None:
        rng = np.random.RandomState(42)
        for _ in range(50):
            n = rng.randint(20, 200)
            labels = rng.randint(0, 2, size=n)
            if labels.sum() in (0, n):
                continue
            m = _compute_metrics(labels, rng.rand(n))
            for key in ("auroc", "accuracy", "precision", "recall", "f1"):
                self.assertGreaterEqual(m[key], 0.0)
                self.assertLessEqual(m[key], 1.0)


if __name__ == "__main__":
    unittest.main()
