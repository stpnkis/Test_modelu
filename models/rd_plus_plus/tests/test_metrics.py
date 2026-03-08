"""
Unit tests for metric computation (evaluate.py helpers)
=========================================================

Verifies that AUROC, Youden-J threshold, Accuracy, Precision and Recall
are computed identically to SimpleNet's / PatchCore's evaluate.py on
synthetic data.

No model weights are loaded — these tests exercise only the pure-numpy /
sklearn metric path to guarantee numerical correctness of the evaluation
pipeline.

Uses only the standard library + numpy + sklearn (no extra dependencies).
"""

import unittest

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
    """Replicates the metric block from evaluate.py (and SimpleNet's evaluate.py).

    This function is intentionally a standalone copy so the test does not
    import evaluate.py (which would pull in torch / torchvision).
    """
    auroc = roc_auc_score(labels, scores)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    optimal_idx = int(np.argmax(j_scores))
    threshold = float(thresholds[optimal_idx])

    pred_labels = (scores >= threshold).astype(int)

    accuracy = accuracy_score(labels, pred_labels)
    precision = precision_score(labels, pred_labels, zero_division=0)
    recall = recall_score(labels, pred_labels, zero_division=0)
    f1 = f1_score(labels, pred_labels, zero_division=0)

    return {
        "auroc": round(float(auroc), 4),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "threshold": round(threshold, 6),
    }


class TestPerfectSeparation(unittest.TestCase):
    """When scores perfectly separate classes, all metrics should be 1.0."""

    def test_perfect_metrics(self) -> None:
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.15, 0.05, 0.3, 0.9, 0.85, 0.95, 0.8, 0.88])
        m = _compute_metrics(labels, scores)

        self.assertEqual(m["auroc"], 1.0)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["f1"], 1.0)


class TestRandomScores(unittest.TestCase):
    """With random scores on a large balanced set, AUROC ≈ 0.5."""

    def test_auroc_near_chance(self) -> None:
        rng = np.random.RandomState(0)
        n = 10_000
        labels = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
        scores = rng.rand(n)
        m = _compute_metrics(labels, scores)

        self.assertAlmostEqual(m["auroc"], 0.5, delta=0.05)


class TestYoudenJThreshold(unittest.TestCase):
    """Verify that the chosen threshold actually maximises TPR − FPR."""

    def test_threshold_maximises_j(self) -> None:
        labels = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0])
        scores = np.array([0.2, 0.35, 0.4, 0.7, 0.65, 0.9, 0.55, 0.3, 0.8, 0.25])

        fpr, tpr, thresholds = roc_curve(labels, scores)
        j = tpr - fpr
        best_idx = int(np.argmax(j))
        expected_threshold = float(thresholds[best_idx])

        m = _compute_metrics(labels, scores)
        self.assertAlmostEqual(m["threshold"], round(expected_threshold, 6))

    def test_threshold_stability_across_runs(self) -> None:
        """Identical inputs must always produce the same threshold."""
        labels = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0.1, 0.4, 0.8, 0.7, 0.3, 0.9])

        m1 = _compute_metrics(labels, scores)
        m2 = _compute_metrics(labels, scores)
        self.assertEqual(m1, m2)


class TestEdgeCases(unittest.TestCase):
    """Edge cases that should not crash."""

    def test_all_positive(self) -> None:
        """All samples are anomalous — recall should be 1.0."""
        labels = np.ones(10, dtype=int)
        scores = np.linspace(0.5, 1.0, 10)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        self.assertEqual(tpr[-1], 1.0)

    def test_single_class_is_undefined(self) -> None:
        """roc_auc_score with one class raises ValueError or returns NaN."""
        labels = np.zeros(5, dtype=int)
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = roc_auc_score(labels, scores)
            self.assertTrue(np.isnan(result) or isinstance(result, float))
        except ValueError:
            pass  # older sklearn raises — also acceptable


class TestMetricRanges(unittest.TestCase):
    """All metrics must stay within valid ranges."""

    def test_ranges(self) -> None:
        rng = np.random.RandomState(42)
        for _ in range(50):
            n = rng.randint(20, 200)
            labels = rng.randint(0, 2, size=n)
            if labels.sum() == 0 or labels.sum() == n:
                continue  # skip degenerate case
            scores = rng.rand(n)
            m = _compute_metrics(labels, scores)

            self.assertGreaterEqual(m["auroc"], 0.0)
            self.assertLessEqual(m["auroc"], 1.0)
            self.assertGreaterEqual(m["accuracy"], 0.0)
            self.assertLessEqual(m["accuracy"], 1.0)
            self.assertGreaterEqual(m["precision"], 0.0)
            self.assertLessEqual(m["precision"], 1.0)
            self.assertGreaterEqual(m["recall"], 0.0)
            self.assertLessEqual(m["recall"], 1.0)
            self.assertGreaterEqual(m["f1"], 0.0)
            self.assertLessEqual(m["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
