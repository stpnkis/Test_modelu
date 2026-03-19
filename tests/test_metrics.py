"""
Tests for shared.metrics — image-level and pixel-level evaluation.

Covers:
  - AUROC range [0, 1]
  - Perfect classifier
  - Random classifier
  - F1/precision/recall correctness
  - AU-PRO: None inputs → N/A
  - AU-PRO: mismatched lengths → N/A
  - AU-PRO: basic synthetic computation
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.metrics import compute_image_metrics, compute_aupro


class TestImageMetrics:
    """Image-level AUROC, precision, recall, F1."""

    def test_perfect_classifier(self):
        labels = [0, 0, 0, 1, 1, 1]
        scores = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]
        result = compute_image_metrics(labels, scores, threshold=0.5)

        assert result["auroc"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_auroc_range(self):
        rng = np.random.RandomState(42)
        labels = rng.randint(0, 2, 100).tolist()
        scores = rng.rand(100).tolist()
        result = compute_image_metrics(labels, scores, threshold=0.5)

        assert 0.0 <= result["auroc"] <= 1.0

    def test_all_normal_classified_as_normal(self):
        labels = [0, 0, 0, 1, 1, 1]
        scores = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]
        result = compute_image_metrics(labels, scores, threshold=0.5)

        assert result["n_test_ok"] == 3
        assert result["n_test_nok"] == 3
        assert result["n_test_total"] == 6

    def test_threshold_stored_in_result(self):
        labels = [0, 1]
        scores = [0.1, 0.9]
        result = compute_image_metrics(labels, scores, threshold=0.5)
        assert result["threshold"] == 0.5

    def test_numpy_inputs(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        result = compute_image_metrics(labels, scores, threshold=0.5)
        assert "auroc" in result

    def test_high_threshold_all_normal(self):
        """Very high threshold → all predicted normal → recall=0 for anomalies."""
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.3, 0.4]
        result = compute_image_metrics(labels, scores, threshold=100.0)
        assert result["recall"] == 0.0


class TestAUPRO:
    """Pixel-level AU-PRO metric."""

    def test_none_inputs_returns_na(self):
        result = compute_aupro(None, None)
        assert result["aupro"] == "N/A"

    def test_empty_lists_returns_na(self):
        result = compute_aupro([], [])
        assert result["aupro"] == "N/A"

    def test_mismatched_lengths_returns_na(self):
        amaps = [np.zeros((32, 32))]
        masks = [np.zeros((32, 32)), np.ones((32, 32))]
        result = compute_aupro(amaps, masks)
        assert result["aupro"] == "N/A"
        assert "Mismatch" in result.get("aupro_reason", "")

    def test_graded_aupro(self):
        """Graded anomaly map with smooth values should give meaningful AU-PRO."""
        rng = np.random.RandomState(123)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 1

        # Graded anomaly map: high in defect area, gradient in between
        amap = rng.rand(64, 64) * 0.3  # normal region: low scores
        amap[20:40, 20:40] = 0.5 + rng.rand(20, 20) * 0.5  # defect: high

        result = compute_aupro([amap], [mask])
        assert isinstance(result["aupro"], float)
        assert result["aupro"] > 0.0  # Should have nonzero area

    def test_aupro_range(self):
        """AU-PRO should be in [0, 1] for reasonable inputs."""
        rng = np.random.RandomState(42)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        amap = rng.rand(64, 64)

        result = compute_aupro([amap], [mask])
        if isinstance(result["aupro"], float):
            assert 0.0 <= result["aupro"] <= 1.0
