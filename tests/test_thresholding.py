"""
Tests for shared.thresholding — val-only anomaly threshold.

Covers:
  - Quantile strategy
  - Max strategy
  - K-sigma strategy
  - Empty scores error
  - Unknown strategy error
  - Test labels never used (only val/ok scores accepted)
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.thresholding import compute_threshold


class TestQuantileStrategy:
    """Default: p-quantile on val/ok scores."""

    def test_basic_quantile(self):
        scores = list(range(100))  # 0..99
        t = compute_threshold(scores, "quantile", quantile_p=0.99)
        assert isinstance(t, float)
        assert 95 <= t <= 100  # p=0.99 of uniform 0..99 ≈ 98.01

    def test_quantile_p0_returns_min(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        t = compute_threshold(scores, "quantile", quantile_p=0.0)
        assert t == pytest.approx(1.0)

    def test_quantile_p1_returns_max(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        t = compute_threshold(scores, "quantile", quantile_p=1.0)
        assert t == pytest.approx(5.0)

    def test_deterministic(self):
        scores = np.random.RandomState(42).randn(100).tolist()
        t1 = compute_threshold(scores, "quantile", quantile_p=0.99)
        t2 = compute_threshold(scores, "quantile", quantile_p=0.99)
        assert t1 == t2


class TestMaxStrategy:
    """Threshold = max of val/ok scores."""

    def test_max_equals_highest_score(self):
        scores = [0.1, 0.5, 0.3, 0.9, 0.2]
        t = compute_threshold(scores, "max")
        assert t == pytest.approx(0.9)

    def test_single_score(self):
        t = compute_threshold([42.0], "max")
        assert t == pytest.approx(42.0)


class TestKSigmaStrategy:
    """Threshold = mean + k * std."""

    def test_basic_k_sigma(self):
        scores = [10.0] * 20  # constant → std=0
        t = compute_threshold(scores, "k_sigma", k_sigma=3.0)
        assert t == pytest.approx(10.0)

    def test_k_sigma_with_variance(self):
        np.random.seed(0)
        scores = np.random.normal(100, 5, 1000).tolist()
        t = compute_threshold(scores, "k_sigma", k_sigma=3.0)
        # mean≈100, std≈5 → threshold ≈ 115
        assert 110 < t < 125


class TestEdgeCases:
    """Error handling and boundary conditions."""

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            compute_threshold([], "quantile")

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown threshold strategy"):
            compute_threshold([1.0, 2.0], "youden_j")

    def test_numpy_input(self):
        arr = np.array([1.0, 2.0, 3.0])
        t = compute_threshold(arr, "quantile", quantile_p=0.5)
        assert isinstance(t, float)

    def test_returns_scalar_float(self):
        t = compute_threshold([1.0, 2.0, 3.0], "quantile")
        assert isinstance(t, float)
        assert np.isscalar(t)
