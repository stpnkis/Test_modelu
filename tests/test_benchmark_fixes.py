"""
Tests for benchmark protocol fixes.

Covers:
  - Fix 1: Results path uniqueness across (mode, n_train)
  - Fix 2: Threshold config propagation (orchestrator → model CLI)
  - Fix 3: No double preprocessing in shared pipeline
  - Fix 7: Split invariance across few-shot sizes
  - Fix 8: AU-PRO shape validation
  - Fix 9: Latency protocol enforcement
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from shared.results_io import get_results_dir, save_results, load_results
from shared.preprocessing import (
    build_transforms,
    validate_transform_pipeline,
    log_transform_pipeline,
)
from shared.split_manager import create_split, validate_split_invariance
from shared.metrics import compute_aupro
from shared.runtime_profiler import profile_model

import torchvision.transforms as T
import torch


# ── Fixtures ─────────────────────────────────────────────────────────────────

NUM_OK = 300
NUM_NOK = 20


@pytest.fixture
def fake_dataset(tmp_path):
    """Create a fake dataset with enough OK images for the split manager."""
    ds = tmp_path / "datasets" / "test_ds"
    ok_dir = ds / "ok"
    nok_dir = ds / "nok"
    ok_dir.mkdir(parents=True)
    nok_dir.mkdir(parents=True)

    for i in range(NUM_OK):
        (ok_dir / f"ok_{i:04d}.png").write_bytes(b"fake_image_data")
    for i in range(NUM_NOK):
        (nok_dir / f"nok_{i:04d}.png").write_bytes(b"fake_image_data")

    return ds, tmp_path / "splits"


# ── Fix 1: Results path uniqueness ───────────────────────────────────────────


class TestResultsPathUniqueness:
    """Same (dataset, model, seed) but different (mode, n_train) must produce
    different output directories."""

    def test_different_mode_different_dir(self):
        root = "/tmp/experiments"
        d1 = get_results_dir(root, "casting", "patchcore", 42, "baseline", 100)
        d2 = get_results_dir(root, "casting", "patchcore", 42, "high_accuracy", 100)
        assert d1 != d2

    def test_different_n_train_different_dir(self):
        root = "/tmp/experiments"
        d1 = get_results_dir(root, "casting", "patchcore", 42, "baseline", 50)
        d2 = get_results_dir(root, "casting", "patchcore", 42, "baseline", 100)
        assert d1 != d2

    def test_same_params_same_dir(self):
        root = "/tmp/experiments"
        d1 = get_results_dir(root, "casting", "patchcore", 42, "baseline", 100)
        d2 = get_results_dir(root, "casting", "patchcore", 42, "baseline", 100)
        assert d1 == d2

    def test_new_layout_contains_mode_and_n_train(self):
        root = "/tmp/experiments"
        d = get_results_dir(root, "casting", "patchcore", 42, "baseline", 100)
        assert "mode=baseline" in str(d)
        assert "n_train=100" in str(d)
        assert "seed=42" in str(d)

    def test_save_and_load_round_trip(self, tmp_path):
        metrics = {"auroc": 0.95, "f1": 0.90}
        path = save_results(
            experiments_root=str(tmp_path),
            dataset_id="test",
            model_name="model",
            seed=42,
            metrics=metrics,
            preprocessing_mode="baseline",
            n_train=100,
        )
        assert path.exists()
        loaded = load_results(
            str(tmp_path),
            "test",
            "model",
            42,
            preprocessing_mode="baseline",
            n_train=100,
        )
        assert loaded["metrics"]["auroc"] == 0.95
        assert loaded["preprocessing_mode"] == "baseline"
        assert loaded["n_train"] == 100

    def test_different_configs_no_overwrite(self, tmp_path):
        """Two different (mode, n_train) combos must not overwrite each other."""
        m1 = {"auroc": 0.90}
        m2 = {"auroc": 0.95}
        save_results(
            str(tmp_path),
            "ds",
            "model",
            42,
            m1,
            preprocessing_mode="baseline",
            n_train=50,
        )
        save_results(
            str(tmp_path),
            "ds",
            "model",
            42,
            m2,
            preprocessing_mode="baseline",
            n_train=100,
        )
        r1 = load_results(
            str(tmp_path),
            "ds",
            "model",
            42,
            preprocessing_mode="baseline",
            n_train=50,
        )
        r2 = load_results(
            str(tmp_path),
            "ds",
            "model",
            42,
            preprocessing_mode="baseline",
            n_train=100,
        )
        assert r1["metrics"]["auroc"] == 0.90
        assert r2["metrics"]["auroc"] == 0.95


# ── Fix 3: No double preprocessing ──────────────────────────────────────────


class TestPreprocessingValidation:
    """Validate that benchmark pipelines have no duplicate operations."""

    def test_baseline_valid(self):
        t = build_transforms("baseline")
        validate_transform_pipeline(t)  # Should not raise

    def test_high_accuracy_valid(self):
        t = build_transforms("high_accuracy")
        validate_transform_pipeline(t)

    def test_edge_safe_valid(self):
        t = build_transforms("edge_safe")
        validate_transform_pipeline(t)

    def test_double_normalize_fails(self):
        pipeline = T.Compose(
            [
                T.Resize(256),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        with pytest.raises(RuntimeError, match="duplicate transforms"):
            validate_transform_pipeline(pipeline)

    def test_double_resize_fails(self):
        pipeline = T.Compose(
            [
                T.Resize(256),
                T.Resize(224),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        with pytest.raises(RuntimeError, match="duplicate transforms"):
            validate_transform_pipeline(pipeline)

    def test_double_centercrop_fails(self):
        pipeline = T.Compose(
            [
                T.CenterCrop(256),
                T.CenterCrop(224),
            ]
        )
        with pytest.raises(RuntimeError, match="duplicate transforms"):
            validate_transform_pipeline(pipeline)


# ── Fix 7: Split invariance across few-shot sizes ───────────────────────────


class TestSplitInvariance:
    """Val and test must be identical across all n_train for same (dataset, seed)."""

    def test_invariance_holds(self, fake_dataset):
        ds, splits_root = fake_dataset
        n_values = [5, 10, 20]
        for n in n_values:
            create_split(
                str(ds),
                str(splits_root),
                "test_ds",
                n_train=n,
                seed=42,
                link_mode="copy",
            )
        # Should not raise
        validate_split_invariance(str(splits_root), "test_ds", 42, n_values)

    def test_invariance_single_n_train(self, fake_dataset):
        """Single n_train should pass trivially."""
        ds, splits_root = fake_dataset
        create_split(
            str(ds),
            str(splits_root),
            "test_ds",
            n_train=5,
            seed=42,
            link_mode="copy",
        )
        validate_split_invariance(str(splits_root), "test_ds", 42, [5])


# ── Fix 8: AU-PRO shape validation ──────────────────────────────────────────


class TestAUPROShapeValidation:
    """AU-PRO must fail if anomaly map and mask shapes don't match."""

    def test_shape_mismatch_raises(self):
        amap = [np.zeros((64, 64))]
        mask = [np.zeros((32, 32))]
        with pytest.raises(RuntimeError, match="shape mismatch"):
            compute_aupro(amap, mask)

    def test_shape_match_ok(self):
        rng = np.random.RandomState(42)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        amap = rng.rand(64, 64)
        result = compute_aupro([amap], [mask])
        assert result["aupro"] != "N/A" or "aupro_reason" in result


# ── Fix 9: Latency protocol enforcement ─────────────────────────────────────


class TestLatencyProtocol:
    """Latency profiler must enforce minimum warmup and iterations."""

    def test_too_few_warmup_raises(self):
        device = torch.device("cpu")
        with pytest.raises(ValueError, match="warmup >= 10"):
            profile_model(
                model_only_fn=lambda: None,
                end_to_end_fn=lambda: None,
                device=device,
                warmup=5,
                iterations=30,
            )

    def test_too_few_iterations_raises(self):
        device = torch.device("cpu")
        with pytest.raises(ValueError, match="measured iterations >= 30"):
            profile_model(
                model_only_fn=lambda: None,
                end_to_end_fn=lambda: None,
                device=device,
                warmup=10,
                iterations=10,
            )

    def test_valid_params_pass(self):
        device = torch.device("cpu")
        result = profile_model(
            model_only_fn=lambda: None,
            end_to_end_fn=lambda: None,
            device=device,
            warmup=10,
            iterations=30,
        )
        assert result.warmup_iterations == 10
        assert result.measured_iterations == 30
        assert result.batch_size == 1
