"""
Smoke test — full pipeline on a synthetic mini dataset.

Generates a tiny fake dataset in a temp dir, runs the shared pipeline
(validation → split → preprocessing → thresholding → metrics), and
verifies the entire chain works end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from shared.dataset_schema import validate_dataset
from shared.split_manager import create_split, load_split_manifest
from shared.preprocessing import build_transforms, get_image_size
from shared.thresholding import compute_threshold
from shared.metrics import compute_image_metrics
from shared.dataset_registry import DatasetConfig, get_dataset_config


# Mock config for the synthetic dataset: custom_holdout with 300 ok, 10 nok
_SYNTH_CONFIG = DatasetConfig(
    dataset_id="synth",
    split_policy="custom_holdout",
    official_train_ok=0,
    official_test_ok=0,
    total_nok=10,
    val_ok_count=30,
    main_train_ok=220,  # 300 - 50 (test) - 30 (val)
    custom_test_ok_count=50,
    few_shot_levels=[5, 10, 25],
    has_masks=False,
)


def _patch_registry(monkeypatch):
    _original = get_dataset_config

    def _patched(dataset_id):
        if dataset_id == "synth":
            return _SYNTH_CONFIG
        return _original(dataset_id)

    monkeypatch.setattr("shared.split_manager.get_dataset_config", _patched)


def _make_synthetic_image(path: Path, size: int = 64, anomalous: bool = False):
    """Create a synthetic image. Anomalous ones have a bright square."""
    rng = np.random.RandomState(hash(str(path)) % (2**31))
    arr = rng.randint(50, 150, (size, size, 3), dtype=np.uint8)
    if anomalous:
        arr[10:30, 10:30, :] = 255  # bright patch = defect
    img = Image.fromarray(arr, "RGB")
    img.save(path)


@pytest.fixture
def mini_dataset(tmp_path, monkeypatch):
    """Build a 300-ok + 10-nok synthetic dataset."""
    ds_dir = tmp_path / "datasets" / "synth"
    ok_dir = ds_dir / "ok"
    nok_dir = ds_dir / "nok"
    ok_dir.mkdir(parents=True)
    nok_dir.mkdir(parents=True)

    for i in range(300):
        _make_synthetic_image(ok_dir / f"ok_{i:04d}.png")
    for i in range(10):
        _make_synthetic_image(nok_dir / f"nok_{i:04d}.png", anomalous=True)

    _patch_registry(monkeypatch)

    return {
        "dataset_dir": ds_dir,
        "splits_root": tmp_path / "splits",
        "experiments_root": tmp_path / "experiments",
    }


class TestFullPipeline:
    """End-to-end: validate → split → preprocess → threshold → metrics."""

    def test_pipeline(self, mini_dataset):
        ds_dir = mini_dataset["dataset_dir"]
        splits_root = mini_dataset["splits_root"]
        seed = 42
        n_train = 5

        # 1. Validate dataset
        stats = validate_dataset(ds_dir)
        assert stats["ok_count"] == 300
        assert stats["nok_count"] == 10

        # 2. Create split
        manifest = create_split(
            str(ds_dir),
            str(splits_root),
            "synth",
            n_train=n_train,
            seed=seed,
            link_mode="copy",
        )
        assert manifest["counts"]["train_ok"] == n_train
        assert manifest["counts"]["val_ok"] > 0
        assert manifest["counts"]["test_nok"] == 10

        # Verify manifest can be loaded back
        loaded = load_split_manifest(str(splits_root), "synth", seed, n_train)
        assert loaded["dataset_id"] == "synth"

        # 3. Load & preprocess images
        split_dir = Path(manifest["split_dir"])
        transform = build_transforms("baseline")
        image_size = get_image_size("baseline")
        assert image_size == 224

        # Load val images
        val_dir = split_dir / "val" / "ok"
        val_images = sorted(val_dir.iterdir())
        assert len(val_images) > 0

        val_tensors = []
        for img_path in val_images:
            img = Image.open(img_path).convert("RGB")
            t = transform(img)
            assert t.shape == (3, 224, 224)
            val_tensors.append(t)

        # Load test images
        test_ok_dir = split_dir / "test" / "ok"
        test_nok_dir = split_dir / "test" / "nok"

        test_labels = []
        test_tensors = []
        for img_path in sorted(test_ok_dir.iterdir()):
            t = transform(Image.open(img_path).convert("RGB"))
            test_tensors.append(t)
            test_labels.append(0)
        for img_path in sorted(test_nok_dir.iterdir()):
            t = transform(Image.open(img_path).convert("RGB"))
            test_tensors.append(t)
            test_labels.append(1)

        assert len(test_labels) > 0

        # 4. Simulate scoring (simple pixel-sum heuristic)
        val_scores = [float(t.sum()) for t in val_tensors]
        test_scores = [float(t.sum()) for t in test_tensors]

        # 5. Compute threshold from val/ok only
        threshold = compute_threshold(val_scores, "quantile", quantile_p=0.99)
        assert isinstance(threshold, float)

        # 6. Compute metrics
        result = compute_image_metrics(test_labels, test_scores, threshold)
        assert "auroc" in result
        assert "f1" in result
        assert "precision" in result
        assert "recall" in result
        assert 0.0 <= result["auroc"] <= 1.0
        assert result["n_test_total"] == len(test_labels)

    def test_pipeline_reproducible(self, mini_dataset):
        """Same seed → identical results."""
        ds_dir = mini_dataset["dataset_dir"]
        splits_root = mini_dataset["splits_root"]

        def run_pipeline(seed=42, n_train=5):
            m = create_split(
                str(ds_dir),
                str(splits_root),
                "synth",
                n_train=n_train,
                seed=seed,
                link_mode="copy",
            )
            split_dir = Path(m["split_dir"])
            transform = build_transforms("baseline")

            val_imgs = sorted((split_dir / "val" / "ok").iterdir())
            val_scores = [
                float(transform(Image.open(p).convert("RGB")).sum()) for p in val_imgs
            ]
            return compute_threshold(val_scores, "quantile", quantile_p=0.99)

        t1 = run_pipeline()
        t2 = run_pipeline()
        assert t1 == t2

    def test_preprocessing_modes(self, mini_dataset):
        """All preprocessing modes produce tensors of expected size."""
        ds_dir = mini_dataset["dataset_dir"]
        img_path = sorted((ds_dir / "ok").iterdir())[0]
        img = Image.open(img_path).convert("RGB")

        for mode, expected_size in [
            ("baseline", 224),
            ("high_accuracy", 448),
            ("edge_safe", 224),
        ]:
            transform = build_transforms(mode)
            t = transform(img)
            assert t.shape == (
                3,
                expected_size,
                expected_size,
            ), f"Mode '{mode}' produced {t.shape}, expected (3, {expected_size}, {expected_size})"
