"""
Tests for shared.split_manager — deterministic split generation.

Covers:
  - Determinism (same seed → same split)
  - Train/val/test disjointness
  - Correct counts
  - Few-shot invariant (val+test constant across n_train)
  - Symlink/hardlink/copy modes
  - Manifest I/O
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from shared.split_manager import create_split, load_split_manifest, get_split_dir


# Enough OK images: need TEST_OK_COUNT(200) + val(~15%) + train
# 300 ok → test=200, val=45, train pool=55
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


class TestSplitDeterminism:
    """Same seed should produce the exact same split."""

    def test_same_seed_same_split(self, fake_dataset):
        ds, splits_root = fake_dataset
        m1 = create_split(
            str(ds), str(splits_root), "test_ds", n_train=5, seed=42, link_mode="copy"
        )
        # Re-create
        m2 = create_split(
            str(ds), str(splits_root), "test_ds", n_train=5, seed=42, link_mode="copy"
        )

        assert m1["files"]["train_ok"] == m2["files"]["train_ok"]
        assert m1["files"]["val_ok"] == m2["files"]["val_ok"]
        assert m1["files"]["test_ok"] == m2["files"]["test_ok"]
        assert m1["files"]["test_nok"] == m2["files"]["test_nok"]

    def test_different_seed_different_train(self, fake_dataset):
        ds, splits_root = fake_dataset
        m1 = create_split(
            str(ds), str(splits_root), "test_ds", n_train=5, seed=42, link_mode="copy"
        )
        m2 = create_split(
            str(ds), str(splits_root), "test_ds", n_train=5, seed=99, link_mode="copy"
        )

        # At least some files should differ
        assert (
            m1["files"]["train_ok"] != m2["files"]["train_ok"]
            or m1["files"]["val_ok"] != m2["files"]["val_ok"]
        )


class TestDisjointness:
    """Train, val, and test must be strictly disjoint."""

    def test_no_overlap(self, fake_dataset):
        ds, splits_root = fake_dataset
        m = create_split(
            str(ds), str(splits_root), "test_ds", n_train=5, seed=42, link_mode="copy"
        )

        train = set(m["files"]["train_ok"])
        val = set(m["files"]["val_ok"])
        test_ok = set(m["files"]["test_ok"])

        assert train.isdisjoint(val), "Train and val overlap"
        assert train.isdisjoint(test_ok), "Train and test_ok overlap"
        assert val.isdisjoint(test_ok), "Val and test_ok overlap"


class TestCorrectCounts:
    """Split counts must match requested values."""

    def test_train_count(self, fake_dataset):
        ds, splits_root = fake_dataset
        m = create_split(
            str(ds), str(splits_root), "test_ds", n_train=5, seed=42, link_mode="copy"
        )
        assert m["counts"]["train_ok"] == 5

    def test_nok_all_in_test(self, fake_dataset):
        ds, splits_root = fake_dataset
        m = create_split(
            str(ds), str(splits_root), "test_ds", n_train=5, seed=42, link_mode="copy"
        )
        assert m["counts"]["test_nok"] == NUM_NOK  # All NOK images

    def test_val_count_nonzero(self, fake_dataset):
        ds, splits_root = fake_dataset
        m = create_split(
            str(ds), str(splits_root), "test_ds", n_train=5, seed=42, link_mode="copy"
        )
        assert m["counts"]["val_ok"] > 0

    def test_error_too_many_train(self, fake_dataset):
        ds, splits_root = fake_dataset
        with pytest.raises(ValueError, match="Not enough OK"):
            create_split(
                str(ds),
                str(splits_root),
                "test_ds",
                n_train=5000,
                seed=42,
                link_mode="copy",
            )


class TestFewShotInvariant:
    """For the same seed, val and test must be constant across all n_train."""

    def test_val_test_constant_across_n_train(self, fake_dataset):
        ds, splits_root = fake_dataset
        m5 = create_split(
            str(ds), str(splits_root), "test_ds", n_train=5, seed=42, link_mode="copy"
        )
        m10 = create_split(
            str(ds), str(splits_root), "test_ds", n_train=10, seed=42, link_mode="copy"
        )

        assert m5["files"]["val_ok"] == m10["files"]["val_ok"]
        assert m5["files"]["test_ok"] == m10["files"]["test_ok"]
        assert m5["files"]["test_nok"] == m10["files"]["test_nok"]

    def test_train_differs_across_n_train(self, fake_dataset):
        ds, splits_root = fake_dataset
        m5 = create_split(
            str(ds), str(splits_root), "test_ds", n_train=5, seed=42, link_mode="copy"
        )
        m10 = create_split(
            str(ds), str(splits_root), "test_ds", n_train=10, seed=42, link_mode="copy"
        )

        # n_train=10 should be a superset of n_train=5
        assert set(m5["files"]["train_ok"]).issubset(set(m10["files"]["train_ok"]))
        assert len(m10["files"]["train_ok"]) > len(m5["files"]["train_ok"])


class TestLinkModes:
    """Test symlink, hardlink, and copy modes."""

    def test_copy_mode(self, fake_dataset):
        ds, splits_root = fake_dataset
        m = create_split(
            str(ds), str(splits_root), "test_ds", n_train=5, seed=42, link_mode="copy"
        )
        split_dir = Path(m["split_dir"])
        train_files = list((split_dir / "train" / "ok").iterdir())
        assert len(train_files) == 5
        # Should be regular files (not symlinks) in copy mode
        for f in train_files:
            assert f.is_file()

    def test_symlink_mode(self, fake_dataset):
        ds, splits_root = fake_dataset
        m = create_split(
            str(ds),
            str(splits_root),
            "test_ds",
            n_train=5,
            seed=42,
            link_mode="symlink",
        )
        split_dir = Path(m["split_dir"])
        train_files = list((split_dir / "train" / "ok").iterdir())
        assert len(train_files) == 5
        for f in train_files:
            assert f.is_symlink()


class TestManifestIO:
    """Test manifest save/load round-trip."""

    def test_round_trip(self, fake_dataset):
        ds, splits_root = fake_dataset
        m = create_split(
            str(ds), str(splits_root), "test_ds", n_train=5, seed=42, link_mode="copy"
        )
        loaded = load_split_manifest(str(splits_root), "test_ds", 42, 5)
        assert loaded["dataset_id"] == "test_ds"
        assert loaded["seed"] == 42
        assert loaded["n_train"] == 5
        assert loaded["files"]["train_ok"] == m["files"]["train_ok"]
