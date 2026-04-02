"""
Tests for shared.split_manager — deterministic split generation.

Covers:
  - Determinism (same seed → same split)
  - Train/val/test disjointness
  - Correct counts
  - Few-shot invariant (val+test constant across n_train)
  - Symlink/hardlink/copy modes
  - Manifest I/O
  - Official test preservation (MVTec policy)
  - Custom holdout (Casting policy)
  - Nested few-shot subsets
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from shared.split_manager import create_split, load_split_manifest, get_split_dir
from shared.dataset_registry import DatasetConfig, DATASET_CONFIGS


# ── Fixtures ─────────────────────────────────────────────────────────────────

# Simulates a dataset with official test policy:
#   official_train_ok=50, official_test_ok=10 → total ok=60
#   nok=15, val_ok=10, main_train_ok=40
MOCK_OFFICIAL_CONFIG = DatasetConfig(
    dataset_id="test_official",
    split_policy="official_test",
    official_train_ok=50,
    official_test_ok=10,
    total_nok=15,
    val_ok_count=10,
    main_train_ok=40,
    few_shot_levels=[5, 10, 25, 40],
    source="test",
    has_masks=False,
)

# Simulates a dataset with custom holdout policy:
#   total ok=100, nok=20
#   custom_test_ok=15, val_ok=10, main_train_ok=75
MOCK_CUSTOM_CONFIG = DatasetConfig(
    dataset_id="test_custom",
    split_policy="custom_holdout",
    official_train_ok=0,
    official_test_ok=0,
    total_nok=20,
    val_ok_count=10,
    main_train_ok=75,
    custom_test_ok_count=15,
    few_shot_levels=[5, 10, 25, 50, 75],
    source="test",
    has_masks=False,
)


@pytest.fixture
def official_dataset(tmp_path):
    """Create a fake dataset matching MOCK_OFFICIAL_CONFIG (60 OK, 15 NOK)."""
    ds = tmp_path / "datasets" / "test_official"
    ok_dir = ds / "ok"
    nok_dir = ds / "nok"
    ok_dir.mkdir(parents=True)
    nok_dir.mkdir(parents=True)
    for i in range(60):
        (ok_dir / f"ok_{i:04d}.png").write_bytes(b"fake_image_data")
    for i in range(15):
        (nok_dir / f"nok_{i:04d}.png").write_bytes(b"fake_image_data")
    return ds, tmp_path / "splits"


@pytest.fixture
def custom_dataset(tmp_path):
    """Create a fake dataset matching MOCK_CUSTOM_CONFIG (100 OK, 20 NOK)."""
    ds = tmp_path / "datasets" / "test_custom"
    ok_dir = ds / "ok"
    nok_dir = ds / "nok"
    ok_dir.mkdir(parents=True)
    nok_dir.mkdir(parents=True)
    for i in range(100):
        (ok_dir / f"ok_{i:04d}.png").write_bytes(b"fake_image_data")
    for i in range(20):
        (nok_dir / f"nok_{i:04d}.png").write_bytes(b"fake_image_data")
    return ds, tmp_path / "splits"


def _patch_registry(config: DatasetConfig):
    """Context manager to inject a test config into the dataset registry."""
    patched = {**DATASET_CONFIGS, config.dataset_id: config}
    return patch("shared.split_manager.get_dataset_config", side_effect=lambda did: patched[did])


# ── Official Test Policy Tests ───────────────────────────────────────────────


class TestOfficialTestPolicy:
    """Tests for datasets with official test sets (MVTec-like)."""

    def test_official_test_preserved(self, official_dataset):
        """Official test/ok must be the last N images from sorted ok list."""
        ds, splits_root = official_dataset
        with _patch_registry(MOCK_OFFICIAL_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_official",
                n_train=20, seed=42, link_mode="copy",
            )
        # Official test should be exactly 10 images
        assert m["counts"]["test_ok"] == 10
        # All NOK in test
        assert m["counts"]["test_nok"] == 15

    def test_official_test_independent_of_seed(self, official_dataset):
        """Official test set must be the same across all seeds."""
        ds, splits_root = official_dataset
        manifests = {}
        for seed in [42, 1337, 2026]:
            with _patch_registry(MOCK_OFFICIAL_CONFIG):
                m = create_split(
                    str(ds), str(splits_root), "test_official",
                    n_train=20, seed=seed, link_mode="copy",
                )
            manifests[seed] = m

        ref_test_ok = sorted(manifests[42]["files"]["test_ok"])
        for seed in [1337, 2026]:
            assert sorted(manifests[seed]["files"]["test_ok"]) == ref_test_ok

    def test_val_from_train_pool_only(self, official_dataset):
        """val/ok must come from the train pool, not the test pool."""
        ds, splits_root = official_dataset
        with _patch_registry(MOCK_OFFICIAL_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_official",
                n_train=20, seed=42, link_mode="copy",
            )

        val_set = set(m["files"]["val_ok"])
        test_ok_set = set(m["files"]["test_ok"])
        assert val_set.isdisjoint(test_ok_set), "val/ok must not overlap with test/ok"

    def test_main_train_uses_full_pool(self, official_dataset):
        """Main benchmark should use all available train images."""
        ds, splits_root = official_dataset
        with _patch_registry(MOCK_OFFICIAL_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_official",
                n_train=40, seed=42, link_mode="copy",
            )
        assert m["counts"]["train_ok"] == 40
        assert m["counts"]["val_ok"] == 10


class TestCustomHoldoutPolicy:
    """Tests for datasets without official test sets (Casting-like)."""

    def test_custom_holdout_counts(self, custom_dataset):
        """Custom holdout must create correct test/val/train sizes."""
        ds, splits_root = custom_dataset
        with _patch_registry(MOCK_CUSTOM_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_custom",
                n_train=50, seed=42, link_mode="copy",
            )
        assert m["counts"]["test_ok"] == 15
        assert m["counts"]["val_ok"] == 10
        assert m["counts"]["train_ok"] == 50
        assert m["counts"]["test_nok"] == 20

    def test_custom_holdout_stable_across_n_train(self, custom_dataset):
        """Test/ok and val/ok must be the same across different n_train for same seed."""
        ds, splits_root = custom_dataset
        manifests = {}
        for n in [10, 25, 50]:
            with _patch_registry(MOCK_CUSTOM_CONFIG):
                m = create_split(
                    str(ds), str(splits_root), "test_custom",
                    n_train=n, seed=42, link_mode="copy",
                )
            manifests[n] = m

        ref_test_ok = sorted(manifests[10]["files"]["test_ok"])
        ref_val_ok = sorted(manifests[10]["files"]["val_ok"])
        for n in [25, 50]:
            assert sorted(manifests[n]["files"]["test_ok"]) == ref_test_ok
            assert sorted(manifests[n]["files"]["val_ok"]) == ref_val_ok


class TestSplitDeterminism:
    """Same seed should produce the exact same split."""

    def test_same_seed_same_split(self, custom_dataset):
        ds, splits_root = custom_dataset
        with _patch_registry(MOCK_CUSTOM_CONFIG):
            m1 = create_split(
                str(ds), str(splits_root), "test_custom",
                n_train=5, seed=42, link_mode="copy",
            )
            m2 = create_split(
                str(ds), str(splits_root), "test_custom",
                n_train=5, seed=42, link_mode="copy",
            )

        assert m1["files"]["train_ok"] == m2["files"]["train_ok"]
        assert m1["files"]["val_ok"] == m2["files"]["val_ok"]
        assert m1["files"]["test_ok"] == m2["files"]["test_ok"]
        assert m1["files"]["test_nok"] == m2["files"]["test_nok"]

    def test_different_seed_different_train(self, custom_dataset):
        ds, splits_root = custom_dataset
        with _patch_registry(MOCK_CUSTOM_CONFIG):
            m1 = create_split(
                str(ds), str(splits_root), "test_custom",
                n_train=5, seed=42, link_mode="copy",
            )
            m2 = create_split(
                str(ds), str(splits_root), "test_custom",
                n_train=5, seed=99, link_mode="copy",
            )

        assert (
            m1["files"]["train_ok"] != m2["files"]["train_ok"]
            or m1["files"]["val_ok"] != m2["files"]["val_ok"]
        )


class TestDisjointness:
    """Train, val, and test must be strictly disjoint."""

    def test_no_overlap_official(self, official_dataset):
        ds, splits_root = official_dataset
        with _patch_registry(MOCK_OFFICIAL_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_official",
                n_train=20, seed=42, link_mode="copy",
            )

        train = set(m["files"]["train_ok"])
        val = set(m["files"]["val_ok"])
        test_ok = set(m["files"]["test_ok"])

        assert train.isdisjoint(val), "Train and val overlap"
        assert train.isdisjoint(test_ok), "Train and test_ok overlap"
        assert val.isdisjoint(test_ok), "Val and test_ok overlap"

    def test_no_overlap_custom(self, custom_dataset):
        ds, splits_root = custom_dataset
        with _patch_registry(MOCK_CUSTOM_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_custom",
                n_train=20, seed=42, link_mode="copy",
            )

        train = set(m["files"]["train_ok"])
        val = set(m["files"]["val_ok"])
        test_ok = set(m["files"]["test_ok"])

        assert train.isdisjoint(val), "Train and val overlap"
        assert train.isdisjoint(test_ok), "Train and test_ok overlap"
        assert val.isdisjoint(test_ok), "Val and test_ok overlap"


class TestFewShotNesting:
    """Few-shot subsets must be nested: k=5 ⊂ k=10 ⊂ k=25."""

    def test_nested_subsets_official(self, official_dataset):
        ds, splits_root = official_dataset
        manifests = {}
        for n in [5, 10, 25, 40]:
            with _patch_registry(MOCK_OFFICIAL_CONFIG):
                m = create_split(
                    str(ds), str(splits_root), "test_official",
                    n_train=n, seed=42, link_mode="copy",
                )
            manifests[n] = m

        sizes = [5, 10, 25, 40]
        for i in range(len(sizes) - 1):
            smaller = set(manifests[sizes[i]]["files"]["train_ok"])
            larger = set(manifests[sizes[i + 1]]["files"]["train_ok"])
            assert smaller.issubset(larger), (
                f"n_train={sizes[i]} not a subset of n_train={sizes[i+1]}"
            )

    def test_nested_subsets_custom(self, custom_dataset):
        ds, splits_root = custom_dataset
        manifests = {}
        for n in [5, 10, 25, 50, 75]:
            with _patch_registry(MOCK_CUSTOM_CONFIG):
                m = create_split(
                    str(ds), str(splits_root), "test_custom",
                    n_train=n, seed=42, link_mode="copy",
                )
            manifests[n] = m

        sizes = [5, 10, 25, 50, 75]
        for i in range(len(sizes) - 1):
            smaller = set(manifests[sizes[i]]["files"]["train_ok"])
            larger = set(manifests[sizes[i + 1]]["files"]["train_ok"])
            assert smaller.issubset(larger), (
                f"n_train={sizes[i]} not a subset of n_train={sizes[i+1]}"
            )


class TestFewShotInvariant:
    """For the same seed, val and test must be constant across all n_train."""

    def test_val_test_constant_across_n_train(self, official_dataset):
        ds, splits_root = official_dataset
        with _patch_registry(MOCK_OFFICIAL_CONFIG):
            m5 = create_split(
                str(ds), str(splits_root), "test_official",
                n_train=5, seed=42, link_mode="copy",
            )
            m25 = create_split(
                str(ds), str(splits_root), "test_official",
                n_train=25, seed=42, link_mode="copy",
            )

        assert m5["files"]["val_ok"] == m25["files"]["val_ok"]
        assert m5["files"]["test_ok"] == m25["files"]["test_ok"]
        assert m5["files"]["test_nok"] == m25["files"]["test_nok"]


class TestCorrectCounts:
    """Split counts must match requested values."""

    def test_train_count(self, official_dataset):
        ds, splits_root = official_dataset
        with _patch_registry(MOCK_OFFICIAL_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_official",
                n_train=20, seed=42, link_mode="copy",
            )
        assert m["counts"]["train_ok"] == 20

    def test_nok_all_in_test(self, official_dataset):
        ds, splits_root = official_dataset
        with _patch_registry(MOCK_OFFICIAL_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_official",
                n_train=20, seed=42, link_mode="copy",
            )
        assert m["counts"]["test_nok"] == 15

    def test_error_too_many_train(self, official_dataset):
        ds, splits_root = official_dataset
        with _patch_registry(MOCK_OFFICIAL_CONFIG):
            with pytest.raises(ValueError, match="Not enough"):
                create_split(
                    str(ds), str(splits_root), "test_official",
                    n_train=5000, seed=42, link_mode="copy",
                )


class TestLinkModes:
    """Test symlink, hardlink, and copy modes."""

    def test_copy_mode(self, custom_dataset):
        ds, splits_root = custom_dataset
        with _patch_registry(MOCK_CUSTOM_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_custom",
                n_train=5, seed=42, link_mode="copy",
            )
        split_dir = Path(m["split_dir"])
        train_files = list((split_dir / "train" / "ok").iterdir())
        assert len(train_files) == 5
        for f in train_files:
            assert f.is_file()

    def test_symlink_mode(self, custom_dataset):
        ds, splits_root = custom_dataset
        with _patch_registry(MOCK_CUSTOM_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_custom",
                n_train=5, seed=42, link_mode="symlink",
            )
        split_dir = Path(m["split_dir"])
        train_files = list((split_dir / "train" / "ok").iterdir())
        assert len(train_files) == 5
        for f in train_files:
            assert f.is_symlink()


class TestManifestIO:
    """Test manifest save/load round-trip."""

    def test_round_trip(self, custom_dataset):
        ds, splits_root = custom_dataset
        with _patch_registry(MOCK_CUSTOM_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_custom",
                n_train=5, seed=42, link_mode="copy",
            )
        loaded = load_split_manifest(str(splits_root), "test_custom", 42, 5)
        assert loaded["dataset_id"] == "test_custom"
        assert loaded["seed"] == 42
        assert loaded["n_train"] == 5
        assert loaded["files"]["train_ok"] == m["files"]["train_ok"]

    def test_manifest_has_split_policy(self, custom_dataset):
        ds, splits_root = custom_dataset
        with _patch_registry(MOCK_CUSTOM_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_custom",
                n_train=5, seed=42, link_mode="copy",
            )
        assert "split_policy" in m
        assert m["split_policy"] == "custom_holdout"


# ── MVTec prefix-detection tests ─────────────────────────────────────────────

# Small MVTec-style config: 10 test_* + 50 train_* = 60 OK, 15 NOK
MOCK_MVTEC_SMALL_CONFIG = DatasetConfig(
    dataset_id="test_mvtec_small",
    split_policy="official_test",
    official_train_ok=50,
    official_test_ok=10,
    total_nok=15,
    val_ok_count=10,
    main_train_ok=40,
    few_shot_levels=[5, 10, 25, 40],
    source="test",
    has_masks=False,
)

# Wood-scale MVTec config: 19 test_* + 247 train_* = 266 OK, 60 NOK
MOCK_WOOD_CONFIG = DatasetConfig(
    dataset_id="test_wood_mvtec",
    split_policy="official_test",
    official_train_ok=247,
    official_test_ok=19,
    total_nok=60,
    val_ok_count=50,
    main_train_ok=197,
    few_shot_levels=[10, 25, 50, 100, 197],
    source="test",
    has_masks=False,
)


@pytest.fixture
def mvtec_dataset(tmp_path):
    """MVTec-style dataset with test_NNN.png / train_NNN.png OK filenames."""
    ds = tmp_path / "datasets" / "test_mvtec_small"
    ok_dir = ds / "ok"
    nok_dir = ds / "nok"
    ok_dir.mkdir(parents=True)
    nok_dir.mkdir(parents=True)
    for i in range(10):
        (ok_dir / f"test_{i:03d}.png").write_bytes(b"fake")
    for i in range(50):
        (ok_dir / f"train_{i:03d}.png").write_bytes(b"fake")
    for i in range(15):
        (nok_dir / f"nok_{i:04d}.png").write_bytes(b"fake")
    return ds, tmp_path / "splits"


@pytest.fixture
def wood_scale_dataset(tmp_path):
    """Wood-scale MVTec dataset: 19 test_* + 247 train_* OK, 60 NOK."""
    ds = tmp_path / "datasets" / "test_wood_mvtec"
    ok_dir = ds / "ok"
    nok_dir = ds / "nok"
    ok_dir.mkdir(parents=True)
    nok_dir.mkdir(parents=True)
    for i in range(19):
        (ok_dir / f"test_{i:03d}.png").write_bytes(b"fake")
    for i in range(247):
        (ok_dir / f"train_{i:03d}.png").write_bytes(b"fake")
    for i in range(60):
        (nok_dir / f"nok_{i:03d}.png").write_bytes(b"fake")
    return ds, tmp_path / "splits"


class TestMVTecPrefixDetection:
    """
    Verify that prefix-based split detection is used for MVTec-style datasets
    and that the official test/train membership is correct.
    """

    def test_test_ok_contains_only_test_prefix_files(self, mvtec_dataset):
        """test_ok must contain only test_*.png files (official MVTec test)."""
        ds, splits_root = mvtec_dataset
        with _patch_registry(MOCK_MVTEC_SMALL_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_mvtec_small",
                n_train=20, seed=42, link_mode="copy",
            )
        for name in m["files"]["test_ok"]:
            assert name.startswith("test_"), (
                f"test_ok contains non-test_ file: {name}"
            )

    def test_test_ok_contains_no_train_prefix_files(self, mvtec_dataset):
        """test_ok must NOT contain any train_*.png files."""
        ds, splits_root = mvtec_dataset
        with _patch_registry(MOCK_MVTEC_SMALL_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_mvtec_small",
                n_train=20, seed=42, link_mode="copy",
            )
        train_prefix_in_test = [
            n for n in m["files"]["test_ok"] if n.startswith("train_")
        ]
        assert train_prefix_in_test == [], (
            f"train_ files leaked into test_ok: {train_prefix_in_test}"
        )

    def test_train_ok_contains_only_train_prefix_files(self, mvtec_dataset):
        """train_ok must contain only train_*.png files."""
        ds, splits_root = mvtec_dataset
        with _patch_registry(MOCK_MVTEC_SMALL_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_mvtec_small",
                n_train=20, seed=42, link_mode="copy",
            )
        for name in m["files"]["train_ok"]:
            assert name.startswith("train_"), (
                f"train_ok contains non-train_ file: {name}"
            )

    def test_val_ok_contains_no_test_prefix_files(self, mvtec_dataset):
        """val_ok must NOT contain any test_*.png files."""
        ds, splits_root = mvtec_dataset
        with _patch_registry(MOCK_MVTEC_SMALL_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_mvtec_small",
                n_train=20, seed=42, link_mode="copy",
            )
        test_prefix_in_val = [
            n for n in m["files"]["val_ok"] if n.startswith("test_")
        ]
        assert test_prefix_in_val == [], (
            f"test_ files leaked into val_ok: {test_prefix_in_val}"
        )

    def test_val_ok_contains_only_train_prefix_files(self, mvtec_dataset):
        """val_ok must contain only train_*.png files (sampled from train pool)."""
        ds, splits_root = mvtec_dataset
        with _patch_registry(MOCK_MVTEC_SMALL_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_mvtec_small",
                n_train=20, seed=42, link_mode="copy",
            )
        for name in m["files"]["val_ok"]:
            assert name.startswith("train_"), (
                f"val_ok contains non-train_ file: {name}"
            )

    def test_test_ok_count_matches_registry(self, mvtec_dataset):
        """test_ok count must equal cfg.official_test_ok."""
        ds, splits_root = mvtec_dataset
        with _patch_registry(MOCK_MVTEC_SMALL_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_mvtec_small",
                n_train=20, seed=42, link_mode="copy",
            )
        assert m["counts"]["test_ok"] == MOCK_MVTEC_SMALL_CONFIG.official_test_ok

    def test_train_ok_count_matches_request(self, mvtec_dataset):
        """train_ok count must equal the requested n_train."""
        ds, splits_root = mvtec_dataset
        with _patch_registry(MOCK_MVTEC_SMALL_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_mvtec_small",
                n_train=30, seed=42, link_mode="copy",
            )
        assert m["counts"]["train_ok"] == 30

    def test_disjoint_all_sets(self, mvtec_dataset):
        """train_ok, val_ok, test_ok must be mutually disjoint."""
        ds, splits_root = mvtec_dataset
        with _patch_registry(MOCK_MVTEC_SMALL_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_mvtec_small",
                n_train=20, seed=42, link_mode="copy",
            )
        train = set(m["files"]["train_ok"])
        val = set(m["files"]["val_ok"])
        test_ok = set(m["files"]["test_ok"])
        assert train.isdisjoint(val), "train_ok and val_ok overlap"
        assert train.isdisjoint(test_ok), "train_ok and test_ok overlap"
        assert val.isdisjoint(test_ok), "val_ok and test_ok overlap"

    def test_train_val_partition_official_train_pool(self, mvtec_dataset):
        """train_ok + val_ok must together be a subset of all train_* files."""
        ds, splits_root = mvtec_dataset
        with _patch_registry(MOCK_MVTEC_SMALL_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_mvtec_small",
                n_train=40, seed=42, link_mode="copy",
            )
        train_names = set(m["files"]["train_ok"])
        val_names = set(m["files"]["val_ok"])
        used = train_names | val_names
        # Every file in train or val must have a train_ prefix
        for name in used:
            assert name.startswith("train_"), (
                f"Non-train_ file {name!r} found in train+val pool"
            )
        # Together they must equal the full train pool (all 50 train_* files)
        assert len(used) == MOCK_MVTEC_SMALL_CONFIG.official_train_ok

    def test_test_ok_is_complete_official_test_set(self, mvtec_dataset):
        """test_ok must contain EVERY official test image (all test_* files)."""
        ds, splits_root = mvtec_dataset
        with _patch_registry(MOCK_MVTEC_SMALL_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_mvtec_small",
                n_train=20, seed=42, link_mode="copy",
            )
        expected = {f"test_{i:03d}.png" for i in range(10)}
        assert set(m["files"]["test_ok"]) == expected

    def test_test_ok_seed_invariant(self, mvtec_dataset):
        """test_ok must be identical across seeds."""
        ds, splits_root = mvtec_dataset
        test_sets = []
        for seed in [42, 1337, 2026, 99999]:
            with _patch_registry(MOCK_MVTEC_SMALL_CONFIG):
                m = create_split(
                    str(ds), str(splits_root), "test_mvtec_small",
                    n_train=20, seed=seed, link_mode="copy",
                )
            test_sets.append(set(m["files"]["test_ok"]))
        assert all(s == test_sets[0] for s in test_sets), (
            "test_ok differs across seeds — official test is not preserved"
        )

    def test_nok_all_in_test_unchanged(self, mvtec_dataset):
        """All NOK images must end up in test_nok, untouched."""
        ds, splits_root = mvtec_dataset
        with _patch_registry(MOCK_MVTEC_SMALL_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_mvtec_small",
                n_train=20, seed=42, link_mode="copy",
            )
        assert m["counts"]["test_nok"] == 15

    # ── Regression: old positional bug ───────────────────────────────

    def test_regression_old_positional_slicing_would_corrupt_split(
        self, mvtec_dataset
    ):
        """
        Regression guard: demonstrate that alphabetical positional slicing
        would have placed test_* images in the train pool.

        With sorted filenames test_000..test_009 sort BEFORE train_000..train_049.
        Old code: official_train_pool = ok_images[:50]  ← contains ALL 10 test_*
        Fixed code: uses prefix detection → train_pool = only train_* files.

        This test verifies the FIXED behaviour: no test_ file in train_ok or val_ok.
        It will FAIL if the old slice-based logic is re-introduced.
        """
        ds, splits_root = mvtec_dataset
        with _patch_registry(MOCK_MVTEC_SMALL_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_mvtec_small",
                n_train=40, seed=42, link_mode="copy",
            )

        # Under old code, ok_images[:50] would include test_000..test_009
        # (because test_ < train_ alphabetically), so train_ok and val_ok
        # would contain test_ files.  Verify this does NOT happen.
        test_in_train = [n for n in m["files"]["train_ok"] if n.startswith("test_")]
        test_in_val = [n for n in m["files"]["val_ok"] if n.startswith("test_")]
        assert test_in_train == [], (
            f"REGRESSION: test_ files in train_ok: {test_in_train}. "
            "Positional slicing bug is still active."
        )
        assert test_in_val == [], (
            f"REGRESSION: test_ files in val_ok: {test_in_val}. "
            "Positional slicing bug is still active."
        )

    def test_regression_old_positional_slicing_would_put_train_files_in_test(
        self, mvtec_dataset
    ):
        """
        Regression guard: old code produced test_ok = train_040..train_049
        (the last 10 train_* files after the positional slice).
        Fixed code: test_ok = test_000..test_009.
        """
        ds, splits_root = mvtec_dataset
        with _patch_registry(MOCK_MVTEC_SMALL_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_mvtec_small",
                n_train=20, seed=42, link_mode="copy",
            )
        train_in_test = [n for n in m["files"]["test_ok"] if n.startswith("train_")]
        assert train_in_test == [], (
            f"REGRESSION: train_ files in test_ok: {train_in_test}. "
            "Official test images are corrupted with training images."
        )


class TestWoodRegression:
    """
    End-to-end regression test using wood-scale (247 train_* + 19 test_*) data.

    Verifies correctness of the official MVTec split for the exact file
    counts that caused the bug (ticket: forensic audit finding A1).
    """

    def test_wood_test_ok_is_19_official_test_images(self, wood_scale_dataset):
        """test_ok must be exactly the 19 test_*.png files."""
        ds, splits_root = wood_scale_dataset
        with _patch_registry(MOCK_WOOD_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_wood_mvtec",
                n_train=197, seed=42, link_mode="copy",
            )
        assert m["counts"]["test_ok"] == 19
        for name in m["files"]["test_ok"]:
            assert name.startswith("test_"), (
                f"Non-test_ file in test_ok: {name}"
            )

    def test_wood_train_ok_is_197_official_train_images(self, wood_scale_dataset):
        """train_ok must be 197 of the 247 train_*.png files."""
        ds, splits_root = wood_scale_dataset
        with _patch_registry(MOCK_WOOD_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_wood_mvtec",
                n_train=197, seed=42, link_mode="copy",
            )
        assert m["counts"]["train_ok"] == 197
        for name in m["files"]["train_ok"]:
            assert name.startswith("train_"), (
                f"Non-train_ file in train_ok: {name}"
            )

    def test_wood_val_ok_is_50_official_train_images(self, wood_scale_dataset):
        """val_ok must be exactly 50 of the 247 train_*.png files."""
        ds, splits_root = wood_scale_dataset
        with _patch_registry(MOCK_WOOD_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_wood_mvtec",
                n_train=197, seed=42, link_mode="copy",
            )
        assert m["counts"]["val_ok"] == 50
        for name in m["files"]["val_ok"]:
            assert name.startswith("train_"), (
                f"test_ file leaked into val_ok: {name}"
            )

    def test_wood_no_test_images_in_train_or_val(self, wood_scale_dataset):
        """No official test image (test_*.png) may appear in train_ok or val_ok."""
        ds, splits_root = wood_scale_dataset
        with _patch_registry(MOCK_WOOD_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_wood_mvtec",
                n_train=197, seed=42, link_mode="copy",
            )
        test_in_train = [n for n in m["files"]["train_ok"] if n.startswith("test_")]
        test_in_val = [n for n in m["files"]["val_ok"] if n.startswith("test_")]
        assert test_in_train == [], f"test_ files in train_ok: {test_in_train}"
        assert test_in_val == [], f"test_ files in val_ok: {test_in_val}"

    def test_wood_train_val_cover_full_train_pool(self, wood_scale_dataset):
        """train_ok + val_ok = all 247 train_*.png files (for n_train=197)."""
        ds, splits_root = wood_scale_dataset
        with _patch_registry(MOCK_WOOD_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_wood_mvtec",
                n_train=197, seed=42, link_mode="copy",
            )
        used = set(m["files"]["train_ok"]) | set(m["files"]["val_ok"])
        expected = {f"train_{i:03d}.png" for i in range(247)}
        assert used == expected, (
            f"train+val does not cover the full official train pool. "
            f"Missing: {expected - used}"
        )

    def test_wood_disjoint(self, wood_scale_dataset):
        ds, splits_root = wood_scale_dataset
        with _patch_registry(MOCK_WOOD_CONFIG):
            m = create_split(
                str(ds), str(splits_root), "test_wood_mvtec",
                n_train=197, seed=42, link_mode="copy",
            )
        train = set(m["files"]["train_ok"])
        val = set(m["files"]["val_ok"])
        test_ok = set(m["files"]["test_ok"])
        assert train.isdisjoint(val)
        assert train.isdisjoint(test_ok)
        assert val.isdisjoint(test_ok)

    def test_wood_test_ok_seed_invariant(self, wood_scale_dataset):
        """Official test set must be the same for seeds 42, 1337, 2026."""
        ds, splits_root = wood_scale_dataset
        test_sets = []
        for seed in [42, 1337, 2026]:
            with _patch_registry(MOCK_WOOD_CONFIG):
                m = create_split(
                    str(ds), str(splits_root), "test_wood_mvtec",
                    n_train=197, seed=seed, link_mode="copy",
                )
            test_sets.append(frozenset(m["files"]["test_ok"]))
        assert len(set(test_sets)) == 1, (
            "test_ok differs across seeds — official test is not preserved"
        )
