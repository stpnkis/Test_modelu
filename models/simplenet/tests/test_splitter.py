"""
Unit tests for dataset_splitter.py
====================================

Verifies key invariants of the splitting pipeline:
  - Deterministic output for a fixed seed.
  - Correct number of symlinks in train/test directories.
  - All symlinks resolve to real files.
  - Different seeds produce different train sets.

Uses only the standard library + pathlib (no pytest).
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

# Ensure the src/ directory is importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset_splitter import create_split, get_image_files  # noqa: E402


class _SplitterTestBase(unittest.TestCase):
    """Base class that creates a temporary fake dataset before each test."""

    N_OK = 420        # must be > 200 (test_size) + max(n_train) used in tests
    N_NOK = 50

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.dataset_dir = Path(self.tmp) / "dataset"
        self.ok_dir = self.dataset_dir / "ok"
        self.nok_dir = self.dataset_dir / "nok"
        self.ok_dir.mkdir(parents=True)
        self.nok_dir.mkdir(parents=True)

        # Create small dummy JPEG files (1-byte content is enough for the splitter)
        for i in range(self.N_OK):
            (self.ok_dir / f"ok_{i:04d}.jpg").write_bytes(b"\xff")
        for i in range(self.N_NOK):
            (self.nok_dir / f"nok_{i:04d}.jpg").write_bytes(b"\xff")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestCreateSplit(_SplitterTestBase):
    """Tests for create_split()."""

    def test_correct_counts(self) -> None:
        """Train/test directories contain the expected number of symlinks."""
        n_train = 50
        out = Path(self.tmp) / "split"
        stats = create_split(str(self.dataset_dir), str(out), n_train=n_train, seed=42)

        self.assertEqual(stats["n_train_ok"], n_train)
        self.assertEqual(stats["n_test_ok"], 200)
        self.assertEqual(stats["n_test_nok"], self.N_NOK)
        self.assertEqual(stats["n_test_total"], 200 + self.N_NOK)

        # Verify actual file counts match stats
        train_files = list((out / "train" / "good").iterdir())
        test_good = list((out / "test" / "good").iterdir())
        test_def = list((out / "test" / "defective").iterdir())

        self.assertEqual(len(train_files), n_train)
        self.assertEqual(len(test_good), 200)
        self.assertEqual(len(test_def), self.N_NOK)

    def test_symlinks_resolve(self) -> None:
        """Every symlink in the split directory points to an existing file."""
        out = Path(self.tmp) / "split"
        create_split(str(self.dataset_dir), str(out), n_train=50, seed=42)

        for symlink in out.rglob("*"):
            if symlink.is_symlink():
                target = symlink.resolve()
                self.assertTrue(target.exists(), f"Broken symlink: {symlink} -> {target}")

    def test_deterministic_seed(self) -> None:
        """Same seed produces byte-identical splits (same filenames)."""
        out_a = Path(self.tmp) / "split_a"
        out_b = Path(self.tmp) / "split_b"

        create_split(str(self.dataset_dir), str(out_a), n_train=50, seed=42)
        create_split(str(self.dataset_dir), str(out_b), n_train=50, seed=42)

        for subdir in ["train/good", "test/good", "test/defective"]:
            names_a = sorted(p.name for p in (out_a / subdir).iterdir())
            names_b = sorted(p.name for p in (out_b / subdir).iterdir())
            self.assertEqual(names_a, names_b, f"Mismatch in {subdir}")

    def test_different_seed_different_train(self) -> None:
        """Different seeds pick different training images."""
        out_a = Path(self.tmp) / "split_a"
        out_b = Path(self.tmp) / "split_b"

        create_split(str(self.dataset_dir), str(out_a), n_train=50, seed=42)
        create_split(str(self.dataset_dir), str(out_b), n_train=50, seed=123)

        names_a = set(p.name for p in (out_a / "train" / "good").iterdir())
        names_b = set(p.name for p in (out_b / "train" / "good").iterdir())
        # Extremely unlikely (but not impossible) that two seeds give the
        # exact same 50-element subset of 220 images.
        self.assertNotEqual(names_a, names_b)

    def test_no_overlap_train_test(self) -> None:
        """Training and test-good sets must be disjoint."""
        out = Path(self.tmp) / "split"
        create_split(str(self.dataset_dir), str(out), n_train=50, seed=42)

        train_names = {p.name for p in (out / "train" / "good").iterdir()}
        test_names = {p.name for p in (out / "test" / "good").iterdir()}
        self.assertEqual(len(train_names & test_names), 0)

    def test_all_nok_in_test(self) -> None:
        """Every NOK image must appear in test/defective/."""
        out = Path(self.tmp) / "split"
        create_split(str(self.dataset_dir), str(out), n_train=50, seed=42)

        test_def_names = {p.name for p in (out / "test" / "defective").iterdir()}
        source_nok_names = {p.name for p in self.nok_dir.iterdir()}
        self.assertEqual(test_def_names, source_nok_names)

    def test_value_error_too_many_train(self) -> None:
        """Requesting more training images than available raises ValueError."""
        out = Path(self.tmp) / "split"
        with self.assertRaises(ValueError):
            create_split(str(self.dataset_dir), str(out), n_train=999, seed=42)

    def test_value_error_too_few_train(self) -> None:
        """Requesting fewer than MIN_TRAIN_IMAGES raises ValueError."""
        out = Path(self.tmp) / "split"
        with self.assertRaises(ValueError):
            create_split(str(self.dataset_dir), str(out), n_train=10, seed=42)
        with self.assertRaises(ValueError):
            create_split(str(self.dataset_dir), str(out), n_train=49, seed=42)
        # Boundary: exactly 50 should NOT raise
        create_split(str(self.dataset_dir), str(out), n_train=50, seed=42)


class TestGetImageFiles(_SplitterTestBase):
    """Tests for the get_image_files helper."""

    def test_only_images(self) -> None:
        """Non-image files are excluded."""
        (self.ok_dir / "readme.txt").write_text("not an image")
        files = get_image_files(self.ok_dir)
        self.assertTrue(all(f.suffix.lower() in {".jpg"} for f in files))


if __name__ == "__main__":
    unittest.main()
