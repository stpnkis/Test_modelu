"""
Unit tests for ``dataset_splitter.py``
========================================

Verifies key invariants of the splitting pipeline:

- Deterministic output for a fixed seed.
- Correct number of symlinks in train / test directories.
- All symlinks point to existing files.
- Different seeds produce different training sets.
- ``n_train = 1`` (single-shot) is valid.
- No overlap between training and test-good sets.
- All NOK images are included in the test set.

Runs with ``unittest`` (no pytest required), no GPU needed.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
import sys

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset_splitter import create_split, get_image_files  # noqa: E402


class _SplitterTestBase(unittest.TestCase):
    """Base class — creates a temporary fake dataset before each test."""

    N_OK = 420   # must be > 200 (test_size) + max(n_train) used in tests
    N_NOK = 50

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.dataset_dir = Path(self.tmp) / "dataset"
        self.ok_dir = self.dataset_dir / "ok"
        self.nok_dir = self.dataset_dir / "nok"
        self.ok_dir.mkdir(parents=True)
        self.nok_dir.mkdir(parents=True)

        for i in range(self.N_OK):
            (self.ok_dir / f"ok_{i:04d}.jpg").write_bytes(b"\xff")
        for i in range(self.N_NOK):
            (self.nok_dir / f"nok_{i:04d}.jpg").write_bytes(b"\xff")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestCreateSplit(_SplitterTestBase):

    def test_correct_counts(self) -> None:
        out = Path(self.tmp) / "split"
        stats = create_split(str(self.dataset_dir), str(out), n_train=4, seed=42)

        self.assertEqual(stats["n_train_ok"], 4)
        self.assertEqual(stats["n_test_ok"], 200)
        self.assertEqual(stats["n_test_nok"], self.N_NOK)
        self.assertEqual(stats["n_test_total"], 200 + self.N_NOK)

        self.assertEqual(len(list((out / "train" / "good").iterdir())), 4)
        self.assertEqual(len(list((out / "test" / "good").iterdir())), 200)
        self.assertEqual(len(list((out / "test" / "defective").iterdir())), self.N_NOK)

    def test_single_shot(self) -> None:
        out = Path(self.tmp) / "split"
        stats = create_split(str(self.dataset_dir), str(out), n_train=1, seed=42)
        self.assertEqual(stats["n_train_ok"], 1)
        self.assertEqual(len(list((out / "train" / "good").iterdir())), 1)

    def test_symlinks_resolve(self) -> None:
        out = Path(self.tmp) / "split"
        create_split(str(self.dataset_dir), str(out), n_train=4, seed=42)
        for link in out.rglob("*"):
            if link.is_symlink():
                self.assertTrue(link.resolve().exists(), f"Broken: {link}")

    def test_deterministic_seed(self) -> None:
        a = Path(self.tmp) / "a"
        b = Path(self.tmp) / "b"
        create_split(str(self.dataset_dir), str(a), n_train=8, seed=42)
        create_split(str(self.dataset_dir), str(b), n_train=8, seed=42)
        for sub in ("train/good", "test/good", "test/defective"):
            self.assertEqual(
                sorted(p.name for p in (a / sub).iterdir()),
                sorted(p.name for p in (b / sub).iterdir()),
            )

    def test_different_seed_different_train(self) -> None:
        a = Path(self.tmp) / "a"
        b = Path(self.tmp) / "b"
        create_split(str(self.dataset_dir), str(a), n_train=8, seed=42)
        create_split(str(self.dataset_dir), str(b), n_train=8, seed=123)
        na = {p.name for p in (a / "train" / "good").iterdir()}
        nb = {p.name for p in (b / "train" / "good").iterdir()}
        self.assertNotEqual(na, nb)

    def test_no_overlap_train_test(self) -> None:
        out = Path(self.tmp) / "split"
        create_split(str(self.dataset_dir), str(out), n_train=16, seed=42)
        train = {p.name for p in (out / "train" / "good").iterdir()}
        test = {p.name for p in (out / "test" / "good").iterdir()}
        self.assertEqual(len(train & test), 0)

    def test_all_nok_in_test(self) -> None:
        out = Path(self.tmp) / "split"
        create_split(str(self.dataset_dir), str(out), n_train=4, seed=42)
        test_def = {p.name for p in (out / "test" / "defective").iterdir()}
        source_nok = {p.name for p in self.nok_dir.iterdir()}
        self.assertEqual(test_def, source_nok)

    def test_error_too_many_train(self) -> None:
        with self.assertRaises(ValueError):
            create_split(str(self.dataset_dir), str(Path(self.tmp) / "x"),
                         n_train=999, seed=42)

    def test_error_zero_train(self) -> None:
        with self.assertRaises(ValueError):
            create_split(str(self.dataset_dir), str(Path(self.tmp) / "x"),
                         n_train=0, seed=42)

    def test_test_set_stable_across_sizes(self) -> None:
        """Same seed → same 200 test-good images regardless of n_train."""
        a = Path(self.tmp) / "a"
        b = Path(self.tmp) / "b"
        create_split(str(self.dataset_dir), str(a), n_train=4, seed=42)
        create_split(str(self.dataset_dir), str(b), n_train=16, seed=42)
        self.assertEqual(
            sorted(p.name for p in (a / "test" / "good").iterdir()),
            sorted(p.name for p in (b / "test" / "good").iterdir()),
        )


class TestGetImageFiles(_SplitterTestBase):

    def test_only_images(self) -> None:
        (self.ok_dir / "readme.txt").write_text("not an image")
        files = get_image_files(self.ok_dir)
        self.assertTrue(all(f.suffix.lower() == ".jpg" for f in files))


if __name__ == "__main__":
    unittest.main()
