"""
Tests for shared.dataset_schema — fail-fast dataset validation.

Covers:
  - Valid dataset passes
  - Missing ok/ directory
  - Missing nok/ directory
  - Empty ok/ directory
  - Non-image files cause error
  - Orphan masks (mask without matching NOK image)
  - Stats returned correctly
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.dataset_schema import DatasetValidationError, validate_dataset


@pytest.fixture
def valid_dataset(tmp_path):
    """Create a minimal valid dataset."""
    ok_dir = tmp_path / "ok"
    nok_dir = tmp_path / "nok"
    ok_dir.mkdir()
    nok_dir.mkdir()

    for i in range(10):
        (ok_dir / f"ok_{i:03d}.png").write_bytes(b"fake")
    for i in range(5):
        (nok_dir / f"nok_{i:03d}.png").write_bytes(b"fake")

    return tmp_path


class TestValidDataset:
    def test_passes_and_returns_stats(self, valid_dataset):
        stats = validate_dataset(valid_dataset)
        assert stats["ok_count"] == 10
        assert stats["nok_count"] == 5
        assert stats["has_masks"] is False
        assert stats["mask_count"] == 0


class TestMissingDirectories:
    def test_missing_ok(self, tmp_path):
        (tmp_path / "nok").mkdir()
        (tmp_path / "nok" / "img.png").write_bytes(b"fake")
        with pytest.raises(DatasetValidationError, match="Missing 'ok/'"):
            validate_dataset(tmp_path)

    def test_missing_nok(self, tmp_path):
        (tmp_path / "ok").mkdir()
        (tmp_path / "ok" / "img.png").write_bytes(b"fake")
        with pytest.raises(DatasetValidationError, match="Missing 'nok/'"):
            validate_dataset(tmp_path)

    def test_nonexistent_root(self, tmp_path):
        with pytest.raises(DatasetValidationError, match="does not exist"):
            validate_dataset(tmp_path / "nonexistent")


class TestEmptyDirectories:
    def test_empty_ok(self, tmp_path):
        (tmp_path / "ok").mkdir()
        (tmp_path / "nok").mkdir()
        (tmp_path / "nok" / "img.png").write_bytes(b"fake")
        with pytest.raises(DatasetValidationError, match="No images found"):
            validate_dataset(tmp_path)


class TestInvalidFiles:
    def test_non_image_files(self, tmp_path):
        ok = tmp_path / "ok"
        nok = tmp_path / "nok"
        ok.mkdir()
        nok.mkdir()
        (ok / "readme.txt").write_text("hello")
        (ok / "img.png").write_bytes(b"fake")
        (nok / "defect.png").write_bytes(b"fake")
        with pytest.raises(DatasetValidationError, match="non-image file"):
            validate_dataset(tmp_path)


class TestMaskCorrespondence:
    def test_valid_masks(self, tmp_path):
        ok = tmp_path / "ok"
        nok = tmp_path / "nok"
        masks = tmp_path / "masks"
        ok.mkdir()
        nok.mkdir()
        masks.mkdir()

        (ok / "ok_001.png").write_bytes(b"fake")
        (nok / "nok_001.png").write_bytes(b"fake")
        (masks / "nok_001.png").write_bytes(b"fake")

        stats = validate_dataset(tmp_path)
        assert stats["has_masks"] is True
        assert stats["mask_count"] == 1

    def test_orphan_masks(self, tmp_path):
        ok = tmp_path / "ok"
        nok = tmp_path / "nok"
        masks = tmp_path / "masks"
        ok.mkdir()
        nok.mkdir()
        masks.mkdir()

        (ok / "ok_001.png").write_bytes(b"fake")
        (nok / "nok_001.png").write_bytes(b"fake")
        (masks / "orphan_mask.png").write_bytes(b"fake")

        with pytest.raises(DatasetValidationError, match="without a matching NOK"):
            validate_dataset(tmp_path)
