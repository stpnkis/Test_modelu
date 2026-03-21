"""
Deterministic split manager — single source of truth for train/val/test splits.

Produces a JSON manifest that all models must consume.  No model is
permitted to use its own local split logic.

Split structure:
    train/ok/      — normal images for training
    val/ok/        — normal images for threshold calibration
    test/ok/       — normal images for evaluation
    test/nok/      — defective images for evaluation

Few-shot invariant:
    For the same ``(dataset_id, seed)``, val and test remain constant
    across all ``n_train`` values.  Only the train/ok subset changes.
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from shared.dataset_schema import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

VAL_OK_FRACTION = 0.15  # 15 % of OK images reserved for validation
TEST_OK_COUNT = 200  # Fixed number of OK images in the test set


def _list_images(directory: Path) -> List[Path]:
    """Return sorted image files from *directory*."""
    return sorted(
        f
        for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


# ── Link helpers ─────────────────────────────────────────────────────────────


def _create_link(src: Path, dst: Path, mode: str) -> None:
    """Create a link from *src* to *dst* using *mode* with auto-fallback."""
    try:
        if mode == "symlink":
            dst.symlink_to(src)
        elif mode == "hardlink":
            os.link(src, dst)
        elif mode == "copy":
            shutil.copy2(src, dst)
        else:
            raise ValueError(f"Unknown link_mode: {mode}")
    except OSError:
        # Fallback chain: symlink → hardlink → copy
        if mode == "symlink":
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        elif mode == "hardlink":
            shutil.copy2(src, dst)


# ── Manifest I/O ─────────────────────────────────────────────────────────────


def save_manifest(manifest: Dict[str, Any], path: Path) -> None:
    """Save split manifest as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Split manifest saved → %s", path)


def load_manifest(path: Path) -> Dict[str, Any]:
    """Load and return a split manifest."""
    with open(path) as f:
        return json.load(f)


# ── Main API ─────────────────────────────────────────────────────────────────


def create_split(
    dataset_dir: str | Path,
    splits_root: str | Path,
    dataset_id: str,
    n_train: int,
    seed: int = 42,
    link_mode: str = "symlink",
) -> Dict[str, Any]:
    """Create a deterministic train/val/test split and write a JSON manifest.

    The few-shot invariant is guaranteed: for a fixed ``(dataset_id, seed)``,
    the val and test pools are determined first and remain constant.
    Only ``train/ok`` varies with ``n_train``.

    Args:
        dataset_dir:  Root of the dataset (contains ``ok/``, ``nok/``).
        splits_root:  Root directory for all splits.
        dataset_id:   Unique name for this dataset.
        n_train:      Number of OK images for the training set.
        seed:         Random seed for reproducibility.
        link_mode:    ``"symlink"`` | ``"hardlink"`` | ``"copy"`` with
                      automatic fallback.

    Returns:
        The manifest dictionary (also saved to disk).

    Raises:
        ValueError:  If there are not enough OK images.
    """
    dataset_dir = Path(dataset_dir).resolve()
    splits_root = Path(splits_root)

    if n_train < 1:
        raise ValueError(f"n_train must be >= 1, got {n_train}.")

    ok_images = _list_images(dataset_dir / "ok")
    nok_images = _list_images(dataset_dir / "nok")

    logger.info(
        "Creating split: dataset=%s  n_train=%d  seed=%d  " "ok=%d  nok=%d",
        dataset_id,
        n_train,
        seed,
        len(ok_images),
        len(nok_images),
    )

    # ── Deterministic shuffle ────────────────────────────────────────
    rng = random.Random(seed)
    shuffled = ok_images.copy()
    rng.shuffle(shuffled)

    # ── Pool allocation (few-shot invariant) ─────────────────────────
    #    1. test_ok  = first TEST_OK_COUNT images  (or fewer if dataset is small)
    #    2. val_ok   = next VAL_OK_FRACTION of total ok count
    #    3. train_ok = next n_train images from the remainder
    test_ok_count = min(TEST_OK_COUNT, len(shuffled))
    val_ok_count = max(1, int(len(ok_images) * VAL_OK_FRACTION))

    needed = test_ok_count + val_ok_count + n_train
    if needed > len(shuffled):
        raise ValueError(
            f"Not enough OK images: need test={test_ok_count} + "
            f"val={val_ok_count} + train={n_train} = {needed}, "
            f"have {len(shuffled)}."
        )

    test_ok = shuffled[:test_ok_count]
    val_ok = shuffled[test_ok_count : test_ok_count + val_ok_count]
    train_ok = shuffled[
        test_ok_count + val_ok_count : test_ok_count + val_ok_count + n_train
    ]

    # ── Build directory tree ─────────────────────────────────────────
    split_dir = splits_root / dataset_id / f"seed{seed}" / f"n{n_train}"
    if split_dir.exists():
        shutil.rmtree(split_dir)

    dirs = {
        "train_ok": split_dir / "train" / "ok",
        "val_ok": split_dir / "val" / "ok",
        "test_ok": split_dir / "test" / "ok",
        "test_nok": split_dir / "test" / "nok",
    }
    for d in dirs.values():
        d.mkdir(parents=True)

    # Masks (optional)
    masks_src = dataset_dir / "masks"
    has_masks = masks_src.is_dir()
    if has_masks:
        test_masks_dir = split_dir / "test" / "masks"
        test_masks_dir.mkdir(parents=True)

    # ── Link files ───────────────────────────────────────────────────
    def _link_files(file_list: List[Path], dest_dir: Path) -> List[str]:
        names = []
        for f in file_list:
            _create_link(f, dest_dir / f.name, link_mode)
            names.append(f.name)
        return names

    train_names = _link_files(train_ok, dirs["train_ok"])
    val_names = _link_files(val_ok, dirs["val_ok"])
    test_ok_names = _link_files(test_ok, dirs["test_ok"])
    test_nok_names = _link_files(nok_images, dirs["test_nok"])

    # Link masks for NOK test images
    if has_masks:
        nok_stems = {p.stem for p in nok_images}
        for mask_path in sorted(masks_src.iterdir()):
            if mask_path.is_file() and mask_path.stem in nok_stems:
                _create_link(mask_path, test_masks_dir / mask_path.name, link_mode)

    # ── Manifest ─────────────────────────────────────────────────────
    manifest = {
        "dataset_id": dataset_id,
        "seed": seed,
        "n_train": n_train,
        "link_mode": link_mode,
        "split_dir": str(split_dir),
        "has_masks": has_masks,
        "counts": {
            "train_ok": len(train_names),
            "val_ok": len(val_names),
            "test_ok": len(test_ok_names),
            "test_nok": len(test_nok_names),
            "test_total": len(test_ok_names) + len(test_nok_names),
        },
        "files": {
            "train_ok": train_names,
            "val_ok": val_names,
            "test_ok": test_ok_names,
            "test_nok": test_nok_names,
        },
    }

    manifest_path = split_dir / "manifest.json"
    save_manifest(manifest, manifest_path)

    logger.info(
        "Split created: train=%d  val=%d  test_ok=%d  test_nok=%d",
        len(train_names),
        len(val_names),
        len(test_ok_names),
        len(test_nok_names),
    )
    return manifest


def get_split_dir(
    splits_root: str | Path,
    dataset_id: str,
    seed: int,
    n_train: int,
) -> Path:
    """Return the expected split directory path."""
    return Path(splits_root) / dataset_id / f"seed{seed}" / f"n{n_train}"


def load_split_manifest(
    splits_root: str | Path,
    dataset_id: str,
    seed: int,
    n_train: int,
) -> Dict[str, Any]:
    """Load a previously created split manifest."""
    split_dir = get_split_dir(splits_root, dataset_id, seed, n_train)
    return load_manifest(split_dir / "manifest.json")


def validate_split_invariance(
    splits_root: str | Path,
    dataset_id: str,
    seed: int,
    n_train_values: List[int],
) -> None:
    """Validate the few-shot invariant: for a fixed (dataset_id, seed),
    val and test must be identical across all n_train values.

    Raises:
        RuntimeError: If val or test files differ across n_train values.
    """
    if len(n_train_values) < 2:
        return

    ref_n = n_train_values[0]
    ref_manifest = load_split_manifest(splits_root, dataset_id, seed, ref_n)
    ref_val = sorted(ref_manifest["files"]["val_ok"])
    ref_test_ok = sorted(ref_manifest["files"]["test_ok"])
    ref_test_nok = sorted(ref_manifest["files"]["test_nok"])

    for n in n_train_values[1:]:
        m = load_split_manifest(splits_root, dataset_id, seed, n)
        val = sorted(m["files"]["val_ok"])
        test_ok = sorted(m["files"]["test_ok"])
        test_nok = sorted(m["files"]["test_nok"])

        if val != ref_val:
            raise RuntimeError(
                f"Split invariance violated: val/ok differs between "
                f"n_train={ref_n} and n_train={n} for "
                f"dataset={dataset_id}, seed={seed}."
            )
        if test_ok != ref_test_ok:
            raise RuntimeError(
                f"Split invariance violated: test/ok differs between "
                f"n_train={ref_n} and n_train={n} for "
                f"dataset={dataset_id}, seed={seed}."
            )
        if test_nok != ref_test_nok:
            raise RuntimeError(
                f"Split invariance violated: test/nok differs between "
                f"n_train={ref_n} and n_train={n} for "
                f"dataset={dataset_id}, seed={seed}."
            )

    logger.info(
        "Split invariance validated: dataset=%s seed=%d n_train_values=%s",
        dataset_id,
        seed,
        n_train_values,
    )
