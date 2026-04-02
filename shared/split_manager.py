"""
Deterministic split manager — single source of truth for train/val/test splits.

Produces a JSON manifest that all models must consume.  No model is
permitted to use its own local split logic.

Split structure:
    train/ok/      — normal images for training
    val/ok/        — normal images for threshold calibration
    test/ok/       — normal images for evaluation
    test/nok/      — defective images for evaluation

Supports two split policies (see ``shared/dataset_registry.py``):

  ``official_test`` — MVTec AD, VisA:
      The official test set (test/ok + test/nok) is preserved exactly.
      val/ok is carved from the official train/ok pool only.
      test images are NEVER reshuffled.

  ``custom_holdout`` — Kaggle Casting:
      A fixed hold-out test set is created once per seed, then frozen.
      val/ok and test/ok are drawn from the OK pool before training.

Few-shot invariant:
    For the same ``(dataset_id, seed)``, val and test remain constant
    across all ``n_train`` values.  Only the train/ok subset changes.

Few-shot subset nesting:
    k=10 ⊂ k=25 ⊂ k=50 ⊂ … ⊂ k=full.
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
from shared.dataset_registry import get_dataset_config, DatasetConfig

logger = logging.getLogger(__name__)


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
            # Use relative symlinks so they resolve correctly both on the
            # host and inside Docker containers (where mount points differ).
            rel = os.path.relpath(src, dst.parent)
            dst.symlink_to(rel)
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

    Dispatches to the correct split policy based on the dataset registry.

    The few-shot invariant is guaranteed: for a fixed ``(dataset_id, seed)``,
    the val and test pools are determined first and remain constant.
    Only ``train/ok`` varies with ``n_train``.

    Few-shot subsets are nested: k=10 ⊂ k=25 ⊂ k=50 ⊂ ... ⊂ k=full.

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
        KeyError:    If dataset_id is not in the registry.
    """
    cfg = get_dataset_config(dataset_id)

    if cfg.split_policy == "official_test":
        return _create_split_official_test(
            dataset_dir, splits_root, dataset_id, n_train, seed, link_mode, cfg
        )
    elif cfg.split_policy == "custom_holdout":
        return _create_split_custom_holdout(
            dataset_dir, splits_root, dataset_id, n_train, seed, link_mode, cfg
        )
    else:
        raise ValueError(f"Unknown split policy: {cfg.split_policy}")


def _create_split_official_test(
    dataset_dir: str | Path,
    splits_root: str | Path,
    dataset_id: str,
    n_train: int,
    seed: int,
    link_mode: str,
    cfg: DatasetConfig,
) -> Dict[str, Any]:
    """Split for datasets with official test sets (MVTec AD, VisA).

    The official test/ok and test/nok are preserved exactly.
    val/ok and train/ok are carved from the official train/ok pool only.
    The seed controls which train/ok images become val/ok, and which
    subset of the remaining pool is used for training at each n_train.

    Pool allocation:
      1. Identify official train_ok and test_ok by filename prefix (MVTec)
         or by positional convention (fallback for non-prefixed datasets)
      2. Shuffle only the train_ok pool with the seed
      3. val_ok = first cfg.val_ok_count images from shuffled train pool
      4. train_ok = next n_train images (nested subsets guaranteed)
      5. test_ok = official test OK (untouched)
      6. test_nok = all NOK images (untouched)
    """
    dataset_dir = Path(dataset_dir).resolve()
    splits_root = Path(splits_root)

    if n_train < 1:
        raise ValueError(f"n_train must be >= 1, got {n_train}.")

    ok_images = _list_images(dataset_dir / "ok")
    nok_images = _list_images(dataset_dir / "nok")

    # ── Verify total OK count against registry ───────────────────────
    total_ok = len(ok_images)
    expected_ok = cfg.official_train_ok + cfg.official_test_ok
    if total_ok != expected_ok:
        raise ValueError(
            f"Dataset '{dataset_id}': expected {expected_ok} OK images "
            f"({cfg.official_train_ok} train + {cfg.official_test_ok} test), "
            f"found {total_ok}. Check dataset or update registry."
        )

    # ── Identify official train/test pools ──────────────────────────
    # Primary strategy: MVTec-style prefix detection.
    # OK images are named test_*.png (official test) and train_*.png
    # (official train).  We detect membership by filename prefix so that
    # alphabetical sort order NEVER corrupts the split.  (test_* sorts
    # before train_* alphabetically, so any positional slice would be wrong.)
    #
    # Fallback: positional convention — last cfg.official_test_ok images in
    # sorted order.  Only used when the ok/ directory does NOT contain files
    # with both test_ and train_ prefixes (e.g. a different naming scheme).
    test_prefix_ok = [f for f in ok_images if f.name.startswith("test_")]
    train_prefix_ok = [f for f in ok_images if f.name.startswith("train_")]
    all_covered_by_prefix = (
        len(test_prefix_ok) + len(train_prefix_ok) == total_ok
    )

    if all_covered_by_prefix and test_prefix_ok and train_prefix_ok:
        # Explicit membership: filenames carry their own provenance label.
        official_test_ok = test_prefix_ok
        official_train_pool = train_prefix_ok
        logger.info(
            "Official split [%s]: prefix detection — train_*=%d  test_*=%d",
            dataset_id,
            len(official_train_pool),
            len(official_test_ok),
        )
    else:
        # Fallback: positional convention.
        # Only reached for non-MVTec datasets without prefix filenames.
        official_train_pool = ok_images[: cfg.official_train_ok]
        official_test_ok = ok_images[cfg.official_train_ok :]
        logger.warning(
            "Official split [%s]: no test_/train_ prefix found — "
            "falling back to positional convention "
            "(train=%d, test=%d). Verify dataset filenames.",
            dataset_id,
            len(official_train_pool),
            len(official_test_ok),
        )

    # ── Validate pool sizes against registry ─────────────────────────
    if len(official_test_ok) != cfg.official_test_ok:
        raise ValueError(
            f"Dataset '{dataset_id}': official test_ok size mismatch — "
            f"detected {len(official_test_ok)} files, registry expects "
            f"{cfg.official_test_ok}. Check dataset filenames or registry."
        )
    if len(official_train_pool) != cfg.official_train_ok:
        raise ValueError(
            f"Dataset '{dataset_id}': official train_ok pool size mismatch — "
            f"detected {len(official_train_pool)} files, registry expects "
            f"{cfg.official_train_ok}. Check dataset filenames or registry."
        )

    logger.info(
        "Official test split: %d OK + %d NOK (preserved, not reshuffled)",
        len(official_test_ok),
        len(nok_images),
    )

    # ── Deterministic shuffle of train pool only ─────────────────────
    rng = random.Random(seed)
    shuffled_train = official_train_pool.copy()
    rng.shuffle(shuffled_train)

    # ── Pool allocation (few-shot invariant) ─────────────────────────
    val_ok_count = cfg.val_ok_count
    needed = val_ok_count + n_train
    if needed > len(shuffled_train):
        raise ValueError(
            f"Not enough train/ok images: need val={val_ok_count} + "
            f"train={n_train} = {needed}, have {len(shuffled_train)} "
            f"in the official train pool for '{dataset_id}'."
        )

    val_ok = shuffled_train[:val_ok_count]
    train_ok = shuffled_train[val_ok_count : val_ok_count + n_train]

    # ── Safety guards: strict provenance checks ───────────────────────
    _validate_official_test_provenance(
        dataset_id=dataset_id,
        train_ok=train_ok,
        val_ok=val_ok,
        test_ok=official_test_ok,
    )

    return _write_split(
        dataset_dir=dataset_dir,
        splits_root=splits_root,
        dataset_id=dataset_id,
        n_train=n_train,
        seed=seed,
        link_mode=link_mode,
        train_ok=train_ok,
        val_ok=val_ok,
        test_ok=official_test_ok,
        nok_images=nok_images,
        split_policy="official_test",
    )


def _validate_official_test_provenance(
    *,
    dataset_id: str,
    train_ok: List[Path],
    val_ok: List[Path],
    test_ok: List[Path],
) -> None:
    """Raise ValueError if any official-test provenance rule is violated.

    Checks (only those applicable given the detected filename style):
      1. test_ok contains no train_*.png files
      2. train_ok contains no test_*.png files
      3. val_ok contains no test_*.png files
      4. train_ok, val_ok, test_ok are mutually disjoint
    """
    test_names = {p.name for p in test_ok}
    train_names = {p.name for p in train_ok}
    val_names = {p.name for p in val_ok}

    # Prefix leakage checks (only meaningful when prefix naming is used)
    bad = [p.name for p in test_ok if p.name.startswith("train_")]
    if bad:
        raise ValueError(
            f"Dataset '{dataset_id}': {len(bad)} train_* file(s) found in "
            f"test_ok — official test pool is contaminated: {bad[:5]}"
        )
    bad = [p.name for p in train_ok if p.name.startswith("test_")]
    if bad:
        raise ValueError(
            f"Dataset '{dataset_id}': {len(bad)} test_* file(s) found in "
            f"train_ok — official test images leaked into training: {bad[:5]}"
        )
    bad = [p.name for p in val_ok if p.name.startswith("test_")]
    if bad:
        raise ValueError(
            f"Dataset '{dataset_id}': {len(bad)} test_* file(s) found in "
            f"val_ok — official test images leaked into validation: {bad[:5]}"
        )

    # Disjointness
    tv = train_names & val_names
    if tv:
        raise ValueError(
            f"Dataset '{dataset_id}': train_ok and val_ok overlap "
            f"({len(tv)} file(s)): {sorted(tv)[:5]}"
        )
    tt = train_names & test_names
    if tt:
        raise ValueError(
            f"Dataset '{dataset_id}': train_ok and test_ok overlap "
            f"({len(tt)} file(s)): {sorted(tt)[:5]}"
        )
    vt = val_names & test_names
    if vt:
        raise ValueError(
            f"Dataset '{dataset_id}': val_ok and test_ok overlap "
            f"({len(vt)} file(s)): {sorted(vt)[:5]}"
        )


def _create_split_custom_holdout(
    dataset_dir: str | Path,
    splits_root: str | Path,
    dataset_id: str,
    n_train: int,
    seed: int,
    link_mode: str,
    cfg: DatasetConfig,
) -> Dict[str, Any]:
    """Split for datasets without official test sets (Kaggle Casting).

    Creates one fixed hold-out test from the OK pool, then carves val/ok
    and train/ok from the remainder. The hold-out is seed-dependent but
    stable across n_train values (few-shot invariant).

    Pool allocation:
      1. Shuffle all OK with seed
      2. test_ok = first cfg.custom_test_ok_count
      3. val_ok = next cfg.val_ok_count
      4. train_ok = next n_train (nested subsets)
      5. test_nok = all NOK images
    """
    dataset_dir = Path(dataset_dir).resolve()
    splits_root = Path(splits_root)

    if n_train < 1:
        raise ValueError(f"n_train must be >= 1, got {n_train}.")

    ok_images = _list_images(dataset_dir / "ok")
    nok_images = _list_images(dataset_dir / "nok")

    # ── Deterministic shuffle ────────────────────────────────────────
    rng = random.Random(seed)
    shuffled = ok_images.copy()
    rng.shuffle(shuffled)

    # ── Pool allocation (few-shot invariant) ─────────────────────────
    test_ok_count = cfg.custom_test_ok_count
    val_ok_count = cfg.val_ok_count
    needed = test_ok_count + val_ok_count + n_train
    if needed > len(shuffled):
        raise ValueError(
            f"Not enough OK images for '{dataset_id}': need "
            f"test={test_ok_count} + val={val_ok_count} + train={n_train} "
            f"= {needed}, have {len(shuffled)}."
        )

    test_ok = shuffled[:test_ok_count]
    val_ok = shuffled[test_ok_count : test_ok_count + val_ok_count]
    train_ok = shuffled[
        test_ok_count + val_ok_count : test_ok_count + val_ok_count + n_train
    ]

    return _write_split(
        dataset_dir=dataset_dir,
        splits_root=splits_root,
        dataset_id=dataset_id,
        n_train=n_train,
        seed=seed,
        link_mode=link_mode,
        train_ok=train_ok,
        val_ok=val_ok,
        test_ok=test_ok,
        nok_images=nok_images,
        split_policy="custom_holdout",
    )


def _write_split(
    *,
    dataset_dir: Path,
    splits_root: Path,
    dataset_id: str,
    n_train: int,
    seed: int,
    link_mode: str,
    train_ok: List[Path],
    val_ok: List[Path],
    test_ok: List[Path],
    nok_images: List[Path],
    split_policy: str,
) -> Dict[str, Any]:
    """Build the split directory tree, link files, and write the manifest."""
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
    has_masks = masks_src.is_dir() and any(masks_src.iterdir())
    if has_masks:
        test_masks_dir = split_dir / "test" / "masks"
        test_masks_dir.mkdir(parents=True)

    # ── Link files ───────────────────────────────────────────────────
    def _link_files(file_list: List[Path], dest_dir: Path) -> List[str]:
        names = []
        for f in file_list:
            # Normalize extension to lowercase so anomalib recognises files
            # whose originals use uppercase extensions (e.g. .JPG → .jpg).
            norm_name = f.stem + f.suffix.lower()
            _create_link(f, dest_dir / norm_name, link_mode)
            names.append(norm_name)
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
                norm_mask = mask_path.stem + mask_path.suffix.lower()
                _create_link(mask_path, test_masks_dir / norm_mask, link_mode)

    # ── Manifest ─────────────────────────────────────────────────────
    manifest = {
        "dataset_id": dataset_id,
        "seed": seed,
        "n_train": n_train,
        "link_mode": link_mode,
        "split_policy": split_policy,
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
        "Split created [%s]: train=%d  val=%d  test_ok=%d  test_nok=%d",
        split_policy,
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
