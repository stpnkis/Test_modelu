"""
Dataset schema validation — fail-fast checks before any experiment runs.

Validates:
- existence of ok/ and nok/ directories
- file counts (non-empty)
- supported image extensions
- naming consistency
- mask-to-NOK correspondence (if masks/ present)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class DatasetValidationError(Exception):
    """Raised when the dataset fails validation."""


def _list_images(directory: Path) -> List[Path]:
    """Return sorted image files in *directory*."""
    return sorted(
        f
        for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_dataset(dataset_dir: str | Path) -> dict:
    """Validate a dataset directory and return summary stats.

    Args:
        dataset_dir: Path to ``datasets/<dataset_id>/`` containing
                     ``ok/``, ``nok/``, and optionally ``masks/``.

    Returns:
        Dictionary with keys ``ok_count``, ``nok_count``, ``mask_count``,
        ``has_masks``.

    Raises:
        DatasetValidationError: with a clear, actionable message.
    """
    root = Path(dataset_dir).resolve()
    errors: List[str] = []

    # ── Existence checks ─────────────────────────────────────────────
    ok_dir = root / "ok"
    nok_dir = root / "nok"
    masks_dir = root / "masks"

    if not root.is_dir():
        raise DatasetValidationError(
            f"Dataset directory does not exist: {root}\n"
            f"  → Create it and place images in ok/ and nok/ subdirectories."
        )

    if not ok_dir.is_dir():
        errors.append(
            f"Missing 'ok/' directory in {root}.\n"
            f"  → Create {ok_dir} and place normal (defect-free) images inside."
        )
    if not nok_dir.is_dir():
        errors.append(
            f"Missing 'nok/' directory in {root}.\n"
            f"  → Create {nok_dir} and place defective images inside."
        )

    if errors:
        raise DatasetValidationError("\n".join(errors))

    # ── File counts ──────────────────────────────────────────────────
    ok_images = _list_images(ok_dir)
    nok_images = _list_images(nok_dir)

    if len(ok_images) == 0:
        errors.append(
            f"No images found in {ok_dir}.\n"
            f"  → Place at least one image with extension {IMAGE_EXTENSIONS}."
        )
    if len(nok_images) == 0:
        errors.append(
            f"No images found in {nok_dir}.\n"
            f"  → Place at least one defective image with extension {IMAGE_EXTENSIONS}."
        )

    # ── Extension consistency ────────────────────────────────────────
    all_images = ok_images + nok_images
    non_image_files = [
        f
        for f in list(ok_dir.iterdir()) + list(nok_dir.iterdir())
        if f.is_file() and f.suffix.lower() not in IMAGE_EXTENSIONS
    ]
    if non_image_files:
        names = [f.name for f in non_image_files[:5]]
        errors.append(
            f"Found {len(non_image_files)} non-image file(s) in ok/ or nok/: {names}\n"
            f"  → Remove them or convert to a supported format {IMAGE_EXTENSIONS}."
        )

    # ── Mask correspondence ──────────────────────────────────────────
    has_masks = masks_dir.is_dir()
    mask_count = 0
    if has_masks:
        mask_files = _list_images(masks_dir)
        mask_count = len(mask_files)
        nok_stems = {f.stem for f in nok_images}
        mask_stems = {f.stem for f in mask_files}

        orphan_masks = mask_stems - nok_stems
        if orphan_masks:
            examples = sorted(orphan_masks)[:5]
            errors.append(
                f"Found {len(orphan_masks)} mask(s) without a matching NOK image: {examples}\n"
                f"  → Each mask filename (stem) must match a NOK image filename."
            )

    if errors:
        raise DatasetValidationError("\n".join(errors))

    stats = {
        "ok_count": len(ok_images),
        "nok_count": len(nok_images),
        "mask_count": mask_count,
        "has_masks": has_masks,
    }
    logger.info("Dataset validated: %s — %s", root.name, stats)
    return stats
