"""
AnomalyDINO — Dataset Splitter
================================

Creates a deterministic train / test split of the casting dataset for
a given number of normal (OK) reference images.

Split strategy
--------------
- A **fixed pool of 200 OK images** is always reserved for testing.
- ``n_train`` OK images are selected for training (reference).
- **All NOK images** go into the test set.
- Symlinks are used — no data is duplicated.
- The same ``seed`` guarantees the same test set across all runs,
  making experiments with different ``n_train`` directly comparable.

The logic is **identical** to PatchCore's and SimpleNet's
``dataset_splitter.py`` so that all models share the same test images.

Output structure::

    <output_dir>/
        train/
            good/         ← n_train OK images (symlinks)
        test/
            good/         ← 200 OK images (symlinks)
            defective/    ← all NOK images (symlinks)

Usage (standalone)::

    python src/dataset_splitter.py \\
        --dataset-dir dataset --output-dir splits/n50 --n-train 50

Typically called automatically by ``experiment_runner.py``.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict, List, Set

# ── Constants ────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_image_files(directory: Path) -> List[Path]:
    """Return a sorted list of image files in *directory*."""
    return sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


# ── Main splitter ────────────────────────────────────────────────────────────

def create_split(
    dataset_dir: str,
    output_dir: str,
    n_train: int,
    seed: int = 42,
) -> Dict[str, int]:
    """Create a train / test split with *n_train* OK reference images.

    Args:
        dataset_dir: Path to ``dataset/`` containing ``ok/`` and ``nok/``.
        output_dir:  Where the split will be written.
        n_train:     How many OK images to use for training.
        seed:        Random seed for reproducibility.

    Returns:
        Dictionary of split statistics (counts).

    Raises:
        ValueError: If not enough images are available.
    """
    if n_train < 1:
        raise ValueError(f"n_train must be ≥ 1, got {n_train}")

    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir)

    ok_images = get_image_files(dataset_dir / "ok")
    nok_images = get_image_files(dataset_dir / "nok")

    print(f"[split] Found {len(ok_images)} OK images, {len(nok_images)} NOK images")

    # ── Shuffle and split ────────────────────────────────────────────────
    random.seed(seed)
    shuffled = ok_images.copy()
    random.shuffle(shuffled)

    test_size = 200
    if len(shuffled) < test_size + n_train:
        raise ValueError(
            f"Not enough OK images: need {test_size} (test) + {n_train} (train) "
            f"= {test_size + n_train}, have {len(shuffled)}."
        )

    test_ok = shuffled[:test_size]
    train_ok = shuffled[test_size : test_size + n_train]

    # ── Create output directories ────────────────────────────────────────
    if output_dir.exists():
        shutil.rmtree(output_dir)

    train_good_dir = output_dir / "train" / "good"
    test_good_dir = output_dir / "test" / "good"
    test_defective_dir = output_dir / "test" / "defective"

    train_good_dir.mkdir(parents=True)
    test_good_dir.mkdir(parents=True)
    test_defective_dir.mkdir(parents=True)

    # ── Symlink images ───────────────────────────────────────────────────
    for img in train_ok:
        (train_good_dir / img.name).symlink_to(img)
    for img in test_ok:
        (test_good_dir / img.name).symlink_to(img)
    for img in nok_images:
        (test_defective_dir / img.name).symlink_to(img)

    stats: Dict[str, int] = {
        "n_train_ok": len(train_ok),
        "n_test_ok": len(test_ok),
        "n_test_nok": len(nok_images),
        "n_test_total": len(test_ok) + len(nok_images),
    }
    print(f"[split] Created split at {output_dir}: {stats}")
    return stats


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Split dataset for AnomalyDINO experiment.",
    )
    parser.add_argument(
        "--dataset-dir", default="dataset",
        help="Path to dataset/ with ok/ and nok/ subdirs  (default: dataset)",
    )
    parser.add_argument(
        "--output-dir", default="splits/default",
        help="Where to write the split  (default: splits/default)",
    )
    parser.add_argument(
        "--n-train", type=int, required=True,
        help="Number of OK images for training (reference)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed  (default: 42)",
    )
    args = parser.parse_args()

    create_split(args.dataset_dir, args.output_dir, args.n_train, args.seed)
