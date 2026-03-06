"""
Dataset Splitter for SimpleNet Experiment
==========================================

Splits the casting dataset into train/test sets for a given number
of OK training samples.

Behavior:
  - Randomly selects N images from dataset/ok/ for training.
  - A fixed pool of 200 OK images is reserved for test/good/.
  - ALL NOK images go to test/defective/.

The split uses symlinks to avoid duplicating image data.

**Identical split logic to PatchCore** — deterministic with the same seed
so that both models are evaluated on exactly the same test set.

Guard: n_train must be ≥ MIN_TRAIN_IMAGES (50) for reliable results.

Output directory structure:
    splits/n{N}/
        train/
            good/           <- N randomly selected OK images
        test/
            good/           <- 200 fixed OK images (not used for training)
            defective/      <- all NOK images
"""

import random
import shutil
from pathlib import Path

# Common image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Must match train.MIN_TRAIN_IMAGES.  Duplicated here to avoid
# importing torch (heavy) inside a lightweight splitting utility.
MIN_TRAIN_IMAGES = 50


def get_image_files(directory: Path) -> list[Path]:
    """Return sorted list of image files in a directory."""
    return sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def create_split(
    dataset_dir: str,
    output_dir: str,
    n_train: int,
    seed: int = 42,
) -> dict:
    """
    Create a train/test split with N OK images for training.

    Args:
        dataset_dir: Path to dataset/ folder containing ok/ and nok/ subdirs.
        output_dir:  Path where the split will be created.
        n_train:     Number of OK images to use for training.
        seed:        Random seed for reproducibility.

    Returns:
        Dictionary with split statistics.

    Raises:
        ValueError: If n_train < MIN_TRAIN_IMAGES or not enough images.
    """
    # --- Guard: minimum training size ---
    if n_train < MIN_TRAIN_IMAGES:
        msg = (
            f"\n{'=' * 60}\n"
            f"  SPLIT REJECTED: n_train={n_train} < {MIN_TRAIN_IMAGES}\n"
            f"\n"
            f"  SimpleNet requires at least {MIN_TRAIN_IMAGES} OK images\n"
            f"  for reliable anomaly detection. Fewer images produce\n"
            f"  an unreliable feature bank.\n"
            f"\n"
            f"  Please provide at least {MIN_TRAIN_IMAGES} OK images.\n"
            f"{'=' * 60}"
        )
        raise ValueError(msg)

    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir)

    ok_dir = dataset_dir / "ok"
    nok_dir = dataset_dir / "nok"

    # ---------- Collect image paths ----------
    ok_images = get_image_files(ok_dir)
    nok_images = get_image_files(nok_dir)

    print(f"[split] Found {len(ok_images)} OK images, {len(nok_images)} NOK images")

    if n_train > len(ok_images):
        raise ValueError(
            f"Requested {n_train} training images, but only "
            f"{len(ok_images)} OK images are available."
        )

    # ---------- Shuffle and split ----------
    random.seed(seed)
    shuffled = ok_images.copy()
    random.shuffle(shuffled)

    # CRITICAL: Fixed test set of 200 OK images across all experiments.
    # If the test distribution changed per run, Accuracy/Precision would
    # shift randomly, making experiments scientifically incomparable.
    test_size = 200
    if len(shuffled) < test_size + n_train:
        raise ValueError(
            f"Not enough OK images for test={test_size} + train={n_train}. "
            f"Have {len(shuffled)}."
        )

    test_ok = shuffled[:test_size]
    available_train = shuffled[test_size:]

    train_ok = available_train[:n_train]

    # ---------- Create output directories ----------
    # Remove previous split to ensure a clean state
    if output_dir.exists():
        shutil.rmtree(output_dir)

    train_good_dir = output_dir / "train" / "good"
    test_good_dir = output_dir / "test" / "good"
    test_defective_dir = output_dir / "test" / "defective"

    train_good_dir.mkdir(parents=True)
    test_good_dir.mkdir(parents=True)
    test_defective_dir.mkdir(parents=True)

    # ---------- Create symlinks (no data duplication) ----------
    for img in train_ok:
        (train_good_dir / img.name).symlink_to(img)

    for img in test_ok:
        (test_good_dir / img.name).symlink_to(img)

    for img in nok_images:
        (test_defective_dir / img.name).symlink_to(img)

    # ---------- Summary ----------
    stats = {
        "n_train_ok": len(train_ok),
        "n_test_ok": len(test_ok),
        "n_test_nok": len(nok_images),
        "n_test_total": len(test_ok) + len(nok_images),
    }
    print(f"[split] Created split at {output_dir}: {stats}")
    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Split casting dataset for SimpleNet experiment"
    )
    parser.add_argument(
        "--dataset-dir", type=str, default="dataset",
        help="Path to dataset/ folder with ok/ and nok/ subdirs (default: dataset)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="splits/default",
        help="Where to write the split (default: splits/default)",
    )
    parser.add_argument(
        "--n-train", type=int, required=True,
        help="Number of OK images to use for training",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()
    create_split(args.dataset_dir, args.output_dir, args.n_train, args.seed)
