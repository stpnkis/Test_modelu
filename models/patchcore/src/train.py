"""
PatchCore Training Script
==========================

Trains a PatchCore anomaly detection model on a dataset split.

How PatchCore works (simplified):
  1. A pretrained CNN backbone (e.g. WideResNet-50) extracts patch-level
     features from every training image.
  2. All patch features are collected into a "memory bank".
  3. Coreset subsampling reduces the memory bank to a manageable size
     while preserving coverage of the feature space.
  4. At inference, test-image patches are compared to the nearest
     neighbours in the memory bank.  High distance = anomaly.

Because PatchCore does NOT update network weights, training is a single
forward pass through the data (1 epoch).  It is very fast.

This script uses the anomalib library which wraps PatchCore in a
PyTorch Lightning module and provides:
  - Folder datamodule  (loads images from custom directory layout)
  - Engine             (handles training loop, checkpointing, logging)
"""

import logging
from pathlib import Path

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

logger = logging.getLogger(__name__)


def train_patchcore(
    split_dir: str,
    output_dir: str,
    image_size: int = 256,
    backbone: str = "wide_resnet50_2",
    batch_size: int = 32,
) -> str:
    """
    Train PatchCore on a prepared dataset split.

    Args:
        split_dir:   Path to split directory (must contain train/ and test/).
        output_dir:  Where to save the model checkpoint and training logs.
        image_size:  Resize all images to this size (square).
        backbone:    Torchvision model used as feature extractor.
        batch_size:  Batch size for feature extraction.

    Returns:
        Path to the saved model checkpoint (.ckpt file).
    """
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)

    logger.info(f"Training PatchCore  |  split={split_dir}  backbone={backbone}")

    # ------------------------------------------------------------------
    # Step 1: Data module
    # ------------------------------------------------------------------
    # The Folder datamodule loads images from a custom directory layout.
    #   normal_dir      -> training images  (only OK / "good")
    #   abnormal_dir    -> test defective   (NOK)
    #   normal_test_dir -> test OK          (good images held out from training)
    #
    # Note: our images are grayscale but the backbone expects 3-channel RGB.
    # anomalib's data pipeline automatically converts grayscale -> RGB.
    datamodule = Folder(
        name="casting",
        root=str(split_dir),
        normal_dir="train/good",
        abnormal_dir="test/defective",
        normal_test_dir="test/good",
        task="classification",
        image_size=(image_size, image_size) if isinstance(image_size, int) else image_size,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
    )

    # ------------------------------------------------------------------
    # Step 2: Model
    # ------------------------------------------------------------------
    # layers_to_extract: which backbone layers to collect features from.
    #   - "layer2" gives mid-level features  (good spatial resolution)
    #   - "layer3" gives higher-level features (more semantic)
    # coreset_sampling_ratio: fraction of the full memory bank to keep
    #   after greedy coreset selection.  0.1 = keep ~10 %.
    # num_neighbors: K in the K-NN anomaly scoring.
    model = Patchcore(
        backbone=backbone,
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )

    # ------------------------------------------------------------------
    # Step 3: Training via anomalib Engine
    # ------------------------------------------------------------------
    # PatchCore does not use gradient-based training, so max_epochs=1
    # is correct — a single pass collects all features into the memory
    # bank.  The Engine also runs validation and saves checkpoints.
    engine = Engine(
        task="classification",
        default_root_dir=str(output_dir),
        max_epochs=1,
        devices=1,
        accelerator="auto",   # GPU if available, else CPU
    )

    engine.fit(model=model, datamodule=datamodule)

    # ------------------------------------------------------------------
    # Step 4: Locate saved checkpoint
    # ------------------------------------------------------------------
    ckpt_path = None

    # Primary: use Lightning's checkpoint callback
    cb = engine.trainer.checkpoint_callback
    if cb is not None and cb.best_model_path:
        ckpt_path = cb.best_model_path

    # Fallback: search the output directory for any .ckpt file
    if not ckpt_path:
        candidates = sorted(output_dir.rglob("*.ckpt"))
        if candidates:
            ckpt_path = str(candidates[-1])

    if not ckpt_path:
        raise FileNotFoundError(
            f"No checkpoint found in {output_dir}. Training may have failed."
        )

    logger.info(f"Checkpoint saved: {ckpt_path}")
    return ckpt_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train PatchCore model")
    parser.add_argument("--split-dir", type=str, required=True,
                        help="Path to dataset split directory")
    parser.add_argument("--output-dir", type=str, default="experiments/train_output",
                        help="Path to save model checkpoint and logs")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2")
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    ckpt = train_patchcore(
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        backbone=args.backbone,
        batch_size=args.batch_size,
    )
    print(f"\nModel checkpoint: {ckpt}")
