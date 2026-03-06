"""
Experiment Runner
==================

Automates the full experimental pipeline for the research question:

    "What is the minimum number of OK training samples for
     SimpleNet to achieve reliable AUROC > 0.95?"

For each training-set size N in TRAIN_SIZES:
  1.  Split the dataset  (N OK images for training, rest for test).
  2.  Train SimpleNet    (adaptor + discriminator on feature bank).
  3.  Evaluate on test   (computes AUROC, Precision, Recall, F1).
  4.  Append results to  experiments/results.csv.

Usage (inside the Docker container):
    python src/experiment_runner.py

Or run a single experiment:
    python src/experiment_runner.py --sizes 100

Or run a subset:
    python src/experiment_runner.py --sizes 50 100 150 200
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure sibling modules are importable regardless of working directory
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_splitter import create_split          # noqa: E402
from train import MIN_TRAIN_IMAGES, train_simplenet  # noqa: E402
from evaluate import evaluate_simplenet, save_results  # noqa: E402


# ===========================================================================
# Configuration  (edit these values to customise the experiment)
# ===========================================================================

# Sample-efficiency experiment: how many OK images to test
TRAIN_SIZES = [50, 100, 150, 200]

# Filesystem paths (relative to project root)
DATASET_DIR = "dataset"
SPLITS_DIR = "splits"
EXPERIMENTS_DIR = "experiments"
RESULTS_CSV = "experiments/results.csv"
LOGS_DIR = "experiments/logs"

# SimpleNet hyper-parameters (paper defaults, CVPR 2023 Liu et al.)
IMAGE_SIZE = 256
BACKBONE = "wide_resnet50_2"
BATCH_SIZE = 32
EPOCHS = 200              # max epochs; early stopping may terminate sooner
LR = 2e-4                 # adaptor learning rate
DISC_LR = 1e-4            # discriminator learning rate (lr / 2)
NOISE_STD = 0.015         # Gaussian noise std (paper default)
ADAPTOR_DIM = 512
WEIGHT_DECAY = 1e-5       # L2 regularization
PATIENCE = 30             # early stopping patience (epochs)
N_AUGMENT_PASSES = 1      # no augmentation for controlled experiment

# Reproducibility
SEED = 42


# ===========================================================================
# Logging setup
# ===========================================================================

def setup_logging() -> logging.Logger:
    """Configure logging to stdout AND a timestamped log file."""
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(LOGS_DIR) / f"experiment_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    logger = logging.getLogger("experiment")
    logger.info(f"Log file: {log_file}")
    return logger


# ===========================================================================
# Single experiment
# ===========================================================================

def run_single_experiment(n_train: int, logger: logging.Logger) -> dict:
    """
    Run one full experiment cycle for a given training-set size.

    Args:
        n_train:  Number of OK images to use for training.
        logger:   Logger instance.

    Returns:
        Dictionary of evaluation metrics.
    """
    # --- Guard: reject if n_train < MIN_TRAIN_IMAGES ---
    if n_train < MIN_TRAIN_IMAGES:
        msg = (
            f"\n{'=' * 60}\n"
            f"  EXPERIMENT SKIPPED: n_train={n_train} < {MIN_TRAIN_IMAGES}\n"
            f"\n"
            f"  SimpleNet requires at least {MIN_TRAIN_IMAGES} OK images\n"
            f"  to produce reliable results.  With fewer images the\n"
            f"  feature bank is too sparse for the discriminator to\n"
            f"  learn a meaningful normal/anomaly boundary.\n"
            f"\n"
            f"  Skipping this training size.\n"
            f"{'=' * 60}"
        )
        logger.warning(msg)
        raise ValueError(msg)

    logger.info("=" * 60)
    logger.info(f"  EXPERIMENT:  n_train = {n_train}")
    logger.info("=" * 60)

    split_dir = str(Path(SPLITS_DIR) / f"n{n_train}")
    output_dir = str(Path(EXPERIMENTS_DIR) / f"n{n_train}")

    # ---- 1. Create dataset split ----
    logger.info("Step 1/3  Creating dataset split ...")
    stats = create_split(
        dataset_dir=DATASET_DIR,
        output_dir=split_dir,
        n_train=n_train,
        seed=SEED,
    )

    # ---- 2. Train SimpleNet ----
    logger.info("Step 2/3  Training SimpleNet ...")
    checkpoint_path = train_simplenet(
        split_dir=split_dir,
        output_dir=output_dir,
        image_size=IMAGE_SIZE,
        backbone=BACKBONE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        disc_lr=DISC_LR,
        noise_std=NOISE_STD,
        adaptor_dim=ADAPTOR_DIM,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        n_augment_passes=N_AUGMENT_PASSES,
    )

    # ---- 3. Evaluate model ----
    logger.info("Step 3/3  Evaluating model ...")
    metrics = evaluate_simplenet(
        split_dir=split_dir,
        checkpoint_path=checkpoint_path,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        backbone=BACKBONE,
    )

    # ---- 4. Persist results ----
    save_results(metrics, RESULTS_CSV, n_train)

    logger.info(
        f"  -> n_train={n_train}  AUROC={metrics['auroc']:.4f}  "
        f"Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}  "
        f"F1={metrics['f1']:.4f}"
    )
    return metrics


# ===========================================================================
# Main
# ===========================================================================

def main(train_sizes: list[int] | None = None) -> None:
    """Run the complete experiment suite."""
    logger = setup_logging()

    sizes = train_sizes or TRAIN_SIZES

    logger.info("SimpleNet Sample-Efficiency Experiment")
    logger.info(f"  Training sizes : {sizes}")
    logger.info(f"  Dataset        : {DATASET_DIR}")
    logger.info(f"  Backbone       : {BACKBONE}")
    logger.info(f"  Image size     : {IMAGE_SIZE}")
    logger.info(f"  Max epochs     : {EPOCHS} (early stopping patience={PATIENCE})")
    logger.info(f"  Adaptor LR     : {LR}")
    logger.info(f"  Disc LR        : {DISC_LR}")
    logger.info(f"  Weight decay   : {WEIGHT_DECAY}")
    logger.info(f"  Noise std      : {NOISE_STD}  (paper default)")
    logger.info(f"  Adaptor dim    : {ADAPTOR_DIM}")
    logger.info(f"  Augment passes : {N_AUGMENT_PASSES}")
    logger.info(f"  Seed           : {SEED}")
    logger.info(f"  Min train imgs : {MIN_TRAIN_IMAGES}")
    logger.info(f"  Results file   : {RESULTS_CSV}")

    # Clear previous results so the CSV only contains this run
    results_path = Path(RESULTS_CSV)
    if results_path.exists():
        results_path.unlink()
        logger.info("Cleared previous results file.")

    # ---- Run experiments ----
    all_results: dict[int, dict] = {}

    for n_train in sizes:
        try:
            metrics = run_single_experiment(n_train, logger)
            all_results[n_train] = metrics
        except Exception:
            logger.exception(f"FAILED: n_train={n_train}")
            continue

    # ---- Print summary table ----
    logger.info("")
    logger.info("=" * 72)
    logger.info("  SUMMARY — Sample Efficiency Experiment")
    logger.info("=" * 72)
    header = (
        f"{'n_train':>8}  {'AUROC':>7}  {'Precision':>9}  "
        f"{'Recall':>7}  {'F1':>7}  {'Accuracy':>8}"
    )
    logger.info(header)
    logger.info("-" * 72)
    for n, m in sorted(all_results.items()):
        logger.info(
            f"{n:>8}  {m['auroc']:>7.4f}  {m['precision']:>9.4f}  "
            f"{m['recall']:>7.4f}  {m['f1']:>7.4f}  {m['accuracy']:>8.4f}"
        )
    logger.info("-" * 72)

    # Flag sizes that meet the AUROC > 0.95 target
    passing = [n for n, m in all_results.items() if m["auroc"] >= 0.95]
    if passing:
        logger.info(
            f"  AUROC ≥ 0.95 achieved at n_train = {sorted(passing)}"
        )
        logger.info(
            f"  Recommended minimum training size: {min(passing)} OK images"
        )
    else:
        logger.info("  WARNING: No configuration achieved AUROC ≥ 0.95")
        logger.info("  Consider: more training images, augmentation, or noise_std tuning")

    logger.info("")
    logger.info(f"Full results saved to: {RESULTS_CSV}")


# ===========================================================================
# CLI
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SimpleNet sample-efficiency experiments"
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=None,
        help=(
            "Training-set sizes to test.  "
            f"Default: {TRAIN_SIZES}"
        ),
    )
    args = parser.parse_args()
    main(train_sizes=args.sizes)
