"""
Experiment Runner
==================

Automates the full experimental pipeline for the research question:

    "How does PatchCore performance degrade as the number of
     OK training samples decreases?"

For each training-set size N in TRAIN_SIZES:
  1.  Split the dataset  (N OK images for training, rest for test).
  2.  Train PatchCore    (builds the memory bank).
  3.  Evaluate on test   (computes AUROC, accuracy, precision, recall).
  4.  Append results to  experiments/results.csv.

Usage (inside the Docker container):
    python src/experiment_runner.py

Or run a single experiment from the command line:
    python src/experiment_runner.py --sizes 50
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure sibling modules are importable regardless of working directory
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_splitter import create_split   # noqa: E402
from train import train_patchcore           # noqa: E402
from evaluate import evaluate_patchcore, save_results  # noqa: E402


# ===========================================================================
# Configuration  (edit these values to customise the experiment)
# ===========================================================================

# How many OK images to train on in each run
TRAIN_SIZES = [10, 20, 50, 100, 200]

# Filesystem paths (relative to project root)
DATASET_DIR = "dataset"
SPLITS_DIR = "splits"
EXPERIMENTS_DIR = "experiments"
RESULTS_CSV = "experiments/results.csv"
LOGS_DIR = "experiments/logs"

# PatchCore hyper-parameters
IMAGE_SIZE = 256
BACKBONE = "wide_resnet50_2"
BATCH_SIZE = 32

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

    # ---- 2. Train PatchCore ----
    logger.info("Step 2/3  Training PatchCore ...")
    checkpoint_path = train_patchcore(
        split_dir=split_dir,
        output_dir=output_dir,
        image_size=IMAGE_SIZE,
        backbone=BACKBONE,
        batch_size=BATCH_SIZE,
    )

    # ---- 3. Evaluate model ----
    logger.info("Step 3/3  Evaluating model ...")
    metrics = evaluate_patchcore(
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
        f"Acc={metrics['accuracy']:.4f}  "
        f"Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}"
    )
    return metrics


# ===========================================================================
# Main
# ===========================================================================

def main(train_sizes: list[int] | None = None) -> None:
    """Run the complete experiment suite."""
    logger = setup_logging()

    sizes = train_sizes or TRAIN_SIZES

    logger.info("PatchCore Sample-Efficiency Experiment")
    logger.info(f"  Training sizes : {sizes}")
    logger.info(f"  Dataset        : {DATASET_DIR}")
    logger.info(f"  Backbone       : {BACKBONE}")
    logger.info(f"  Image size     : {IMAGE_SIZE}")
    logger.info(f"  Seed           : {SEED}")
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
    logger.info("=" * 62)
    logger.info("  SUMMARY")
    logger.info("=" * 62)
    header = f"{'n_train':>8}  {'AUROC':>7}  {'Accuracy':>8}  {'Precision':>9}  {'Recall':>7}"
    logger.info(header)
    logger.info("-" * 62)
    for n, m in sorted(all_results.items()):
        logger.info(
            f"{n:>8}  {m['auroc']:>7.4f}  {m['accuracy']:>8.4f}  "
            f"{m['precision']:>9.4f}  {m['recall']:>7.4f}"
        )
    logger.info("")
    logger.info(f"Full results saved to: {RESULTS_CSV}")


# ===========================================================================
# CLI
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run PatchCore sample-efficiency experiments"
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
