"""
AnomalyDINO — Experiment Runner
=================================

Automates the full experimental pipeline:

    **"How does AnomalyDINO performance scale with the number of
    normal (OK) training images?"**

For each ``n_train`` in :data:`TRAIN_SIZES`:

1. Create a dataset split (``dataset_splitter.py``).
2. Build a DINOv2 memory bank (``train.py``).
3. Evaluate on a fixed test set (``evaluate.py``).
4. Append results to ``experiments/results.csv``.

Usage (inside Docker)::

    # Run all sizes with default ViT-B backbone
    python src/experiment_runner.py

    # Single size
    python src/experiment_runner.py --sizes 50

    # Override backbone
    python src/experiment_runner.py --backbone dinov2_vits14

    # Custom results file
    python src/experiment_runner.py --results-csv experiments/results_custom.csv

Configuration constants at the top of this file can be edited directly
or overridden via CLI flags.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── Ensure sibling modules are importable ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_splitter import create_split                          # noqa: E402
from train import DEFAULT_BACKBONE, DEFAULT_IMAGE_SIZE, train_anomalydino  # noqa: E402
from evaluate import evaluate_anomalydino, save_results            # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# Configuration  (edit here — or override via CLI flags)
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_SIZES: List[int] = [10, 50, 100, 150, 200]
"""Number of OK reference images for each experiment run."""

# Filesystem paths (relative to project root = /app inside Docker)
DATASET_DIR:    str = "dataset"
SPLITS_DIR:     str = "splits"
EXPERIMENTS_DIR: str = "experiments"
RESULTS_CSV:    str = "experiments/results.csv"
LOGS_DIR:       str = "experiments/logs"

# Model hyper-parameters
IMAGE_SIZE: int = DEFAULT_IMAGE_SIZE   # 448  (448/14 = 32×32 patches)
BACKBONE:   str = DEFAULT_BACKBONE     # "dinov2_vitb14"

# Reproducibility
SEED: int = 42


# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging() -> logging.Logger:
    """Configure logging to **stdout** and a timestamped log file."""
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
    exp_logger = logging.getLogger("experiment")
    exp_logger.info("Log file: %s", log_file)
    return exp_logger


# ══════════════════════════════════════════════════════════════════════════════
# Single experiment
# ══════════════════════════════════════════════════════════════════════════════

def run_single_experiment(n_train: int, log: logging.Logger) -> dict:
    """Run one train → evaluate cycle for a given *n_train*.

    Returns:
        Dictionary of evaluation metrics.
    """
    log.info("=" * 60)
    log.info("  EXPERIMENT:  n_train = %d", n_train)
    log.info("=" * 60)

    split_dir = str(Path(SPLITS_DIR) / f"n{n_train}")
    output_dir = str(Path(EXPERIMENTS_DIR) / f"n{n_train}")

    # 1/3 — Dataset split
    log.info("Step 1/3  Creating dataset split …")
    create_split(
        dataset_dir=DATASET_DIR,
        output_dir=split_dir,
        n_train=n_train,
        seed=SEED,
    )

    # 2/3 — Build memory bank
    log.info("Step 2/3  Building AnomalyDINO memory bank …")
    checkpoint_path = train_anomalydino(
        split_dir=split_dir,
        output_dir=output_dir,
        image_size=IMAGE_SIZE,
        backbone=BACKBONE,
    )

    # 3/3 — Evaluate
    log.info("Step 3/3  Evaluating model …")
    metrics = evaluate_anomalydino(
        split_dir=split_dir,
        checkpoint_path=checkpoint_path,
        image_size=IMAGE_SIZE,
        backbone=BACKBONE,
    )

    # Persist row
    save_results(metrics, RESULTS_CSV, n_train, backbone=BACKBONE)

    log.info(
        "  → n_train=%d  AUROC=%.4f  Prec=%.4f  Rec=%.4f  "
        "F1=%.4f  Inf=%.1f ms/img",
        n_train,
        metrics["auroc"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["avg_inference_ms"],
    )
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(train_sizes: Optional[List[int]] = None) -> None:
    """Run the complete experiment suite (all sizes)."""
    log = setup_logging()
    sizes = train_sizes or TRAIN_SIZES

    log.info("AnomalyDINO — Sample-Efficiency Experiment")
    log.info("  Sizes      : %s", sizes)
    log.info("  Dataset    : %s", DATASET_DIR)
    log.info("  Backbone   : %s", BACKBONE)
    log.info("  Image size : %d", IMAGE_SIZE)
    log.info("  Seed       : %d", SEED)
    log.info("  Results    : %s", RESULTS_CSV)

    # Clear previous results so the CSV matches this run exactly
    results_path = Path(RESULTS_CSV)
    if results_path.exists():
        results_path.unlink()
        log.info("Cleared previous results file.")

    # ── Run experiments ──────────────────────────────────────────────────
    all_results: Dict[int, dict] = {}
    for n_train in sizes:
        try:
            all_results[n_train] = run_single_experiment(n_train, log)
        except Exception:
            log.exception("FAILED: n_train=%d", n_train)
            continue

    # ── Summary table ────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 82)
    log.info("  SUMMARY — Sample-Efficiency Experiment  (%s)", BACKBONE)
    log.info("=" * 82)
    log.info(
        "%8s  %7s  %9s  %7s  %7s  %8s  %8s  %8s",
        "n_train", "AUROC", "Precision", "Recall", "F1",
        "Accuracy", "Inf ms", "Build s",
    )
    log.info("-" * 82)
    for n, m in sorted(all_results.items()):
        log.info(
            "%8d  %7.4f  %9.4f  %7.4f  %7.4f  %8.4f  %8.1f  %8.2f",
            n,
            m["auroc"], m["precision"], m["recall"], m["f1"],
            m["accuracy"], m["avg_inference_ms"],
            m.get("build_time_s") or 0,
        )
    log.info("-" * 82)

    passing = sorted(n for n, m in all_results.items() if m["auroc"] >= 0.95)
    if passing:
        log.info("  AUROC ≥ 0.95 at n_train = %s", passing)
        log.info("  Recommended minimum: %d OK images", min(passing))
    else:
        log.info("  WARNING: No config achieved AUROC ≥ 0.95")
        log.info("  Consider: more images / ViT-B backbone / image_size tuning")

    log.info("")
    log.info("Full results saved to: %s", RESULTS_CSV)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run AnomalyDINO sample-efficiency experiments.",
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=None,
        help=f"Training-set sizes to test  (default: {TRAIN_SIZES})",
    )
    parser.add_argument(
        "--results-csv", default=None,
        help=f"Override results CSV path  (default: {RESULTS_CSV})",
    )
    parser.add_argument(
        "--backbone", default=None,
        choices=["dinov2_vits14", "dinov2_vitb14"],
        help=f"Override backbone  (default: {BACKBONE})",
    )
    parser.add_argument(
        "--image-size", type=int, default=None,
        help=f"Override image size  (default: {IMAGE_SIZE})",
    )
    args = parser.parse_args()

    if args.backbone:
        BACKBONE = args.backbone
    if args.image_size:
        IMAGE_SIZE = args.image_size
    if args.results_csv:
        RESULTS_CSV = args.results_csv

    main(train_sizes=args.sizes)
