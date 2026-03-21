"""
Results I/O — structured storage of per-seed experiment results.

Storage layout:
    experiments/<dataset_id>/<model_name>/mode=<mode>/n_train=<n>/seed=<seed>/
        results.json       — full metrics + runtime + config + environment
        predictions.csv    — per-image raw scores (image_path, score, label, prediction)
        log.txt            — log output

Legacy layout (still readable):
    experiments/<dataset_id>/<model_name>/<seed>/
        results.json

Results from different datasets **never** overwrite each other.
Results from different (mode, n_train) combinations **never** overwrite each other.
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Environment metadata ─────────────────────────────────────────────────────


def _get_git_sha() -> str:
    """Return the current git HEAD SHA, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _get_environment_info() -> Dict[str, Any]:
    """Collect hardware + software environment metadata for auditability."""
    import torch

    env: Dict[str, Any] = {
        "git_sha": _get_git_sha(),
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env["cuda_version"] = torch.version.cuda or "unknown"
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_count"] = torch.cuda.device_count()
        env["gpu_memory_total_mb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1
        )
    return env


def get_results_dir(
    experiments_root: str | Path,
    dataset_id: str,
    model_name: str,
    seed: int,
    preprocessing_mode: str = "baseline",
    n_train: int = 100,
) -> Path:
    """Return the results directory for a specific run.

    New layout: ``<root>/<dataset>/<model>/mode=<mode>/n_train=<n>/seed=<seed>/``
    """
    return (
        Path(experiments_root)
        / dataset_id
        / model_name
        / f"mode={preprocessing_mode}"
        / f"n_train={n_train}"
        / f"seed={seed}"
    )


def get_results_dir_legacy(
    experiments_root: str | Path,
    dataset_id: str,
    model_name: str,
    seed: int,
) -> Path:
    """Return the legacy results directory (for reading old results)."""
    return Path(experiments_root) / dataset_id / model_name / str(seed)


def save_results(
    experiments_root: str | Path,
    dataset_id: str,
    model_name: str,
    seed: int,
    metrics: Dict[str, Any],
    runtime: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    preprocessing_mode: str = "baseline",
    n_train: int = 100,
) -> Path:
    """Save a single experiment run's results to JSON.

    Automatically includes git SHA, hardware info, and torch/CUDA versions.

    Args:
        experiments_root:   Root experiments directory.
        dataset_id:         Dataset identifier.
        model_name:         Model name (e.g. ``"patchcore"``).
        seed:               Random seed used.
        metrics:            Image-level (and pixel-level) metrics dict.
        runtime:            Runtime profiling dict.
        config:             Configuration snapshot.
        preprocessing_mode: Preprocessing mode used.
        n_train:            Number of training OK images.

    Returns:
        Path to the saved results file.
    """
    results_dir = get_results_dir(
        experiments_root,
        dataset_id,
        model_name,
        seed,
        preprocessing_mode=preprocessing_mode,
        n_train=n_train,
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset_id": dataset_id,
        "model_name": model_name,
        "seed": seed,
        "preprocessing_mode": preprocessing_mode,
        "n_train": n_train,
        "timestamp": datetime.now().isoformat(),
        "environment": _get_environment_info(),
        "metrics": metrics,
        "runtime": runtime or {},
        "config": config or {},
    }

    results_path = results_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    logger.info("Results saved → %s", results_path)
    return results_path


def load_results(
    experiments_root: str | Path,
    dataset_id: str,
    model_name: str,
    seed: int,
    preprocessing_mode: str = "baseline",
    n_train: int = 100,
) -> Dict[str, Any]:
    """Load results for a specific run. Falls back to legacy layout."""
    results_dir = get_results_dir(
        experiments_root,
        dataset_id,
        model_name,
        seed,
        preprocessing_mode=preprocessing_mode,
        n_train=n_train,
    )
    results_path = results_dir / "results.json"
    if not results_path.exists():
        # Fallback to legacy layout
        legacy_dir = get_results_dir_legacy(
            experiments_root,
            dataset_id,
            model_name,
            seed,
        )
        results_path = legacy_dir / "results.json"
    with open(results_path) as f:
        return json.load(f)


# ── Per-image predictions CSV ────────────────────────────────────────────────


def save_predictions_csv(
    experiments_root: str | Path,
    dataset_id: str,
    model_name: str,
    seed: int,
    image_paths: List[str],
    scores: List[float],
    labels: List[int],
    predictions: List[int],
    preprocessing_mode: str = "baseline",
    n_train: int = 100,
) -> Path:
    """Save per-image raw scores and predictions as a CSV audit artifact.

    Columns: image_path, score, label, prediction

    Args:
        image_paths:  Paths to test images.
        scores:       Raw anomaly scores (float).
        labels:       Ground-truth labels (0=normal, 1=anomalous).
        predictions:  Binary predictions at the threshold.

    Returns:
        Path to the saved CSV file.
    """
    results_dir = get_results_dir(
        experiments_root,
        dataset_id,
        model_name,
        seed,
        preprocessing_mode=preprocessing_mode,
        n_train=n_train,
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "score", "label", "prediction"])
        for path, score, label, pred in zip(image_paths, scores, labels, predictions):
            writer.writerow([path, f"{score:.6f}", label, pred])

    logger.info("Predictions CSV saved → %s  (%d rows)", csv_path, len(image_paths))
    return csv_path
