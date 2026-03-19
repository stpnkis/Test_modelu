"""
Results I/O — structured storage of per-seed experiment results.

Storage layout:
    experiments/<dataset_id>/<model_name>/<seed>/
        results.json       — full metrics + runtime + config
        log.txt            — log output

Results from different datasets **never** overwrite each other.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_results_dir(
    experiments_root: str | Path,
    dataset_id: str,
    model_name: str,
    seed: int,
) -> Path:
    """Return the results directory for a specific run."""
    return Path(experiments_root) / dataset_id / model_name / str(seed)


def save_results(
    experiments_root: str | Path,
    dataset_id: str,
    model_name: str,
    seed: int,
    metrics: Dict[str, Any],
    runtime: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a single experiment run's results to JSON.

    Args:
        experiments_root: Root experiments directory.
        dataset_id:       Dataset identifier.
        model_name:       Model name (e.g. ``"patchcore"``).
        seed:             Random seed used.
        metrics:          Image-level (and pixel-level) metrics dict.
        runtime:          Runtime profiling dict.
        config:           Configuration snapshot.

    Returns:
        Path to the saved results file.
    """
    results_dir = get_results_dir(experiments_root, dataset_id, model_name, seed)
    results_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset_id": dataset_id,
        "model_name": model_name,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
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
) -> Dict[str, Any]:
    """Load results for a specific run."""
    results_dir = get_results_dir(experiments_root, dataset_id, model_name, seed)
    results_path = results_dir / "results.json"
    with open(results_path) as f:
        return json.load(f)
