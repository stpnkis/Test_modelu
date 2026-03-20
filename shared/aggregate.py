"""
Aggregation — compute mean ± std across seeds and format for LaTeX.

Reads per-seed results from the experiments directory and produces
summary tables.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Metrics to aggregate
AGGREGATE_KEYS = [
    "auroc",
    "precision",
    "recall",
    "f1",
    "model_only_latency_ms",
    "end_to_end_latency_ms",
    "gpu_peak_memory_mb",
    "fit_time_s",
]


def aggregate_seeds(
    experiments_root: str | Path,
    dataset_id: str,
    model_name: str,
    seeds: List[int],
    preprocessing_mode: str = "baseline",
    n_train: int = 100,
) -> Dict[str, Any]:
    """Aggregate results across seeds for a single model+dataset.

    Returns:
        Dictionary with ``mean``, ``std``, and ``per_seed`` sub-dicts.
    """
    experiments_root = Path(experiments_root)
    per_seed_data = {}
    values_by_key: Dict[str, List[float]] = {k: [] for k in AGGREGATE_KEYS}

    for seed in seeds:
        # Try new layout first, then fallback to legacy
        results_path = (
            experiments_root
            / dataset_id
            / model_name
            / f"mode={preprocessing_mode}"
            / f"n_train={n_train}"
            / f"seed={seed}"
            / "results.json"
        )
        if not results_path.exists():
            # Legacy fallback
            results_path = (
                experiments_root / dataset_id / model_name / str(seed) / "results.json"
            )
        if not results_path.exists():
            logger.warning("Missing results for seed %d: %s", seed, results_path)
            continue

        with open(results_path) as f:
            data = json.load(f)

        per_seed_data[seed] = data

        # Collect numeric metric values
        metrics = data.get("metrics", {})
        runtime = data.get("runtime", {})
        merged = {**metrics, **runtime}

        for key in AGGREGATE_KEYS:
            val = merged.get(key)
            if val is not None and val != "N/A":
                values_by_key[key].append(float(val))

    # Handle AU-PRO separately (may be N/A)
    aupro_values = []
    aupro_reason = None
    for data in per_seed_data.values():
        aupro_val = data.get("metrics", {}).get("aupro")
        if aupro_val is not None and aupro_val != "N/A":
            aupro_values.append(float(aupro_val))
        else:
            aupro_reason = data.get("metrics", {}).get("aupro_reason", "N/A")

    # Compute mean ± std
    summary: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "model_name": model_name,
        "seeds": seeds,
        "n_seeds_found": len(per_seed_data),
        "mean": {},
        "std": {},
    }

    for key in AGGREGATE_KEYS:
        vals = values_by_key[key]
        if vals:
            summary["mean"][key] = round(float(np.mean(vals)), 4)
            summary["std"][key] = (
                round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0
            )
        else:
            summary["mean"][key] = "N/A"
            summary["std"][key] = "N/A"

    # AU-PRO
    if aupro_values:
        summary["mean"]["aupro"] = round(float(np.mean(aupro_values)), 4)
        summary["std"]["aupro"] = (
            round(float(np.std(aupro_values, ddof=1)), 4)
            if len(aupro_values) > 1
            else 0.0
        )
    else:
        summary["mean"]["aupro"] = "N/A"
        summary["std"]["aupro"] = "N/A"
        if aupro_reason:
            summary["aupro_reason"] = aupro_reason

    summary["per_seed"] = per_seed_data

    return summary


def format_latex_row(summary: Dict[str, Any]) -> str:
    """Format a summary as a LaTeX table row.

    Format: ``model & AUROC & AU-PRO & Prec & Rec & F1 & latency \\\\``
    """

    def _fmt(key: str) -> str:
        mean = summary["mean"].get(key, "N/A")
        std = summary["std"].get(key, "N/A")
        if mean == "N/A":
            return "N/A"
        return f"${mean:.4f} \\pm {std:.4f}$"

    cols = [
        summary["model_name"],
        _fmt("auroc"),
        _fmt("aupro"),
        _fmt("precision"),
        _fmt("recall"),
        _fmt("f1"),
        _fmt("model_only_latency_ms"),
        _fmt("end_to_end_latency_ms"),
        _fmt("gpu_peak_memory_mb"),
    ]
    return " & ".join(cols) + " \\\\"


def save_summary(
    summary: Dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save aggregated summary to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary saved → %s", output_path)
