"""
Aggregation — compute mean ± std across seeds and format for LaTeX.

Reads per-seed results from the experiments directory and produces
summary tables.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Consistency validation ───────────────────────────────────────────────────


def _validate_seed_consistency(
    per_seed_data: Dict[int, Dict[str, Any]],
    expected_seeds: List[int],
    dataset_id: str,
    model_name: str,
    preprocessing_mode: str,
    n_train: int,
) -> None:
    """Validate that all seed results are consistent and match expectations.

    Guards against stale artifacts, mixed configurations, and incomplete runs.

    Raises:
        ValueError: If any consistency check fails.
    """
    # ---- Duplicate seeds in the request itself ----
    if len(expected_seeds) != len(set(expected_seeds)):
        raise ValueError(
            f"Duplicate seeds in expected list: {expected_seeds}"
        )

    # ---- Exact seed-set match ----
    actual_seeds = set(per_seed_data.keys())
    expected_set = set(expected_seeds)
    if actual_seeds != expected_set:
        missing = sorted(expected_set - actual_seeds)
        extra = sorted(actual_seeds - expected_set)
        parts = []
        if missing:
            parts.append(f"missing seeds: {missing}")
        if extra:
            parts.append(f"unexpected seeds: {extra}")
        raise ValueError(
            f"Seed mismatch for {model_name}/{dataset_id}: {'; '.join(parts)}. "
            f"Expected exactly: {sorted(expected_seeds)}"
        )

    # ---- Top-level field consistency across seeds ----
    EXPECTED_FIELDS = {
        "model_name": model_name,
        "dataset_id": dataset_id,
        "n_train": n_train,
        "preprocessing_mode": preprocessing_mode,
    }
    for seed, data in per_seed_data.items():
        for field, expected_val in EXPECTED_FIELDS.items():
            actual_val = data.get(field)
            if actual_val is not None and actual_val != expected_val:
                raise ValueError(
                    f"Inconsistent '{field}' in seed={seed} results: "
                    f"expected {expected_val!r}, got {actual_val!r}. "
                    f"Possible stale artifact at {data.get('timestamp', '?')}."
                )

    # ---- Threshold config consistency across seeds ----
    threshold_strategies: dict[int, str] = {}
    threshold_quantiles: dict[int, float] = {}
    for seed, data in per_seed_data.items():
        config = data.get("config", {})
        ts = config.get("threshold_strategy")
        tq = config.get("threshold_quantile_p")
        if ts is not None:
            threshold_strategies[seed] = ts
        if tq is not None:
            threshold_quantiles[seed] = tq

    if len(set(threshold_strategies.values())) > 1:
        raise ValueError(
            f"Inconsistent threshold_strategy across seeds: {threshold_strategies}"
        )
    if len(set(threshold_quantiles.values())) > 1:
        raise ValueError(
            f"Inconsistent threshold_quantile_p across seeds: {threshold_quantiles}"
        )

    logger.info(
        "Consistency OK: %s/%s mode=%s n_train=%d seeds=%s",
        dataset_id, model_name, preprocessing_mode, n_train,
        sorted(expected_seeds),
    )

# Metrics to aggregate
AGGREGATE_KEYS = [
    "auroc",
    "average_precision",
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
            raise FileNotFoundError(
                f"Missing results for seed {seed}: {results_path}. "
                f"All {len(seeds)} seeds must complete before aggregation. "
                f"Re-run the failed seed or check for errors."
            )

        with open(results_path) as f:
            data = json.load(f)

        per_seed_data[seed] = data

    # ---- Validate consistency before aggregating ----
    _validate_seed_consistency(
        per_seed_data,
        expected_seeds=seeds,
        dataset_id=dataset_id,
        model_name=model_name,
        preprocessing_mode=preprocessing_mode,
        n_train=n_train,
    )

    for seed, data in per_seed_data.items():
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
        "preprocessing_mode": preprocessing_mode,
        "n_train": n_train,
        "seeds": seeds,
        "n_seeds_found": len(per_seed_data),
        "n_seeds_expected": len(seeds),
        "aggregation_timestamp": datetime.now().isoformat(),
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

    Format: ``model & AUROC & AP & AU-PRO & Prec & Rec & F1 & latency \\\\``
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
        _fmt("average_precision"),
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
