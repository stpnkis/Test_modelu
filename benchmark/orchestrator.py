"""
Orchestrator — runs benchmarks for a given dataset × model × seeds.

This module handles:
  1. Dataset validation
  2. Split creation for all seeds / n_train values
  3. Dispatching training + evaluation to the appropriate model container
  4. Result collection and aggregation
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.aggregate import aggregate_seeds, save_summary
from shared.dataset_schema import validate_dataset
from shared.results_io import get_results_dir
from shared.split_manager import create_split, get_split_dir

logger = logging.getLogger(__name__)


# ── Pre/post-run audit helpers ───────────────────────────────────────────────


def _audit_pre_run(
    splits_root: str,
    dataset_id: str,
    seed: int,
    n_train: int,
) -> None:
    """Verify that the split manifest exists before dispatching a model run."""
    split_dir = get_split_dir(splits_root, dataset_id, seed, n_train)
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Pre-run audit FAILED: split manifest not found at {manifest_path}. "
            f"Cannot run model without a valid split."
        )
    logger.debug("Pre-run audit OK: split manifest exists at %s", manifest_path)


def _audit_post_run(
    experiments_root: str,
    dataset_id: str,
    model_name: str,
    seed: int,
    preprocessing_mode: str,
    n_train: int,
) -> None:
    """Verify that a model run produced the expected artifacts."""
    results_dir = get_results_dir(
        experiments_root, dataset_id, model_name, seed,
        preprocessing_mode=preprocessing_mode, n_train=n_train,
    )
    results_json = results_dir / "results.json"
    predictions_csv = results_dir / "predictions.csv"

    missing = []
    if not results_json.exists():
        missing.append(str(results_json))
    if not predictions_csv.exists():
        missing.append(str(predictions_csv))

    if missing:
        raise FileNotFoundError(
            f"Post-run audit FAILED for {model_name} seed={seed}: "
            f"missing artifacts: {missing}"
        )
    logger.debug(
        "Post-run audit OK: %s seed=%d -> results.json + predictions.csv present",
        model_name, seed,
    )


def _resolve_model_dir(model_name: str, repo_root: Path) -> Path:
    """Return the model directory for docker-compose."""
    return repo_root / "models" / model_name


def run_model_in_docker(
    model_name: str,
    dataset_id: str,
    seed: int,
    n_train: int,
    preprocessing_mode: str,
    repo_root: Path,
    extra_args: Optional[List[str]] = None,
    threshold_strategy: str = "quantile",
    threshold_quantile_p: float = 0.98,
) -> int:
    """Launch a model's train+evaluate pipeline inside its Docker container.

    Args:
        model_name:          e.g. ``"patchcore"``, ``"anomalydino"``
        dataset_id:          Dataset identifier.
        seed:                Random seed.
        n_train:             Number of training OK images.
        preprocessing_mode:  Preprocessing mode name.
        repo_root:           Repository root path.
        extra_args:          Additional CLI args to pass to the model script.
        threshold_strategy:  Threshold strategy name.
        threshold_quantile_p: Quantile p for threshold strategy.

    Returns:
        Subprocess return code (0 = success).
    """
    model_dir = _resolve_model_dir(model_name, repo_root)

    cmd = [
        "docker",
        "compose",
        "-f",
        str(model_dir / "docker-compose.yml"),
        "run",
        "--rm",
        model_name,
        "python",
        "src/run_benchmark.py",
        "--dataset-id",
        dataset_id,
        "--seed",
        str(seed),
        "--n-train",
        str(n_train),
        "--preprocessing-mode",
        preprocessing_mode,
        "--threshold-strategy",
        threshold_strategy,
        "--threshold-quantile-p",
        str(threshold_quantile_p),
    ]
    if extra_args:
        cmd.extend(extra_args)

    logger.info("Running: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=False,
    )
    return result.returncode


def orchestrate(
    dataset_id: str,
    models: List[str],
    seeds: List[int],
    n_train: int,
    preprocessing_mode: str,
    datasets_root: str,
    splits_root: str,
    experiments_root: str,
    repo_root: Optional[str] = None,
    threshold_strategy: str = "quantile",
    threshold_quantile_p: float = 0.98,
) -> Dict[str, Any]:
    """Run the full benchmark for one dataset.

    Steps:
      1. Validate dataset
      2. Create splits for all seeds
      3. Run each model for each seed
      4. Aggregate and save summaries

    Returns:
        Dictionary of per-model summaries.
    """
    if repo_root is None:
        repo_root = str(Path(__file__).resolve().parent.parent)
    repo_root_path = Path(repo_root)

    dataset_dir = Path(datasets_root) / dataset_id

    # 1. Validate
    logger.info("Validating dataset: %s", dataset_id)
    validate_dataset(dataset_dir)

    # 2. Create splits for all seeds
    for seed in seeds:
        logger.info("Creating split: seed=%d  n_train=%d", seed, n_train)
        create_split(
            dataset_dir=str(dataset_dir),
            splits_root=splits_root,
            dataset_id=dataset_id,
            n_train=n_train,
            seed=seed,
        )

    # 3. Run each model
    logger.info(
        "Run config: experiments_root=%s  seeds=%s  n_train=%d  "
        "preprocessing=%s  threshold=%s(p=%.4f)",
        experiments_root, seeds, n_train, preprocessing_mode,
        threshold_strategy, threshold_quantile_p,
    )

    results_map: Dict[str, Dict[str, Any]] = {}
    for model_name in models:
        logger.info("=" * 60)
        logger.info("  MODEL: %s", model_name)
        logger.info("=" * 60)

        seed_failed = False
        failed_seed = None
        failed_rc = None
        for seed in seeds:
            logger.info("  Seed: %d", seed)

            # Pre-run audit: split manifest must exist
            _audit_pre_run(splits_root, dataset_id, seed, n_train)

            rc = run_model_in_docker(
                model_name=model_name,
                dataset_id=dataset_id,
                seed=seed,
                n_train=n_train,
                preprocessing_mode=preprocessing_mode,
                repo_root=repo_root_path,
                threshold_strategy=threshold_strategy,
                threshold_quantile_p=threshold_quantile_p,
            )
            if rc != 0:
                logger.error(
                    "FAILED: model=%s seed=%d returned code %d — "
                    "aborting remaining seeds for this model.",
                    model_name,
                    seed,
                    rc,
                )
                seed_failed = True
                failed_seed = seed
                failed_rc = rc
                break

            # Post-run audit: results.json + predictions.csv must exist
            _audit_post_run(
                experiments_root, dataset_id, model_name, seed,
                preprocessing_mode, n_train,
            )

        if seed_failed:
            results_map[model_name] = {
                "error": (
                    f"Docker run failed for seed {failed_seed} "
                    f"(exit code {failed_rc})"
                ),
                "model_name": model_name,
                "dataset_id": dataset_id,
            }
            logger.warning(
                "Skipping aggregation for %s due to seed failure.", model_name
            )
            continue

        # 4. Aggregate (with consistency validation)
        logger.info(
            "Aggregating %s: seeds=%s  n_train=%d  mode=%s",
            model_name, seeds, n_train, preprocessing_mode,
        )
        summary = aggregate_seeds(
            experiments_root=experiments_root,
            dataset_id=dataset_id,
            model_name=model_name,
            seeds=seeds,
            preprocessing_mode=preprocessing_mode,
            n_train=n_train,
        )

        # Post-aggregation audit: seed count must match
        if summary["n_seeds_found"] != summary["n_seeds_expected"]:
            raise RuntimeError(
                f"Post-aggregation audit FAILED for {model_name}: "
                f"expected {summary['n_seeds_expected']} seeds, "
                f"found {summary['n_seeds_found']}"
            )

        summary_path = (
            Path(experiments_root)
            / dataset_id
            / model_name
            / f"mode={preprocessing_mode}"
            / f"n_train={n_train}"
            / "summary.json"
        )
        save_summary(summary, summary_path)
        results_map[model_name] = summary

    return results_map


def orchestrate_few_shot(
    dataset_id: str,
    models: List[str],
    seeds: List[int],
    few_shot_sizes: List[int],
    preprocessing_mode: str,
    datasets_root: str,
    splits_root: str,
    experiments_root: str,
    repo_root: Optional[str] = None,
    threshold_strategy: str = "quantile",
    threshold_quantile_p: float = 0.98,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Run the few-shot sample-efficiency study.

    Runs the full benchmark for each n_train in *few_shot_sizes*.

    Returns:
        Nested dict: ``{model_name: {n_train: summary_dict}}``.
    """
    results: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for n_train in few_shot_sizes:
        logger.info("=" * 70)
        logger.info("  FEW-SHOT STUDY: n_train=%d", n_train)
        logger.info("=" * 70)
        summaries = orchestrate(
            dataset_id=dataset_id,
            models=models,
            seeds=seeds,
            n_train=n_train,
            preprocessing_mode=preprocessing_mode,
            datasets_root=datasets_root,
            splits_root=splits_root,
            experiments_root=experiments_root,
            repo_root=repo_root,
            threshold_strategy=threshold_strategy,
            threshold_quantile_p=threshold_quantile_p,
        )
        for model_name, summary in summaries.items():
            results.setdefault(model_name, {})[n_train] = summary
    return results
