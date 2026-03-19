#!/usr/bin/env python3
"""
Benchmark CLI — main entry point for running anomaly-detection benchmarks.

Usage examples:

    # Single model on a dataset
    python benchmark/run.py --dataset casting --model patchcore

    # All models
    python benchmark/run.py --dataset casting --all-models

    # Full control
    python benchmark/run.py --dataset casting --model patchcore \\
        --n-train-ok 100 --seeds 42 1337 2026 --preprocessing-mode baseline

    # Aggregate only (no training)
    python benchmark/run.py --dataset casting --aggregate-only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ── Ensure shared/ and benchmark/ are importable ────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml

from shared.aggregate import aggregate_seeds, format_latex_row, save_summary
from shared.dataset_schema import validate_dataset
from shared.split_manager import create_split
from benchmark.orchestrator import orchestrate, run_model_in_docker

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = REPO_ROOT / "benchmark" / "config.yaml"
ALL_MODELS = ["patchcore", "rd_plus_plus", "simplenet", "anomalydino"]


def load_config(path: Path) -> dict:
    """Load YAML config with defaults."""
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Anomaly Detection Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset ID (must exist under datasets/<id>/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to benchmark (e.g. patchcore, anomalydino)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all models",
    )
    parser.add_argument(
        "--n-train-ok",
        type=int,
        default=None,
        help="Number of OK training images (overrides config.yaml)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to use (overrides config.yaml)",
    )
    parser.add_argument(
        "--preprocessing-mode",
        type=str,
        default=None,
        help="Preprocessing mode: baseline | high_accuracy | edge_safe",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only aggregate existing results (no training)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the dataset (no training)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)

    # ── Load config ──────────────────────────────────────────────────
    cfg = load_config(Path(args.config))
    paths = cfg.get("paths", {})

    dataset_id = args.dataset
    seeds = args.seeds or cfg.get("seeds", [42, 1337, 2026])
    n_train = args.n_train_ok or cfg.get("n_train_ok", 100)
    preprocessing_mode = args.preprocessing_mode or cfg.get(
        "preprocessing_mode", "baseline"
    )
    datasets_root = paths.get("datasets_root", "datasets")
    splits_root = paths.get("splits_root", "splits")
    experiments_root = paths.get("experiments_root", "experiments")
    threshold_strategy = cfg.get("threshold_strategy", "quantile")
    threshold_quantile_p = cfg.get("threshold_quantile_p", 0.99)

    # Resolve models
    if args.all_models:
        models = cfg.get("models", ALL_MODELS)
    elif args.model:
        models = [args.model]
    else:
        parser.error("Specify --model <name> or --all-models")
        return

    dataset_dir = Path(datasets_root) / dataset_id

    # ── Validate only ────────────────────────────────────────────────
    if args.validate_only:
        stats = validate_dataset(dataset_dir)
        print(json.dumps(stats, indent=2))
        return

    # ── Aggregate only ───────────────────────────────────────────────
    if args.aggregate_only:
        for model_name in models:
            summary = aggregate_seeds(experiments_root, dataset_id, model_name, seeds)
            summary_path = (
                Path(experiments_root) / dataset_id / model_name / "summary.json"
            )
            save_summary(summary, summary_path)
            print(f"\n{model_name}:")
            print(f"  AUROC:  {summary['mean'].get('auroc', 'N/A')}")
            print(f"  F1:     {summary['mean'].get('f1', 'N/A')}")
            print(f"  AU-PRO: {summary['mean'].get('aupro', 'N/A')}")
            print(f"\n  LaTeX:  {format_latex_row(summary)}")
        return

    # ── Full benchmark ───────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("  BENCHMARK: dataset=%s  models=%s", dataset_id, models)
    logger.info("  seeds=%s  n_train=%d  mode=%s", seeds, n_train, preprocessing_mode)
    logger.info("=" * 70)

    results_map = orchestrate(
        dataset_id=dataset_id,
        models=models,
        seeds=seeds,
        n_train=n_train,
        preprocessing_mode=preprocessing_mode,
        datasets_root=datasets_root,
        splits_root=splits_root,
        experiments_root=experiments_root,
        repo_root=str(REPO_ROOT),
        threshold_strategy=threshold_strategy,
        threshold_quantile_p=threshold_quantile_p,
    )

    # ── Print summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)
    for model_name, summary in results_map.items():
        print(f"\n{model_name}:")
        mean = summary.get("mean", {})
        std = summary.get("std", {})
        for key in [
            "auroc",
            "precision",
            "recall",
            "f1",
            "aupro",
            "model_only_latency_ms",
            "end_to_end_latency_ms",
            "gpu_peak_memory_mb",
            "fit_time_s",
        ]:
            m = mean.get(key, "N/A")
            s = std.get(key, "N/A")
            if m != "N/A":
                print(f"  {key:30s} {m:.4f} ± {s:.4f}")
            else:
                print(f"  {key:30s} N/A")

    print("\n\nLaTeX table rows:")
    for model_name, summary in results_map.items():
        print(format_latex_row(summary))


if __name__ == "__main__":
    main()
