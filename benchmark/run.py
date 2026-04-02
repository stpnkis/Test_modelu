#!/usr/bin/env python3
"""
Benchmark CLI — main entry point for running anomaly-detection benchmarks.

Supports two benchmark modes:
    ``main``    — primary model-selection benchmark (max available train data)
    ``fewshot`` — supplementary low-data robustness study (nested subsets)

Usage examples:

    # Main benchmark (default): uses max train data per dataset registry
    python benchmark/run.py --dataset wood --model patchcore --mode main

    # Main benchmark with all models
    python benchmark/run.py --dataset wood --all-models --mode main

    # Few-shot study: runs all nested sizes from dataset registry
    python benchmark/run.py --dataset wood --model patchcore --mode fewshot

    # Explicit override (advanced)
    python benchmark/run.py --dataset wood --model patchcore \\
        --n-train-ok 100 --seeds 42 1337 2026

    # Aggregate only (no training)
    python benchmark/run.py --dataset wood --aggregate-only --mode main
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
from shared.dataset_registry import (
    get_dataset_config,
    get_main_n_train,
    get_few_shot_levels,
)
from shared.split_manager import create_split
from benchmark.orchestrator import (
    orchestrate,
    orchestrate_few_shot,
    run_model_in_docker,
)

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
        "--mode",
        type=str,
        default=None,
        choices=["main", "fewshot"],
        help="Benchmark mode: main (max train data) or fewshot (nested subsets). "
        "Overrides config.yaml benchmark_mode.",
    )
    parser.add_argument(
        "--n-train-ok",
        type=int,
        default=None,
        help="Number of OK training images (overrides registry default). "
        "In main mode, defaults to the dataset's main_train_ok from the registry.",
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
    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="(DEPRECATED: use --mode fewshot) "
        "Run few-shot study — equivalent to --mode fewshot.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)

    # ── Load config ──────────────────────────────────────────────────
    cfg = load_config(Path(args.config))
    paths = cfg.get("paths", {})

    dataset_id = args.dataset
    seeds = args.seeds or cfg.get("seeds", [42, 1337, 2026])
    preprocessing_mode = args.preprocessing_mode or cfg.get(
        "preprocessing_mode", "baseline"
    )

    # ── Resolve benchmark mode ──────────────────────────────────────
    if args.few_shot:
        logger.warning("--few-shot is deprecated. Use --mode fewshot instead.")
    benchmark_mode = (
        args.mode
        or ("fewshot" if args.few_shot else None)
        or cfg.get("benchmark_mode", "main")
    )

    # ── Resolve n_train from registry or override ───────────────────
    try:
        ds_cfg = get_dataset_config(dataset_id)
    except KeyError as e:
        parser.error(str(e))
        return

    if args.n_train_ok is not None:
        n_train = args.n_train_ok
    elif benchmark_mode == "main":
        n_train = ds_cfg.main_train_ok
    else:
        # fewshot mode — n_train is not used directly; few_shot_levels are
        n_train = ds_cfg.main_train_ok  # placeholder for aggregate-only

    few_shot_levels = list(ds_cfg.few_shot_levels)

    logger.info(
        "Mode: %s  dataset: %s  n_train: %d  few_shot_levels: %s",
        benchmark_mode,
        dataset_id,
        n_train,
        few_shot_levels if benchmark_mode == "fewshot" else "N/A",
    )

    # ── Validate config early ───────────────────────────────────────────────
    VALID_PREPROCESSING_MODES = {"baseline", "high_accuracy", "edge_safe"}
    VALID_THRESHOLD_STRATEGIES = {"quantile", "max", "k_sigma"}
    if preprocessing_mode not in VALID_PREPROCESSING_MODES:
        parser.error(
            f"Invalid --preprocessing-mode '{preprocessing_mode}'. "
            f"Must be one of: {sorted(VALID_PREPROCESSING_MODES)}"
        )
    threshold_strategy_raw = cfg.get("threshold_strategy", "quantile")
    if threshold_strategy_raw not in VALID_THRESHOLD_STRATEGIES:
        parser.error(
            f"Invalid threshold_strategy '{threshold_strategy_raw}' in config. "
            f"Must be one of: {sorted(VALID_THRESHOLD_STRATEGIES)}"
        )

    datasets_root = paths.get("datasets_root", "datasets")
    splits_root = paths.get("splits_root", "splits")
    experiments_root = paths.get("experiments_root", "experiments")
    threshold_strategy = cfg.get("threshold_strategy", "quantile")
    threshold_quantile_p = cfg.get("threshold_quantile_p", 0.98)

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
        logger.info("=" * 70)
        logger.info("  AGGREGATE-ONLY RUN (no training)")
        logger.info(
            "  dataset=%s  models=%s  seeds=%s  n_train=%d  mode=%s",
            dataset_id, models, seeds, n_train, preprocessing_mode,
        )
        logger.info("  experiments_root=%s", experiments_root)
        logger.info("=" * 70)
        for model_name in models:
            summary = aggregate_seeds(
                experiments_root,
                dataset_id,
                model_name,
                seeds,
                preprocessing_mode=preprocessing_mode,
                n_train=n_train,
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
            print(f"\n{model_name}:")
            print(f"  AUROC:  {summary['mean'].get('auroc', 'N/A')}")
            print(f"  F1:     {summary['mean'].get('f1', 'N/A')}")
            print(f"  AU-PRO: {summary['mean'].get('aupro', 'N/A')}")
            print(f"\n  LaTeX:  {format_latex_row(summary)}")
        return

    # ── Few-shot study ─────────────────────────────────────────────
    if benchmark_mode == "fewshot":
        logger.info("=" * 70)
        logger.info(
            "  FEW-SHOT STUDY: dataset=%s  models=%s  levels=%s",
            dataset_id,
            models,
            few_shot_levels,
        )
        logger.info("=" * 70)

        results = orchestrate_few_shot(
            dataset_id=dataset_id,
            models=models,
            seeds=seeds,
            few_shot_sizes=few_shot_levels,
            preprocessing_mode=preprocessing_mode,
            datasets_root=datasets_root,
            splits_root=splits_root,
            experiments_root=experiments_root,
            repo_root=str(REPO_ROOT),
            threshold_strategy=threshold_strategy,
            threshold_quantile_p=threshold_quantile_p,
        )

        # Print summary
        for model_name, by_n in results.items():
            print(f"\n{'='*60}")
            print(f"  {model_name} — Few-shot results")
            print(f"{'='*60}")
            for n, summary in sorted(by_n.items()):
                auroc = summary.get("mean", {}).get("auroc", "N/A")
                f1 = summary.get("mean", {}).get("f1", "N/A")
                print(f"  n_train={n:>4d}  AUROC={auroc}  F1={f1}")
        return

    # ── Full benchmark ───────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info(
        "  BENCHMARK RUN [%s]: dataset=%s  models=%s",
        benchmark_mode.upper(),
        dataset_id,
        models,
    )
    logger.info(
        "  seeds=%s  n_train=%d  preprocessing=%s",
        seeds, n_train, preprocessing_mode,
    )
    logger.info(
        "  threshold=%s(p=%.4f)  experiments_root=%s",
        threshold_strategy, threshold_quantile_p, experiments_root,
    )
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
            "average_precision",
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
        if "mean" in summary:
            print(format_latex_row(summary))
        else:
            print(f"{model_name} & \\multicolumn{{9}}{{c}}{{FAILED}} \\\\")


if __name__ == "__main__":
    main()
