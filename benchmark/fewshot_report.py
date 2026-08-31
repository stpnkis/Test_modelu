#!/usr/bin/env python3
"""
Few-shot report generator.

Reads all few-shot experiment summaries and produces:
  1. Per-dataset tables (AUROC per model × n_train)
  2. Aggregate table (mean AUROC across datasets) for the LaTeX thesis table

Usage:
    python benchmark/fewshot_report.py [--experiments-root experiments]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared.dataset_registry import DATASET_CONFIGS, get_dataset_config

# Canonical model order and display names
MODEL_ORDER = ["patchcore", "rd_plus_plus", "simplenet", "anomalydino"]
MODEL_DISPLAY = {
    "patchcore": "PatchCore",
    "rd_plus_plus": "RD++",
    "simplenet": "SimpleNet",
    "anomalydino": "AnomalyDINO",
}

# Target few-shot levels for the summary table
TARGET_LEVELS = [10, 25, 50, 100, 200]


def load_summary(experiments_root: Path, dataset_id: str, model: str,
                 n_train: int, mode: str = "baseline") -> dict | None:
    """Load summary.json for a given (dataset, model, n_train) tuple."""
    summary_path = (
        experiments_root / dataset_id / model
        / f"mode={mode}" / f"n_train={n_train}" / "summary.json"
    )
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return None


def collect_all_results(experiments_root: Path,
                        datasets: list[str] | None = None,
                        mode: str = "baseline",
                        ) -> dict:
    """Collect all few-shot results.

    Returns:
        {dataset_id: {model: {n_train: auroc_mean}}}
    """
    if datasets is None:
        datasets = [d for d in DATASET_CONFIGS if d != "vial"]  # skip unfinished

    results = {}
    for ds_id in datasets:
        cfg = get_dataset_config(ds_id)
        results[ds_id] = {}
        for model in MODEL_ORDER:
            results[ds_id][model] = {}
            for n_train in cfg.few_shot_levels:
                summary = load_summary(experiments_root, ds_id, model, n_train, mode)
                if summary and "mean" in summary:
                    auroc = summary["mean"].get("auroc")
                    if auroc is not None:
                        results[ds_id][model][n_train] = auroc
    return results


def compute_aggregate(results: dict) -> dict:
    """Compute per-level, per-model aggregate AUROC (mean over datasets).

    Returns:
        {n_train_or_"full": {model: (mean_auroc, n_datasets, std_auroc)}}
    """
    aggregate = defaultdict(lambda: defaultdict(list))

    for ds_id, models_data in results.items():
        cfg = get_dataset_config(ds_id)
        full_level = cfg.main_train_ok

        for model in MODEL_ORDER:
            model_data = models_data.get(model, {})

            # Collect target levels
            for target_n in TARGET_LEVELS:
                if target_n in model_data:
                    aggregate[target_n][model].append(model_data[target_n])

            # Full level
            if full_level in model_data:
                aggregate["full"][model].append(model_data[full_level])

    # Compute means and stds
    summary = {}
    for level in list(TARGET_LEVELS) + ["full"]:
        summary[level] = {}
        for model in MODEL_ORDER:
            values = aggregate[level][model]
            if values:
                import statistics
                mean_v = statistics.mean(values)
                std_v = statistics.stdev(values) if len(values) > 1 else 0.0
                summary[level][model] = (mean_v, len(values), std_v)
            else:
                summary[level][model] = None
    return summary


def format_text_table(aggregate: dict, title: str = "Few-shot Aggregate") -> str:
    """Format a plain-text ASCII table."""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"  {title}")
    lines.append(f"{'='*80}")

    header = f"{'n_train':<12}" + "".join(
        f"{MODEL_DISPLAY[m]:>16}" for m in MODEL_ORDER
    )
    lines.append(header)
    lines.append("-" * len(header))

    for level in list(TARGET_LEVELS) + ["full"]:
        label = "Full" if level == "full" else f"{level} OK"
        row = f"{label:<12}"
        for model in MODEL_ORDER:
            entry = aggregate[level][model]
            if entry:
                mean_v, n_ds, std_v = entry
                row += f"{mean_v:>11.4f} ({n_ds:d})"
            else:
                row += f"{'N/A':>16}"
        lines.append(row)

    lines.append("")
    lines.append("  Numbers in parentheses = number of datasets contributing.")
    lines.append("  'Full' = each dataset's max training set size.")
    return "\n".join(lines)


def format_per_dataset_table(results: dict, ds_id: str) -> str:
    """Format per-dataset table."""
    cfg = get_dataset_config(ds_id)
    lines = []
    lines.append(f"\n--- {ds_id} (full={cfg.main_train_ok}) ---")

    header = f"{'n_train':<12}" + "".join(
        f"{MODEL_DISPLAY[m]:>14}" for m in MODEL_ORDER
    )
    lines.append(header)
    lines.append("-" * len(header))

    for n in cfg.few_shot_levels:
        label = f"{n} OK"
        if n == cfg.main_train_ok:
            label += " (full)"
        row = f"{label:<12}"
        for model in MODEL_ORDER:
            auroc = results.get(ds_id, {}).get(model, {}).get(n)
            if auroc is not None:
                row += f"{auroc:>14.4f}"
            else:
                row += f"{'N/A':>14}"
        lines.append(row)
    return "\n".join(lines)


def format_latex_table(aggregate: dict) -> str:
    """Generate the LaTeX table body for the thesis."""
    lines = []
    lines.append("% Auto-generated by benchmark/fewshot_report.py")
    lines.append("% Aggregate image-level AUROC (mean across datasets)")
    lines.append("")

    for level in list(TARGET_LEVELS) + ["full"]:
        label = "Full" if level == "full" else f"{level} OK"
        cells = []
        for model in MODEL_ORDER:
            entry = aggregate[level][model]
            if entry:
                mean_v, n_ds, std_v = entry
                cells.append(f"{mean_v:.4f}")
            else:
                cells.append(r"\na")
        row = f"{label:<12} & " + " & ".join(cells) + r" \\"
        lines.append(row)

    return "\n".join(lines)


def format_json_report(results: dict, aggregate: dict) -> dict:
    """Create JSON-serializable report."""
    report = {
        "per_dataset": {},
        "aggregate": {},
    }

    for ds_id, models_data in results.items():
        cfg = get_dataset_config(ds_id)
        report["per_dataset"][ds_id] = {
            "full_n_train": cfg.main_train_ok,
            "few_shot_levels": list(cfg.few_shot_levels),
            "models": {},
        }
        for model in MODEL_ORDER:
            model_results = {}
            for n, auroc in sorted(models_data.get(model, {}).items()):
                model_results[str(n)] = round(auroc, 4)
            report["per_dataset"][ds_id]["models"][model] = model_results

    for level in list(TARGET_LEVELS) + ["full"]:
        key = str(level)
        report["aggregate"][key] = {}
        for model in MODEL_ORDER:
            entry = aggregate[level][model]
            if entry:
                mean_v, n_ds, std_v = entry
                report["aggregate"][key][model] = {
                    "mean_auroc": round(mean_v, 4),
                    "std_auroc": round(std_v, 4),
                    "n_datasets": n_ds,
                }
            else:
                report["aggregate"][key][model] = None

    return report


def main():
    parser = argparse.ArgumentParser(description="Few-shot report generator")
    parser.add_argument("--experiments-root", default="experiments",
                        help="Path to experiments root")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Specific datasets to include (default: all registered)")
    parser.add_argument("--mode", default="baseline",
                        help="Preprocessing mode")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON report to stdout")
    parser.add_argument("--latex", action="store_true",
                        help="Output LaTeX table rows")
    parser.add_argument("--save", type=str, default=None,
                        help="Save JSON report to file")
    args = parser.parse_args()

    experiments_root = Path(args.experiments_root)

    # Determine which datasets to include
    if args.datasets:
        datasets = args.datasets
    else:
        # Auto-detect: include datasets that have at least one result
        datasets = []
        for ds_id in DATASET_CONFIGS:
            ds_dir = experiments_root / ds_id
            if ds_dir.exists() and any(ds_dir.rglob("summary.json")):
                datasets.append(ds_id)
        datasets.sort()

    print(f"Datasets included: {datasets}")

    results = collect_all_results(experiments_root, datasets, args.mode)
    aggregate = compute_aggregate(results)

    # Per-dataset tables
    for ds_id in datasets:
        print(format_per_dataset_table(results, ds_id))

    # Aggregate table
    print(format_text_table(aggregate, "Few-shot Aggregate (mean AUROC across datasets)"))

    # LaTeX output
    if args.latex:
        print("\n% LaTeX table rows:")
        print(format_latex_table(aggregate))

    # JSON output
    report = format_json_report(results, aggregate)
    if args.json:
        print(json.dumps(report, indent=2))

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON report saved to: {save_path}")


if __name__ == "__main__":
    main()
