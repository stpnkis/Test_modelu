#!/usr/bin/env bash
# setup.sh — Create required directories after a fresh git clone.
# Run once: bash setup.sh

set -e

echo "Creating directory structure..."

# Datasets directory (user places datasets here)
mkdir -p datasets

# Centralized splits and experiments
mkdir -p splits
mkdir -p experiments

# Legacy per-model directories (backward compatibility)
for model in anomalydino simplenet patchcore rd_plus_plus; do
    mkdir -p "experiments/$model/logs"
done

# Documentation
mkdir -p docs

echo ""
echo "Done. Directory structure ready."
echo ""
echo "Next steps:"
echo "  1. Place your dataset in datasets/<dataset_id>/{ok,nok}"
echo "  2. Run benchmark:"
echo "     python benchmark/run.py --dataset <dataset_id> --model patchcore"
echo "     python benchmark/run.py --dataset <dataset_id> --all-models"
echo ""
echo "  Or run a single model directly:"
echo "     cd models/anomalydino && docker compose build"
echo "     docker compose run --rm anomalydino python src/run_benchmark.py \\"
echo "       --dataset-id <dataset_id> --seed 42 --n-train 100"
