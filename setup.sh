#!/usr/bin/env bash
# setup.sh — Initialise workspace after a fresh git clone.
# Run once: bash setup.sh

set -e

echo "Creating directory structure..."

# Datasets directory (user places datasets here)
mkdir -p datasets

# Centralized splits and experiments
mkdir -p splits
mkdir -p experiments

# Documentation
mkdir -p docs

# Install host-side Python dependencies (if a virtualenv is active)
if command -v pip &>/dev/null; then
    echo ""
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
else
    echo ""
    echo "WARNING: pip not found — install dependencies manually:"
    echo "  pip install -r requirements.txt"
fi

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
