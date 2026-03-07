#!/usr/bin/env bash
# setup.sh — Create required output directories after a fresh git clone.
# Run once: bash setup.sh

set -e

echo "Creating experiment and split output directories..."

for model in anomalydino simplenet patchcore; do
    mkdir -p "experiments/$model/logs"
    mkdir -p "splits/$model"
done

echo "Done. Directory structure ready."
echo ""
echo "Next steps:"
echo "  cd models/anomalydino && docker compose build"
echo "  docker compose run --rm anomalydino python src/experiment_runner.py"
