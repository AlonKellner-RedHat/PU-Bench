#!/bin/bash
#
# Fresh VPU Variants Rerun
#
# Runs all 6 VPU variants on 6 core datasets with 5 seeds
# Expected experiments: ~2,100 (6 datasets × 7 c-values × 6 priors × 5 seeds)
# Estimated time: 20-24 hours sequential, 3-4 hours parallel (8 cores)
#

set -e  # Exit on error

# VPU methods to run
METHODS="vpu vpu_mean vpu_mean_prior vpu_nomixup vpu_nomixup_mean vpu_nomixup_mean_prior"

# Core datasets
DATASETS=(
    "config/vpu_rerun/mnist.yaml"
    "config/vpu_rerun/fashionmnist.yaml"
    "config/vpu_rerun/imdb.yaml"
    "config/vpu_rerun/20news.yaml"
    "config/vpu_rerun/mushrooms.yaml"
    "config/vpu_rerun/spambase.yaml"
)

echo "========================================="
echo "VPU Variants Fresh Rerun"
echo "========================================="
echo ""
echo "Methods: $METHODS"
echo "Datasets: ${#DATASETS[@]}"
echo "  - MNIST"
echo "  - FashionMNIST"
echo "  - IMDB"
echo "  - 20News"
echo "  - Mushrooms"
echo "  - Spambase"
echo ""
echo "Seeds: 42, 123, 456, 789, 2024"
echo "C values: 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9"
echo "Priors: natural, 0.1, 0.3, 0.5, 0.7, 0.9"
echo ""
echo "Expected: ~2,100 experiments"
echo "Estimated time: 20-24 hours sequential, 3-4 hours parallel"
echo ""
echo "========================================="
echo ""

# Create log directory
mkdir -p logs/vpu_rerun

# Run all datasets
echo "Starting rerun at $(date)"
echo ""

python run_train.py \
    --dataset-config "${DATASETS[@]}" \
    --methods $METHODS \
    2>&1 | tee logs/vpu_rerun/rerun_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "========================================="
echo "Rerun complete at $(date)"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Validate: python scripts/validate_metrics_coverage.py"
echo "2. Extract: python analysis/extract_comprehensive_metrics.py"
echo "3. Plot: python analysis/plot_vpu_comprehensive_with_priors.py"
