#!/bin/bash
#
# Simple VPU Parallel Execution (requires GNU parallel)
#
# Install: brew install parallel
#
# Much simpler and more robust than manual process management
#

set -e

METHODS="vpu vpu_mean vpu_mean_prior vpu_nomixup vpu_nomixup_mean vpu_nomixup_mean_prior"

DATASETS=(
    "config/vpu_rerun/mnist.yaml"
    "config/vpu_rerun/fashionmnist.yaml"
    "config/vpu_rerun/imdb.yaml"
    "config/vpu_rerun/20news.yaml"
    "config/vpu_rerun/mushrooms.yaml"
    "config/vpu_rerun/spambase.yaml"
)

# Check for GNU parallel
if ! command -v parallel &> /dev/null; then
    echo "❌ GNU parallel not found"
    echo ""
    echo "Install with: brew install parallel"
    echo ""
    echo "Or use: bash scripts/run_vpu_parallel.sh (fallback version)"
    exit 1
fi

echo "========================================="
echo "VPU Parallel Execution (GNU Parallel)"
echo "========================================="
echo ""
echo "Running 3 experiments simultaneously"
echo "Press Ctrl+C to stop (all workers will be killed)"
echo ""
echo "========================================="
echo ""

mkdir -p logs/vpu_rerun/parallel

# Define worker function
run_worker() {
    local config=$1
    local dataset=$(basename "$config" .yaml)
    local logfile="logs/vpu_rerun/parallel/${dataset}.log"

    echo "[$(date '+%H:%M:%S')] Starting $dataset"

    python run_train.py \
        --dataset-config "$config" \
        --methods $METHODS \
        --resume \
        > "$logfile" 2>&1

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✓ Completed $dataset"
    else
        echo "[$(date '+%H:%M:%S')] ✗ Failed $dataset (exit: $exit_code)"
    fi

    return $exit_code
}

export -f run_worker
export METHODS

# Run with GNU parallel
# --jobs 3: Run 3 at once
# --line-buffer: Show output in real-time
# --progress: Show progress bar
# {/.} is basename without extension
printf '%s\n' "${DATASETS[@]}" | \
    parallel --jobs 3 \
             --line-buffer \
             --tagstring '[{/.}]' \
             --progress \
             run_worker {}

echo ""
echo "========================================="
echo "All experiments complete!"
echo "========================================="
echo ""
echo "Check logs: logs/vpu_rerun/parallel/*.log"
echo "Validate: python scripts/validate_metrics_coverage.py"
echo ""
