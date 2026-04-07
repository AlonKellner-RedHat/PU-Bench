#!/bin/bash
# Simple single-threaded Phase 2 execution - no parallelization

OUTPUT_DIR="results_comprehensive"
SHUFFLE_SEED=12345

echo "=========================================="
echo "Phase 2: Simple Sequential Execution"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Datasets: CIFAR10, AlzheimerMRI"
echo "  Methods: vpu, vpu_nomixup, vpu_mean_prior, vpu_nomixup_mean_prior, nnpu, distpu"
echo "  Execution: Sequential (no parallelization)"
echo "  Resume: Enabled"
echo ""
echo "Starting at: $(date)"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p logs/comprehensive

# Simple single command - no workers, no parallelization
uv run python run_train.py \
    --dataset-config config/comprehensive/cifar10_comprehensive.yaml config/comprehensive/alzheimermri_comprehensive.yaml \
    --methods "vpu,vpu_nomixup,vpu_mean_prior,vpu_nomixup_mean_prior,nnpu,distpu" \
    --output-dir "$OUTPUT_DIR" \
    --shuffle-seed "$SHUFFLE_SEED" \
    --resume

echo ""
echo "Completed at: $(date)"
