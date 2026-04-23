#!/bin/bash
# Phase 1 Extended: Add method_prior=0.353 variants
# Runs only VPU mean-prior methods with --resume to add 0.353 experiments

OUTPUT_DIR="results_phase1_extended"
SHUFFLE_SEED=54321  # Same seed as original Phase 1 Extended
NUM_WORKERS=8

echo "============================================"
echo "Phase 1 Extended: Adding method_prior=0.353"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Datasets: 7 (MNIST, FashionMNIST, IMDB, 20News, Mushrooms, Spambase, Connect4)"
echo "  Methods: 2 (VPU mean-prior variants only)"
echo "    - vpu_mean_prior (will create 0.353 variant)"
echo "    - vpu_nomixup_mean_prior (will create 0.353 variant)"
echo "  Seeds: 10"
echo "  C values: 3 [0.1, 0.3, 0.5]"
echo "  Additional experiments: 420 (7 × 10 × 3 × 2)"
echo "  Workers: ${NUM_WORKERS}"
echo "  Estimated time: ~30 minutes (with 8 workers)"
echo ""
echo "Note: --resume flag will skip already-completed experiments"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p logs/phase1_extended_0353

# Only VPU mean-prior methods (will create variants based on method_prior_values in config)
METHODS="vpu_mean_prior,vpu_nomixup_mean_prior"

# Config files for 7 Phase 1 datasets (space-separated string for export)
CONFIGS="config/comprehensive/mnist_comprehensive.yaml config/comprehensive/fashionmnist_comprehensive.yaml config/comprehensive/imdb_comprehensive.yaml config/comprehensive/20news_comprehensive.yaml config/comprehensive/mushrooms_comprehensive.yaml config/comprehensive/spambase_comprehensive.yaml config/comprehensive/connect4_comprehensive.yaml"

# Function to run a single worker
function run_worker() {
    worker_id=$1

    echo "Starting Phase 1 Extended (0.353) worker ${worker_id}..."

    python run_train.py \
        --dataset-config $CONFIGS \
        --methods "$METHODS" \
        --output-dir "$OUTPUT_DIR" \
        --shuffle-seed "$SHUFFLE_SEED" \
        --num-workers "$NUM_WORKERS" \
        --worker-id "$worker_id" \
        --resume \
        > "logs/phase1_extended_0353/worker_${worker_id}.log" 2>&1

    echo "Worker ${worker_id} completed!"
}

# Export function and variables for parallel
export -f run_worker
export OUTPUT_DIR SHUFFLE_SEED NUM_WORKERS METHODS CONFIGS

# Run all workers in parallel
echo "Launching ${NUM_WORKERS} parallel workers..."
parallel -j "$NUM_WORKERS" run_worker ::: $(seq 0 $((NUM_WORKERS - 1)))

echo ""
echo "All Phase 1 Extended (0.353) experiments complete!"
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "  1. Verify completeness: python analysis/generate_comprehensive_table.py --validate-only"
echo "  2. Regenerate tables: python analysis/generate_comprehensive_table.py"
echo ""
