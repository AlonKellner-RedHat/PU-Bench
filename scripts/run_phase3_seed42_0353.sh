#!/bin/bash
# Phase 3 Seed 42: Add method_prior=0.353 variants
# Runs only VPU mean-prior methods with --resume to add 0.353 experiments for seed 42

OUTPUT_DIR="results_phase3"
SHUFFLE_SEED=42  # Phase 3 seed
NUM_WORKERS=8

echo "============================================"
echo "Phase 3 (Seed 42): Adding method_prior=0.353"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Datasets: 7 (MNIST, FashionMNIST, IMDB, 20News, Mushrooms, Spambase, Connect4)"
echo "  Methods: 2 (VPU mean-prior variants only)"
echo "    - vpu_mean_prior (will create 0.353 variant)"
echo "    - vpu_nomixup_mean_prior (will create 0.353 variant)"
echo "  Seeds: 1 [42]"
echo "  Label frequencies (c): 7 [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]"
echo "  True priors (π): 7 [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]"
echo "  Additional experiments: 686 (7 × 1 × 7 × 7 × 2)"
echo "  Workers: ${NUM_WORKERS}"
echo "  Estimated time: ~55 minutes (with 8 workers)"
echo ""
echo "Note: --resume flag will skip already-completed experiments"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p logs/phase3_seed42_0353

# Only VPU mean-prior methods (will create variants based on method_prior_values in config)
METHODS="vpu_mean_prior,vpu_nomixup_mean_prior"

# Config files for 7 Phase 3 datasets
CONFIGS="config/phase3/mnist_phase3.yaml config/phase3/fashionmnist_phase3.yaml config/phase3/imdb_phase3.yaml config/phase3/20news_phase3.yaml config/phase3/mushrooms_phase3.yaml config/phase3/spambase_phase3.yaml config/phase3/connect4_phase3.yaml"

# Function to run a single worker
function run_worker() {
    worker_id=$1

    echo "Starting Phase 3 Seed 42 (0.353) worker ${worker_id}..."

    python run_train.py \
        --dataset-config $CONFIGS \
        --methods "$METHODS" \
        --output-dir "$OUTPUT_DIR" \
        --shuffle-seed "$SHUFFLE_SEED" \
        --num-workers "$NUM_WORKERS" \
        --worker-id "$worker_id" \
        --resume \
        > "logs/phase3_seed42_0353/worker_${worker_id}.log" 2>&1

    echo "Worker ${worker_id} completed!"
}

# Export function and variables for parallel
export -f run_worker
export OUTPUT_DIR SHUFFLE_SEED NUM_WORKERS METHODS CONFIGS

# Run all workers in parallel
echo "Launching ${NUM_WORKERS} parallel workers..."
parallel -j "$NUM_WORKERS" run_worker ::: $(seq 0 $((NUM_WORKERS - 1)))

echo ""
echo "All Phase 3 Seed 42 (0.353) experiments complete!"
echo "Results saved to: $OUTPUT_DIR/"
echo ""
