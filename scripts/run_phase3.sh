#!/bin/bash
# Phase 3: VPU Methods Deep Dive with Prior/Label-Frequency Grid

OUTPUT_DIR="results_phase3"
SHUFFLE_SEED=99999  # Unique seed for Phase 3
NUM_WORKERS=8

echo "============================================"
echo "Phase 3: VPU Prior/Label-Frequency Analysis"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Datasets: 7 (MNIST, FashionMNIST, IMDB, 20News, Mushrooms, Spambase, Connect4)"
echo "  Methods: 6 (VPU variants + Oracle-PN + Naive-PN)"
echo "    - vpu, vpu_nomixup"
echo "    - vpu_mean_prior (auto, 0.5)"
echo "    - vpu_nomixup_mean_prior (auto, 0.5)"
echo "    - oracle_bce, pn_naive"
echo "  Seeds: 1 [42]"
echo "  Label frequencies (c): 7 [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]"
echo "  True priors (π): 7 [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]"
echo "  Total: 2,744 experiments"
echo "  Workers: ${NUM_WORKERS}"
echo "  Estimated time: ~15-20 hours (with 8 workers)"
echo ""
echo "Goal: Analyze VPU performance across full prior/label-frequency space"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p logs/phase3

# VPU methods only
VPU_METHODS="vpu,vpu_nomixup,vpu_mean_prior,vpu_nomixup_mean_prior"

# Oracle and baseline
ORACLE_METHODS="oracle_bce,pn_naive"

ALL_METHODS="${VPU_METHODS},${ORACLE_METHODS}"

# Config files for 7 Phase 3 datasets (space-separated string for export)
CONFIGS="config/phase3/mnist_phase3.yaml config/phase3/fashionmnist_phase3.yaml config/phase3/imdb_phase3.yaml config/phase3/20news_phase3.yaml config/phase3/mushrooms_phase3.yaml config/phase3/spambase_phase3.yaml config/phase3/connect4_phase3.yaml"

# Function to run a single worker
function run_worker() {
    worker_id=$1

    echo "Starting Phase 3 worker ${worker_id}..."

    python run_train.py \
        --dataset-config $CONFIGS \
        --methods "$ALL_METHODS" \
        --output-dir "$OUTPUT_DIR" \
        --shuffle-seed "$SHUFFLE_SEED" \
        --num-workers "$NUM_WORKERS" \
        --worker-id "$worker_id" \
        --resume \
        > "logs/phase3/worker_${worker_id}.log" 2>&1

    echo "Worker ${worker_id} completed!"
}

# Export function and variables for parallel
export -f run_worker
export OUTPUT_DIR SHUFFLE_SEED NUM_WORKERS ALL_METHODS CONFIGS

# Run all workers in parallel
echo "Launching ${NUM_WORKERS} parallel workers..."
parallel -j "$NUM_WORKERS" run_worker ::: $(seq 0 $((NUM_WORKERS - 1)))

echo ""
echo "All Phase 3 experiments complete!"
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "  1. Analyze prior sensitivity: scripts/analyze_phase3_priors.py"
echo "  2. Analyze label-frequency impact: scripts/analyze_phase3_label_freq.py"
echo "  3. Generate heatmaps: scripts/plot_phase3_heatmaps.py"
echo ""
