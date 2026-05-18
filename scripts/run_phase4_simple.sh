#!/bin/bash
# Phase 4: Simple parallel execution - run until complete

OUTPUT_DIR="results_phase4"
SHUFFLE_SEED=77777
NUM_WORKERS=12

echo "============================================"
echo "Phase 4: Simple Parallel Execution"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Workers: ${NUM_WORKERS}"
echo "  Output: ${OUTPUT_DIR}/"
echo "  Resume mode: enabled"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p logs/phase4

# Config files for 7 Phase 4 datasets (space-separated)
CONFIGS="config/phase4/mnist_phase4.yaml config/phase4/fashionmnist_phase4.yaml config/phase4/imdb_phase4.yaml config/phase4/20news_phase4.yaml config/phase4/mushrooms_phase4.yaml config/phase4/spambase_phase4.yaml config/phase4/connect4_phase4.yaml"

ALL_METHODS="nnpu,nnpusb,lbe,distpu,selfpu,p3mixe,p3mixc,robustpu,vpu_mean_prior,vpu_nomixup_mean_prior"

# Function to run a single worker
function run_worker() {
    worker_id=$1

    echo "[$(date +%H:%M:%S)] Starting worker ${worker_id}..."

    .venv/bin/python run_train.py \
        --dataset-config $CONFIGS \
        --methods "$ALL_METHODS" \
        --output-dir "$OUTPUT_DIR" \
        --shuffle-seed "$SHUFFLE_SEED" \
        --num-workers "$NUM_WORKERS" \
        --worker-id "$worker_id" \
        --resume \
        > "logs/phase4/worker_${worker_id}_simple.log" 2>&1

    echo "[$(date +%H:%M:%S)] Worker ${worker_id} completed!"
}

# Export function and variables for parallel
export -f run_worker
export OUTPUT_DIR SHUFFLE_SEED NUM_WORKERS ALL_METHODS CONFIGS

# Run all workers in parallel
echo "Launching ${NUM_WORKERS} parallel workers..."
echo "Workers will run until all experiments complete."
echo ""

parallel -j "$NUM_WORKERS" run_worker ::: $(seq 0 $((NUM_WORKERS - 1)))

echo ""
echo "All Phase 4 experiments complete!"
echo "Results saved to: $OUTPUT_DIR/"
