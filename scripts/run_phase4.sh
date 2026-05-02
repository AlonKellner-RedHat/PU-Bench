#!/bin/bash
# Phase 4: Prior-Based Methods Robustness Analysis

OUTPUT_DIR="results_phase4"
SHUFFLE_SEED=77777  # Unique seed for Phase 4
NUM_WORKERS=12

echo "============================================"
echo "Phase 4: Prior Robustness Grid Analysis"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Datasets: 7 (MNIST, FashionMNIST, IMDB, 20News, Mushrooms, Spambase, Connect4)"
echo "  Methods: 10 (8 baseline prior-based + 2 VPU variants)"
echo "    Baseline: nnpu, nnpusb, lbe, distpu, selfpu, p3mixe, p3mixc, robustpu"
echo "    VPU: vpu_mean_prior, vpu_nomixup_mean_prior"
echo "  Seeds: 1 [42]"
echo "  Label frequencies (c): 3 [0.01, 0.5, 0.99]"
echo "  True priors (π): 7 [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]"
echo "  Method_prior values: 13 [ninths, 0, 1, auto, true]"
echo "  Total: ~19,110 experiments"
echo "  Workers: ${NUM_WORKERS}"
echo "  Estimated time: ~8-11 hours (with 12 workers)"
echo ""
echo "Goal: Identify optimal robust prior for each method type"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p logs/phase4

# All 10 prior-based methods
BASELINE_METHODS="nnpu,nnpusb,lbe,distpu,selfpu,p3mixe,p3mixc,robustpu"
VPU_METHODS="vpu_mean_prior,vpu_nomixup_mean_prior"

ALL_METHODS="${BASELINE_METHODS},${VPU_METHODS}"

# Config files for 7 Phase 4 datasets (space-separated)
CONFIGS="config/phase4/mnist_phase4.yaml config/phase4/fashionmnist_phase4.yaml config/phase4/imdb_phase4.yaml config/phase4/20news_phase4.yaml config/phase4/mushrooms_phase4.yaml config/phase4/spambase_phase4.yaml config/phase4/connect4_phase4.yaml"

# Launch all workers in parallel (direct background jobs instead of GNU parallel to avoid temp disk issues)
echo "Launching ${NUM_WORKERS} parallel workers..."

pids=()
for worker_id in $(seq 0 $((NUM_WORKERS - 1))); do
    echo "Starting Phase 4 worker ${worker_id}..."

    .venv/bin/python run_train.py \
        --dataset-config $CONFIGS \
        --methods "$ALL_METHODS" \
        --output-dir "$OUTPUT_DIR" \
        --shuffle-seed "$SHUFFLE_SEED" \
        --num-workers "$NUM_WORKERS" \
        --worker-id "$worker_id" \
        --resume \
        > "logs/phase4/worker_${worker_id}.log" 2>&1 &

    pids+=($!)
done

echo "All workers launched. Waiting for completion..."

# Wait for all workers to complete
for pid in "${pids[@]}"; do
    wait $pid
done

echo ""
echo "All Phase 4 experiments complete!"
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "  1. Analyze optimal method_prior per method type"
echo "  2. Compare 2/3 hypothesis across all methods"
echo "  3. Identify method-specific optimal priors"
echo ""
