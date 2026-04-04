#!/bin/bash
# Comprehensive experiment suite - randomized execution with 4 workers

set -e  # Exit on error

OUTPUT_DIR="results_comprehensive"
SHUFFLE_SEED=12345  # For reproducible randomization
NUM_WORKERS=4

echo "=========================================="
echo "Comprehensive PU Learning Comparison"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Datasets: 9 (MNIST, FashionMNIST, IMDB, 20News, Mushrooms, Spambase, Connect4, CIFAR10, AlzheimerMRI)"
echo "  Methods: 10 (8 VPU + nnPU + Dist-PU)"
echo "  Seeds: 5 [42, 456, 789, 1024, 2048]"
echo "  C values: 3 [0.01, 0.1, 0.5]"
echo "  True priors: 5 [0.1, 0.3, 0.5, 0.7, 0.9]"
echo "  Total: 6,750 experiments"
echo "  Workers: ${NUM_WORKERS}"
echo "  Estimated time: ~21 hours"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p logs/comprehensive

# Method list (6 base methods, will expand to 10 via config)
# vpu_mean_prior and vpu_nomixup_mean_prior will be run with method_prior=[0.5, null, 1.0] via config
METHODS="vpu,vpu_nomixup,vpu_mean_prior,vpu_nomixup_mean_prior,nnpu,distpu"

# Config files for all 9 datasets
CONFIGS="config/comprehensive/mnist_comprehensive.yaml config/comprehensive/fashionmnist_comprehensive.yaml config/comprehensive/imdb_comprehensive.yaml config/comprehensive/20news_comprehensive.yaml config/comprehensive/mushrooms_comprehensive.yaml config/comprehensive/spambase_comprehensive.yaml config/comprehensive/connect4_comprehensive.yaml config/comprehensive/cifar10_comprehensive.yaml config/comprehensive/alzheimermri_comprehensive.yaml"

# Launch workers in background
echo "Launching $NUM_WORKERS workers..."
for worker_id in $(seq 0 $((NUM_WORKERS - 1))); do
    echo "  Starting worker ${worker_id}..."

    python run_train.py \
        --dataset-config $CONFIGS \
        --methods "$METHODS" \
        --output-dir "$OUTPUT_DIR" \
        --shuffle-seed "$SHUFFLE_SEED" \
        --num-workers "$NUM_WORKERS" \
        --worker-id "$worker_id" \
        --resume \
        > "logs/comprehensive/worker_${worker_id}.log" 2>&1 &

    # Save PID
    echo $! > "logs/comprehensive/worker_${worker_id}.pid"
done

echo ""
echo "All workers launched!"
echo "Worker PIDs:"
for worker_id in $(seq 0 $((NUM_WORKERS - 1))); do
    pid=$(cat "logs/comprehensive/worker_${worker_id}.pid" 2>/dev/null || echo "unknown")
    echo "  Worker ${worker_id}: PID ${pid}"
done

echo ""
echo "Monitoring progress (Ctrl+C to stop monitoring, workers will continue)..."
echo "Worker logs: logs/comprehensive/worker_*.log"
echo ""

# Wait for all workers to complete
wait

echo ""
echo "All workers completed!"
echo "Results saved to: $OUTPUT_DIR/"

# Count total results
total_results=$(find "$OUTPUT_DIR" -name "*.json" | wc -l)
echo "Total result files: ${total_results}"

# Clean up PID files
rm -f logs/comprehensive/worker_*.pid
