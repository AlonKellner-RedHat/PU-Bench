#!/bin/bash
# Phase 1: Lightweight datasets only (text/tabular) with 4 workers

set -e

OUTPUT_DIR="results_comprehensive"
SHUFFLE_SEED=12345
NUM_WORKERS=4

echo "=========================================="
echo "Phase 1: Lightweight Datasets (Text/Tabular)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Datasets: 7 (MNIST, FashionMNIST, IMDB, 20News, Mushrooms, Spambase, Connect4)"
echo "  Excluded: CIFAR10, AlzheimerMRI (memory-intensive image datasets - deferred to Phase 2)"
echo "  Methods: 10 total via config expansion"
echo "  Seeds: 5 [42, 456, 789, 1024, 2048]"
echo "  C values: 3 [0.01, 0.1, 0.5]"
echo "  True priors: 5 [0.1, 0.3, 0.5, 0.7, 0.9]"
echo "  Workers: ${NUM_WORKERS}"
echo "  Expected: ~9,450 experiments"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p logs/comprehensive

METHODS="vpu,vpu_nomixup,vpu_mean_prior,vpu_nomixup_mean_prior,nnpu,distpu"

# Only lightweight datasets (exclude CIFAR10 and AlzheimerMRI)
CONFIGS="config/comprehensive/mnist_comprehensive.yaml config/comprehensive/fashionmnist_comprehensive.yaml config/comprehensive/imdb_comprehensive.yaml config/comprehensive/20news_comprehensive.yaml config/comprehensive/mushrooms_comprehensive.yaml config/comprehensive/spambase_comprehensive.yaml config/comprehensive/connect4_comprehensive.yaml"

echo "Launching $NUM_WORKERS workers..."
for worker_id in $(seq 0 $((NUM_WORKERS - 1))); do
    echo "  Starting worker ${worker_id}..."

    nohup uv run python run_train.py \
        --dataset-config $CONFIGS \
        --methods "$METHODS" \
        --output-dir "$OUTPUT_DIR" \
        --shuffle-seed "$SHUFFLE_SEED" \
        --num-workers "$NUM_WORKERS" \
        --worker-id "$worker_id" \
        --resume \
        > "logs/comprehensive/phase1_worker_${worker_id}.log" 2>&1 &

    echo $! > "logs/comprehensive/phase1_worker_${worker_id}.pid"
done

echo ""
echo "Phase 1 workers launched!"
echo "Worker PIDs:"
for worker_id in $(seq 0 $((NUM_WORKERS - 1))); do
    pid=$(cat "logs/comprehensive/phase1_worker_${worker_id}.pid")
    echo "  Worker ${worker_id}: PID ${pid}"
done

echo ""
echo "Monitor with: bash scripts/monitor_phase1.sh"
echo "Logs: logs/comprehensive/phase1_worker_*.log"
