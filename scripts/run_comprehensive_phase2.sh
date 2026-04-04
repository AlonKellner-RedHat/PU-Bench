#!/bin/bash
# Phase 2: Image datasets only with 2 workers (memory-intensive)

set -e

OUTPUT_DIR="results_comprehensive"
SHUFFLE_SEED=12345
NUM_WORKERS=2  # Reduced to avoid OOM

echo "=========================================="
echo "Phase 2: Image Datasets (Memory-Intensive)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Datasets: CIFAR10 (RGB images 3×32×32), AlzheimerMRI (grayscale 1×128×128)"
echo "  Methods: 10 total via config expansion"
echo "  Seeds: 5 [42, 456, 789, 1024, 2048]"
echo "  C values: 3 [0.01, 0.1, 0.5]"
echo "  True priors: 5 [0.1, 0.3, 0.5, 0.7, 0.9]"
echo "  Workers: ${NUM_WORKERS} (reduced to avoid OOM)"
echo "  Expected: ~2,700 experiments"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p logs/comprehensive

METHODS="vpu,vpu_nomixup,vpu_mean_prior,vpu_nomixup_mean_prior,nnpu,distpu"

# Only image datasets (CIFAR10 and AlzheimerMRI)
CONFIGS="config/comprehensive/cifar10_comprehensive.yaml config/comprehensive/alzheimermri_comprehensive.yaml"

echo "Launching $NUM_WORKERS workers for image datasets..."
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
        > "logs/comprehensive/phase2_worker_${worker_id}.log" 2>&1 &

    echo $! > "logs/comprehensive/phase2_worker_${worker_id}.pid"
done

echo ""
echo "Phase 2 workers launched!"
echo "Worker PIDs:"
for worker_id in $(seq 0 $((NUM_WORKERS - 1))); do
    pid=$(cat "logs/comprehensive/phase2_worker_${worker_id}.pid")
    echo "  Worker ${worker_id}: PID ${pid}"
done

echo ""
echo "Monitor with: bash scripts/monitor_phase2.sh"
echo "Logs: logs/comprehensive/phase2_worker_*.log"
