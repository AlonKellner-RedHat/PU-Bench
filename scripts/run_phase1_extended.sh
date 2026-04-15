#!/bin/bash
# Phase 1 Extended: 10 seeds, 25 methods, val_max_f1 early stopping

OUTPUT_DIR="results_phase1_extended"
SHUFFLE_SEED=54321  # Different from Phase 2
NUM_WORKERS=8  # Increased for faster completion

echo "============================================"
echo "Phase 1 Extended: Comprehensive Baselines"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Datasets: 7 (MNIST, FashionMNIST, IMDB, 20News, Mushrooms, Spambase, Connect4)"
echo "  Methods: 18 (12 baselines + 6 VPU variants)"
echo "  Excluded: LAGAM (crashes), CGENPU/PULCPBF/PAN/HOLISTICPU/PULDA (too slow), VAEPU (hangs on MNIST/FashionMNIST)"
echo "  Seeds: 10"
echo "  C values: 3 [0.1, 0.3, 0.5]"
echo "  Total: 3,780 experiments"
echo "  Workers: ${NUM_WORKERS}"
echo "  Estimated time: ~6-8 hours (with 8 workers)"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p logs/phase1_extended

# All 18 methods (LAGAM removed due to segfault, CGENPU/PULCPBF/PAN/HOLISTICPU/PULDA removed due to extreme slowness, VAEPU removed due to MNIST/FashionMNIST hangs)
# Baselines (12):
BASELINE_METHODS="nnpu,nnpusb,bbepu,lbe,puet,distpu,selfpu,p3mixe,p3mixc"
BASELINE_METHODS+=",robustpu,pn_naive,oracle_bce"

# VPU variants (6):
VPU_METHODS="vpu,vpu_nomixup"

# Note: vpu_nomixup_mean_prior and vpu_mean_prior will be run with method_prior=[auto, 0.5] via config
VPU_MP_METHODS="vpu_nomixup_mean_prior,vpu_mean_prior"

ALL_METHODS="${BASELINE_METHODS},${VPU_METHODS},${VPU_MP_METHODS}"

# Config files for 7 Phase 1 datasets (as space-separated string)
CONFIGS="config/comprehensive/mnist_comprehensive.yaml config/comprehensive/fashionmnist_comprehensive.yaml config/comprehensive/imdb_comprehensive.yaml config/comprehensive/20news_comprehensive.yaml config/comprehensive/mushrooms_comprehensive.yaml config/comprehensive/spambase_comprehensive.yaml config/comprehensive/connect4_comprehensive.yaml"

# Function to run a single worker
function run_worker() {
    worker_id=$1

    echo "Starting Phase 1 Extended worker ${worker_id}..."

    python run_train.py \
        --dataset-config $CONFIGS \
        --methods "$ALL_METHODS" \
        --output-dir "$OUTPUT_DIR" \
        --shuffle-seed "$SHUFFLE_SEED" \
        --num-workers "$NUM_WORKERS" \
        --worker-id "$worker_id" \
        --resume \
        > "logs/phase1_extended/worker_${worker_id}.log" 2>&1

    echo "Worker ${worker_id} completed!"
}

# Export function and variables for parallel
export -f run_worker
export OUTPUT_DIR SHUFFLE_SEED NUM_WORKERS ALL_METHODS CONFIGS

# Run all workers in parallel
echo "Launching ${NUM_WORKERS} parallel workers..."
parallel -j "$NUM_WORKERS" run_worker ::: $(seq 0 $((NUM_WORKERS - 1)))

echo ""
echo "All Phase 1 Extended experiments complete!"
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "To recompute metrics from saved scores:"
echo "  python analysis/recompute_metrics.py --results-dir $OUTPUT_DIR --output ${OUTPUT_DIR}_recomputed.json"
