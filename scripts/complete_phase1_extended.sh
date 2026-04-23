#!/bin/bash
# Complete Phase 1 Extended with missing methods

OUTPUT_DIR="results_phase1_extended"
SHUFFLE_SEED=54321
NUM_WORKERS=6

echo "============================================"
echo "Phase 1 Extended: Complete Missing Methods"
echo "============================================"
echo ""
echo "Missing: 1,478 method runs"
echo "  - cgenpu: 210"
echo "  - holisticpu: 210"
echo "  - lagam: 210"
echo "  - pan: 210"
echo "  - pulcpbf: 210"
echo "  - pulda: 210"
echo "  - puet: 30"
echo "  - vaepu: 188"
echo ""
echo "Using fixed --resume logic to fill missing methods"
echo ""

mkdir -p logs/phase1_extended_complete

# All base methods
BASE_METHODS="nnpu,nnpusb,bbepu,lbe,puet,distpu,pulda,selfpu,p3mixe,p3mixc"
BASE_METHODS+=",robustpu,holisticpu,lagam,pulcpbf,vaepu,pan,cgenpu,pn_naive,oracle_bce"

# VPU variants
VPU_METHODS="vpu,vpu_nomixup"

# Mean-prior methods (will run with all priors via config)
VPU_MP_METHODS="vpu_nomixup_mean_prior,vpu_mean_prior"

ALL_METHODS="${BASE_METHODS},${VPU_METHODS},${VPU_MP_METHODS}"

# Config files for 7 Phase 1 datasets
CONFIGS=(
    "config/comprehensive/mnist_comprehensive.yaml"
    "config/comprehensive/fashionmnist_comprehensive.yaml"
    "config/comprehensive/imdb_comprehensive.yaml"
    "config/comprehensive/20news_comprehensive.yaml"
    "config/comprehensive/mushrooms_comprehensive.yaml"
    "config/comprehensive/spambase_comprehensive.yaml"
    "config/comprehensive/connect4_comprehensive.yaml"
)

# Run with parallel workers
function run_worker() {
    worker_id=$1

    echo "Starting Phase 1 Extended worker ${worker_id}..."

    python -u run_train.py \
        --dataset-config "${CONFIGS[@]}" \
        --methods "$ALL_METHODS" \
        --output-dir "$OUTPUT_DIR" \
        --shuffle-seed "$SHUFFLE_SEED" \
        --num-workers "$NUM_WORKERS" \
        --worker-id "$worker_id" \
        --resume \
        > "logs/phase1_extended_complete/worker_${worker_id}.log" 2>&1 &

    echo $! > "logs/phase1_extended_complete/worker_${worker_id}.pid"
}

# Export for parallel
export -f run_worker
export OUTPUT_DIR SHUFFLE_SEED NUM_WORKERS ALL_METHODS
export -a CONFIGS

# Launch workers
echo "Launching ${NUM_WORKERS} parallel workers..."
for worker_id in $(seq 0 $((NUM_WORKERS - 1))); do
    run_worker $worker_id
done

echo ""
echo "Workers launched in background. Monitor with:"
echo "  tail -f logs/phase1_extended_complete/worker_0.log"
echo ""
echo "Check progress with:"
echo "  python scripts/count_phase1_method_runs.py"
