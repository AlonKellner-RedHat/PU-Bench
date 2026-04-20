#!/bin/bash
# Re-run Phase 1 Extended: 20News only (after bug fix)

OUTPUT_DIR="results_phase1_extended"
SHUFFLE_SEED=54321  # Same as Phase 1 Extended
NUM_WORKERS=8

echo "============================================"
echo "Re-running 20News: Phase 1 Extended"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Dataset: 20News (bug fix applied)"
echo "  Methods: 18 (excluding VAE-PU, PUET)"
echo "  Seeds: 10"
echo "  C values: 3 [0.1, 0.3, 0.5]"
echo "  Total: ~540 experiments"
echo "  Workers: ${NUM_WORKERS}"
echo "  Estimated time: ~2-3 hours (with 8 workers)"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p logs/phase1_extended_20news_rerun

# All 18 methods (same as Phase 1 Extended, excluding LAGAM, slow methods, VAE-PU, PUET)
# Baselines (12):
BASELINE_METHODS="nnpu,nnpusb,bbepu,lbe,distpu,selfpu,p3mixe,p3mixc"
BASELINE_METHODS+=",robustpu,pn_naive,oracle_bce"

# VPU variants (6):
VPU_METHODS="vpu,vpu_nomixup"

# Note: vpu_nomixup_mean_prior and vpu_mean_prior will be run with method_prior=[auto, 0.5] via config
VPU_MP_METHODS="vpu_nomixup_mean_prior,vpu_mean_prior"

ALL_METHODS="${BASELINE_METHODS},${VPU_METHODS},${VPU_MP_METHODS}"

# Config file for 20News only
CONFIG="config/comprehensive/20news_comprehensive.yaml"

# Function to run a single worker
function run_worker() {
    worker_id=$1

    echo "Starting 20News re-run worker ${worker_id}..."

    python run_train.py \
        --dataset-config "$CONFIG" \
        --methods "$ALL_METHODS" \
        --output-dir "$OUTPUT_DIR" \
        --shuffle-seed "$SHUFFLE_SEED" \
        --num-workers "$NUM_WORKERS" \
        --worker-id "$worker_id" \
        --resume \
        > "logs/phase1_extended_20news_rerun/worker_${worker_id}.log" 2>&1

    echo "Worker ${worker_id} completed!"
}

# Export function and variables for parallel
export -f run_worker
export OUTPUT_DIR SHUFFLE_SEED NUM_WORKERS ALL_METHODS CONFIG

# Run all workers in parallel
echo "Launching ${NUM_WORKERS} parallel workers..."
parallel -j "$NUM_WORKERS" run_worker ::: $(seq 0 $((NUM_WORKERS - 1)))

echo ""
echo "All 20News experiments complete!"
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "  1. Verify results: python analysis/generate_comprehensive_table.py"
echo "  2. Check 20News performance is good across all seeds"
echo ""
