#!/bin/bash
# Phase 3: Complete Seed 42 Only
# Focus on finishing the last 8 experiments for seed 42

OUTPUT_DIR="results_phase3"
SHUFFLE_SEED=42  # Only seed 42
NUM_WORKERS=8

echo "============================================"
echo "Phase 3: Completing Seed 42 Only"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Datasets: 7 (MNIST, FashionMNIST, IMDB, 20News, Mushrooms, Spambase, Connect4)"
echo "  Methods: All VPU variants + baselines"
echo "  Seeds: 1 [42] ONLY"
echo "  Label frequencies (c): 7 [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]"
echo "  True priors (π): 7 [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]"
echo "  method_prior: All 6 variants (auto, 0.353, 0.5, 0.69, 1.0)"
echo ""
echo "Current Status:"
echo "  Seed 42: $(find results_phase3/seed_42 -name "*.json" 2>/dev/null | wc -l) / ~2,058"
echo "  Missing: ~8 experiments"
echo ""
echo "Goal: Complete all seed 42 experiments before proceeding to other seeds"
echo ""
echo "Workers: ${NUM_WORKERS}"
echo ""
echo "Note: --resume flag will skip already-completed experiments"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p logs/phase3_seed42_only

# All methods
METHODS="vpu,vpu_nomixup,vpu_mean_prior,vpu_nomixup_mean_prior,oracle_bce,pn_naive"

# Temporarily modify configs to only use seed 42
# Create temporary config files
TEMP_CONFIGS=""
for config in config/phase3/*.yaml; do
    temp_config="${config%.yaml}_seed42_temp.yaml"
    # Copy config and modify random_seeds to only [42]
    sed 's/random_seeds: \[42, 456, 789, 1024, 2048\]/random_seeds: [42]/' "$config" > "$temp_config"
    TEMP_CONFIGS="$TEMP_CONFIGS $temp_config"
done

# Function to run a single worker
function run_worker() {
    worker_id=$1

    echo "Starting Phase 3 Seed 42 worker ${worker_id}..."

    python run_train.py \
        --dataset-config $TEMP_CONFIGS \
        --methods "$METHODS" \
        --output-dir "$OUTPUT_DIR" \
        --shuffle-seed "$SHUFFLE_SEED" \
        --num-workers "$NUM_WORKERS" \
        --worker-id "$worker_id" \
        --resume \
        > "logs/phase3_seed42_only/worker_${worker_id}.log" 2>&1

    echo "Worker ${worker_id} completed!"
}

# Export function and variables for parallel
export -f run_worker
export OUTPUT_DIR SHUFFLE_SEED NUM_WORKERS METHODS TEMP_CONFIGS

# Run all workers in parallel
echo "Launching ${NUM_WORKERS} parallel workers..."
parallel -j "$NUM_WORKERS" run_worker ::: $(seq 0 $((NUM_WORKERS - 1)))

# Clean up temporary configs
rm -f config/phase3/*_seed42_temp.yaml

echo ""
echo "Seed 42 completion check:"
echo "  Total files: $(find results_phase3/seed_42 -name "*.json" 2>/dev/null | wc -l)"
echo "  Expected: ~2,058"
echo ""
echo "Seed 42 method_prior breakdown:"
echo "  Base:   $(find results_phase3/seed_42 -name "*.json" ! -name "*methodprior*" 2>/dev/null | wc -l) / 343"
echo "  auto:   $(find results_phase3/seed_42 -name "*methodprior_auto.json" 2>/dev/null | wc -l) / 343"
echo "  0.353:  $(find results_phase3/seed_42 -name "*methodprior0.353.json" 2>/dev/null | wc -l) / 343"
echo "  0.5:    $(find results_phase3/seed_42 -name "*methodprior0.5.json" 2>/dev/null | wc -l) / 343"
echo "  0.69:   $(find results_phase3/seed_42 -name "*methodprior0.69.json" 2>/dev/null | wc -l) / 343"
echo "  1.0:    $(find results_phase3/seed_42 -name "*methodprior1.json" 2>/dev/null | wc -l) / 343"
echo ""
echo "All Seed 42 experiments complete!"
echo "Results saved to: $OUTPUT_DIR/seed_42/"
echo ""
