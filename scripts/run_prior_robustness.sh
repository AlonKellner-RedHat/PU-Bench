#!/bin/bash
# Prior Robustness Experiments for vpu_nomixup_mean_prior

DATASETS=(MNIST FashionMNIST IMDB 20News Mushrooms Spambase)
METHODS="vpu_nomixup vpu_nomixup_mean vpu_nomixup_mean_prior"
OUTPUT_DIR="results_robustness"

echo "========================================"
echo "Prior Robustness Experiments"
echo "========================================"
echo ""
echo "Datasets: ${DATASETS[@]}"
echo "Methods: $METHODS"
echo "Output Directory: $OUTPUT_DIR (isolated from results/)"
echo "Total base configs: 54 (6 datasets × 3 seeds × 3 c_values)"
echo "Method runs: 486 (54 × 9)"
echo ""

mkdir -p logs/prior_robustness
mkdir -p "$OUTPUT_DIR"

# Define function before using it
function run_dataset() {
    dataset=$1
    config="config/vpu_rerun/robustness/${dataset,,}_robustness.yaml"

    echo "Running ${dataset} robustness experiments..."
    python run_train.py \
        --dataset-config "$config" \
        --methods $METHODS \
        --output-dir "$OUTPUT_DIR" \
        --resume \
        > "logs/prior_robustness/${dataset,,}.log" 2>&1

    echo "Completed ${dataset}"
}

# Export function for parallel
export -f run_dataset
export OUTPUT_DIR
export METHODS

# Parallel execution using GNU parallel
parallel -j 3 run_dataset ::: "${DATASETS[@]}"

echo ""
echo "All robustness experiments complete!"
echo "Results saved to: $OUTPUT_DIR/"
