#!/bin/bash
# Prior Robustness Experiments for vpu_nomixup_mean_prior

METHODS="vpu_nomixup vpu_nomixup_mean vpu_nomixup_mean_prior"
OUTPUT_DIR="results_robustness"

echo "========================================"
echo "Prior Robustness Experiments"
echo "========================================"
echo ""
echo "Datasets: MNIST FashionMNIST IMDB 20News Mushrooms Spambase"
echo "Methods: $METHODS"
echo "Output Directory: $OUTPUT_DIR (isolated from results/)"
echo "Total base configs: 54 (6 datasets × 3 seeds × 3 c_values)"
echo "Method runs: 486 (54 × 9)"
echo ""

mkdir -p logs/prior_robustness
mkdir -p "$OUTPUT_DIR"

# Function to convert dataset name to lowercase config filename
get_config_name() {
    case "$1" in
        MNIST) echo "mnist" ;;
        FashionMNIST) echo "fashionmnist" ;;
        IMDB) echo "imdb" ;;
        20News) echo "20news" ;;
        Mushrooms) echo "mushrooms" ;;
        Spambase) echo "spambase" ;;
        *) echo "$(echo "$1" | tr '[:upper:]' '[:lower:]')" ;;
    esac
}

# Define function before using it
run_dataset() {
    dataset=$1
    config_name=$(get_config_name "$dataset")
    config="config/vpu_rerun/robustness/${config_name}_robustness.yaml"

    echo "Running ${dataset} robustness experiments..."
    python run_train.py \
        --dataset-config "$config" \
        --methods $METHODS \
        --output-dir "$OUTPUT_DIR" \
        --resume \
        > "logs/prior_robustness/${config_name}.log" 2>&1

    echo "Completed ${dataset}"
}

# Export function for parallel
export -f run_dataset
export -f get_config_name
export OUTPUT_DIR
export METHODS

# Parallel execution using GNU parallel
parallel -j 3 run_dataset ::: MNIST FashionMNIST IMDB 20News Mushrooms Spambase

echo ""
echo "All robustness experiments complete!"
echo "Results saved to: $OUTPUT_DIR/"
