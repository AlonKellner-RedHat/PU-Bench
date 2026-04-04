#!/bin/bash
# Run missing experiments for 4-way method comparison

DATASETS=(MNIST FashionMNIST IMDB 20News Mushrooms Spambase)
OUTPUT_DIR="results_cartesian"

echo "========================================"
echo "Method Comparison Experiments"
echo "========================================"
echo ""
echo "Adding missing methods to cartesian grid:"
echo "  1. vpu_nomixup (baseline)"
echo "  2. vpu_nomixup_mean_prior with auto (true prior)"
echo ""
echo "Grid per method:"
echo "  Datasets: 6"
echo "  Seeds: [42, 456, 789]"
echo "  Label freq (c): [0.1, 0.5]"
echo "  True priors: [0.1, 0.3, 0.5, 0.7, 0.9]"
echo "  Total per method: 180 experiments"
echo "  Total new: 360 experiments"
echo ""

mkdir -p logs/comparison

# Function to run vpu_nomixup baseline
function run_baseline() {
    dataset=$1

    case "$dataset" in
        MNIST) config_name="mnist" ;;
        FashionMNIST) config_name="fashionmnist" ;;
        IMDB) config_name="imdb" ;;
        20News) config_name="20news" ;;
        Mushrooms) config_name="mushrooms" ;;
        Spambase) config_name="spambase" ;;
    esac

    config="config/prior_cartesian/${config_name}_baseline.yaml"

    echo "Running ${dataset} (vpu_nomixup)..."
    python run_train.py \
        --dataset-config "$config" \
        --methods "vpu_nomixup" \
        --output-dir "$OUTPUT_DIR" \
        --resume \
        > "logs/comparison/${config_name}_baseline.log" 2>&1

    echo "Completed ${dataset} baseline"
}

# Function to run vpu_nomixup_mean_prior with auto
function run_auto() {
    dataset=$1

    case "$dataset" in
        MNIST) config_name="mnist" ;;
        FashionMNIST) config_name="fashionmnist" ;;
        IMDB) config_name="imdb" ;;
        20News) config_name="20news" ;;
        Mushrooms) config_name="mushrooms" ;;
        Spambase) config_name="spambase" ;;
    esac

    config="config/prior_cartesian/${config_name}_cartesian.yaml"

    echo "Running ${dataset} (vpu_nomixup_mean_prior auto)..."
    python run_train.py \
        --dataset-config "$config" \
        --methods "vpu_nomixup_mean_prior" \
        --output-dir "$OUTPUT_DIR" \
        --resume \
        > "logs/comparison/${config_name}_auto.log" 2>&1

    echo "Completed ${dataset} auto"
}

# Export functions for parallel
export -f run_baseline
export -f run_auto
export OUTPUT_DIR

echo "Starting baseline experiments (vpu_nomixup)..."
parallel -j 6 run_baseline ::: "${DATASETS[@]}"

echo ""
echo "Starting auto prior experiments (vpu_nomixup_mean_prior with auto)..."
parallel -j 6 run_auto ::: "${DATASETS[@]}"

echo ""
echo "All comparison experiments complete!"
echo "Results saved to: $OUTPUT_DIR/"
