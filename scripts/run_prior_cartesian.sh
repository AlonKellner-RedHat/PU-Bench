#!/bin/bash
# Cartesian Product Prior Experiments - All Datasets

DATASETS=(MNIST FashionMNIST IMDB 20News Mushrooms Spambase)
METHODS="vpu_nomixup_mean_prior"
OUTPUT_DIR="results_cartesian"

echo "========================================"
echo "Cartesian Product Prior Experiments"
echo "========================================"
echo ""
echo "Datasets: ${DATASETS[@]}"
echo "Methods: $METHODS"
echo "Output: $OUTPUT_DIR"
echo ""
echo "Grid:"
echo "  Datasets: 6"
echo "  True priors: [0.1, 0.3, 0.5, 0.7, 0.9]"
echo "  Method priors: [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]"
echo "  Label frequencies: [0.1, 0.5]"
echo "  Seeds: [42, 456, 789]"
echo "  Total: ~1,080 experiments"
echo "  Estimated time: ~2.25 hours (6 parallel workers)"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p logs/prior_cartesian

# Function to run a single dataset
function run_dataset() {
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

    echo "Running ${dataset}..."
    python run_train.py \
        --dataset-config "$config" \
        --methods "$METHODS" \
        --output-dir "$OUTPUT_DIR" \
        --resume \
        > "logs/prior_cartesian/${config_name}.log" 2>&1

    echo "Completed ${dataset}"
}

# Export function and variables for parallel
export -f run_dataset
export OUTPUT_DIR
export METHODS

# Run all datasets in parallel (6 workers)
parallel -j 6 run_dataset ::: "${DATASETS[@]}"

echo ""
echo "All experiments complete!"
echo "Results saved to: $OUTPUT_DIR/"
