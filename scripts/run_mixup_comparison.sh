#!/bin/bash
# Run mixup variants for 8-way comparison (4 no-mixup + 4 with-mixup)

DATASETS=(MNIST FashionMNIST IMDB 20News Mushrooms Spambase)
OUTPUT_DIR="results_cartesian"

echo "========================================"
echo "Mixup Variants Experiments"
echo "========================================"
echo ""
echo "Adding 4 mixup variants to complete 8-way comparison:"
echo "  1. vpu (classic VPU with mixup)"
echo "  2. vpu_mean (with mean + mixup, no prior)"
echo "  3. vpu_mean_prior (auto) (with mean + mixup + true prior)"
echo "  4. vpu_mean_prior (0.5) (with mean + mixup + prior=0.5)"
echo ""
echo "Grid per method:"
echo "  Datasets: 6"
echo "  Seeds: [42, 456, 789]"
echo "  Label freq (c): [0.1, 0.5]"
echo "  True priors: [0.1, 0.3, 0.5, 0.7, 0.9]"
echo "  Total per method: 180 experiments"
echo "  Total new: 720 experiments"
echo ""

mkdir -p logs/mixup_comparison

# Function to run vpu baseline with mixup
function run_vpu() {
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

    echo "Running ${dataset} (vpu - classic with mixup)..."
    python run_train.py \
        --dataset-config "$config" \
        --methods "vpu" \
        --output-dir "$OUTPUT_DIR" \
        --resume \
        > "logs/mixup_comparison/${config_name}_vpu.log" 2>&1

    echo "Completed ${dataset} vpu"
}

# Function to run vpu_mean (no prior parameter)
function run_vpu_mean() {
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

    echo "Running ${dataset} (vpu_mean - mean+mixup, no prior)..."
    python run_train.py \
        --dataset-config "$config" \
        --methods "vpu_mean" \
        --output-dir "$OUTPUT_DIR" \
        --resume \
        > "logs/mixup_comparison/${config_name}_vpu_mean.log" 2>&1

    echo "Completed ${dataset} vpu_mean"
}

# Function to run vpu_mean_prior with auto
function run_vpu_mean_prior_auto() {
    dataset=$1

    case "$dataset" in
        MNIST) config_name="mnist" ;;
        FashionMNIST) config_name="fashionmnist" ;;
        IMDB) config_name="imdb" ;;
        20News) config_name="20news" ;;
        Mushrooms) config_name="mushrooms" ;;
        Spambase) config_name="spambase" ;;
    esac

    config="config/prior_cartesian/${config_name}_mixup_auto.yaml"

    echo "Running ${dataset} (vpu_mean_prior auto - mixup+auto)..."
    python run_train.py \
        --dataset-config "$config" \
        --methods "vpu_mean_prior" \
        --output-dir "$OUTPUT_DIR" \
        --resume \
        > "logs/mixup_comparison/${config_name}_vpu_mean_prior_auto.log" 2>&1

    echo "Completed ${dataset} vpu_mean_prior auto"
}

# Function to run vpu_mean_prior with 0.5
function run_vpu_mean_prior_05() {
    dataset=$1

    case "$dataset" in
        MNIST) config_name="mnist" ;;
        FashionMNIST) config_name="fashionmnist" ;;
        IMDB) config_name="imdb" ;;
        20News) config_name="20news" ;;
        Mushrooms) config_name="mushrooms" ;;
        Spambase) config_name="spambase" ;;
    esac

    config="config/prior_cartesian/${config_name}_mixup_05.yaml"

    echo "Running ${dataset} (vpu_mean_prior 0.5 - mixup+0.5)..."
    python run_train.py \
        --dataset-config "$config" \
        --methods "vpu_mean_prior" \
        --output-dir "$OUTPUT_DIR" \
        --resume \
        > "logs/mixup_comparison/${config_name}_vpu_mean_prior_05.log" 2>&1

    echo "Completed ${dataset} vpu_mean_prior 0.5"
}

# Export functions for parallel
export -f run_vpu
export -f run_vpu_mean
export -f run_vpu_mean_prior_auto
export -f run_vpu_mean_prior_05
export OUTPUT_DIR

echo "Phase 1: vpu (classic VPU with mixup)..."
parallel -j 6 run_vpu ::: "${DATASETS[@]}"

echo ""
echo "Phase 2: vpu_mean (mean+mixup, no prior)..."
parallel -j 6 run_vpu_mean ::: "${DATASETS[@]}"

echo ""
echo "Phase 3: vpu_mean_prior (auto)..."
parallel -j 6 run_vpu_mean_prior_auto ::: "${DATASETS[@]}"

echo ""
echo "Phase 4: vpu_mean_prior (0.5)..."
parallel -j 6 run_vpu_mean_prior_05 ::: "${DATASETS[@]}"

echo ""
echo "All mixup comparison experiments complete!"
echo "Results saved to: $OUTPUT_DIR/"
