#!/bin/bash
# Quick test to validate prior robustness setup

echo "Testing prior robustness implementation..."
echo ""

# Create test output directory
TEST_DIR="results_robustness_test"
mkdir -p "$TEST_DIR"

echo "1. Testing single MNIST experiment with method_prior=0.5..."
python run_train.py \
    --dataset-config config/vpu_rerun/robustness/mnist_robustness.yaml \
    --methods vpu_nomixup_mean_prior \
    --output-dir "$TEST_DIR" \
    2>&1 | head -20

echo ""
echo "2. Checking if results were created in $TEST_DIR..."
if [ -d "$TEST_DIR/seed_42" ]; then
    echo "✓ Results directory created"
    echo "Files created:"
    find "$TEST_DIR" -name "*.json" | head -5
else
    echo "✗ No results directory found"
fi

echo ""
echo "3. Checking for methodprior in filenames..."
find "$TEST_DIR" -name "*methodprior*.json" | head -5

echo ""
echo "Test complete. Clean up with: rm -rf $TEST_DIR"
