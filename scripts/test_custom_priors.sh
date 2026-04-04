#!/bin/bash
# Test custom priors on problematic datasets

echo "========================================"
echo "Testing Custom Priors on Problematic Datasets"
echo "========================================"
echo ""

mkdir -p logs/test_priors
mkdir -p test_results

# Test configurations
declare -a TESTS=(
    "Connect4:0.1"
    "Connect4:0.5"
    "Connect4:0.9"
    "CIFAR10:0.5"
    "AlzheimerMRI:0.1"
    "AlzheimerMRI:0.5"
    "AlzheimerMRI:0.9"
)

PASSED=0
FAILED=0

for test in "${TESTS[@]}"; do
    IFS=':' read -r dataset prior <<< "$test"
    dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    config="config/test/${dataset_lower}_test_prior${prior}.yaml"

    echo "----------------------------------------"
    echo "Testing ${dataset} with prior=${prior}"
    echo "Config: ${config}"
    echo "----------------------------------------"

    # Run single training experiment
    python run_train.py \
        --dataset-config "$config" \
        --methods "vpu_nomixup" \
        --output-dir "test_results" \
        2>&1 | tee "logs/test_priors/${dataset_lower}_prior${prior}.log"

    # Check exit status
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ ${dataset} prior=${prior} SUCCESS"
        PASSED=$((PASSED + 1))

        # Verify the result JSON exists and has the expected prior
        result_file=$(find test_results -name "*.json" -newer logs/test_priors/${dataset_lower}_prior${prior}.log | head -n 1)
        if [ -n "$result_file" ]; then
            # Extract actual measured prior from JSON
            actual_prior=$(python3 -c "import json; data = json.load(open('$result_file')); print(data['runs']['vpu_nomixup']['dataset']['train']['prior'])" 2>/dev/null)
            if [ -n "$actual_prior" ]; then
                echo "  Measured training prior: ${actual_prior}"
                # Check if within ±0.05 of target
                tolerance_check=$(python3 -c "print('OK' if abs($actual_prior - $prior) < 0.05 else 'WARN')")
                if [ "$tolerance_check" = "OK" ]; then
                    echo "  ✓ Prior matches target (within ±0.05)"
                else
                    echo "  ⚠ Prior differs from target by more than ±0.05"
                fi
            fi
        fi
    else
        echo "✗ ${dataset} prior=${prior} FAILED"
        FAILED=$((FAILED + 1))
        echo ""
        echo "Error details from log:"
        tail -n 20 "logs/test_priors/${dataset_lower}_prior${prior}.log"
        echo ""
    fi
    echo ""
done

echo "========================================"
echo "Test Summary"
echo "========================================"
echo "PASSED: ${PASSED}"
echo "FAILED: ${FAILED}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✓ All tests passed!"
    exit 0
else
    echo "✗ Some tests failed. Check logs in logs/test_priors/"
    exit 1
fi
