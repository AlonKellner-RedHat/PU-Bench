#!/bin/bash
# Verify completeness of cartesian product experiments

RESULTS_DIR="results_cartesian"

echo "========================================"
echo "Cartesian Experiments Verification"
echo "========================================"
echo ""

# Expected grid dimensions
SEEDS=(42 456 789)
C_VALUES=(0.1 0.5)
TRUE_PRIORS=(0.1 0.3 0.5 0.7 0.9)
METHOD_PRIORS=(0.1 0.3 0.5 0.7 0.9 1.0)
DATASETS=(MNIST FashionMNIST IMDB 20News Mushrooms Spambase)

EXPECTED_PER_DATASET=180
EXPECTED_TOTAL=1080

# Count actual results
TOTAL_COUNT=$(find "$RESULTS_DIR" -name "*.json" -type f | wc -l | tr -d ' ')

echo "Total experiments: $TOTAL_COUNT / $EXPECTED_TOTAL"
echo ""

# Check each dataset
echo "Completeness by dataset:"
for dataset in "${DATASETS[@]}"; do
    count=$(find "$RESULTS_DIR" -name "${dataset}*.json" -type f | wc -l | tr -d ' ')

    if [ "$count" -eq "$EXPECTED_PER_DATASET" ]; then
        status="✓"
    else
        status="✗"
    fi

    printf "  %s %-15s: %3d / %3d\n" "$status" "$dataset" "$count" "$EXPECTED_PER_DATASET"
done

echo ""

# Identify missing combinations
echo "Checking for missing combinations..."
MISSING=0

for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for c in "${C_VALUES[@]}"; do
            for tp in "${TRUE_PRIORS[@]}"; do
                for mp in "${METHOD_PRIORS[@]}"; do
                    # Construct expected filename
                    expected="${dataset}_case-control_random_c${c}_seed${seed}_trueprior${tp}_methodprior${mp}.json"

                    if [ ! -f "$RESULTS_DIR/seed_${seed}/$expected" ]; then
                        echo "  MISSING: $expected"
                        MISSING=$((MISSING + 1))
                    fi
                done
            done
        done
    done
done

echo ""
if [ "$MISSING" -eq 0 ]; then
    echo "✓ All $EXPECTED_TOTAL experiments completed successfully!"
else
    echo "✗ Missing $MISSING experiments"
fi

echo ""
echo "Data quality checks:"

# Check for corrupted JSON files
CORRUPTED=0
for json_file in $(find "$RESULTS_DIR" -name "*.json" -type f); do
    if ! jq empty "$json_file" 2>/dev/null; then
        echo "  CORRUPTED: $(basename $json_file)"
        CORRUPTED=$((CORRUPTED + 1))
    fi
done

if [ "$CORRUPTED" -eq 0 ]; then
    echo "  ✓ All JSON files valid"
else
    echo "  ✗ $CORRUPTED corrupted files"
fi

# Sample resampling accuracy
echo ""
echo "Resampling accuracy (sample from seed 42, c=0.1):"
for tp in "${TRUE_PRIORS[@]}"; do
    # Try to find an example for this true_prior
    sample=$(find "$RESULTS_DIR/seed_42" -name "*_seed42_trueprior${tp}_methodprior0.5.json" | head -1)

    if [ -n "$sample" ]; then
        target=$(cat "$sample" | jq -r '.runs.vpu_nomixup_mean_prior.hyperparameters.target_prevalence_train')
        actual=$(cat "$sample" | jq -r '.runs.vpu_nomixup_mean_prior.dataset.train.prior')
        error=$(echo "scale=3; ($actual - $target) * 100" | bc)
        printf "  π_target=%3.1f → π_actual=%5.3f (error: %+5.1f%%)\n" "$target" "$actual" "$error"
    fi
done

echo ""
echo "========================================"
