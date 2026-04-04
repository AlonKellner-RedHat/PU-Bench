#!/bin/bash
# Monitor progress of cartesian product experiments

RESULTS_DIR="results_cartesian"
EXPECTED_TOTAL=1080

echo "========================================"
echo "Cartesian Experiments Progress"
echo "========================================"
echo ""

# Count completed experiments
COMPLETED=$(find "$RESULTS_DIR" -name "*.json" -type f | wc -l | tr -d ' ')
PERCENT=$(echo "scale=1; $COMPLETED * 100 / $EXPECTED_TOTAL" | bc)

echo "Completed: $COMPLETED / $EXPECTED_TOTAL ($PERCENT%)"
echo ""

# Count by dataset
echo "By Dataset:"
for dataset in MNIST FashionMNIST IMDB 20News Mushrooms Spambase; do
    count=$(find "$RESULTS_DIR" -name "${dataset}*.json" -type f | wc -l | tr -d ' ')
    dataset_percent=$(echo "scale=1; $count * 100 / 180" | bc)
    printf "  %-15s: %3d / 180 (%5.1f%%)\n" "$dataset" "$count" "$dataset_percent"
done

echo ""
echo "Expected dimensions per dataset:"
echo "  Seeds: 3 [42, 456, 789]"
echo "  Label freq (c): 2 [0.1, 0.5]"
echo "  True priors: 5 [0.1, 0.3, 0.5, 0.7, 0.9]"
echo "  Method priors: 6 [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]"
echo "  Total per dataset: 3 × 2 × 5 × 6 = 180"
echo "  Total all datasets: 180 × 6 = 1,080"
echo ""

# Estimate completion time
if [ "$COMPLETED" -gt 0 ]; then
    OLDEST=$(find "$RESULTS_DIR" -name "*.json" -type f -print0 | xargs -0 ls -t | tail -1)
    NEWEST=$(find "$RESULTS_DIR" -name "*.json" -type f -print0 | xargs -0 ls -t | head -1)

    if [ -n "$OLDEST" ] && [ -n "$NEWEST" ]; then
        START_TIME=$(stat -f %B "$OLDEST" 2>/dev/null || stat -c %Y "$OLDEST" 2>/dev/null)
        CURRENT_TIME=$(date +%s)
        ELAPSED=$((CURRENT_TIME - START_TIME))

        if [ "$COMPLETED" -gt 10 ]; then
            AVG_TIME_PER_EXP=$(echo "scale=2; $ELAPSED / $COMPLETED" | bc)
            REMAINING=$((EXPECTED_TOTAL - COMPLETED))
            EST_REMAINING=$((REMAINING * ${AVG_TIME_PER_EXP%.*} / 60))

            echo "Estimated time remaining: ~$EST_REMAINING minutes"
        fi
    fi
fi

echo ""
echo "Recent activity:"
find "$RESULTS_DIR" -name "*.json" -type f -print0 | xargs -0 ls -lt | head -5
