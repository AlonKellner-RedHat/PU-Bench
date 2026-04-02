#!/bin/bash
# Monitor prior robustness experiment progress

RESULTS_DIR="results_robustness"
EXPECTED_TOTAL=486

echo "========================================"
echo "Prior Robustness Experiments - Progress"
echo "========================================"
echo ""

# Count completed experiments
if [ -d "$RESULTS_DIR" ]; then
    COMPLETED=$(find "$RESULTS_DIR" -name "*.json" -type f | wc -l | tr -d ' ')
    PERCENT=$(awk "BEGIN {printf \"%.1f\", ($COMPLETED/$EXPECTED_TOTAL)*100}")

    echo "Progress: $COMPLETED / $EXPECTED_TOTAL experiments ($PERCENT%)"
    echo ""

    # Count by seed
    echo "By seed:"
    for seed in 42 456 789; do
        if [ -d "$RESULTS_DIR/seed_$seed" ]; then
            COUNT=$(find "$RESULTS_DIR/seed_$seed" -name "*.json" -type f | wc -l | tr -d ' ')
            echo "  seed_$seed: $COUNT files"
        fi
    done
    echo ""

    # Recent completions (last 5)
    echo "Recent completions:"
    find "$RESULTS_DIR" -name "*.json" -type f -print0 | \
        xargs -0 ls -lt | head -6 | tail -5 | \
        awk '{print "  " $9}' | sed 's|.*/||'

    echo ""

    # Check logs for active processes
    echo "Active processes:"
    ps aux | grep -E "run_train.py|run_prior_robustness" | grep -v grep | wc -l | xargs echo "  Python processes:"

else
    echo "Results directory not created yet."
    echo "Experiments may still be starting..."
fi

echo ""
echo "To view live logs:"
echo "  tail -f logs/prior_robustness/*.log"
echo ""
echo "To check this status again:"
echo "  bash scripts/check_robustness_progress.sh"
