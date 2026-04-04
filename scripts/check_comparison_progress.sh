#!/bin/bash
# Monitor progress of comparison experiments

RESULTS_DIR="results_cartesian"

echo "========================================"
echo "4-Way Method Comparison Progress"
echo "========================================"
echo ""

echo "Expected experiments:"
echo "  Per method: 180 (6 datasets × 3 seeds × 2 c × 5 true_priors)"
echo "  Total new: 360"
echo ""

# Count results by method
echo "By Method:"

# vpu_nomixup (baseline) - new
NOMIXUP=$(find "$RESULTS_DIR" -name "*.json" -exec grep -l '"vpu_nomixup"' {} \; 2>/dev/null | grep -v "vpu_nomixup_mean" | wc -l | tr -d ' ')
echo "  vpu_nomixup (baseline):          $NOMIXUP / 180"

# vpu_nomixup_mean (method_prior=1.0) - already exists
NOMIXUP_MEAN=$(find "$RESULTS_DIR" -name "*methodprior1.json" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "  vpu_nomixup_mean (prior=1.0):    $NOMIXUP_MEAN / 180 (from cartesian)"

# vpu_nomixup_mean_prior auto - new
MEAN_PRIOR_AUTO=$(find "$RESULTS_DIR" -name "*methodprior_auto.json" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "  vpu_nomixup_mean_prior (auto):   $MEAN_PRIOR_AUTO / 180"

# vpu_nomixup_mean_prior 0.5 - already exists
MEAN_PRIOR_05=$(find "$RESULTS_DIR" -name "*methodprior0.5.json" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "  vpu_nomixup_mean_prior (0.5):    $MEAN_PRIOR_05 / 180 (from cartesian)"

echo ""
echo "New experiments needed:"
NEW_NEEDED=$((360 - NOMIXUP - MEAN_PRIOR_AUTO))
echo "  Remaining: $NEW_NEEDED / 360"

if [ "$NEW_NEEDED" -le 0 ]; then
    echo ""
    echo "✅ All comparison experiments complete!"
else
    PERCENT=$(echo "scale=1; ($NOMIXUP + $MEAN_PRIOR_AUTO) * 100 / 360" | bc)
    echo "  Progress: ${PERCENT}%"
fi

echo ""
echo "Recent activity (new baseline):"
find "$RESULTS_DIR" -name "*.json" -exec grep -l '"vpu_nomixup"' {} \; 2>/dev/null | grep -v "vpu_nomixup_mean" | xargs ls -lt 2>/dev/null | head -3

echo ""
echo "Recent activity (new auto):"
find "$RESULTS_DIR" -name "*methodprior_auto.json" -type f | xargs ls -lt 2>/dev/null | head -3
