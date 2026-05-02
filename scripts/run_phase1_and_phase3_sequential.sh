#!/bin/bash
# Run Phase 1 Extended and Phase 3 sequentially for new method_prior values

echo "=========================================="
echo "Sequential Phase 1 + Phase 3 Execution"
echo "=========================================="
echo ""
echo "Running Phase 1 Extended first..."
echo ""

# Run Phase 1 Extended
bash scripts/run_phase1_extended.sh > logs/phase1_extended_0666_077.log 2>&1

echo ""
echo "Phase 1 Extended complete!"
echo ""
echo "Starting Phase 3..."
echo ""

# Run Phase 3 after Phase 1 completes
bash scripts/run_phase3.sh > logs/phase3_0666_077.log 2>&1

echo ""
echo "Phase 3 complete!"
echo ""
echo "All experiments finished!"
