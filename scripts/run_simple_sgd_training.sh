#!/bin/bash
# Simplified meta-training with SGD, K=1, random initialization
# Configuration: batch_size=12, K=1, SGD, random init

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="meta_training_output"
LOG_FILE="${OUTPUT_DIR}/sgd_training_${TIMESTAMP}.log"

mkdir -p "$OUTPUT_DIR"

echo "======================================================================="
echo "SIMPLIFIED META-TRAINING WITH SGD (1800 iterations)"
echo "======================================================================="
echo "Config: monotonic_basis_meta_simple_sgd.yaml"
echo ""
echo "SIMPLIFIED SETTINGS:"
echo "  - K inner steps: 1 (no second-order gradients)"
echo "  - Batch size: 12"
echo "  - Gradient accumulation: 3 (effective batch=36)"
echo "  - Optimizer: SGD (lr=0.001, momentum=0.9)"
echo "  - Initialization: FULLY RANDOM (no baselines)"
echo "  - Repetitions: 3 (instead of 4)"
echo ""
echo "Output log: ${LOG_FILE}"
echo "======================================================================="
echo ""

# Run meta-training
uv run python scripts/run_meta_learning.py \
    --config config/methods/monotonic_basis_meta_simple_sgd.yaml \
    --checkpoint-dir ./meta_checkpoints \
    --device auto \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "======================================================================="
echo "Training complete!"
echo "======================================================================="
echo "Log saved to: ${LOG_FILE}"
echo "Learned loss saved to: ./learned_losses/"
echo ""
echo "Summary statistics:"
grep "Avg improvement:" "${LOG_FILE}" | awk '
{
  total += $3
  count++
  if ($3 > max || count == 1) max = $3
  if ($3 < min || count == 1) min = $3
}
END {
  printf "Mean: %+.4f | Max: %+.4f | Min: %+.4f\n", total/count, max, min
}'
echo "======================================================================="
