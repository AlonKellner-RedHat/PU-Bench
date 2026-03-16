#!/bin/bash
# Full meta-training with 1800 iterations (one epoch over checkpoint pool)
# Configuration: batch_size=8, K=4, gradient_accumulation=4, diverse_baselines

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="meta_training_output"
LOG_FILE="${OUTPUT_DIR}/training_${TIMESTAMP}.log"

mkdir -p "$OUTPUT_DIR"

echo "======================================================================="
echo "FULL META-TRAINING RUN (1800 iterations)"
echo "======================================================================="
echo "Config: monotonic_basis_meta_safe.yaml"
echo "  - Batch size: 8"
echo "  - K inner steps: 4"
echo "  - Gradient accumulation: 4"
echo "  - Total iterations: 1800"
echo "  - Initialization: diverse_baselines (4 reps: uPU + PUDRa + VPU)"
echo "  - Optimizer: AdamW (lr=5e-5, wd=0.01)"
echo ""
echo "Output log: ${LOG_FILE}"
echo "======================================================================="
echo ""

# Run meta-training
uv run python scripts/run_meta_learning.py \
    --config config/methods/monotonic_basis_meta_safe.yaml \
    --checkpoint-dir ./meta_checkpoints \
    --device auto \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "======================================================================="
echo "Training complete!"
echo "======================================================================="
echo "Log saved to: ${LOG_FILE}"
echo "Learned loss saved to: ./learned_losses/"
echo "======================================================================="
