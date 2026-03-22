#!/bin/bash
# Phase 1: Quick Validation of Batch Size Sensitivity
# Tests VPU vs VPU-Mean at batch sizes 16, 64, 256

set -e

echo "========================================================================"
echo "Batch Size Sensitivity: Phase 1 Quick Validation"
echo "========================================================================"
echo ""
echo "Hypothesis: VPU is more sensitive to batch size than VPU-Mean"
echo ""
echo "Batch sizes: 16, 64, 256 (default)"
echo "Datasets: MNIST, IMDB"
echo "Methods: vpu (batch 16/64/256), vpu_mean (batch 16/64/256)"
echo "Seeds: 42, 123"
echo ""
echo "Total runs: 3 batch sizes × 2 datasets × 2 methods × 2 seeds = 24 runs"
echo "Estimated time: ~1 hour"
echo "========================================================================"
echo ""

python run_train.py \
  --dataset-config \
    config/datasets_batch_size/batch_validation_mnist.yaml \
    config/datasets_batch_size/batch_validation_imdb.yaml \
  --methods \
    vpu \
    vpu_batch16 \
    vpu_batch64 \
    vpu_mean \
    vpu_mean_batch16 \
    vpu_mean_batch64 \
  --resume

echo ""
echo "========================================================================"
echo "Quick validation complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "1. Run: python scripts/analyze_batch_size_sensitivity.py"
echo "2. Check if VPU shows performance degradation at small batch sizes"
echo "3. If validated, proceed to full batch size sweep"
echo ""
