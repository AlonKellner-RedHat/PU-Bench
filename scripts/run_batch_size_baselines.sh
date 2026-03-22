#!/bin/bash

# Re-run batch 256 baselines to get AP and max_f1 metrics
# This complements the batch size validation by regenerating the baseline (batch 256)
# results with the new threshold-independent metrics.

echo "========================================================================"
echo "Re-running Batch 256 Baselines with AP/max_f1 Metrics"
echo "========================================================================"
echo ""
echo "This will regenerate results for vpu and vpu_mean at batch size 256"
echo "with the new threshold-independent metrics (AP, max_f1)"
echo ""
echo "Datasets: MNIST, IMDB"
echo "Methods: vpu (batch 256), vpu_mean (batch 256)"
echo "Seeds: 42, 123"
echo "Total runs: 2 methods × 2 datasets × 2 seeds = 8 runs"
echo "Estimated time: ~20-30 minutes"
echo "========================================================================"
echo ""

python run_train.py \
  --dataset-config \
    config/datasets_batch_size/batch_validation_mnist.yaml \
    config/datasets_batch_size/batch_validation_imdb.yaml \
  --methods \
    vpu \
    vpu_mean \
  --resume
