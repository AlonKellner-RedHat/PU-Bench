#!/bin/bash

echo "========================================================================"
echo "Extreme Batch Size Sensitivity Testing"
echo "========================================================================"
echo ""
echo "Testing hypothesis: Does performance continue improving at very small batches?"
echo ""
echo "Previous results showed batch 256 is best by AP/AUC."
echo "Testing smaller batch sizes: 2, 4, 8 (skipping 1 - incompatible with MixUp)"
echo ""
echo "Batch sizes: 2, 4, 8"
echo "Datasets: MNIST, IMDB"
echo "Methods: vpu, vpu_mean (at each batch size)"
echo "Seeds: 42, 123"
echo "Total runs: 3 batch sizes × 2 datasets × 2 methods × 2 seeds = 24 runs"
echo "Estimated time: ~1.5-2 hours"
echo "========================================================================"
echo ""

python run_train.py \
  --dataset-config \
    config/datasets_batch_size/batch_validation_mnist.yaml \
    config/datasets_batch_size/batch_validation_imdb.yaml \
  --methods \
    vpu_batch2 vpu_batch4 vpu_batch8 \
    vpu_mean_batch2 vpu_mean_batch4 vpu_mean_batch8 \
  --resume
